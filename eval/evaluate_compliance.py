"""
Structural Compliance Evaluation

Computes compliance error between predicted and ground-truth topologies
using the ATOMS FEM solver (from NITO).

The pipeline:
  1. Load predicted topology images (saved by the training callback)
  2. Convert from image format to solver-compatible format
  3. Run FEM analysis to compute compliance for both pred and GT
  4. Compute relative error statistics

Usage:
    python -m eval.evaluate_compliance \
        --images_dir logs/<experiment>/imagesALL/<dataset> \
        --gt_dir datasets/structural/Test_OOD \
        --output_dir eval/results/<experiment>

    # Or evaluate from raw numpy arrays:
    python -m eval.evaluate_compliance \
        --topologies_pred results/topologies_pred.npy \
        --topologies_gt results/topologies_gt.npy \
        --gt_dir datasets/structural/Test_OOD \
        --output_dir eval/results/<experiment>
"""
import argparse
import os
import sys
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.ERROR)
sys.path.insert(0, os.path.dirname(__file__))

from ATOMS.solver import Solver
from ATOMS.MaterialModels import SingleMaterial
from ATOMS.geometry import generate_structured_mesh
from ATOMS.utils import filter_2D_structured


def image_to_solver_format(img_arr):
    """Convert callback-saved image to solver-compatible 1D array.
    
    The callback saves images via np.flipud(data.T), so we invert:
    recovered = np.flipud(saved_image).T → flatten C-order.
    """
    recovered = np.flipud(img_arr).T
    return recovered.flatten(order='C')


def load_topologies_from_images(images_dir):
    """Load prediction/reference topologies from callback-saved PNG images."""
    pred_files = sorted([f for f in os.listdir(images_dir) if f.endswith('_pred.png')])
    print(f"Found {len(pred_files)} prediction images in {images_dir}")

    preds, refs = [], []
    for f in tqdm(pred_files, desc="Loading images"):
        idx = f.replace('_pred.png', '')
        pred_img = np.array(Image.open(os.path.join(images_dir, f)).convert('L')) / 255.0
        ref_img = np.array(Image.open(os.path.join(images_dir, f'{idx}_ref.png')).convert('L')) / 255.0
        preds.append(image_to_solver_format(pred_img))
        refs.append(image_to_solver_format(ref_img))

    return np.array(preds, dtype=object), np.array(refs, dtype=object)


def compute_compliance(rho, shape, vf, BCs, loads):
    """Compute compliance for a single topology using ATOMS FEM solver."""
    nelx, nely = int(shape[0]), int(shape[1])
    dim = shape.astype(float) / float(shape.max())
    shape_int = np.array([nelx, nely], dtype=int)
    nodes, elements = generate_structured_mesh(dim=dim, nel=shape_int)
    elements = elements.astype(np.int64)

    r_min = 1.5
    filter_kernel = filter_2D_structured(
        elements=elements, nodes=nodes,
        nelx=nelx, nely=nely,
        r_min=r_min * (1.0 / shape.max())
    )

    material = SingleMaterial(E=1.0, nu=0.3, penalty=3.0,
                              volume_fraction=vf, void=1e-9)

    solver = Solver(
        mesh=(nodes, elements),
        filter_kernel=filter_kernel,
        material_model=material,
        structured=True,
        max_iter=0,
        move=0.2,
        solver='cholesky'
    )

    solver.reset_BC()
    solver.reset_F()

    BCs_array = np.atleast_2d(BCs)
    solver.add_BCs(BCs_array[:, 0:2], BCs_array[:, 2:4])

    loads_array = np.atleast_2d(loads)
    solver.add_Forces(loads_array[:, 0:2], loads_array[:, 2:4])

    rho_arr = np.array(rho, dtype=np.float64).reshape(-1, 1)
    _, compliance, _, _, flag = solver.FEA(rho=rho_arr)
    return compliance, flag


def evaluate_compliance(topologies_pred, topologies_gt, shapes, vfs, BCs, loads,
                        outlier_threshold=1000.0):
    """Evaluate compliance error between predicted and GT topologies.
    
    Args:
        topologies_pred: array of predicted topology vectors
        topologies_gt: array of ground-truth topology vectors
        shapes: (N, 2) mesh shapes
        vfs: (N,) volume fractions
        BCs: (N,) boundary conditions (each is array of [x, y, bc_x, bc_y])
        loads: (N,) loads (each is array of [x, y, fx, fy])
        outlier_threshold: exclude samples with relative error > this (%)
        
    Returns:
        dict with compliance arrays and statistics
    """
    n = len(topologies_pred)
    c_pred = np.full(n, np.nan)
    c_gt = np.full(n, np.nan)

    for i in tqdm(range(n), desc="Computing compliance"):
        try:
            cp, flag_p = compute_compliance(
                topologies_pred[i], shapes[i], vfs[i], BCs[i], loads[i])
            cg, flag_g = compute_compliance(
                topologies_gt[i], shapes[i], vfs[i], BCs[i], loads[i])
            c_pred[i] = cp
            c_gt[i] = cg
        except Exception as e:
            print(f"  Sample {i} failed: {e}")

    # Compute relative error
    valid = ~np.isnan(c_pred) & ~np.isnan(c_gt) & (c_gt > 0)
    rel_err = (c_pred[valid] - c_gt[valid]) / c_gt[valid] * 100

    # Filter outliers (following NITO protocol)
    mask = rel_err <= outlier_threshold
    clean_err = rel_err[mask]

    results = {
        'c_pred': c_pred,
        'c_gt': c_gt,
        'rel_err_all': rel_err,
        'rel_err_clean': clean_err,
        'n_total': n,
        'n_valid': int(valid.sum()),
        'n_outliers': int(len(rel_err) - len(clean_err)),
        'mean': float(clean_err.mean()) if len(clean_err) > 0 else float('nan'),
        'median': float(np.median(clean_err)) if len(clean_err) > 0 else float('nan'),
        'std': float(clean_err.std()) if len(clean_err) > 0 else float('nan'),
    }
    return results


def print_results(results, name=""):
    """Pretty-print compliance evaluation results."""
    print(f"\n{'='*60}")
    print(f"COMPLIANCE ERROR — {name}" if name else "COMPLIANCE ERROR")
    print(f"{'='*60}")
    print(f"  Total samples:    {results['n_total']}")
    print(f"  Valid FEM solves: {results['n_valid']}")
    print(f"  Outliers (>1000%): {results['n_outliers']}")
    print(f"  ---")
    print(f"  Mean Err_C:   {results['mean']:.2f}%")
    print(f"  Median Err_C: {results['median']:.2f}%")
    print(f"  Std Err_C:    {results['std']:.2f}%")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate structural compliance error')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--images_dir', type=str,
                       help='Directory with callback-saved *_pred.png/*_ref.png images')
    group.add_argument('--topologies_pred', type=str,
                       help='Path to predicted topologies .npy file')

    parser.add_argument('--topologies_gt', type=str, default=None,
                        help='Path to GT topologies .npy (if using --topologies_pred)')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Directory with BCs, loads, shapes, vfs .npy files')
    parser.add_argument('--output_dir', type=str, default='eval/results',
                        help='Output directory for results')
    parser.add_argument('--outlier_threshold', type=float, default=1000.0,
                        help='Exclude samples with relative error > threshold %%')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load topologies
    if args.images_dir:
        topos_pred, topos_gt = load_topologies_from_images(args.images_dir)
    else:
        topos_pred = np.load(args.topologies_pred, allow_pickle=True)
        topos_gt = np.load(args.topologies_gt, allow_pickle=True)

    # Load problem data
    shapes = np.load(os.path.join(args.gt_dir, 'shapes.npy'))
    vfs = np.load(os.path.join(args.gt_dir, 'vfs.npy'))
    BCs = np.load(os.path.join(args.gt_dir, 'boundary_conditions.npy'), allow_pickle=True)
    loads_data = np.load(os.path.join(args.gt_dir, 'loads.npy'))

    n = min(len(topos_pred), len(shapes))
    print(f"Evaluating {n} samples...")

    results = evaluate_compliance(
        topos_pred[:n], topos_gt[:n], shapes[:n], vfs[:n], BCs[:n], loads_data[:n],
        outlier_threshold=args.outlier_threshold
    )

    name = os.path.basename(args.images_dir) if args.images_dir else "custom"
    print_results(results, name)

    # Save
    np.save(os.path.join(args.output_dir, 'compliances_pred.npy'), results['c_pred'])
    np.save(os.path.join(args.output_dir, 'compliances_gt.npy'), results['c_gt'])
    np.savez(os.path.join(args.output_dir, 'compliance_stats.npz'),
             mean=results['mean'], median=results['median'], std=results['std'],
             n_valid=results['n_valid'], n_outliers=results['n_outliers'])
    print(f"Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
