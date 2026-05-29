"""
Cross-Entropy Evaluation for Topology Optimization Models

Computes binary cross-entropy between predicted soft probabilities and
ground-truth binary topologies on the OOD test set.

For BFM (our method): use 1-step inference (num_inference_steps=1) to get
soft probabilities — do NOT apply Bernoulli sampling, which would push
values toward 0/1 and give misleading CE.

For other flow-matching models (DiT, UDiT, PDE-T): the entropy is computed
from the final soft output before thresholding.

Usage:
    # From callback-saved images (uses pixel values as probabilities):
    python -m eval.evaluate_cross_entropy \
        --images_dir logs/<experiment>/imagesALL/<dataset>

    # From raw numpy predictions:
    python -m eval.evaluate_cross_entropy \
        --predictions results/predictions.npy \
        --targets results/targets.npy
"""
import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def binary_cross_entropy(pred, target, epsilon=1e-7):
    """Compute per-sample binary cross-entropy."""
    pred = np.clip(np.array(pred, dtype=np.float64), epsilon, 1 - epsilon)
    target = np.array(target, dtype=np.float64).flatten()
    return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))


def load_predictions_from_images(images_dir):
    """Load prediction probabilities and GT from callback-saved images."""
    pred_files = sorted([f for f in os.listdir(images_dir) if f.endswith('_pred.png')])
    print(f"Found {len(pred_files)} prediction images")

    preds, refs = [], []
    for f in tqdm(pred_files, desc="Loading images"):
        idx = f.replace('_pred.png', '')
        pred = np.array(Image.open(os.path.join(images_dir, f)).convert('L')) / 255.0
        ref = np.array(Image.open(os.path.join(images_dir, f'{idx}_ref.png')).convert('L')) / 255.0
        preds.append(pred.flatten())
        refs.append(ref.flatten())

    return preds, refs


def evaluate_cross_entropy(predictions, targets):
    """Compute cross-entropy statistics.
    
    Args:
        predictions: list of soft probability arrays
        targets: list of binary target arrays
        
    Returns:
        dict with CE statistics
    """
    ces = []
    for pred, target in tqdm(zip(predictions, targets), total=len(predictions),
                             desc="Computing CE"):
        ces.append(binary_cross_entropy(pred, target))

    ces = np.array(ces)
    return {
        'per_sample': ces,
        'mean': float(ces.mean()),
        'median': float(np.median(ces)),
        'std': float(ces.std()),
        'min': float(ces.min()),
        'max': float(ces.max()),
        'n_samples': len(ces),
    }


def print_results(results, name=""):
    """Pretty-print CE results."""
    print(f"\n{'='*60}")
    print(f"CROSS-ENTROPY — {name}" if name else "CROSS-ENTROPY")
    print(f"{'='*60}")
    print(f"  Samples: {results['n_samples']}")
    print(f"  Mean CE:   {results['mean']:.4f}")
    print(f"  Median CE: {results['median']:.4f}")
    print(f"  Std CE:    {results['std']:.4f}")
    print(f"  Min CE:    {results['min']:.4f}")
    print(f"  Max CE:    {results['max']:.4f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate cross-entropy')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--images_dir', type=str,
                       help='Directory with *_pred.png/*_ref.png images')
    group.add_argument('--predictions', type=str,
                       help='Path to predictions .npy file')

    parser.add_argument('--targets', type=str, default=None,
                        help='Path to targets .npy (if using --predictions)')
    parser.add_argument('--output_dir', type=str, default='eval/results',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.images_dir:
        preds, refs = load_predictions_from_images(args.images_dir)
        name = os.path.basename(args.images_dir)
    else:
        preds = np.load(args.predictions, allow_pickle=True)
        refs = np.load(args.targets, allow_pickle=True)
        name = os.path.basename(args.predictions).replace('.npy', '')

    results = evaluate_cross_entropy(preds, refs)
    print_results(results, name)

    np.save(os.path.join(args.output_dir, 'cross_entropy_per_sample.npy'),
            results['per_sample'])
    np.savez(os.path.join(args.output_dir, 'cross_entropy_stats.npz'),
             mean=results['mean'], median=results['median'],
             std=results['std'], n_samples=results['n_samples'])
    print(f"Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
