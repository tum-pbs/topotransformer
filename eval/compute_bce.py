"""
Compute 1-Step BCE (Cross-Entropy) — Primary evaluation metric for paper Tables A1, 3, and Fig 3.

Method:
  1. Sample noise from Bernoulli(0.5) prior
  2. Concatenate [noise, conditioning_signal] → 2-channel input
  3. Forward model at t=eps (single denoising step from noise)
  4. Compute BCE_with_logits(model_output, target) per sample
  5. Report median (= paper's "Cross-Entropy" metric)

Usage:
  # CFD experiment (Ours + sensitivity on OOD-Hard test set)
  python eval/compute_bce.py \
    --checkpoint logs/cfd-ours-sensitivity/checkpoint-ema/last.ckpt \
    --config config/experiments/cfd-ours-sensitivity.yaml \
    --test-dataset datasets/cfd-test-threeOutlet-500.hdf5 \
    --train-dataset datasets/cfd-train-10k.hdf5 \
    --seed 13

  # Structural experiment
  python eval/compute_bce.py \
    --checkpoint logs/structural-sensitivity/checkpoint-ema/last.ckpt \
    --config config/experiments/structural-sensitivity.yaml \
    --test-dataset datasets/structural_test_ood/ \
    --train-dataset datasets/structuralsq-47k-v4.hdf5 \
    --seed 13 --domain structural
"""

import argparse
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import h5py
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_file_refs(d, base_dir='.'):
    """Recursively resolve _file references in config dicts (mimics parse_config)."""
    if not isinstance(d, dict):
        return d
    if '_file' in d:
        file_path = os.path.join(base_dir, d['_file'])
        with open(file_path) as f:
            file_content = yaml.safe_load(f)
        if isinstance(file_content, dict):
            merged = {**file_content}
            for k, v in d.items():
                if k != '_file':
                    if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                        merged[k] = {**merged[k], **v}
                    else:
                        merged[k] = v
            d = merged
    return {k: _resolve_file_refs(v, base_dir) for k, v in d.items()}


def load_config(config_path):
    """Load config YAML, resolve _file references and extract model-relevant settings."""
    from omegaconf import OmegaConf
    from pdetransformer.utils import parse_config

    cfg = OmegaConf.load(config_path)
    cfg = parse_config(cfg)

    # Provide all missing placeholders to allow full resolution
    defaults = {
        'runtime': {'checkpoint_dir': '/tmp/unused', 'resume': False, 'logdir': '/tmp'},
        'paths': {'PBDL_index': './datasets/'},
        'machine': {'devices': 1, 'accelerator': 'gpu', 'strategy': 'auto'},
    }
    for key, val in defaults.items():
        OmegaConf.update(cfg, key, val, force_add=True)
    if 'dataset_names' not in cfg or cfg.dataset_names is None:
        OmegaConf.update(cfg, 'dataset_names', ['dummy_dummy'], force_add=True)

    # Strip callbacks/logger — they may reference unavailable keys
    for key in list(cfg.keys()):
        if key in ('callbacks', 'logger'):
            try:
                OmegaConf.to_container(cfg[key], resolve=True)
            except Exception:
                cfg[key] = {}

    OmegaConf.resolve(cfg)
    return OmegaConf.to_container(cfg, resolve=True)


def load_model_bfm(config, checkpoint_path, device='cuda'):
    """Load Bernoulli Flow Matching model from checkpoint."""
    from topotransformer.core.mixed_channels.train_bernoulli_flow import BernoulliFlowDiffusion

    model_params = config['model']['params']
    inner_model_cfg = model_params['model']

    model = BernoulliFlowDiffusion(
        model=inner_model_cfg,
        learning_rate=config['trainer']['base_learning_rate'],
        eps=config.get('bernoulli_eps', 1e-6),
        prior_type=config.get('prior_type', 'uniform'),
        volume_fraction=config.get('volume_fraction', 0.5),
        volume_fraction_std=config.get('volume_fraction_std', 0.1),
        num_sampling_steps=config.get('num_sampling_steps', 50),
        sampling_method=config.get('sampling_method', 'heun'),
        threshold=config.get('threshold', None),
        denormalize_target=config.get('denormalize_target', False),
        target_channel_idx=config.get('target_channel_idx', 4),
        monitor='val/loss',
        ckpt_path='/tmp/unused',
    )

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.to(device).eval()
    return model


def load_model_singlestep(config, checkpoint_path, device='cuda'):
    """Load SingleStepDiffusion model (DiT, UDiT, PDE-T) from checkpoint."""
    from topotransformer.core.mixed_channels import SingleStepDiffusion

    model_params = config['model']['params']

    model = SingleStepDiffusion(
        model=model_params['model'],
        objective=model_params['objective'],
        monitor=model_params.get('monitor', 'val/loss_epoch'),
        ckpt_path='/tmp/unused',
        image_key=model_params.get('image_key', 0),
        optimizer=model_params.get('optimizer', 'adamw'),
    )

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.to(device).eval()
    return model


def get_norm_stats(train_dataset_path):
    """Load normalization statistics from training dataset HDF5."""
    with h5py.File(train_dataset_path, 'r') as h:
        mean = h['norm_fields_sca_mean'][:].flatten().astype(np.float32)
        std = h['norm_fields_sca_std'][:].flatten().astype(np.float32)
    return mean, std


def load_cfd_test_data(test_dataset_path, sel_channels_input, sel_channels_target, norm_mean, norm_std):
    """Load and normalize CFD test data."""
    with h5py.File(test_dataset_path, 'r') as h:
        nsims = len([k for k in h['sims'].keys()])
        conditions = []
        targets = []

        for i in range(nsims):
            sim = h[f'sims/sim{i}'][:].astype(np.float32)

            # Get conditioning channels (from t=0, normalized)
            cond_channels = []
            for ch_idx in sel_channels_input:
                if ch_idx < 0:
                    ch_idx = sim.shape[1] + ch_idx  # handle negative indexing
                ch_data = (sim[0, ch_idx] - norm_mean[ch_idx]) / norm_std[ch_idx]
                cond_channels.append(ch_data)
            conditions.append(np.stack(cond_channels, axis=0))

            # Get target (from t=1)
            tgt_ch = sel_channels_target[0]
            targets.append(np.clip(sim[1, tgt_ch], 0, 1))

    return np.array(conditions), np.array(targets), nsims


def load_structural_test_data(test_dataset_path, sel_channels_input, sel_channels_target, norm_mean, norm_std):
    """Load structural test data (NPY files or HDF5)."""
    if os.path.isdir(test_dataset_path):
        # NPY directory format
        files = sorted([f for f in os.listdir(test_dataset_path) if f.endswith('.npy')])
        conditions = []
        targets = []
        for f in files:
            data = np.load(os.path.join(test_dataset_path, f)).astype(np.float32)
            # data shape: (channels, H, W) or (2, channels, H, W)
            if data.ndim == 4:
                data = data[0]  # take t=0

            cond_channels = []
            for ch_idx in sel_channels_input:
                ch_data = (data[ch_idx] - norm_mean[ch_idx]) / norm_std[ch_idx]
                cond_channels.append(ch_data)
            conditions.append(np.stack(cond_channels, axis=0))
            targets.append(np.clip(data[sel_channels_target[0]], 0, 1))

        return np.array(conditions), np.array(targets), len(files)
    else:
        # HDF5 format
        return load_cfd_test_data(test_dataset_path, sel_channels_input, sel_channels_target, norm_mean, norm_std)


def compute_bce_bfm(model, conditions, targets, batch_size=256, seed=13, device='cuda',
                    domain='cfd', temperature=1.0):
    """Compute 1-step BCE for BFM model."""
    nsims = len(conditions)
    all_bce = []

    # Structural models were trained without class labels (None)
    # CFD models use pde_idx=28
    use_labels = (domain == 'cfd')

    for batch_start in range(0, nsims, batch_size):
        batch_end = min(batch_start + batch_size, nsims)
        B = batch_end - batch_start

        cond_t = torch.from_numpy(conditions[batch_start:batch_end]).to(device)
        tgt_t = torch.from_numpy(targets[batch_start:batch_end]).unsqueeze(1).to(device)
        labels = torch.full((B,), 28, dtype=torch.long, device=device) if use_labels else None

        with torch.no_grad():
            gen = torch.Generator(device=device).manual_seed(seed)
            x_0 = torch.bernoulli(
                torch.full((B, 1, targets.shape[1], targets.shape[2]), 0.5, device=device),
                generator=gen
            )

            # Input = [x_0, cond] (concatenated channels)
            input_tensor = torch.cat([x_0, cond_t], dim=1)
            t_input = torch.full((B,), model.eps, device=device)

            output = model.forward(input_tensor, t_input, class_labels=labels)
            logits = output.sample if hasattr(output, 'sample') else output

            if temperature != 1.0:
                logits = logits / temperature

            bce = F.binary_cross_entropy_with_logits(logits, tgt_t, reduction='none')
            bce_per_sample = bce.mean(dim=(1, 2, 3))
            all_bce.extend(bce_per_sample.cpu().numpy().tolist())

    return np.array(all_bce)


def compute_bce_singlestep(model, conditions, targets, batch_size=64, seed=13, device='cuda',
                           num_inference_steps=100):
    """Compute BCE for SingleStepDiffusion model (DiT/UDiT/PDE-T) using full ODE sampling."""
    nsims = len(conditions)
    all_bce = []

    for batch_start in range(0, nsims, batch_size):
        batch_end = min(batch_start + batch_size, nsims)
        B = batch_end - batch_start

        cond_t = torch.from_numpy(conditions[batch_start:batch_end]).to(device)
        tgt_t = torch.from_numpy(targets[batch_start:batch_end]).unsqueeze(1).to(device)

        with torch.no_grad():
            gen = torch.Generator(device=device).manual_seed(seed)
            labels = torch.full((B,), 28, dtype=torch.long, device=device)

            # Full multi-step ODE sampling
            pred = model.predict_step(
                cond_t, target_channels=1,
                num_inference_steps=num_inference_steps,
                generator=gen, class_labels=labels
            )

            bce = F.binary_cross_entropy_with_logits(pred, tgt_t, reduction='none')
            bce_per_sample = bce.mean(dim=(1, 2, 3))
            all_bce.extend(bce_per_sample.cpu().numpy().tolist())

    return np.array(all_bce)


def main():
    parser = argparse.ArgumentParser(description='Compute 1-Step BCE (Cross-Entropy) metric')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--config', required=True, help='Path to experiment config YAML')
    parser.add_argument('--test-dataset', required=True, help='Path to test dataset (HDF5 or dir)')
    parser.add_argument('--train-dataset', required=True, help='Path to train dataset (for norm stats)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=13, help='Random seed for noise sampling')
    parser.add_argument('--domain', choices=['cfd', 'structural'], default='cfd',
                        help='Domain (cfd or structural)')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Logit temperature scaling (>1 softens predictions)')
    args = parser.parse_args()

    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    sel_channels_input = config['sel_channels_input']
    sel_channels_target = config['sel_channels_target']

    # Determine model type
    model_target = config['model']['target']
    is_bfm = 'bernoulli_flow' in model_target.lower()

    print(f"Loading model: {model_target}")
    print(f"  Checkpoint: {args.checkpoint}")
    if is_bfm:
        model = load_model_bfm(config, args.checkpoint, args.device)
    else:
        model = load_model_singlestep(config, args.checkpoint, args.device)

    print(f"Loading normalization stats from: {args.train_dataset}")
    norm_mean, norm_std = get_norm_stats(args.train_dataset)

    print(f"Loading test data: {args.test_dataset}")
    if args.domain == 'cfd':
        conditions, targets, nsims = load_cfd_test_data(
            args.test_dataset, sel_channels_input, sel_channels_target, norm_mean, norm_std
        )
    else:
        conditions, targets, nsims = load_structural_test_data(
            args.test_dataset, sel_channels_input, sel_channels_target, norm_mean, norm_std
        )

    print(f"  Loaded {nsims} samples, cond shape: {conditions.shape}, target shape: {targets.shape}")
    print(f"  Conditioning channels: {sel_channels_input}")
    print(f"  Target channel: {sel_channels_target}")
    print(f"  Seed: {args.seed}")
    print(f"  Temperature: {args.temperature}")

    print(f"\nComputing 1-step BCE...")
    if is_bfm:
        bce_values = compute_bce_bfm(model, conditions, targets, args.batch_size, args.seed,
                                     args.device, domain=args.domain, temperature=args.temperature)
    else:
        bce_values = compute_bce_singlestep(model, conditions, targets, args.batch_size, args.seed, args.device)

    print(f"\n{'='*50}")
    print(f"RESULTS: 1-Step BCE (Cross-Entropy)")
    print(f"{'='*50}")
    print(f"  Samples:  {len(bce_values)}")
    print(f"  Mean:     {np.mean(bce_values):.4f}")
    print(f"  Median:   {np.median(bce_values):.4f}  ← Paper metric")
    print(f"  Std:      {np.std(bce_values):.4f}")
    print(f"  Min:      {np.min(bce_values):.4f}")
    print(f"  Max:      {np.max(bce_values):.4f}")
    print(f"{'='*50}")

    return np.median(bce_values)


if __name__ == '__main__':
    main()
