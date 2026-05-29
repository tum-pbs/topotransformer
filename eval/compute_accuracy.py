"""
Compute Pixel Accuracy — Full 50-step ODE sampling evaluation.

For BFM models: runs the full ODE (50 Heun steps) from Bernoulli prior to prediction,
then computes pixel-level accuracy: (pred > 0.5) == (target > 0.5).

Usage:
  python eval/compute_accuracy.py \
    --checkpoint logs/cfd-ours-sensitivity/checkpoint-ema/last.ckpt \
    --config config/experiments/cfd-ours-sensitivity.yaml \
    --test-dataset datasets/cfd-test-threeOutlet-500.hdf5 \
    --train-dataset datasets/cfd-train-10k.hdf5 \
    --seed 13
"""

import argparse
import sys
import os
import numpy as np
import torch
import h5py
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    OmegaConf.update(cfg, 'runtime', {'checkpoint_dir': '/tmp/unused'}, force_add=True)
    OmegaConf.update(cfg, 'paths', {'PBDL_index': './datasets/'}, force_add=True)
    OmegaConf.update(cfg, 'dataset_names', ['dummy_dummy'], force_add=True)
    OmegaConf.resolve(cfg)
    return OmegaConf.to_container(cfg, resolve=True)


def load_model_bfm(config, checkpoint_path, device='cuda'):
    from topotransformer.core.mixed_channels.train_bernoulli_flow import BernoulliFlowDiffusion

    model_params = config['model']['params']
    model = BernoulliFlowDiffusion(
        model=model_params['model'],
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
    return model.to(device).eval()


def get_norm_stats(train_dataset_path):
    with h5py.File(train_dataset_path, 'r') as h:
        mean = h['norm_fields_sca_mean'][:].flatten().astype(np.float32)
        std = h['norm_fields_sca_std'][:].flatten().astype(np.float32)
    return mean, std


def main():
    parser = argparse.ArgumentParser(description='Compute pixel accuracy via full ODE sampling')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--test-dataset', required=True)
    parser.add_argument('--train-dataset', required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--num-steps', type=int, default=50, help='Number of ODE steps')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    config = load_config(args.config)
    sel_channels_input = config['sel_channels_input']
    sel_channels_target = config['sel_channels_target']

    print(f"Loading model from: {args.checkpoint}")
    model = load_model_bfm(config, args.checkpoint, args.device)

    norm_mean, norm_std = get_norm_stats(args.train_dataset)

    # PDE class label
    pde_idx = 28  # CFD

    print(f"Loading test data: {args.test_dataset}")
    with h5py.File(args.test_dataset, 'r') as h:
        nsims = len([k for k in h['sims'].keys()])
        all_accuracies = []

        for batch_start in range(0, nsims, args.batch_size):
            batch_end = min(batch_start + args.batch_size, nsims)
            B = batch_end - batch_start

            conds = []
            targets = []
            for i in range(batch_start, batch_end):
                sim = h[f'sims/sim{i}'][:].astype(np.float32)
                cond_channels = []
                for ch_idx in sel_channels_input:
                    if ch_idx < 0:
                        ch_idx = sim.shape[1] + ch_idx
                    ch_data = (sim[0, ch_idx] - norm_mean[ch_idx]) / norm_std[ch_idx]
                    cond_channels.append(ch_data)
                conds.append(np.stack(cond_channels, axis=0))
                tgt_ch = sel_channels_target[0]
                targets.append((sim[1, tgt_ch] > 0.5).astype(np.float32))

            cond_t = torch.from_numpy(np.array(conds)).to(args.device)
            labels = torch.full((B,), pde_idx, dtype=torch.long, device=args.device)

            with torch.no_grad():
                gen = torch.Generator(device=args.device).manual_seed(args.seed)
                pred = model.predict_step(
                    previous_frame=cond_t,
                    num_inference_steps=args.num_steps,
                    class_labels=labels,
                    output_probabilistic=True,
                    generator=gen,
                )

            pred_np = pred.cpu().numpy().squeeze()
            if pred_np.ndim == 2:
                pred_np = pred_np[None]

            for j in range(B):
                acc = ((pred_np[j] > 0.5).astype(float) == targets[j]).mean()
                all_accuracies.append(acc)

            print(f"  Batch {batch_start}-{batch_end}: "
                  f"acc={np.mean(all_accuracies[-B:])*100:.2f}%")

    print(f"\n{'='*50}")
    print(f"RESULTS: Pixel Accuracy ({args.num_steps}-step ODE)")
    print(f"{'='*50}")
    print(f"  Samples:  {len(all_accuracies)}")
    print(f"  Mean:     {np.mean(all_accuracies)*100:.2f}%")
    print(f"  Median:   {np.median(all_accuracies)*100:.2f}%")
    print(f"  Std:      {np.std(all_accuracies)*100:.2f}%")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
