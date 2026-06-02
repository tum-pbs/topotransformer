# TopoTransformer

**On the Generalization in Topology Optimization via Sensitivity-Conditioned Bernoulli Flow Matching**

*Mohammad Rashed, Duarte F. Valoroso Madeira, Babak Gholami, Caglar Guerbuez, Yunjia Yang, Nils Thuerey*

**ICML 2026** &nbsp;|&nbsp; [Project Page](https://tum-pbs.github.io/topotransformer/) &nbsp;|&nbsp; [Paper (coming soon)](#)

---

## Overview

TopoTransformer is a framework for generative topology optimization that uses sensitivity-conditioned Bernoulli flow matching. We show that the choice of conditioning signal governs out-of-distribution (OOD) generalization in topology prediction, and that adjoint sensitivities are information-theoretically optimal under a causal Markov abstraction. When exact sensitivities are unavailable, pseudo-sensitivities (physical fields related to true sensitivities via monotone transformations) provide a practical and principled alternative.

## Repository Structure

```
├── main.py                          # Training entry point
├── pyproject.toml                   # Dependencies
├── config/
│   ├── experiments/                 # CFD experiment configs (12 configs)
│   │   ├── cfd-ours-{sensitivity,flowmag,pressure}.yaml
│   │   ├── cfd-dit-{sensitivity,flowmag,pressure}.yaml
│   │   ├── cfd-udit-{sensitivity,flowmag,pressure}.yaml
│   │   └── cfd-pdet-{sensitivity,flowmag,pressure}.yaml
│   ├── structural/                  # Structural experiment configs
│   │   └── structural-v10-{sensitivity,sed,displacement}.yaml
│   └── ...                          # Shared defaults (trainer, callbacks, data)
├── topotransformer/                 # Core package
│   ├── core/mixed_channels/         # Model implementations
│   │   ├── FA_cross.py              # Our cross-attention architecture
│   │   ├── train_bernoulli_flow.py  # Bernoulli flow matching training
│   │   └── train_probabilistic.py   # Baseline diffusion training
│   ├── data/                        # Data loading (HDF5 via PBDL)
│   ├── objectives/                  # Bernoulli flow objective
│   ├── metric/                      # Accuracy and entropy metrics
│   └── callback/                    # Visualization callbacks
├── eval/                            # Evaluation scripts
│   ├── compute_bce.py               # Cross-entropy evaluation
│   ├── evaluate_compliance.py       # Structural compliance evaluation
│   ├── evaluate_cross_entropy.py    # Per-sample CE computation
│   ├── plot_cfd_bce.py              # BCE bar chart (Figure 4)
│   └── NewSimulationMacro_withInit.java  # STAR-CCM+ data generation
└── docs/                            # Project page (GitHub Pages)
```

## Installation

```bash
git clone https://github.com/tum-pbs/topotransformer.git
cd topotransformer

# Install the pdetransformer dependency (provides base transformer, samplers, utilities)
pip install pdetransformer

# Install topotransformer
pip install -e .
```

**Requirements:** Python >= 3.11, PyTorch, CUDA-capable GPU.

## Datasets

Datasets are available for download (links coming soon):

| Dataset | Domain | Samples | Resolution |
|---------|--------|---------|------------|
| CFD-TO  | Turbulent channel flow (RANS k-epsilon) | 10,000 train + 2,000 test | 128 x 128 |
| Structural-TO | 2D compliance (cantilever/bridge) | 47,000 train + 992 OOD test | 64 x 64 |

## Training

Each experiment is defined by a single YAML config that specifies the model, data, conditioning signal, and training hyperparameters:

```bash
# CFD: Our model with sensitivity conditioning (primary result)
python main.py --config config/experiments/cfd-ours-sensitivity.yaml --seed 42

# CFD: DiT baseline with flow magnitude conditioning
python main.py --config config/experiments/cfd-dit-flowmag.yaml --seed 42

# Structural: Sensitivity conditioning
python main.py --config config/structural/structural-v10-sensitivity.yaml --seed 42
```

All configs use the same interface. Override any parameter from the command line:

```bash
python main.py --config config/experiments/cfd-ours-sensitivity.yaml batch_size=64 max_epochs=200
```

## Evaluation

### Cross-Entropy (BCE)

```bash
# Evaluate BCE across all test splits
python eval/compute_bce.py \
    --model_dir logs/cfd-ours-sensitivity \
    --dataset datasets/cfd-test.hdf5 \
    --ema
```

### Structural Compliance

```bash
# Evaluate compliance error on OOD structural test set
python eval/evaluate_compliance.py \
    --model_dir logs/structural-v10-sensitivity \
    --dataset datasets/structuralsq-test-ood-v10.hdf5 \
    --gt_dir datasets/structural/Test_OOD
```

## Main Results

### CFD Topology Optimization (Simulation-Level)

| Model | ID Med. (%) | OOD-Hard Med. (%) | OOD-Hard Acc. (%) |
|-------|------------|-------------------|-------------------|
| DiT   | 1.45       | 4.85              | 70.6              |
| UDiT  | 1.65       | 7.00              | 62.7              |
| PDE-T | 1.82       | 4.35              | 68.6              |
| **Ours** | 2.63    | **2.70**          | **74.5**          |

### Structural Topology Optimization

| Model | Params (M) | Mean Err_C (%) | Med. Err_C (%) |
|-------|-----------|----------------|----------------|
| TopoDiff | 121 | 8.57 | 1.14 |
| NITO | 22 | 9.33 | 2.37 |
| **Ours** | 34 | **5.73** | **0.53** |

## Citation

```bibtex
@misc{rashed2026generalizationtopologyoptimizationsensitivityconditioned,
      title={On the Generalization in Topology Optimization via Sensitivity-Conditioned Bernoulli Flow Matching}, 
      author={Mohammad Rashed and Duarte F. Valoroso Madeira and Babak Gholami and Caglar Guerbuez and Yunjia Yang and Nils Thuerey},
      year={2026},
      eprint={2606.02179},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.02179}, 
}
```

## License

This project is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). See [LICENSE](LICENSE) for details.

## Licensing

- Repository root: LICENSE (CC BY-NC-ND 4.0) — code and datasets unless otherwise stated.
- Landing page and static assets: files under `docs/` are derived from the Academic Project Page Template (Eliahu Horwitz) and are licensed under CC BY-SA 4.0. See `docs/LICENSE-CC-BY-SA-4.0` and `docs/ATTRIBUTION.md` for details.
