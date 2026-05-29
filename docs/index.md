---
layout: default
title: Home
---

<div style="text-align: center; margin-bottom: 2em;">
  <h1 style="margin-bottom: 0.2em;">TopoTransformer</h1>
  <p style="font-size: 1.1em; color: #555; max-width: 700px; margin: 0 auto;">
    On the Generalization in Topology Optimization via<br>
    Sensitivity-Conditioned Bernoulli Flow Matching
  </p>
  <p style="margin-top: 0.8em;">
    <strong>Mohammad Rashed</strong>, Duarte F. Valoroso Madeira, Babak Gholami, Caglar Guerbuez, Yunjia Yang, Nils Thuerey
    <br><em>ICML 2026</em>
  </p>
  <p style="margin-top: 1em;">
    <a href="https://github.com/tum-pbs/topotransformer" style="text-decoration: none; padding: 8px 18px; background: #24292e; color: white; border-radius: 6px; font-weight: bold;">GitHub Repository</a>
    &nbsp;&nbsp;
    <a href="#" style="text-decoration: none; padding: 8px 18px; background: #0366d6; color: white; border-radius: 6px; font-weight: bold;">Paper (coming soon)</a>
  </p>
</div>

---

## Abstract

Surrogate models for topology optimization (TO) exhibit highly variable out-of-distribution (OOD) generalization under distribution shifts such as changing loads or boundary conditions, yet the source of this variability remains unclear. We hypothesize that OOD performance is governed by how much information the conditioning signal preserves about the adjoint sensitivity (reduced gradient) that drives classical TO. Modeling the TO pipeline as a causal Markov chain, the Data Processing Inequality establishes that, under this abstraction, the sensitivity field is an information-theoretically optimal conditioning signal for topology prediction. However, computing exact adjoint sensitivities can be expensive or unavailable in practice; we observe that certain physical fields can approximate sensitivities through monotone transformations. To formalize this, we introduce **pseudo-sensitivities** to characterize which fields enable generalization versus those that are information-poor. We then show that a sensitivity-conditioned Bernoulli flow-matching generator empirically confirms these predictions: conditioning on sensitivities yields state-of-the-art OOD performance, while increasingly distant physical fields degrade toward raw parameter conditioning. We further benchmark against competitive baselines, and find the same ordering of conditioning signals and the same OOD trends. Results hold across structural TO benchmarks under load shifts and our new CFD-TO dataset under boundary-condition shifts such as multi-outlet configurations.

---

## Overview

TopoTransformer learns to generate optimized topologies conditioned on physics-derived sensitivity fields using **Bernoulli flow matching** — a binary-valued generative process tailored for topology optimization's discrete nature.

The framework supports multiple physics domains:

| Domain | Input Conditioning | Output |
|--------|-------------------|--------|
| **Fluid** | Sensitivity, pressure, velocity | Optimized channel topology |
| **Structural** | Strain energy density, von Mises stress | Load-bearing structure |
| **Thermal** | Temperature gradient, conductivity | Heat path topology |
| **Electrokinetic** | Electric potential, concentration | Electrode topology |

---

## Key Contributions

<div style="display: flex; flex-wrap: wrap; gap: 1.5em; margin: 1.5em 0;">

<div style="flex: 1; min-width: 280px; padding: 1em; border: 1px solid #e1e4e8; border-radius: 8px;">
<h3>🔄 Bernoulli Flow Matching</h3>
<p>A flow matching formulation operating directly on binary topology fields, avoiding the mismatch between continuous generative models and discrete optimization targets.</p>
</div>

<div style="flex: 1; min-width: 280px; padding: 1em; border: 1px solid #e1e4e8; border-radius: 8px;">
<h3>📐 Sensitivity Conditioning</h3>
<p>Conditions generation on adjoint sensitivity fields, enabling the model to leverage gradient information from the underlying PDE solver for physically-informed topology generation.</p>
</div>

<div style="flex: 1; min-width: 280px; padding: 1em; border: 1px solid #e1e4e8; border-radius: 8px;">
<h3>🌍 Multi-Physics Generalization</h3>
<p>A single architecture handles fluid, structural, thermal, and electrokinetic topology optimization through a unified metadata encoding of PDEs, fields, and boundary conditions.</p>
</div>

</div>

---

## Results

<!-- Placeholder: Add result figures and tables here -->

<div style="text-align: center; padding: 3em 1em; background: #f6f8fa; border-radius: 8px; margin: 1em 0;">
  <p style="color: #999; font-style: italic;">📊 Quantitative results and visualizations coming soon.</p>
</div>

### Fluid Topology Optimization

<!-- Placeholder for fluid results -->
<div style="text-align: center; padding: 2em; background: #f6f8fa; border-radius: 8px; margin: 1em 0;">
  <p style="color: #999; font-style: italic;">🖼️ Fluid topology optimization samples — placeholder</p>
</div>

### Structural Topology Optimization

<!-- Placeholder for structural results -->
<div style="text-align: center; padding: 2em; background: #f6f8fa; border-radius: 8px; margin: 1em 0;">
  <p style="color: #999; font-style: italic;">🖼️ Structural topology optimization samples — placeholder</p>
</div>

---

## Inference-Time User-Controlled Manipulation

A key advantage of conditioning on sensitivity fields is that users can **manipulate the generation at inference time** without retraining. By modifying the conditioning signal, designers can interactively steer the topology optimization process.

### Spatial Blocking

Block specific regions of the design domain to enforce keep-out zones or force material placement — the model respects these constraints while optimizing the remaining topology.

<!-- Placeholder for blocking figure -->
<div style="text-align: center; padding: 2em; background: #f6f8fa; border-radius: 8px; margin: 1em 0;">
  <p style="color: #999; font-style: italic;">🖼️ Region blocking demonstration — placeholder</p>
</div>

### Volume Fraction Control

Continuously control the volume fraction of the generated topology by scaling the sensitivity field, enabling smooth trade-offs between material usage and performance.

<!-- Placeholder for VF control figure -->
<div style="text-align: center; padding: 2em; background: #f6f8fa; border-radius: 8px; margin: 1em 0;">
  <p style="color: #999; font-style: italic;">🖼️ Volume fraction control — placeholder</p>
</div>

### Sensitivity Field Manipulation

Apply spatial biases, gradients, or offsets to the sensitivity field to steer the optimizer toward preferred design regions — enabling designer intent without modifying the underlying physics.

<!-- Placeholder for sensitivity manipulation figure -->
<div style="text-align: center; padding: 2em; background: #f6f8fa; border-radius: 8px; margin: 1em 0;">
  <p style="color: #999; font-style: italic;">🖼️ Sensitivity manipulation (region bias, gradient, inversion) — placeholder</p>
</div>

### Diverse Sampling

Generate multiple diverse topology candidates from the same conditioning signal by varying the random seed, then select the best design or ensemble them.

<!-- Placeholder for multi-sample figure -->
<div style="text-align: center; padding: 2em; background: #f6f8fa; border-radius: 8px; margin: 1em 0;">
  <p style="color: #999; font-style: italic;">🖼️ Multi-sample diverse generation — placeholder</p>
</div>

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/tum-pbs/topotransformer.git
cd topotransformer
pip install pdetransformer
pip install -e .

# Train (example: Bernoulli flow matching on fluid topology)
python main.py \
    --config config/diffusion/pde-mc-s-mse2in1out-sen-material10k-bernoulli-flow.yaml \
             config/models/pde-mc-s-bfm.yaml \
             config/data/data.yaml \
    --name my_experiment --seed 42

# Evaluate only
python main.py --config <config>.yaml --no-train --logdir logs/your_model/
```

See the [full documentation](https://github.com/tum-pbs/topotransformer#readme) for detailed usage.

---

## Reproducing Paper Results

<!-- Placeholder: specific configs and commands for paper experiments -->

<div style="padding: 1.5em; background: #fff8e1; border-left: 4px solid #ffc107; border-radius: 4px; margin: 1em 0;">
  <strong>🚧 Coming soon:</strong> Specific configuration files and commands to reproduce all experiments from the paper, including dataset download instructions and expected metrics.
</div>

---

## Architecture

<!-- Placeholder for architecture diagram -->
<div style="text-align: center; padding: 3em 1em; background: #f6f8fa; border-radius: 8px; margin: 1em 0;">
  <p style="color: #999; font-style: italic;">🏗️ Architecture diagram — placeholder</p>
</div>

The model uses a DiT-style transformer backbone with:
- **Mixed-channel input encoding** — physical fields and topology share the same token space
- **Metadata conditioning** — PDE type, boundary conditions, and physical constants encoded as conditioning vectors
- **Flow matching sampling** — iterative refinement from noise to binary topology via learned velocity fields

---

## Citation

```bibtex
@inproceedings{rashed2026generalization,
    title={On the Generalization in Topology Optimization via
           Sensitivity-Conditioned Bernoulli Flow Matching},
    author={Rashed, Mohammad and Madeira, Duarte F. Valoroso and Gholami, Babak and Guerbuez, Caglar and Yang, Yunjia and Thuerey, Nils},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2026}
}
```

---

## License

This project is released under the [MIT License](https://github.com/tum-pbs/topotransformer/blob/main/LICENSE).

<div style="text-align: center; margin-top: 2em; color: #999; font-size: 0.9em;">
  Technical University of Munich — Physics-based Deep Learning Group
</div>
