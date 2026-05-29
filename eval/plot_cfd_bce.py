#!/usr/bin/env python3
"""
Bar chart comparing CFD cross-entropy (BCE) across models and conditioning signals.
Uses replicated results from the release repo.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Publication-quality styling (ICML format)
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['STIX Two Text', 'STIXGeneral', 'Times New Roman', 'DejaVu Serif', 'serif']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# CFD BCE results (median, OOD-Hard 500 samples)
# From our replicated experiments
data = {
    'DiT': {
        'sensitivity': 0.640,
        'velocity': 0.907,
        'pressure': 5.240,
    },
    'UDiT': {
        'sensitivity': 0.228,
        'velocity': 0.356,
        'pressure': 1.668,
    },
    'PDE-T': {
        'sensitivity': 0.408,
        'velocity': 0.580,
        'pressure': 3.032,
    },
    'Ours': {
        'sensitivity': 0.224,
        'velocity': 0.382,
        'pressure': 0.462,
    },
}

# Colors for each conditioning signal
colors = {
    'pressure': '#E63946',      # Vibrant red
    'velocity': '#2A9D8F',      # Teal
    'sensitivity': '#457B9D',   # Steel blue
}

# Hatching patterns
hatches = {
    'pressure': '',
    'velocity': '///',
    'sensitivity': '...',
}

fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=150)

models = list(data.keys())
targets = ['pressure', 'velocity', 'sensitivity']
target_labels = ['Pressure', 'Flow Magnitude', 'Sensitivity']

x = np.arange(len(models))
width = 0.22

# Cap bar height for readability; label truncated bars with actual value
y_cap = 12  # ×10⁻¹ display units

for i, (target, label) in enumerate(zip(targets, target_labels)):
    values = [data[model][target] * 10 for model in models]  # Scale to ×10⁻¹
    display_vals = [min(v, y_cap) for v in values]
    bars = ax.bar(x + i * width, display_vals, width, label=label,
                  color=colors[target], edgecolor='black', linewidth=0.6,
                  hatch=hatches[target], alpha=0.9)

    for bar, val, dv in zip(bars, values, display_vals):
        if val > y_cap:
            # Truncated bar: label inside with actual value
            ax.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, y_cap * 0.75),
                        ha='center', va='center', fontsize=6.5, fontweight='bold',
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.7, lw=0))
            # Break indicator at top of truncated bar
            bx = bar.get_x()
            bw = bar.get_width()
            ax.plot([bx, bx + bw], [y_cap - 0.08, y_cap - 0.08], color='white', lw=2, zorder=5)
            ax.plot([bx, bx + bw], [y_cap - 0.35, y_cap - 0.35], color='white', lw=2, zorder=5)
        else:
            ax.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, fontweight='medium')

ax.set_xlabel('Model', fontsize=11, fontweight='medium')
ax.set_ylabel(r'Topology Cross-Entropy ($\times 10^{-1}$)', fontsize=11, fontweight='medium')
ax.set_xticks(x + width)
ax.set_xticklabels(models, fontsize=10)
ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray',
          fancybox=False, borderpad=0.4, bbox_to_anchor=(1.0, 1.05))
ax.set_ylim(0, y_cap)

ax.yaxis.grid(True, linestyle='-', alpha=0.3, linewidth=0.5)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

ax.annotate(r'$\downarrow$ Lower is better', xy=(0.88, 0.82), xycoords='axes fraction',
            ha='center', va='top', fontsize=8, color='gray', style='italic')

plt.tight_layout()
plt.savefig('eval/cfd_bce_comparison.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('eval/cfd_bce_comparison.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: eval/cfd_bce_comparison.png and eval/cfd_bce_comparison.pdf")
