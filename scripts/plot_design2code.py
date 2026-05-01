"""
Grouped bar chart: Design2Code benchmark comparison across all models.
Ground truth data from presentation (4B model).
Saves paper/figures/design2code_results.{pdf,png}
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    "GPT-4o",
    "Design2Code-18B",
    "LLaVA 1.6-7B",
    "DeepSeek-VL-7B",
    "Qwen 3 VL 4B\n(Base)",
    "VisionCoder\n(RL only)",
    "VisionCoder\n(SFT+RL)",
]

# Block / Text / Position / Color / CLIP
DATA = np.array([
    [93.0, 98.2, 85.5, 84.1, 90.4],   # GPT-4o
    [78.5, 96.4, 74.3, 67.0, 85.8],   # Design2Code-18B
    [50.4, 87.9, 69.1, 63.4, 84.6],   # LLaVA 1.6-7B
    [39.7, 77.0, 64.6, 63.8, 84.5],   # DeepSeek-VL-7B
    [10.1, 10.8,  8.8,  9.1, 77.2],   # Qwen 3 VL 4B base
    [79.5, 88.4, 70.6, 69.8, 85.9],   # VisionCoder RL only
    [80.5, 91.3, 75.6, 70.2, 89.0],   # VisionCoder SFT+RL
])

METRICS = ["Block", "Text", "Position", "Color", "CLIP"]

# Color palette
PALETTE = [
    "#4477AA",  # GPT-4o (blue)
    "#228B22",  # Design2Code-18B (dark green)
    "#AAAAAA",  # LLaVA (gray)
    "#BBBBBB",  # DeepSeek (light gray)
    "#DDDDDD",  # Qwen base (very light)
    "#E88C30",  # VisionCoder RL (orange)
    "#CC2222",  # VisionCoder SFT+RL (red)
]

n_metrics = len(METRICS)
n_models = len(MODELS)
bar_width = 0.10
group_gap = 0.35
x = np.arange(n_metrics) * (n_models * bar_width + group_gap)

fig, ax = plt.subplots(figsize=(12, 5))

for i, (model, color) in enumerate(zip(MODELS, PALETTE)):
    offset = (i - n_models / 2 + 0.5) * bar_width
    bars = ax.bar(
        x + offset,
        DATA[i],
        width=bar_width,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
        label=model.replace("\n", " "),
    )
    # Bold border for our models
    if i >= 5:
        for bar in bars:
            bar.set_edgecolor("black")
            bar.set_linewidth(1.2)

ax.set_xticks(x)
ax.set_xticklabels(METRICS, fontsize=13)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_ylim(0, 106)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="x", bottom=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# Legend
handles = [
    mpatches.Patch(color=PALETTE[i], label=MODELS[i].replace("\n", " "))
    for i in range(n_models)
]
ax.legend(
    handles=handles,
    loc="upper right",
    fontsize=8.5,
    framealpha=0.9,
    ncol=2,
    columnspacing=0.8,
)

ax.set_title(
    "Design2Code Benchmark (484 examples)",
    fontsize=13,
    fontweight="bold",
    pad=10,
)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT_DIR, f"design2code_results.{ext}"), dpi=200, bbox_inches="tight")

print(f"Saved to {OUT_DIR}/design2code_results.{{pdf,png}}")
