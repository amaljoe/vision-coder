"""
Horizontal bar chart of reward component weights.
Saves paper/figures/reward_weights.{pdf,png}
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

COMPONENTS = [
    ("Format",        0.5,  "Markdown fence + \\texttt{<!DOCTYPE html>}"),
    ("Validity",      0.5,  "html/head/body + tag diversity"),
    ("Structural",    0.5,  "Tag LCS + CSS class Jaccard"),
    ("Position",      1.0,  "Hungarian centroid distance"),
    ("SSIM",          1.3,  "Pixel-level SSIM at 320×240"),
    ("CLIP",          1.3,  "CLIP ViT-B/32 cosine similarity"),
    ("Color",         2.5,  "CIEDE2000 on non-white pixels"),
    ("Text Block",    3.4,  "Hungarian IoU + text similarity"),
]

labels = [c[0] for c in COMPONENTS]
weights = [c[1] for c in COMPONENTS]
descriptions = [c[2] for c in COMPONENTS]

cmap = cm.get_cmap("RdYlGn")
norm_w = [(w - 0.4) / (3.5 - 0.4) for w in weights]
colors = [cmap(v) for v in norm_w]

fig, ax = plt.subplots(figsize=(8, 4.5))

bars = ax.barh(labels, weights, color=colors, edgecolor="white", linewidth=0.8, height=0.65)

for bar, w in zip(bars, weights):
    ax.text(w + 0.05, bar.get_y() + bar.get_height() / 2,
            f"×{w}", va="center", ha="left", fontsize=10, fontweight="bold")

ax.axvline(x=0, color="black", lw=0.8)
ax.set_xlim(0, 4.2)
ax.set_xlabel("Weight", fontsize=11)
ax.set_title(
    f"Reward Component Weights  (Total = {sum(weights):.1f})",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
ax.tick_params(axis="y", labelsize=11)
ax.xaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

ax.invert_yaxis()

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT_DIR, f"reward_weights.{ext}"), dpi=200, bbox_inches="tight")

print(f"Saved to {OUT_DIR}/reward_weights.{{pdf,png}}")
