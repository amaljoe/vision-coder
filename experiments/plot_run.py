"""
Plot training curves from a GRPO training run.
Reads trainer_state.json from the latest checkpoint and saves plots to <run_dir>/plots/.

Usage:
    python3 experiments/plot_run.py
    python3 experiments/plot_run.py --run_dir outputs/vcoder-grpo-clip
    python3 experiments/plot_run.py --run_dir outputs/vcoder-grpo-clip --checkpoint 300
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f9f9f9",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

SMOOTH_WINDOW = 20  # rolling mean window for noisy curves


def smooth(values, window=SMOOTH_WINDOW):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def smooth_steps(steps, values, window=SMOOTH_WINDOW):
    """Return smoothed (steps, values) — steps trimmed to match convolution output."""
    sv = smooth(values, window)
    # align steps to centre of the window
    offset = window // 2
    ss = steps[offset: offset + len(sv)]
    return ss, sv


def extract(logs, key):
    steps, vals = [], []
    for entry in logs:
        if key in entry:
            steps.append(entry["step"])
            vals.append(entry[key])
    return np.array(steps), np.array(vals)


def plot_with_smooth(ax, steps, vals, label=None, color=None, alpha_raw=0.25):
    kw = dict(color=color) if color else {}
    ax.plot(steps, vals, alpha=alpha_raw, linewidth=0.8, **kw)
    ss, sv = smooth_steps(steps, vals)
    ax.plot(ss, sv, linewidth=1.8, label=label, **kw)


# ── individual figures ─────────────────────────────────────────────────────────

def fig_rewards(logs, plots_dir):
    """Total reward + per-component reward means."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Reward Signals", fontsize=13, fontweight="bold")

    components = [
        ("rewards/boosted_clip_reward/mean", "Boosted CLIP Reward", "#e07b39"),
        ("rewards/format_reward/mean",        "Format Reward",        "#4c8cbf"),
        ("rewards/html_validity_reward/mean", "HTML Validity Reward", "#5aab61"),
        ("rewards/structural_similarity_reward/mean", "Structural Similarity", "#9b59b6"),
    ]

    ax_total = axes[0, 0]
    s, v = extract(logs, "reward")
    plot_with_smooth(ax_total, s, v, label="Total Reward", color="#c0392b")
    # shade ±std
    _, vstd = extract(logs, "reward_std")
    if len(vstd) == len(v):
        ss, sv = smooth_steps(s, v)
        _, svstd = smooth_steps(s, vstd)
        ax_total.fill_between(ss, sv - svstd, sv + svstd, alpha=0.15, color="#c0392b")
    ax_total.set_title("Total Reward")
    ax_total.set_xlabel("Step")
    ax_total.legend(loc="upper left")

    flat_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
    for ax, (key, title, color) in zip(flat_axes, components[1:]):
        s, v = extract(logs, key)
        # raw std if available
        std_key = key.replace("/mean", "/std")
        _, vstd = extract(logs, std_key)
        plot_with_smooth(ax, s, v, label=title, color=color)
        if len(vstd) == len(v):
            ss, sv = smooth_steps(s, v)
            _, svstd = smooth_steps(s, vstd)
            ax.fill_between(ss, sv - svstd, sv + svstd, alpha=0.15, color=color)
        ax.set_title(title)
        ax.set_xlabel("Step")

    # overlay all components on top-right for comparison
    ax_cmp = axes[0, 1]
    ax_cmp.clear()
    ax_cmp.set_title("All Component Rewards (smoothed)")
    ax_cmp.set_xlabel("Step")
    ax_cmp.grid(True, color="white", linewidth=1.2)
    for key, title, color in components:
        s, v = extract(logs, key)
        ss, sv = smooth_steps(s, v)
        ax_cmp.plot(ss, sv, linewidth=1.8, label=title, color=color)
    ax_cmp.legend(fontsize=8)

    # boosted clip on bottom-left
    key, title, color = components[0]
    ax_clip = axes[1, 0]
    ax_clip.clear()
    ax_clip.set_title(title)
    ax_clip.set_xlabel("Step")
    ax_clip.grid(True, color="white", linewidth=1.2)
    s, v = extract(logs, key)
    _, vstd = extract(logs, key.replace("/mean", "/std"))
    plot_with_smooth(ax_clip, s, v, label=title, color=color)
    if len(vstd) == len(v):
        ss, sv = smooth_steps(s, v)
        _, svstd = smooth_steps(s, vstd)
        ax_clip.fill_between(ss, sv - svstd, sv + svstd, alpha=0.15, color=color)

    fig.tight_layout()
    out = plots_dir / "rewards.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def fig_training_dynamics(logs, plots_dir):
    """Loss, grad norm, LR, entropy."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training Dynamics", fontsize=13, fontweight="bold")

    panels = [
        ("loss",          "Loss",              "#e74c3c", axes[0, 0]),
        ("grad_norm",     "Gradient Norm",     "#e67e22", axes[0, 1]),
        ("learning_rate", "Learning Rate",     "#2980b9", axes[1, 0]),
        ("entropy",       "Policy Entropy",    "#27ae60", axes[1, 1]),
    ]

    for key, title, color, ax in panels:
        s, v = extract(logs, key)
        if key == "learning_rate":
            # don't smooth LR — show exact schedule
            ax.plot(s, v, linewidth=1.6, color=color)
        else:
            plot_with_smooth(ax, s, v, color=color)
        ax.set_title(title)
        ax.set_xlabel("Step")

    fig.tight_layout()
    out = plots_dir / "training_dynamics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def fig_completions(logs, plots_dir):
    """Completion lengths and clipped ratio."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Completion Statistics", fontsize=13, fontweight="bold")

    # lengths
    for key, label, color in [
        ("completions/mean_length",           "Mean Length",      "#3498db"),
        ("completions/mean_terminated_length","Mean Terminated",   "#e74c3c"),
        ("completions/max_length",            "Max Length",        "#95a5a6"),
        ("completions/min_length",            "Min Length",        "#bdc3c7"),
    ]:
        s, v = extract(logs, key)
        if "mean" in key:
            plot_with_smooth(axes[0], s, v, label=label, color=color)
        else:
            axes[0].plot(s, v, linewidth=0.8, alpha=0.5, label=label, color=color)
    axes[0].set_title("Completion Lengths")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Tokens")
    axes[0].legend(fontsize=8)

    # clipped ratio
    s, v = extract(logs, "completions/clipped_ratio")
    plot_with_smooth(axes[1], s, v, color="#e67e22")
    axes[1].set_title("Clipped Completions Ratio")
    axes[1].set_xlabel("Step")
    axes[1].set_ylim(bottom=0)

    # step time
    s, v = extract(logs, "step_time")
    plot_with_smooth(axes[2], s, v, color="#8e44ad")
    axes[2].set_title("Step Time (s)")
    axes[2].set_xlabel("Step")

    fig.tight_layout()
    out = plots_dir / "completions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def fig_clip_ratios(logs, plots_dir):
    """GRPO clip ratios (high / low / region)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle("GRPO Clip Ratios", fontsize=13, fontweight="bold")

    panels = [
        ("clip_ratio/high_mean",   "High Clip (mean)",   "#e74c3c"),
        ("clip_ratio/high_max",    "High Clip (max)",    "#e74c3c"),
        ("clip_ratio/low_mean",    "Low Clip (mean)",    "#3498db"),
        ("clip_ratio/low_min",     "Low Clip (min)",     "#3498db"),
        ("clip_ratio/region_mean", "Region Clip (mean)", "#27ae60"),
    ]

    for key, label, color in panels:
        s, v = extract(logs, key)
        if np.any(v != 0):  # skip all-zero series
            ls = "--" if "max" in key or "min" in key else "-"
            ax.plot(s, v, linewidth=1.4, linestyle=ls, label=label, color=color, alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_title("Clip Ratios over Training")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = plots_dir / "clip_ratios.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def fig_overview(logs, plots_dir):
    """Single-page summary: 6 key metrics."""
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Training Overview — vcoder-grpo-clip", fontsize=14, fontweight="bold")

    panels = [
        ("reward",                                   "Total Reward",          "#c0392b"),
        ("rewards/boosted_clip_reward/mean",         "CLIP Reward",           "#e07b39"),
        ("rewards/structural_similarity_reward/mean","Structural Similarity", "#9b59b6"),
        ("loss",                                     "Loss",                  "#e74c3c"),
        ("entropy",                                  "Entropy",               "#27ae60"),
        ("completions/mean_length",                  "Mean Completion Length","#3498db"),
    ]

    for i, (key, title, color) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        s, v = extract(logs, key)
        plot_with_smooth(ax, s, v, color=color)
        ax.set_title(title)
        ax.set_xlabel("Step")

    out = plots_dir / "overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ── main ───────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return checkpoints[-1]


def main():
    parser = argparse.ArgumentParser(description="Plot GRPO training curves")
    parser.add_argument("--run_dir", default="outputs/vcoder-grpo-clip",
                        help="Path to the training run directory")
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="Checkpoint number to read (default: latest)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    if args.checkpoint is not None:
        ckpt_dir = run_dir / f"checkpoint-{args.checkpoint}"
    else:
        ckpt_dir = find_latest_checkpoint(run_dir)

    state_path = ckpt_dir / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"trainer_state.json not found at {state_path}")

    print(f"Reading training state from {state_path}")
    with open(state_path) as f:
        state = json.load(f)

    logs = state["log_history"]
    print(f"  {len(logs)} log entries  |  steps 1–{logs[-1]['step']}")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to {plots_dir}/\n")

    fig_overview(logs, plots_dir)
    fig_rewards(logs, plots_dir)
    fig_training_dynamics(logs, plots_dir)
    fig_completions(logs, plots_dir)
    fig_clip_ratios(logs, plots_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
