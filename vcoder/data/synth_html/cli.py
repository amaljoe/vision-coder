"""CLI entrypoint for synthetic HTML generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .constants import DEFAULT_NUM_SAMPLES, DEFAULT_OUTPUT_DIR, DEFAULT_SEED
from .dataset import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic HTML screenshot dataset.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num_samples must be > 0")

    generate_dataset(output_dir=Path(args.output_dir), num_samples=args.num_samples, seed=args.seed)
