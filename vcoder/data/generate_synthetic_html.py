"""Backward-compatible entrypoint for synthetic HTML dataset generation."""

from __future__ import annotations

from vcoder.data.synth_html import DEFAULT_NUM_SAMPLES, DEFAULT_OUTPUT_DIR, DEFAULT_SEED, generate_dataset, main

__all__ = [
    "DEFAULT_NUM_SAMPLES",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SEED",
    "generate_dataset",
    "main",
]

if __name__ == "__main__":
    main()
