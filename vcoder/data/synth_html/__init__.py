"""Synthetic HTML generator package."""

from __future__ import annotations

from .cli import main
from .constants import DEFAULT_NUM_SAMPLES, DEFAULT_OUTPUT_DIR, DEFAULT_SEED
from .dataset import generate_dataset

__all__ = [
    "DEFAULT_NUM_SAMPLES",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SEED",
    "generate_dataset",
    "main",
]
