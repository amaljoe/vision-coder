from __future__ import annotations

import os
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk


SYSTEM_PROMPT = (
    "You are a UI-to-code assistant. Given a screenshot of a website, generate the complete HTML code "
    "with inline Tailwind CSS that reproduces the visual layout. Output only the HTML code wrapped in "
    "```html and ``` tags."
)

DEFAULT_CACHE_DIR = os.path.expanduser("~/workspace/vision-coder/data/websight_cache")


def _make_conversation(example: dict) -> dict:
    """Convert a raw WebSight row into the GRPO conversation format."""
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Generate the HTML code that reproduces this website screenshot."},
            ],
        },
    ]
    return {
        "prompt": conversation,
        "image": example["image"].convert("RGB"),
        "solution": example["text"],
    }


def download_websight_dataset(
    max_samples: int = 2000,
    max_html_chars: int = 3000,
    seed: int = 42,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Dataset:
    """Stream, filter, and save WebSight dataset to disk. Requires internet/proxy.

    Args:
        max_samples: Number of samples to collect (after filtering).
        max_html_chars: Skip HTML longer than this.
        seed: Random seed for shuffling.
        cache_dir: Directory to save the processed dataset.

    Returns:
        The saved Dataset.
    """
    stream = load_dataset(
        "HuggingFaceM4/WebSight", "v0.2", split="train", streaming=True
    )

    collected = []
    count = 0
    for example in stream:
        if len(example["text"]) > max_html_chars:
            count += 1
            continue
        collected.append(_make_conversation(example))
        if len(collected) >= max_samples:
            break

    print(f"Collected {len(collected)} samples, skipped {count} (HTML > {max_html_chars} chars).")

    ds = Dataset.from_list(collected)
    ds = ds.shuffle(seed=seed)
    ds.save_to_disk(cache_dir)
    print(f"Dataset saved to {cache_dir}")
    return ds


def load_websight_dataset(
    max_samples: int = 2000,
    max_html_chars: int = 3000,
    seed: int = 42,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Dataset:
    """Load WebSight dataset from local cache, or stream from HuggingFace if not cached.

    Args:
        max_samples: Number of samples to collect (if downloading).
        max_html_chars: Skip HTML longer than this (if downloading).
        seed: Random seed for shuffling (if downloading).
        cache_dir: Directory for the cached dataset.

    Returns:
        A HuggingFace Dataset with columns: prompt, image, solution.
    """
    if Path(cache_dir).exists():
        print(f"Loading cached dataset from {cache_dir}")
        return load_from_disk(cache_dir)

    print("No cached dataset found, streaming from HuggingFace...")
    return download_websight_dataset(max_samples, max_html_chars, seed, cache_dir)


if __name__ == "__main__":
    ds = download_websight_dataset(max_samples=50)
    print(f"Samples: {len(ds)}")
    print(f"Columns: {ds.column_names}")
    print(f"Image: {ds[0]['image'].size}")
