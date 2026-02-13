from __future__ import annotations

import os
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk
from PIL import Image


# SYSTEM_PROMPT = (
#     "You are a UI-to-code assistant. Given a screenshot of a website, generate the complete HTML code "
#     "with inline Tailwind CSS that reproduces the visual layout. Output only the HTML code wrapped in "
#     "```html and ``` tags."
# )

SYSTEM_PROMPT = (
    "You are a UI-to-code assistant. Given a screenshot of a website, generate the complete HTML code "
    "with inline CSS that reproduces the visual layout. Output only the HTML code wrapped in "
    "```html and ``` tags."
)

DEFAULT_CACHE_DIR = os.path.expanduser("~/workspace/vision-coder/data/websight_cache")

# Qwen3-VL: patch_size=16, merge_size=2 → factor=32
# 128 * 28 * 28 = 100352 pixels → ~364 vision tokens
DEFAULT_MAX_PIXELS = 100352
DEFAULT_MIN_PIXELS = 3136  # 4 * 28 * 28


def _resize_image(
    image: Image.Image,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_width: int | None = None,
) -> Image.Image:
    """Resize image to fit constraints, preserving aspect ratio.

    If max_width is set, resizes so width <= max_width (takes priority over max_pixels).
    Otherwise, resizes so total pixels <= max_pixels.
    Dimensions are aligned to 32 for Qwen3-VL (patch_size=16, merge_size=2).
    """
    image = image.convert("RGB")
    w, h = image.size

    if max_width and w > max_width:
        scale = max_width / w
        new_w = max(32, int(w * scale) // 32 * 32)
        new_h = max(32, int(h * scale) // 32 * 32)
        return image.resize((new_w, new_h), Image.LANCZOS)

    if w * h <= max_pixels:
        return image
    scale = (max_pixels / (w * h)) ** 0.5
    new_w = max(32, int(w * scale) // 32 * 32)
    new_h = max(32, int(h * scale) // 32 * 32)
    return image.resize((new_w, new_h), Image.LANCZOS)


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
        "image": _resize_image(example["image"]),
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
    max_width: int | None = 512,
) -> Dataset:
    """Load WebSight dataset from local cache, or stream from HuggingFace if not cached.

    Args:
        max_samples: Number of samples to collect (if downloading).
        max_html_chars: Skip HTML longer than this (if downloading).
        seed: Random seed for shuffling (if downloading).
        cache_dir: Directory for the cached dataset.
        max_width: Resize images so width <= this value. None to skip.

    Returns:
        A HuggingFace Dataset with columns: prompt, image, solution.
    """
    if Path(cache_dir).exists():
        print(f"Loading cached dataset from {cache_dir}")
        ds = load_from_disk(cache_dir)
        ds = ds.select(range(min(len(ds), max_samples)))
        if len(ds) < max_samples:
            print(f"Warning: cached dataset has only {len(ds)} samples, but max_samples={max_samples}")
    else:
        print("No cached dataset found, streaming from HuggingFace...")
        ds = download_websight_dataset(max_samples, max_html_chars, seed, cache_dir)

    # Resize images (handles both fresh downloads and older full-res caches)
    ds = ds.map(lambda ex: {"image": _resize_image(ex["image"], max_width=max_width)})
    return ds


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser(description="Test loading the WebSight dataset")
    args.add_argument("--max_samples", type=int, default=1000)
    max_samples = args.parse_args().max_samples

    print(f"Downloading {max_samples} samples from the WebSight dataset...")
    ds = download_websight_dataset(max_samples=max_samples)
    print(f"Samples: {len(ds)}")
    print(f"Columns: {ds.column_names}")
    print(f"Image: {ds[0]['image'].size}")
