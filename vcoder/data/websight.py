from __future__ import annotations

from itertools import islice

from datasets import Dataset, load_dataset


SYSTEM_PROMPT = (
    "You are a UI-to-code assistant. Given a screenshot of a website, generate the complete HTML code "
    "with inline Tailwind CSS that reproduces the visual layout. Output only the HTML code wrapped in "
    "```html and ``` tags."
)


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


def load_websight_dataset(
    max_samples: int = 2000,
    max_html_chars: int = 3000,
    seed: int = 42,
) -> Dataset:
    """Load WebSight dataset via streaming to avoid downloading all 300+ GB.

    Uses streaming + islice to fetch only the needed samples, then
    materializes into a regular Dataset for GRPO training.

    Args:
        max_samples: Number of samples to collect (after filtering).
        max_html_chars: Skip HTML longer than this (to fit completion limit).
        seed: Random seed for shuffling the final dataset.

    Returns:
        A HuggingFace Dataset with columns: prompt, image, solution.
    """
    stream = load_dataset(
        "HuggingFaceM4/WebSight", "v0.2", split="train", streaming=True
    )

    # Stream, filter by HTML length only, and collect up to max_samples
    collected = []
    count = 0
    for example in stream:
        if len(example["text"]) > max_html_chars:
            count += 1
            continue
        collected.append(_make_conversation(example))
        if len(collected) >= max_samples:
            break

    print(f"Skipped {count} samples due to HTML length > {max_html_chars} chars.")

    ds = Dataset.from_list(collected)
    ds = ds.shuffle(seed=seed)
    return ds


if __name__ == "__main__":
    ds = load_websight_dataset(max_samples=10)
    print(ds[0])
