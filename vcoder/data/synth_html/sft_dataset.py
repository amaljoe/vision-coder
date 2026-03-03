"""Load synthetic screenshot->HTML samples for supervised fine-tuning (SFT)."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from PIL import Image

SYSTEM_PROMPT = (
    "You are a UI-to-code assistant. Given a screenshot of a website, generate the complete HTML code "
    "with inline CSS that reproduces the visual layout. Output only the HTML code wrapped in "
    "```html and ``` tags."
)
USER_PROMPT = "Generate the HTML code that reproduces this website screenshot."


def _sample_sort_key(path: Path) -> tuple[int, int | str]:
    suffix = path.name.split("_")[-1]
    if suffix.isdigit():
        return (0, int(suffix))
    return (1, path.name)


def _resize_image(image: Image.Image, max_width: int | None = 512) -> Image.Image:
    """Resize image so width <= max_width while preserving aspect ratio.

    Dimensions are aligned to 32 to match Qwen3-VL visual tokenization behavior.
    """
    image = image.convert("RGB")
    if not max_width or image.width <= max_width:
        return image
    scale = max_width / image.width
    new_w = max(32, int(image.width * scale) // 32 * 32)
    new_h = max(32, int(image.height * scale) // 32 * 32)
    return image.resize((new_w, new_h), Image.LANCZOS)


def _discover_samples(dataset_dir: Path) -> list[Path]:
    sample_dirs = []
    for path in sorted(dataset_dir.glob("sample_*"), key=_sample_sort_key):
        if not path.is_dir():
            continue
        if (path / "source.html").exists() and (path / "render.png").exists():
            sample_dirs.append(path)
    return sample_dirs


def load_synthetic_sft_dataset(
    dataset_dir: str | Path,
    max_samples: int = 2000,
    seed: int = 42,
    max_width: int | None = 512,
) -> Dataset:
    """Load generated synthetic samples into prompt/image/solution format.

    Returns a Dataset with columns:
        prompt: chat-style messages (system + user with image placeholder)
        image: PIL image (screenshot)
        solution: target HTML string
        sample_id: sample folder name
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Synthetic dataset directory not found: {dataset_path}")

    samples = _discover_samples(dataset_path)
    if not samples:
        raise RuntimeError(f"No valid synthetic samples found in: {dataset_path}")

    rows = []
    for sample_dir in samples[:max_samples]:
        source_html_path = sample_dir / "source.html"
        render_png_path = sample_dir / "render.png"

        with Image.open(render_png_path) as img:
            image = _resize_image(img, max_width=max_width).copy()

        html = source_html_path.read_text(encoding="utf-8")
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]
        rows.append(
            {
                "prompt": conversation,
                "image": image,
                "solution": html,
                "sample_id": sample_dir.name,
            }
        )

    ds = Dataset.from_list(rows)
    return ds.shuffle(seed=seed)
