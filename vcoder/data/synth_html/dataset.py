"""Dataset writing/orchestration for synthetic HTML generation."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from vcoder.rendering.html_renderer import render_html_to_image

from .constants import INDEX_FIELDS, PROGRESS_EVERY, SIMPLE_LAYOUT_SHARE
from .templates import _build_html


def _build_mix_plan(num_samples: int, seed: int) -> list[str]:
    simple_count = int(round(num_samples * SIMPLE_LAYOUT_SHARE))
    simple_count = max(0, min(simple_count, num_samples))
    tags = ["simple"] * simple_count + ["complex"] * (num_samples - simple_count)
    random.Random(seed ^ 0x5A17).shuffle(tags)
    return tags


def _write_sample(output_dir: Path, sample_id: str, template: str, html: str) -> dict[str, str]:
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    html_path = sample_dir / "source.html"
    png_path = sample_dir / "render.png"
    html_path.write_text(html, encoding="utf-8")
    render_html_to_image(html).save(png_path)

    return {
        "sample_id": sample_id,
        "template": template,
        "source_html": str(html_path),
        "render_png": str(png_path),
    }


def _write_index(output_dir: Path, rows: list[dict[str, str]]) -> Path:
    index_path = output_dir / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(INDEX_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return index_path


def generate_dataset(output_dir: Path, num_samples: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    mix_plan = _build_mix_plan(num_samples, seed)

    for idx in range(num_samples):
        sample_id = f"sample_{idx:04d}"
        template, html = _build_html(seed + idx, mix_plan[idx])
        rows.append(_write_sample(output_dir, sample_id, template, html))

        if (idx + 1) % PROGRESS_EVERY == 0 or idx + 1 == num_samples:
            print(f"[generate] {idx + 1}/{num_samples}")

    index_path = _write_index(output_dir, rows)
    print(f"Saved dataset to: {output_dir}")
    print(f"Index: {index_path}")
