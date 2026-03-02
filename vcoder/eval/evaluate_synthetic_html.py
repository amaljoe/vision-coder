"""Evaluate a model on the synthetic screenshot->HTML dataset.

Inputs (per sample):
    <dataset_dir>/sample_XXXX/source.html
    <dataset_dir>/sample_XXXX/render.png

Outputs:
    <output_dir>/report.csv
    <output_dir>/summary.json
    <output_dir>/predictions/sample_XXXX/predicted.html
    <output_dir>/predictions/sample_XXXX/predicted.png
    <output_dir>/gallery/*.png
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
from collections import Counter
from pathlib import Path
from shutil import copyfile
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageOps

from vcoder.rendering.html_renderer import render_html_to_image
from vcoder.utils.image_utils import compute_clip_similarity

DIRECT_PROMPT = (
    "You are an expert web developer. "
    "Given a screenshot, return one self-contained HTML file with inline CSS that reproduces it. "
    "Output only HTML code."
)
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")

DEFAULT_DATASET_DIR = Path("outputs/synth_html")
DEFAULT_OUTPUT_DIR = Path("outputs/synth_eval")
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 180
DEFAULT_DEVICE = "cpu"
DEFAULT_GALLERY_SIZE = 16

REPORT_FIELDS = [
    "sample_id",
    "status",
    "clip_similarity",
    "numeric_exact_match_rate",
    "source_numeric_count",
    "matched_numeric_count",
    "source_html_path",
    "source_render_path",
    "predicted_html_path",
    "predicted_render_path",
    "error",
]
GALLERY_FIELDS = [
    "sample_id",
    "clip_similarity",
    "numeric_exact_match_rate",
    "tile",
    "source_html",
    "predicted_html",
]


def cleanup_response(text: str) -> str:
    fenced = re.search(r"```html\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    fenced_any = re.search(r"```\s*(<!DOCTYPE|<html)(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced_any:
        return (fenced_any.group(1) + fenced_any.group(2)).strip()

    unclosed = re.match(r"```(?:html)?\s*\n?(.*)", text, re.DOTALL | re.IGNORECASE)
    if unclosed:
        return unclosed.group(1).strip()

    return text.strip()


def image_to_data_url(image_path: Path) -> str:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=95)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def discover_model_id(host: str, port: int, timeout: int) -> str:
    url = f"http://{host}:{port}/v1/models"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    items = payload.get("data", [])
    if not items:
        raise RuntimeError(f"No models returned by {url}")

    model_id = items[0].get("id")
    if not model_id:
        raise RuntimeError(f"Model list from {url} had no id field")
    return model_id


def query_model(
    host: str,
    port: int,
    model_id: str,
    image_path: Path,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    disable_thinking: bool,
) -> str:
    url = f"http://{host}:{port}/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def numeric_exact_match_rate(source_html: str, predicted_html: str) -> tuple[float, int, int]:
    source_numbers = NUMBER_PATTERN.findall(source_html)
    predicted_numbers = NUMBER_PATTERN.findall(predicted_html)
    if not source_numbers:
        return 1.0, 0, 0

    source_counts = Counter(source_numbers)
    predicted_counts = Counter(predicted_numbers)
    matched = sum(min(count, predicted_counts.get(value, 0)) for value, count in source_counts.items())
    rate = matched / len(source_numbers)
    return rate, len(source_numbers), matched


def _sample_sort_key(path: Path) -> tuple[int, int | str]:
    suffix = path.name.split("_")[-1]
    if suffix.isdigit():
        return (0, int(suffix))
    return (1, path.name)


def load_samples(dataset_dir: Path) -> list[Path]:
    sample_dirs = []
    for path in sorted(dataset_dir.glob("sample_*"), key=_sample_sort_key):
        if not path.is_dir():
            continue
        if (path / "source.html").exists() and (path / "render.png").exists():
            sample_dirs.append(path)
    return sample_dirs


def save_gallery_tile(
    sample_id: str,
    source_image: Image.Image,
    pred_image: Image.Image,
    clip_score: float,
    numeric_rate: float,
    out_path: Path,
) -> None:
    tile_w, tile_h = 420, 290
    source_fit = ImageOps.fit(source_image.convert("RGB"), (tile_w, tile_h), Image.Resampling.LANCZOS)
    pred_fit = ImageOps.fit(pred_image.convert("RGB"), (tile_w, tile_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (tile_w * 2, tile_h + 72), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"{sample_id} | CLIP={clip_score:.4f} | numeric_exact={numeric_rate:.4f}", fill=(26, 31, 37))
    draw.text((12, 34), "source", fill=(85, 93, 104))
    draw.text((tile_w + 12, 34), "predicted", fill=(85, 93, 104))
    canvas.paste(source_fit, (0, 54))
    canvas.paste(pred_fit, (tile_w, 54))
    canvas.save(out_path)


def _evaluate_sample(
    sample_dir: Path,
    predictions_dir: Path,
    host: str,
    port: int,
    model_id: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    device: str,
    disable_thinking: bool,
) -> tuple[dict[str, str], dict[str, object] | None]:
    sample_id = sample_dir.name
    source_html_path = sample_dir / "source.html"
    source_png_path = sample_dir / "render.png"

    pred_sample_dir = predictions_dir / sample_id
    pred_sample_dir.mkdir(parents=True, exist_ok=True)
    pred_html_path = pred_sample_dir / "predicted.html"
    pred_png_path = pred_sample_dir / "predicted.png"

    row = {
        "sample_id": sample_id,
        "status": "error",
        "clip_similarity": "",
        "numeric_exact_match_rate": "",
        "source_numeric_count": "0",
        "matched_numeric_count": "0",
        "source_html_path": str(source_html_path),
        "source_render_path": str(source_png_path),
        "predicted_html_path": str(pred_html_path),
        "predicted_render_path": str(pred_png_path),
        "error": "",
    }

    source_html = source_html_path.read_text(encoding="utf-8")
    with Image.open(source_png_path) as source_image_opened:
        source_image = source_image_opened.convert("RGB")

    try:
        raw_output = query_model(
            host=host,
            port=port,
            model_id=model_id,
            image_path=source_png_path,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            disable_thinking=disable_thinking,
        )
        predicted_html = cleanup_response(raw_output)
        pred_html_path.write_text(predicted_html, encoding="utf-8")

        pred_image = render_html_to_image(predicted_html)
        pred_image.save(pred_png_path)

        clip_score = compute_clip_similarity(pred_image, source_image, device=device)
        numeric_rate, source_num_count, matched_num_count = numeric_exact_match_rate(source_html, predicted_html)

        row["status"] = "ok"
        row["clip_similarity"] = f"{clip_score:.6f}"
        row["numeric_exact_match_rate"] = f"{numeric_rate:.6f}"
        row["source_numeric_count"] = str(source_num_count)
        row["matched_numeric_count"] = str(matched_num_count)

        ok_row = {
            "sample_id": sample_id,
            "clip": clip_score,
            "numeric": numeric_rate,
            "source_image": source_image.copy(),
            "pred_image": pred_image.copy(),
            "source_html_path": source_html_path,
            "pred_html_path": pred_html_path,
        }
        return row, ok_row
    except Exception as exc:
        row["error"] = str(exc).replace("\n", " ")[:500]
        return row, None


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    output_dir: Path,
    dataset_dir: Path,
    model_id: str,
    num_samples: int,
    successful_rows: list[dict[str, str]],
) -> Path:
    if successful_rows:
        mean_clip = sum(float(row["clip_similarity"]) for row in successful_rows) / len(successful_rows)
        mean_numeric = sum(float(row["numeric_exact_match_rate"]) for row in successful_rows) / len(successful_rows)
    else:
        mean_clip = 0.0
        mean_numeric = 0.0

    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "num_samples": num_samples,
        "num_success": len(successful_rows),
        "num_failed": num_samples - len(successful_rows),
        "mean_clip_similarity": mean_clip,
        "mean_numeric_exact_match_rate": mean_numeric,
        "model_id": model_id,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def _write_gallery(
    gallery_dir: Path,
    ok_rows: list[dict[str, object]],
    gallery_size: int,
) -> None:
    selected = sorted(ok_rows, key=lambda item: float(item["clip"]), reverse=True)[: max(gallery_size, 0)]
    gallery_rows: list[dict[str, str]] = []

    for entry in selected:
        sample_id = str(entry["sample_id"])
        clip_value = float(entry["clip"])
        numeric_value = float(entry["numeric"])

        tile_path = gallery_dir / f"{sample_id}.png"
        save_gallery_tile(
            sample_id=sample_id,
            source_image=entry["source_image"],
            pred_image=entry["pred_image"],
            clip_score=clip_value,
            numeric_rate=numeric_value,
            out_path=tile_path,
        )

        source_copy = gallery_dir / f"{sample_id}_source.html"
        pred_copy = gallery_dir / f"{sample_id}_predicted.html"
        copyfile(entry["source_html_path"], source_copy)
        copyfile(entry["pred_html_path"], pred_copy)

        gallery_rows.append(
            {
                "sample_id": sample_id,
                "clip_similarity": f"{clip_value:.6f}",
                "numeric_exact_match_rate": f"{numeric_value:.6f}",
                "tile": str(tile_path),
                "source_html": str(source_copy),
                "predicted_html": str(pred_copy),
            }
        )

    _write_csv(gallery_dir / "index.csv", GALLERY_FIELDS, gallery_rows)


def evaluate(
    dataset_dir: Path,
    output_dir: Path,
    host: str,
    port: int,
    model_id: str | None,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    device: str,
    gallery_size: int,
    limit: int | None,
    disable_thinking: bool,
) -> None:
    samples = load_samples(dataset_dir)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise RuntimeError(f"No valid samples found in {dataset_dir}")
    if gallery_size < 0:
        raise ValueError("--gallery_size must be >= 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir = output_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    resolved_model_id = model_id or discover_model_id(host, port, timeout)
    print(f"Using model: {resolved_model_id}")
    print(f"Evaluating {len(samples)} samples")

    rows: list[dict[str, str]] = []
    ok_rows: list[dict[str, object]] = []

    for idx, sample_dir in enumerate(samples, start=1):
        row, ok_row = _evaluate_sample(
            sample_dir=sample_dir,
            predictions_dir=predictions_dir,
            host=host,
            port=port,
            model_id=resolved_model_id,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            device=device,
            disable_thinking=disable_thinking,
        )
        rows.append(row)
        if ok_row is not None:
            ok_rows.append(ok_row)

        if idx % 10 == 0 or idx == len(samples):
            print(f"[eval] {idx}/{len(samples)}")

    report_path = output_dir / "report.csv"
    _write_csv(report_path, REPORT_FIELDS, rows)

    successful_rows = [row for row in rows if row["status"] == "ok"]
    summary_path = _write_summary(
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        model_id=resolved_model_id,
        num_samples=len(samples),
        successful_rows=successful_rows,
    )
    _write_gallery(gallery_dir, ok_rows, gallery_size)

    print(f"Report:  {report_path}")
    print(f"Summary: {summary_path}")
    print(f"Gallery: {gallery_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate screenshot->HTML model on synthetic dataset.")
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=DIRECT_PROMPT)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--gallery_size", type=int, default=DEFAULT_GALLERY_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable_thinking", dest="disable_thinking", action="store_true")
    parser.add_argument("--enable_thinking", dest="disable_thinking", action="store_false")
    parser.set_defaults(disable_thinking=True)
    args = parser.parse_args()

    evaluate(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
        model_id=args.model_id,
        prompt=args.prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        device=args.device,
        gallery_size=args.gallery_size,
        limit=args.limit,
        disable_thinking=args.disable_thinking,
    )


if __name__ == "__main__":
    main()
