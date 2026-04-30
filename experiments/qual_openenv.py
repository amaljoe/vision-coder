"""Qualitative analysis on vision-coder-openenv examples.

For each of the 15 HTML files (easy/medium/hard), renders the reference HTML,
sends it to the VLLM server, renders the prediction, and saves side-by-side
comparison images plus a summary CSV.

Usage:
    python experiments/qual_openenv.py --model vcoder-grpo-clip [--port 8001] [--out_dir assets/qual_openenv/before]
    python experiments/qual_openenv.py --model vcoder-grpo-clip-sft [--port 8001] [--out_dir assets/qual_openenv/after]
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import io
import sys
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vcoder.eval.generate_predictions import DIRECT_PROMPT, MAX_TOKENS, TEMPERATURE, cleanup_response
from vcoder.utils.image_utils import compute_clip_similarity, compute_ssim

OPENENV_DIR = Path("/home/compiling-ganesh/24m0797/workspace/vision-coder-openenv/data")
DIFFICULTIES = ["easy", "medium", "hard"]


async def _render_html(html: str, width: int = 1280, height: int = 1024) -> Image.Image | None:
    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"]
        )
        try:
            page = await browser.new_page(viewport={"width": width, "height": height})
            await page.set_content(html, wait_until="networkidle", timeout=15000)
            png_bytes = await page.screenshot(type="png", full_page=False)
        except Exception as e:
            print(f"  [render] failed: {e}")
            return None
        finally:
            await browser.close()
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def render_html(html: str) -> Image.Image | None:
    return asyncio.run(_render_html(html))


def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def call_vllm(img: Image.Image, host: str, port: int, model_id: str) -> str:
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _image_to_data_url(img)}},
            {"type": "text", "text": DIRECT_PROMPT},
        ]}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def make_comparison(ref: Image.Image, pred: Image.Image | None, label: str, ssim: float, clip: float) -> Image.Image:
    """Create a side-by-side comparison image with score labels."""
    w, h = ref.size
    if pred is None:
        pred = Image.new("RGB", (w, h), (220, 220, 220))
    else:
        pred = pred.resize((w, h), Image.LANCZOS)

    gap = 20
    bar_h = 40
    canvas = Image.new("RGB", (w * 2 + gap, h + bar_h), (255, 255, 255))
    canvas.paste(ref, (0, bar_h))
    canvas.paste(pred, (w + gap, bar_h))

    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), f"{label} | SSIM={ssim:.3f}  CLIP={clip:.3f}", fill=(30, 30, 30))
    draw.text((4, 22), "Reference", fill=(0, 100, 0))
    draw.text((w + gap + 4, 22), "Predicted", fill=(150, 0, 0))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model_id", default="/home/compiling-ganesh/24m0797/workspace/vision-coder/outputs/vcoder-grpo-clip/checkpoint-500")
    parser.add_argument("--out_dir", default="assets/qual_openenv/before")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for difficulty in DIFFICULTIES:
        for i in range(5):
            html_path = OPENENV_DIR / difficulty / f"{i}.html"
            if not html_path.exists():
                print(f"Missing: {html_path}")
                continue

            label = f"{difficulty}/{i}"
            html_text = html_path.read_text(encoding="utf-8")
            print(f"\n[{label}] Rendering reference...")
            ref_img = render_html(html_text)
            if ref_img is None:
                print(f"  reference render failed, skipping")
                continue

            print(f"[{label}] Calling VLLM...")
            try:
                raw = call_vllm(ref_img, args.host, args.port, args.model_id)
                pred_html = cleanup_response(raw)
            except Exception as e:
                print(f"  VLLM error: {e}")
                raw, pred_html = "", ""

            print(f"[{label}] Rendering prediction...")
            pred_img = render_html(pred_html) if pred_html else None

            ssim = compute_ssim(pred_img, ref_img) if pred_img else 0.0
            clip = compute_clip_similarity(pred_img, ref_img) if pred_img else 0.0
            print(f"[{label}] SSIM={ssim:.4f}  CLIP={clip:.4f}")

            # Save comparison image
            comp = make_comparison(ref_img, pred_img, label, ssim, clip)
            out_path = out_dir / f"{difficulty}_{i}.png"
            comp.save(str(out_path))
            print(f"  Saved: {out_path}")

            # Also save individual reference and prediction
            ref_img.save(str(out_dir / f"{difficulty}_{i}_ref.png"))
            if pred_img:
                pred_img.save(str(out_dir / f"{difficulty}_{i}_pred.png"))

            rows.append({"label": label, "difficulty": difficulty, "idx": i, "ssim": ssim, "clip": clip})

    # Save summary CSV
    csv_path = out_dir / "scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "difficulty", "idx", "ssim", "clip"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== Summary ===")
    for r in rows:
        print(f"  {r['label']:12s}  SSIM={r['ssim']:.4f}  CLIP={r['clip']:.4f}")
    if rows:
        avg_ssim = sum(r["ssim"] for r in rows) / len(rows)
        avg_clip = sum(r["clip"] for r in rows) / len(rows)
        print(f"\n  avg SSIM={avg_ssim:.4f}  avg CLIP={avg_clip:.4f}")
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
