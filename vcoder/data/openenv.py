"""Dataset loader for the vision-coder-openenv custom HTML examples.

Renders the 15 HTML files (easy/0-4, medium/0-4, hard/0-4) with Playwright
and returns a HF Dataset in SFT format (prompt + completion) mirroring
the WebSight dataset format so the same sft_training.py pipeline works.
"""
from __future__ import annotations

import asyncio
import io
from pathlib import Path

from datasets import Dataset
from PIL import Image

from vcoder.data.websight import SYSTEM_PROMPT, _resize_image

OPENENV_DIR = Path("/home/compiling-ganesh/24m0797/workspace/vision-coder-openenv/data")
DIFFICULTIES = ["easy", "medium", "hard"]
NUM_PER_DIFFICULTY = 5


async def _render_html_file(html_path: Path, width: int = 1280, height: int = 1024) -> Image.Image | None:
    from playwright.async_api import async_playwright
    html_text = html_path.read_text(encoding="utf-8")
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
        )
        try:
            page = await browser.new_page(viewport={"width": width, "height": height})
            await page.set_content(html_text, wait_until="networkidle", timeout=15000)
            png_bytes = await page.screenshot(type="png", full_page=False)
        except Exception as e:
            print(f"[openenv] render failed for {html_path.name}: {e}")
            return None
        finally:
            await browser.close()
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def render_html_file(html_path: Path) -> Image.Image | None:
    return asyncio.run(_render_html_file(html_path))


def _make_sft_row(html: str, image: Image.Image, max_width: int = 512) -> dict:
    prompt = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Generate the HTML code that reproduces this website screenshot."},
            ],
        },
    ]
    completion = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"```html\n{html}\n```"}],
        }
    ]
    return {
        "prompt": prompt,
        "completion": completion,
        "image": _resize_image(image, max_width=max_width),
        "solution": html,
    }


def load_openenv_sft_dataset(max_width: int = 512) -> Dataset:
    """Load and render the 15 openenv HTML examples as an SFT dataset.

    Returns a Dataset with columns: prompt, completion, image, solution.
    """
    rows = []
    for difficulty in DIFFICULTIES:
        for i in range(NUM_PER_DIFFICULTY):
            html_path = OPENENV_DIR / difficulty / f"{i}.html"
            if not html_path.exists():
                print(f"[openenv] missing: {html_path}")
                continue
            html_text = html_path.read_text(encoding="utf-8")
            print(f"[openenv] rendering {difficulty}/{i}.html ...")
            image = render_html_file(html_path)
            if image is None:
                print(f"[openenv] skipping {html_path} (render failed)")
                continue
            rows.append(_make_sft_row(html_text, image, max_width=max_width))

    print(f"[openenv] loaded {len(rows)} examples")
    ds = Dataset.from_list(rows)
    return ds
