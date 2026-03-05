"""Generate assets/results/ sample images for README.

- Queries both VLLM servers concurrently (10 at once, sync requests via threads)
- 30s read timeout with up to 3 retries on failure
- Renders all HTMLs in parallel using a shared Playwright browser
- tqdm progress bars throughout
- Saves top 3 (by finetuned CLIP score) to assets/results/sample_{1,2,3}/

Usage:
    python3 -u vcoder/demo/gen_samples.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests as req_lib
from PIL import Image
from playwright.async_api import async_playwright
from tqdm import tqdm

os.environ["PLAYWRIGHT_BROWSERS_PATH"] = os.path.expanduser("~/playwright-browsers")

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTSET_DIR = REPO_ROOT.parent / "Design2Code" / "testset_final_extracted"
OUTPUT_DIR = REPO_ROOT / "assets" / "results"
NUM_IMAGES = 10
TOP_K = 3
MAX_RETRIES = 5
CONNECT_TIMEOUT = 10   # seconds to establish TCP connection
READ_TIMEOUT = 300     # seconds — 10 concurrent requests, queue wait + generation can exceed 180s

BASE_CFG = {
    "port": 8000,
    "model_id": "Qwen/Qwen3-VL-2B-Instruct",
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
}
FINETUNED_CFG = {
    "port": 8001,
    "model_id": "/home/compiling-ganesh/24m0797/workspace/vision-coder/outputs/vcoder-grpo-clip/checkpoint-500",
    "extra_body": {},
}

DIRECT_PROMPT = (
    "You are an expert web developer who specializes in HTML and CSS.\n"
    "A user will provide you with a screenshot of a webpage.\n"
    "You need to return a single html file that uses HTML and CSS to reproduce the given website.\n"
    "Include all CSS code in the HTML file itself.\n"
    'If it involves any images, use "rick.jpg" as the placeholder.\n'
    "Some images on the webpage are replaced with a blue rectangle as the placeholder, "
    'use "rick.jpg" for those as well.\n'
    "Do not hallucinate any dependencies to external files. "
    "You do not need to include JavaScript scripts for dynamic interactions.\n"
    "Pay attention to things like size, text, position, and color of all the elements, "
    "as well as the overall layout.\n"
    "Respond with the content of the HTML+CSS file:\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def cleanup_response(text: str) -> str:
    m = re.search(r"```html\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(<!DOCTYPE|<html)(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return (m.group(1) + m.group(2)).strip()
    m = re.match(r"```(?:html)?\s*\n?(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def query_vllm_sync(cfg: dict, img: Image.Image, label: str, pbar: tqdm) -> str:
    """Sync VLLM request with retries. Called from a thread pool."""
    url = f"http://localhost:{cfg['port']}/v1/chat/completions"
    data_url = _image_to_data_url(img)
    payload = {
        "model": cfg["model_id"],
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": DIRECT_PROMPT},
        ]}],
        "temperature": 0.0,
        "max_tokens": 7500,
    }
    payload.update(cfg.get("extra_body", {}))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = req_lib.Session().post(url, json=payload, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            resp.raise_for_status()
            text = cleanup_response(resp.json()["choices"][0]["message"]["content"])
            pbar.write(f"  [{label}] ✓ {len(text)} chars")
            pbar.update(1)
            return text
        except (req_lib.exceptions.ConnectionError, req_lib.exceptions.ReadTimeout) as e:
            wait = 3 * attempt
            pbar.write(f"  [{label}] attempt {attempt}/{MAX_RETRIES} failed ({type(e).__name__}), retry in {wait}s")
            time.sleep(wait)
        except Exception as e:
            wait = 2 * attempt
            pbar.write(f"  [{label}] attempt {attempt}/{MAX_RETRIES} failed: {type(e).__name__}: {e}, retry in {wait}s")
            time.sleep(wait)

    pbar.write(f"  [{label}] all {MAX_RETRIES} attempts failed")
    pbar.update(1)
    return ""


def wait_for_servers(ports: list[int], timeout: int = 300) -> None:
    print("Waiting for VLLM servers...", flush=True)
    for port in ports:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if req_lib.get(f"http://localhost:{port}/health", timeout=5).status_code == 200:
                    print(f"  Port {port}: ready ✓", flush=True)
                    break
            except Exception:
                pass
            time.sleep(5)
        else:
            print(f"ERROR: port {port} not ready after {timeout}s")
            sys.exit(1)


async def render_page(browser, html: str, label: str, pbar: tqdm) -> Image.Image | None:
    try:
        page = await browser.new_page(viewport={"width": 1280, "height": 1024})
        await page.set_content(html, wait_until="load", timeout=15000)
        await page.wait_for_timeout(500)
        png_bytes = await page.screenshot(type="png", full_page=False)
        await page.close()
        pbar.write(f"  [{label}] rendered ✓")
        pbar.update(1)
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except Exception as e:
        pbar.write(f"  [{label}] render failed: {e}")
        pbar.update(1)
        try:
            await page.close()
        except Exception:
            pass
        return None


async def main_async() -> None:
    wait_for_servers([8000, 8001])

    pngs = sorted(
        [p for p in TESTSET_DIR.glob("*.png") if "_marker" not in p.name],
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )[:NUM_IMAGES]
    print(f"\nUsing {len(pngs)} images: {[p.stem for p in pngs]}", flush=True)

    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=NUM_IMAGES * 2)

    # ---- Inference ----
    # Stagger ft requests by 150ms each to avoid overwhelming the TCP backlog
    # on port 8001 (all 10 simultaneous SYN packets fill the OS listen queue).
    # Base model on port 8000 handles 10 concurrent fine.
    total_requests = len(pngs) * 2
    with tqdm(total=total_requests, desc="Inference", unit="req") as pbar:
        base_futures = [
            loop.run_in_executor(
                executor, query_vllm_sync,
                BASE_CFG, Image.open(p).convert("RGB"), f"{p.stem}/base", pbar
            )
            for p in pngs
        ]
        ft_futures = [
            loop.run_in_executor(
                executor, query_vllm_sync,
                FINETUNED_CFG, Image.open(p).convert("RGB"), f"{p.stem}/ft", pbar
            )
            for p in pngs
        ]

        all_html = await asyncio.gather(*base_futures, *ft_futures)

    base_htmls = list(all_html[:len(pngs)])
    ft_htmls   = list(all_html[len(pngs):])

    # ---- Rendering ----
    total_renders = sum(1 for h in base_htmls + ft_htmls if h)
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
    )
    try:
        with tqdm(total=total_renders, desc="Rendering", unit="page") as pbar:
            render_coros = []
            for i, p in enumerate(pngs):
                render_coros.append(
                    render_page(browser, base_htmls[i], f"{p.stem}/base", pbar)
                    if base_htmls[i] else asyncio.sleep(0, result=None)
                )
                render_coros.append(
                    render_page(browser, ft_htmls[i], f"{p.stem}/ft", pbar)
                    if ft_htmls[i] else asyncio.sleep(0, result=None)
                )
            rendered = await asyncio.gather(*render_coros)
    finally:
        await browser.close()
        await pw.stop()

    base_renders = [rendered[i * 2]     for i in range(len(pngs))]
    ft_renders   = [rendered[i * 2 + 1] for i in range(len(pngs))]

    # ---- CLIP scoring ----
    from vcoder.utils.image_utils import compute_clip_similarity
    records = []
    with tqdm(total=len(pngs), desc="CLIP scoring", unit="img") as pbar:
        for i, png_path in enumerate(pngs):
            ref = Image.open(png_path).convert("RGB")
            br = base_renders[i] if isinstance(base_renders[i], Image.Image) else None
            fr = ft_renders[i]   if isinstance(ft_renders[i],   Image.Image) else None
            base_clip = await loop.run_in_executor(None, compute_clip_similarity, br, ref) if br else 0.0
            ft_clip   = await loop.run_in_executor(None, compute_clip_similarity, fr, ref) if fr else 0.0
            pbar.write(f"  [{png_path.stem}] base={base_clip:.4f}  ft={ft_clip:.4f}  Δ={ft_clip - base_clip:+.4f}")
            pbar.update(1)
            records.append({
                "idx": png_path.stem,
                "ref_img": ref,
                "base_rendered": br,
                "ft_rendered": fr,
                "base_clip": base_clip,
                "ft_clip": ft_clip,
                "delta": ft_clip - base_clip,
            })

    # ---- Save top-K ----
    top = sorted(records, key=lambda r: r["ft_clip"], reverse=True)[:TOP_K]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_meta = []
    print(f"\nSaving top {TOP_K} samples...", flush=True)
    for rank, r in enumerate(top, start=1):
        folder = OUTPUT_DIR / f"sample_{rank}"
        folder.mkdir(parents=True, exist_ok=True)
        r["ref_img"].save(folder / "reference.png")
        if r["base_rendered"]:
            r["base_rendered"].save(folder / "base_rendered.png")
        if r["ft_rendered"]:
            r["ft_rendered"].save(folder / "finetuned_rendered.png")
        meta = {
            "rank": rank,
            "testset_idx": r["idx"],
            "base_clip": round(r["base_clip"], 4),
            "finetuned_clip": round(r["ft_clip"], 4),
            "delta": round(r["delta"], 4),
        }
        sample_meta.append(meta)
        print(f"  sample_{rank}/ → idx={r['idx']}  base={r['base_clip']:.4f}  ft={r['ft_clip']:.4f}  Δ={r['delta']:+.4f}", flush=True)

    with open(OUTPUT_DIR / "samples_meta.json", "w") as f:
        json.dump(sample_meta, f, indent=2)

    print(f"\nDone. Saved to {OUTPUT_DIR}", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
