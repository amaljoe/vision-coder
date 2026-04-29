"""Generate HTML predictions for the Design2Code benchmark.

Loads reference images from the Design2Code testset directory (PNG files),
sends them concurrently to a VLLM server, and saves the generated HTML with
matching filenames so the official eval script can compare them directly.

VLLM has a continuous batching engine — sending requests concurrently
(--concurrency 32) saturates GPU throughput instead of waiting 1-by-1.

Usage:
    python vcoder/eval/generate_predictions.py \\
        --model qwen-base \\
        --testset_dir ../Design2Code/testset_final_extracted \\
        --concurrency 32

    python vcoder/eval/generate_predictions.py \\
        --model vcoder-grpo-clip \\
        --testset_dir ../Design2Code/testset_final_extracted \\
        --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re
import sys
from pathlib import Path

import aiohttp
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict] = {
    "qwen-base": {
        "host": "localhost",
        "port": 8000,
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        # Disable Qwen3 thinking mode — CoT tokens eat the generation budget
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "vcoder-grpo-clip": {
        "host": "localhost",
        "port": 8001,
        "model_id": "/home/compiling-ganesh/24m0797/workspace/vision-coder/outputs/vcoder-grpo-clip/checkpoint-500",
        "extra_body": {},
    },
    "vcoder-sft": {
        "host": "localhost",
        "port": 8000,
        "model_id": "amaljoe88/vcoder-sft",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "vcoder-rl": {
        "host": "localhost",
        "port": 8000,
        "model_id": "amaljoe88/vcoder-rl",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "vcoder-sft-rl": {
        "host": "localhost",
        "port": 8000,
        "model_id": "amaljoe88/vcoder-sft-rl",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
}

# Design2Code direct-prompting prompt (matches the paper's evaluation setup)
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

MAX_TOKENS = 7500
TEMPERATURE = 0.0
TIMEOUT = 300  # seconds per request (generous for long generations)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def cleanup_response(text: str) -> str:
    """Strip markdown fences. Handles both complete and truncated (unclosed) fences."""
    # Complete fence: ```html ... ```
    m = re.search(r"```html\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Complete fence, bare html tag: ``` <!DOCTYPE ... ```
    m = re.search(r"```\s*(<!DOCTYPE|<html)(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return (m.group(1) + m.group(2)).strip()
    # Truncated/unclosed fence: ```html at start, no closing ``` (hit token limit)
    m = re.match(r"```(?:html)?\s*\n?(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Raw HTML without any fences
    return text.strip()


async def query_vllm_async(
    session: aiohttp.ClientSession,
    url: str,
    model_id: str,
    data_url: str,
    extra_body: dict | None = None,
) -> str:
    """Send a single request to the VLLM server asynchronously."""
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": DIRECT_PROMPT},
                ],
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    if extra_body:
        payload.update(extra_body)
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


async def process_one(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    url: str,
    model_id: str,
    extra_body: dict,
    png_path: Path,
    out_path: Path,
    stats: dict,
    errors: list,
    pbar: tqdm,
) -> None:
    """Process a single example: load image, query VLLM, save HTML."""
    async with sem:
        try:
            # Load and encode image (blocking I/O — run in thread pool)
            loop = asyncio.get_event_loop()
            data_url = await loop.run_in_executor(
                None,
                lambda: _image_to_data_url(Image.open(png_path).convert("RGB")),
            )
            raw = await query_vllm_async(session, url, model_id, data_url, extra_body)
            html = cleanup_response(raw)
            out_path.write_text(html, encoding="utf-8")
            stats["generated"] += 1
        except asyncio.TimeoutError:
            tqdm.write(f"Timeout: {png_path.name}")
            stats["failed"] += 1
            errors.append({"file": png_path.name, "error": "timeout"})
        except Exception as e:
            tqdm.write(f"Error on {png_path.name}: {e}")
            stats["failed"] += 1
            errors.append({"file": png_path.name, "error": str(e)})
        finally:
            pbar.update(1)


async def run_inference(
    cfg: dict,
    test_pngs: list[Path],
    out_dir: Path,
    concurrency: int,
) -> dict:
    url = f"http://{cfg['host']}:{cfg['port']}/v1/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    stats = {"total": len(test_pngs), "generated": 0, "skipped": 0, "failed": 0}
    errors: list[dict] = []

    # Filter already-done files
    pending = []
    for png_path in test_pngs:
        out_path = out_dir / (png_path.stem + ".html")
        if out_path.exists():
            stats["skipped"] += 1
        else:
            pending.append(png_path)

    print(f"  {stats['skipped']} already done, {len(pending)} to generate.")

    extra_body = cfg.get("extra_body", {})
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        with tqdm(total=len(pending), desc=f"{cfg.get('name', 'model')}") as pbar:
            tasks = [
                process_one(
                    sem, session, url, cfg["model_id"], extra_body,
                    png_path, out_dir / (png_path.stem + ".html"),
                    stats, errors, pbar,
                )
                for png_path in pending
            ]
            await asyncio.gather(*tasks)

    return stats, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Design2Code predictions via VLLM")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument(
        "--testset_dir",
        type=str,
        default=None,
        help="Directory with *.png test images (default: ../Design2Code/testset_final_extracted)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/benchmark",
        help="Root dir for saving predictions",
    )
    parser.add_argument(
        "--concurrency", type=int, default=32,
        help="Number of concurrent requests to VLLM (default: 32)",
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model].copy()
    cfg["name"] = args.model
    if args.host:
        cfg["host"] = args.host
    if args.port:
        cfg["port"] = args.port

    # Resolve testset dir
    if args.testset_dir:
        testset_dir = Path(args.testset_dir)
    else:
        repo_root = Path(__file__).resolve().parents[2]
        testset_dir = repo_root.parent / "Design2Code" / "testset_final_extracted"

    if not testset_dir.exists():
        print(f"ERROR: testset dir not found: {testset_dir}")
        sys.exit(1)

    # Collect PNGs
    test_pngs = sorted(
        [p for p in testset_dir.glob("*.png") if "_marker" not in p.name],
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    if not test_pngs:
        print(f"ERROR: no .png files found in {testset_dir}")
        sys.exit(1)
    if args.num_samples:
        test_pngs = test_pngs[: args.num_samples]

    print(f"Found {len(test_pngs)} test images.")

    # Verify server health
    import requests as _req
    try:
        _req.get(f"http://{cfg['host']}:{cfg['port']}/health", timeout=5).raise_for_status()
        print(f"VLLM server at {cfg['host']}:{cfg['port']} is healthy.")
    except Exception as e:
        print(f"ERROR: VLLM server unreachable: {e}")
        sys.exit(1)

    # Output dir
    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Concurrency: {args.concurrency} parallel requests")
    print(f"Output: {out_dir}\n")

    stats, errors = asyncio.run(run_inference(cfg, test_pngs, out_dir, args.concurrency))

    # Save stats
    stats_path = out_dir / "_stats.json"
    with open(stats_path, "w") as f:
        json.dump({"stats": stats, "errors": errors, "config": cfg}, f, indent=2)

    print(f"\nDone.")
    print(f"  Generated : {stats['generated']}")
    print(f"  Skipped   : {stats['skipped']} (already existed)")
    print(f"  Failed    : {stats['failed']}")
    print(f"  Output    : {out_dir}")


if __name__ == "__main__":
    main()
