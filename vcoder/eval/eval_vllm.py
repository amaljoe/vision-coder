"""vLLM-based evaluation for Vision Coder models.

Sends 64 concurrent async requests to a running vLLM server per model,
computes format/validity/structural/CLIP rewards on WebSight holdout.

Usage:
    # Start vLLM servers first (one per GPU), then:
    python vcoder/eval/eval_vllm.py \
        --servers base:localhost:8000 sft:localhost:8001 rl:localhost:8002 sft_rl:localhost:8003 \
        --num_samples 100 \
        --concurrency 64 \
        --output_json outputs/eval_results.json
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import aiohttp
from PIL import Image
from datasets import load_from_disk
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vcoder.data.websight import DEFAULT_CACHE_DIR, SYSTEM_PROMPT
from vcoder.rewards.format_rewards import format_reward
from vcoder.rewards.validity_rewards import html_validity_reward
from vcoder.rewards.structural_rewards import structural_similarity_reward
from vcoder.utils.image_utils import compute_clip_similarity
from vcoder.utils.html_utils import extract_html_from_completion
from vcoder.rendering.browser_pool import BrowserPool

USER_PROMPT = "Generate the HTML code that reproduces this website screenshot."


def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


async def query_one(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    url: str,
    model_id: str,
    data_url: str,
    idx: int,
) -> tuple[int, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    async with sem:
        try:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return idx, data["choices"][0]["message"]["content"]
        except Exception as e:
            return idx, f"ERROR: {e}"


async def batch_generate(
    host: str,
    port: int,
    model_id: str,
    images: list[Image.Image],
    concurrency: int,
) -> list[str]:
    url = f"http://{host}:{port}/v1/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 8)
    results = [""] * len(images)

    async with aiohttp.ClientSession(connector=connector) as session:
        loop = asyncio.get_event_loop()
        # Pre-encode all images concurrently
        print(f"  Encoding {len(images)} images...")
        data_urls = await asyncio.gather(
            *[loop.run_in_executor(None, image_to_data_url, img) for img in images]
        )
        print(f"  Sending {len(images)} requests (concurrency={concurrency})...")
        tasks = [
            query_one(sem, session, url, model_id, data_url, i)
            for i, data_url in enumerate(data_urls)
        ]
        with tqdm(total=len(tasks), desc="  generating") as pbar:
            for coro in asyncio.as_completed(tasks):
                idx, text = await coro
                results[idx] = text
                pbar.update(1)

    return results


async def render_and_clip(completion: str, ref_image: Image.Image) -> float:
    html = extract_html_from_completion(completion)
    if html is None:
        return 0.0
    pool = await BrowserPool.get_instance()
    import io as _io
    png = await pool.render(html)
    if png is None:
        return 0.0
    try:
        rendered = Image.open(_io.BytesIO(png)).convert("RGB")
        return compute_clip_similarity(rendered, ref_image)
    except Exception:
        return 0.0


async def compute_clip_batch(completions: list[str], images: list[Image.Image]) -> list[float]:
    # Reset singleton and lock — both are bound to the previous event loop
    BrowserPool._instance = None
    BrowserPool._lock = asyncio.Lock()
    tasks = [render_and_clip(c, img) for c, img in zip(completions, images)]
    return await asyncio.gather(*tasks)


def compute_metrics(completions: list[str], solutions: list[str], images: list[Image.Image]) -> dict:
    wrapped = [[{"content": c}] for c in completions]
    fmt = format_reward(wrapped)
    val = html_validity_reward(wrapped)
    struct = structural_similarity_reward(wrapped, solution=solutions)
    clip_scores = asyncio.run(compute_clip_batch(completions, images))

    n = len(completions)
    return {
        "format": sum(fmt) / n,
        "validity": sum(val) / n,
        "structural": sum(struct) / n,
        "clip": sum(clip_scores) / n,
        "total": sum(f + v + s + 3 * c for f, v, s, c in zip(fmt, val, struct, clip_scores)) / n,
    }


def check_server(host: str, port: int) -> str | None:
    """Return model_id from running vLLM server, or None if unreachable."""
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=5) as r:
            data = json.loads(r.read())
            return data["data"][0]["id"]
    except Exception:
        return None


def print_results_table(results: dict[str, dict]):
    metrics = ["format", "validity", "structural", "clip", "total"]
    col_w = 12
    header = f"{'Model':<16}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print("\n" + "=" * len(header))
    print("EVALUATION RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, scores in results.items():
        row = f"{name:<16}" + "".join(f"{scores.get(m, 0):{col_w}.4f}" for m in metrics)
        print(row)
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--servers", nargs="+", required=True,
        metavar="NAME:HOST:PORT",
        help="e.g. base:localhost:8000 sft:localhost:8001",
    )
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output_json", default="outputs/eval_results.json")
    args = parser.parse_args()

    # Parse server specs
    server_specs = []
    for spec in args.servers:
        parts = spec.split(":")
        if len(parts) != 3:
            sys.exit(f"Bad server spec '{spec}', expected NAME:HOST:PORT")
        name, host, port = parts[0], parts[1], int(parts[2])
        model_id = check_server(host, port)
        if model_id is None:
            sys.exit(f"ERROR: vLLM server {name} at {host}:{port} is unreachable")
        print(f"  {name}: {host}:{port} → {model_id}")
        server_specs.append((name, host, port, model_id))

    # Load eval dataset
    ds = load_from_disk(args.cache_dir)
    total = len(ds)
    eval_start = max(0, total - args.num_samples)
    eval_ds = ds.select(range(eval_start, total))
    print(f"\nLoaded {len(eval_ds)} eval samples (indices {eval_start}–{total-1})")

    images = [ex["image"] for ex in eval_ds]
    solutions = [ex["solution"] for ex in eval_ds]

    all_results: dict[str, dict] = {}

    for name, host, port, model_id in server_specs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name} ({model_id})")
        print(f"{'='*60}")

        completions = asyncio.run(
            batch_generate(host, port, model_id, images, args.concurrency)
        )

        print(f"  Computing rewards...")
        scores = compute_metrics(completions, solutions, images)
        all_results[name] = scores
        print(f"  format={scores['format']:.4f}  validity={scores['validity']:.4f}  "
              f"structural={scores['structural']:.4f}  clip={scores['clip']:.4f}  "
              f"total={scores['total']:.4f}")

    print_results_table(all_results)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
