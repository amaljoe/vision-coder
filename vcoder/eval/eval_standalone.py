"""Standalone evaluation script for Vision Coder models.

Evaluates models on a held-out WebSight validation set without needing VLLM.
Generates HTML from screenshots and computes reward metrics.

Usage:
    python vcoder/eval/eval_standalone.py \
        --model_ids qwen-base outputs/vcoder-sft outputs/vcoder-rl outputs/vcoder-sft-rl \
        --model_names base sft rl sft_rl \
        --num_samples 100
"""
from __future__ import annotations

import argparse
import asyncio
import io
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from datasets import load_from_disk

from vcoder.data.websight import DEFAULT_CACHE_DIR, SYSTEM_PROMPT
from vcoder.rewards.format_rewards import format_reward
from vcoder.rewards.validity_rewards import html_validity_reward
from vcoder.rewards.structural_rewards import structural_similarity_reward
from vcoder.utils.image_utils import compute_clip_similarity
from vcoder.utils.html_utils import extract_html_from_completion
from vcoder.rendering.browser_pool import BrowserPool


USER_PROMPT = "Generate the HTML code that reproduces this website screenshot."


def build_messages(image: Image.Image) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": USER_PROMPT}],
        },
    ]


@torch.inference_mode()
def generate_html(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
    max_new_tokens: int = 1024,
    device: str = "cuda",
) -> str:
    messages = build_messages(image)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    # Decode only the generated tokens
    gen_tokens = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen_tokens, skip_special_tokens=True)


async def _render_html_to_image(html: str) -> Image.Image | None:
    pool = await BrowserPool.get_instance()
    png_bytes = await pool.render(html)
    if png_bytes is None:
        return None
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def render_html_sync(html: str) -> Image.Image | None:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run, _render_html_to_image(html)).result(timeout=30)
        return loop.run_until_complete(_render_html_to_image(html))
    except Exception:
        return None


def compute_clip_reward(completion_text: str, ref_image: Image.Image) -> float:
    html = extract_html_from_completion(completion_text)
    if html is None:
        return 0.0
    rendered = render_html_sync(html)
    if rendered is None:
        return 0.0
    try:
        return compute_clip_similarity(rendered, ref_image)
    except Exception:
        return 0.0


def eval_completions(completions: list[str], solutions: list[str], images: list[Image.Image]) -> dict[str, float]:
    wrapped = [[{"content": c}] for c in completions]

    fmt = format_reward(wrapped)
    val = html_validity_reward(wrapped)
    struct = structural_similarity_reward(wrapped, solution=solutions)
    clip_scores = [compute_clip_reward(c, img) for c, img in zip(completions, images)]

    return {
        "format": sum(fmt) / len(fmt),
        "validity": sum(val) / len(val),
        "structural": sum(struct) / len(struct),
        "clip": sum(clip_scores) / len(clip_scores),
        "total": sum(
            f + v + s + 3 * c for f, v, s, c in zip(fmt, val, struct, clip_scores)
        ) / len(completions),
    }


def eval_model(
    model_id: str,
    model_name: str,
    dataset,
    num_samples: int,
    max_new_tokens: int,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name} ({model_id})")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("  Loading processor and model...")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True, local_files_only=True)
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    samples = dataset.select(range(min(num_samples, len(dataset))))
    completions, solutions, images = [], [], []

    print(f"  Generating {len(samples)} predictions...")
    t0 = time.time()
    for i, ex in enumerate(samples):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(samples)}] elapsed={elapsed:.0f}s")
        text = generate_html(model, processor, ex["image"], max_new_tokens=max_new_tokens, device=device)
        completions.append(text)
        solutions.append(ex["solution"])
        images.append(ex["image"])

    print(f"  Generation done in {time.time() - t0:.0f}s. Computing metrics...")
    scores = eval_completions(completions, solutions, images)

    # Free GPU memory before next model
    del model
    torch.cuda.empty_cache()

    print(f"  Results for {model_name}:")
    for k, v in scores.items():
        print(f"    {k:12s}: {v:.4f}")

    return scores


def load_eval_dataset(num_samples: int, eval_split_seed: int = 99, cache_dir: str = DEFAULT_CACHE_DIR):
    """Load held-out eval examples from WebSight cache (last N samples from a different seed)."""
    if not Path(cache_dir).exists():
        print(f"ERROR: dataset cache not found at {cache_dir}")
        print("Run: python vcoder/data/websight.py --max_samples 2500")
        sys.exit(1)
    ds = load_from_disk(cache_dir)
    # Use samples after the training split as eval
    total = len(ds)
    eval_start = max(0, total - num_samples)
    eval_ds = ds.select(range(eval_start, total))
    print(f"Loaded {len(eval_ds)} eval samples (indices {eval_start}–{total-1} of {total}).")
    return eval_ds


def print_results_table(results: dict[str, dict[str, float]]):
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
    parser = argparse.ArgumentParser(description="Standalone eval for VisionCoder models")
    parser.add_argument(
        "--model_ids",
        nargs="+",
        default=["Qwen/Qwen3-VL-2B-Instruct"],
        help="Model IDs or paths",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=None,
        help="Display names for models (must match --model_ids length)",
    )
    parser.add_argument("--num_samples", type=int, default=100, help="Eval samples per model")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output_json", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    if args.model_names is None:
        args.model_names = [Path(m).name if "/" not in m else m.split("/")[-1] for m in args.model_ids]

    if len(args.model_names) != len(args.model_ids):
        parser.error("--model_names must have same number of entries as --model_ids")

    eval_ds = load_eval_dataset(args.num_samples, cache_dir=args.cache_dir)

    all_results: dict[str, dict[str, float]] = {}
    for model_id, model_name in zip(args.model_ids, args.model_names):
        scores = eval_model(model_id, model_name, eval_ds, args.num_samples, args.max_new_tokens)
        all_results[model_name] = scores

    print_results_table(all_results)

    if args.output_json:
        import json
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
