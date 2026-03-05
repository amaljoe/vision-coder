# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VisionCoder fine-tunes [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) using **GRPO** (Group Relative Policy Optimization) with rendering-based rewards to convert UI screenshots into HTML/CSS. Training is done on [WebSight](https://huggingface.co/datasets/HuggingFaceM4/WebSight) and evaluated on the [Design2Code](https://github.com/NoviScl/Design2Code) benchmark (484 examples).

## Setup

```bash
pip install -e . --no-deps
playwright install chromium
```

## Key Commands

### Training (4-GPU)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file configs/accelerate_4gpu.yaml \
    vcoder/pipelines/training.py \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --output_dir outputs/vcoder-grpo-clip \
    --max_samples 2000 --num_train_epochs 1 \
    --batch_size 4 --num_generations 8 --max_completion_length 2048
```

### Evaluation
```bash
# 1. Start VLLM servers
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-2B-Instruct --port 8000 \
    --gpu-memory-utilization 0.45 --max-model-len 8192 --trust-remote-code

CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --model outputs/vcoder-grpo-clip/checkpoint-500 --port 8001 \
    --gpu-memory-utilization 0.45 --max-model-len 8192 --trust-remote-code

# 2. Generate predictions
python3 vcoder/eval/generate_predictions.py \
    --model vcoder-grpo-clip \
    --testset_dir ../Design2Code/testset_final_extracted

# 3. Run Design2Code benchmark
cd ../Design2Code/Design2Code && python3 metrics/multi_processing_eval.py
```

### Plotting
```bash
python3 experiments/plot_run.py                           # latest checkpoint
python3 experiments/plot_run.py --run_dir outputs/vcoder-grpo-clip --checkpoint 300
# Saves 5 plots to <run_dir>/plots/
```

### Tests
```bash
python3 -m pytest tests/test_websight.py -v
```

## Architecture

### Reward System (`vcoder/rewards/`)
Four reward functions are summed, with CLIP weighted 3×:

| Reward | Weight | Method |
|---|---|---|
| `format_reward` | 1× | Regex: checks for `<think>` block + `<html>` fence |
| `html_validity_reward` | 1× | BeautifulSoup parse + structural completeness + tag diversity |
| `structural_similarity_reward` | 1× | LCS on DOM tag sequences + CSS class overlap |
| `boosted_clip_reward` | 3× | CLIP cosine similarity between rendered HTML screenshot and reference |

All rewards are async and return `0.0` on failure. Each receives a batch of `(completions, reference_images)`.

### Rendering Pipeline (`vcoder/rendering/`)
- `BrowserPool` is a **singleton** async Playwright pool (max 8 concurrent renders, 1280×1024, 5s timeout).
- `html_renderer.py` provides a sync wrapper (`render_html_to_image`) that creates a fresh event loop — **do not call from inside an async context**.
- HTML is extracted from markdown fences (` ```html ... ``` `) by `vcoder/utils/html_utils.py` before rendering.

### Training Pipeline (`vcoder/pipelines/training.py`)
- Uses TRL's `GRPOTrainer` with `use_vllm=True` (colocated vLLM generation on same GPU as training).
- Importance sampling is **disabled** (`loss_type="dr_grpo"`) for ~38% speed improvement.
- Dataset columns required by GRPOTrainer: `prompt` (conversation list), `image` (PIL), `solution` (HTML string).

### Data Loading (`vcoder/data/websight.py`)
- Loads WebSight from HuggingFace, filters to HTML < `max_html_chars` chars, resizes images to `max_width` px (aligned to 32px boundaries).
- Returns a `Dataset` with columns: `prompt` (Qwen conversation template), `image`, `solution`.

### Inference (`vcoder/eval/generate_predictions.py`)
- Sends async requests to a VLLM OpenAI-compatible server (default 32 concurrent).
- Saves generated HTML to `outputs/benchmark/<model_name>/<i>.html`.
- Saves per-run stats to `outputs/benchmark/<model_name>/_stats.json`.

## Outputs

- Checkpoints: `outputs/<run_name>/checkpoint-{step}/` (adapter weights + tokenizer)
- `trainer_state.json` inside each checkpoint contains full training log history used by `plot_run.py`
- Benchmark predictions: `outputs/benchmark/<model>/` (HTML files + `_stats.json`)
