# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VisionCoder fine-tunes [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) using **GRPO** (Group Relative Policy Optimization) with rendering-based rewards to convert UI screenshots into HTML/CSS. Training is done on [WebSight](https://huggingface.co/datasets/HuggingFaceM4/WebSight) and evaluated on the [Design2Code](https://github.com/NoviScl/Design2Code) benchmark (484 examples).

## Environment — Apptainer is ALWAYS required

**All GPU workloads (training, inference, rendering/eval) must run inside the Apptainer container.** The host OS lacks required shared libraries (e.g. `libatk-bridge-2.0.so.0` for Playwright, CUDA-linked libs for vllm).

```bash
# Restore the vllm env to /dev/shm (fast, from tarball):
restorevllm          # extracts ~/envs/vllm.tar.zst → /dev/shm/vllm (~30-60s)

# Enter container and activate env:
app                  # alias: apptainer exec --nv ~/images/cuda-custom-amal_latest.sif bash
mamba activate /dev/shm/vllm

# Python/tools inside env:
PYTHON=/dev/shm/vllm/bin/python3
ACCELERATE=/dev/shm/vllm/bin/accelerate
```

After any env change, save with `savevllm`. See `~/workspace/ixbrl-tagging/env.md` for full setup details.

## Setup

```bash
# Inside Apptainer (/dev/shm/vllm active):
pip install -e . --no-deps
```

## Key Commands

All commands below must be run inside Apptainer with `/dev/shm/vllm` active.

### Training (2-GPU)
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file configs/accelerate_2gpu.yaml \
    vcoder/pipelines/training.py \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --output_dir outputs/vcoder-rl
```

### Evaluation
```bash
# 1. Start VLLM servers (one per GPU, inside Apptainer)
#    HF models need local patched cache — see generate_predictions.py MODEL_CONFIGS
CUDA_VISIBLE_DEVICES=0 /dev/shm/vllm/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model amaljoe88/vcoder-sft --port 8000 \
    --gpu-memory-utilization 0.7 --max-model-len 8192 --trust-remote-code --dtype bfloat16

# 2. Generate predictions
/dev/shm/vllm/bin/python3 vcoder/eval/generate_predictions.py \
    --model vcoder-sft \
    --testset_dir ../Design2Code/testset_final_extracted

# 3. Run Design2Code benchmark (inside Apptainer)
cd ../Design2Code/Design2Code
PYTHONPATH=$(pwd) /dev/shm/vllm/bin/python3 metrics/multi_processing_eval.py
```

### HF Model Config Gotchas
Fine-tuned models on HF (`amaljoe88/vcoder-*`) need two patches after `snapshot_download()`:
- `config.json`: `text_config.rope_scaling` is `None` — copy from base model config
- `tokenizer_config.json`: `extra_special_tokens` saved as list — change to `{}`

See the patching script in git history or re-run the patch block from the conversation.

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
