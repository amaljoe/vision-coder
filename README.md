# VisionCoder — RLVR for Screenshot-to-HTML Generation

Fine-tuning a vision-language model with **Reinforcement Learning from Verifiable Rewards (RLVR)** to convert UI screenshots into clean HTML/CSS.

> **Input:** UI screenshot &nbsp;→&nbsp; **Output:** HTML + CSS that visually reproduces it

---

## Results

Trained on 500 steps of GRPO with [HuggingFaceM4/WebSight](https://huggingface.co/datasets/HuggingFaceM4/WebSight) (2 000 samples), evaluated on the [Design2Code](https://github.com/NoviScl/Design2Code) benchmark (484 held-out examples).

| Metric | Qwen3-VL-2B (base) | **VCoder-GRPO-CLIP** | Δ |
|---|---|---|---|
| **Overall** | 0.232 | **0.788** | +240% |
| Block-Match | 0.101 | **0.795** | +687% |
| Text Match | 0.107 | **0.884** | +722% |
| Position | 0.088 | **0.706** | +703% |
| Color | 0.091 | **0.698** | +663% |
| CLIP Similarity | 0.772 | **0.858** | +11% |

The base model struggles to produce structured HTML in the correct format; a single epoch of GRPO training with rendering-based rewards closes most of the gap.

---

## Training Overview

![Training Overview](assets/plots/overview.png)

Key observations across 500 steps:
- **Total reward** rises steadily from ~2.4 to ~5.0
- **CLIP reward** (visual fidelity) climbs from near-zero to ~2.4 after 3× boosting
- **Format + validity** rewards converge to near-perfect within ~100 steps, freeing the model to focus on visual quality
- **Completion length** drops sharply (~1 000 → ~460 tokens) as the model learns to generate clean, minimal HTML
- **Entropy** decreases monotonically, indicating a confident but not collapsed policy

---

## Reward Design

| Reward | Weight | Signal |
|---|---|---|
| `boosted_clip_reward` | 3× | CLIP image-image similarity between rendered HTML and reference screenshot |
| `format_reward` | 1× | Presence of `<think>` + `<html>` structure |
| `html_validity_reward` | 1× | HTML parses without critical errors |
| `structural_similarity_reward` | 1× | DOM-level structural similarity to reference |

All rewards are computed without any human annotation. Rendering is done with a headless Playwright browser pool.

---

## Detailed Training Curves

### Reward Signals
![Reward Signals](assets/plots/rewards.png)

### Training Dynamics
![Training Dynamics](assets/plots/training_dynamics.png)

### Completion Statistics
![Completions](assets/plots/completions.png)

---

## Qualitative Samples

Three examples from the Design2Code testset. Each row: reference screenshot → base model render → VCoder-GRPO-CLIP render.

### Sample 1 (testset idx 3) — CLIP: base 0.828 → ft **0.947** (+0.12)

| Reference | Base (Qwen3-VL-2B) | VCoder-GRPO-CLIP |
|:---:|:---:|:---:|
| ![](assets/results/sample_1/reference.png) | ![](assets/results/sample_1/base_rendered.png) | ![](assets/results/sample_1/finetuned_rendered.png) |

### Sample 2 (testset idx 7) — CLIP: base 0.785 → ft **0.849** (+0.06)

| Reference | Base (Qwen3-VL-2B) | VCoder-GRPO-CLIP |
|:---:|:---:|:---:|
| ![](assets/results/sample_2/reference.png) | ![](assets/results/sample_2/base_rendered.png) | ![](assets/results/sample_2/finetuned_rendered.png) |

### Sample 3 (testset idx 5) — CLIP: base 0.557 → ft **0.792** (+0.23)

| Reference | Base (Qwen3-VL-2B) | VCoder-GRPO-CLIP |
|:---:|:---:|:---:|
| ![](assets/results/sample_3/reference.png) | ![](assets/results/sample_3/base_rendered.png) | ![](assets/results/sample_3/finetuned_rendered.png) |

#### Per-sample CLIP Similarity (10 testset images)

| idx | Base | VCoder-GRPO-CLIP | Δ |
|---|---|---|---|
| 0 | 0.525 | 0.542 | +0.018 |
| 1 | 0.547 | 0.788 | +0.241 |
| 2 | 0.380 | 0.369 | −0.011 |
| **3** | **0.828** | **0.947** | **+0.120** |
| 4 | 0.725 | 0.760 | +0.035 |
| **5** | **0.557** | **0.792** | **+0.235** |
| 6 | 0.789 | 0.633 | −0.156 |
| **7** | **0.785** | **0.849** | **+0.064** |
| 8 | 0.428 | 0.499 | +0.072 |
| 9 | 0.754 | 0.752 | −0.001 |

Bold rows = shown above. Average across 10 samples: base 0.632 → ft **0.693** (+0.061).

---

## Architecture

```
vcoder/
├── pipelines/training.py       # GRPO training entry point (accelerate launch)
├── rewards/
│   ├── visual_rewards.py       # CLIP + SSIM rewards via async Playwright rendering
│   ├── structural_rewards.py   # DOM tree similarity reward
│   ├── validity_rewards.py     # HTML validity reward
│   └── format_rewards.py       # Format / thinking-tag reward
├── rendering/
│   ├── html_renderer.py        # Headless browser rendering
│   └── browser_pool.py         # Async Playwright browser pool
├── data/websight.py             # WebSight dataset loader
├── eval/
│   ├── generate_predictions.py # Batch inference via VLLM server
│   └── extract_testset.py      # Extract Design2Code parquet → PNG/HTML
└── demo/inference.py            # Single-image inference
experiments/
└── plot_run.py                  # Plot training curves from trainer_state.json
```

---

## Setup

```bash
# Install package
pip install -e . --no-deps

# Install Playwright for rendering
playwright install chromium
```

---

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file configs/accelerate_4gpu.yaml \
    vcoder/pipelines/training.py \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --output_dir outputs/vcoder-grpo-clip \
    --max_samples 2000 \
    --num_train_epochs 1 \
    --batch_size 4 \
    --num_generations 8 \
    --max_completion_length 2048
```

---

## Evaluation

**1. Start VLLM inference servers**
```bash
# Base model
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-2B-Instruct --port 8000 \
    --gpu-memory-utilization 0.45 --max-model-len 8192 --trust-remote-code

# Fine-tuned model
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --model outputs/vcoder-grpo-clip/checkpoint-500 --port 8001 \
    --gpu-memory-utilization 0.45 --max-model-len 8192 --trust-remote-code
```

**2. Generate predictions**
```bash
python3 vcoder/eval/generate_predictions.py \
    --model vcoder-grpo-clip \
    --testset_dir ../Design2Code/testset_final_extracted
```

**3. Run Design2Code eval**
```bash
cd ../Design2Code/Design2Code
python3 metrics/multi_processing_eval.py
```

---

## Plot Training Curves

```bash
python3 experiments/plot_run.py                          # latest checkpoint
python3 experiments/plot_run.py --run_dir outputs/vcoder-grpo-clip --checkpoint 300
```

Plots saved to `<run_dir>/plots/`.

---

## Model

- **Base model:** [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- **Training:** GRPO (TRL), 4× A40 GPUs, ~500 steps, ~7 hours
- **Dataset:** 2 000 samples from [HuggingFaceM4/WebSight](https://huggingface.co/datasets/HuggingFaceM4/WebSight)
- **Benchmark:** [Design2Code](https://github.com/NoviScl/Design2Code) — 484 held-out screenshot→HTML pairs

---

## Team

Amal Joe · Job J
