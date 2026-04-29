#!/usr/bin/env bash
# Master script: download data → SFT → RL → SFT+RL → Eval
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_all.sh
# Logs are written to logs/ directory.
set -euo pipefail

PYTHON=/dev/shm/qwen35/bin/python
ACCELERATE=/dev/shm/qwen35/bin/accelerate
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
MAX_SAMPLES=2000
EVAL_SAMPLES=100

mkdir -p logs outputs

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/run_all.log; }

# ---- 1. Download/cache dataset ----
log "Step 1: Downloading WebSight dataset (GRPO + SFT formats)..."

HF_DATASETS_OFFLINE=0 $PYTHON -c "
from vcoder.data.websight import load_websight_dataset, load_websight_sft_dataset
print('Loading GRPO dataset...')
ds = load_websight_dataset(max_samples=$MAX_SAMPLES)
print(f'GRPO dataset: {len(ds)} samples')
print('Loading SFT dataset...')
ds2 = load_websight_sft_dataset(max_samples=$MAX_SAMPLES)
print(f'SFT dataset: {len(ds2)} samples')
" 2>&1 | tee logs/download.log

log "Data ready."

# ---- 2. SFT training ----
log "Step 2: SFT training (base → SFT)..."

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 \
$ACCELERATE launch \
    --config_file configs/accelerate_2gpu.yaml \
    vcoder/pipelines/sft_training.py \
    --model_id "$MODEL_ID" \
    --output_dir outputs/vcoder-sft \
    --max_samples $MAX_SAMPLES \
    --num_train_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --max_length 2048 \
2>&1 | tee logs/sft_training.log

log "SFT training done → outputs/vcoder-sft"

# ---- 3. RL training (base → RL) ----
log "Step 3: RL training (base → RL)..."

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 \
$ACCELERATE launch \
    --config_file configs/accelerate_2gpu.yaml \
    vcoder/pipelines/training.py \
    --model_id "$MODEL_ID" \
    --output_dir outputs/vcoder-rl \
    --max_samples $MAX_SAMPLES \
    --num_train_epochs 1 \
    --batch_size 2 \
    --num_generations 4 \
    --max_completion_length 1024 \
    --vllm_max_model_length 3000 \
2>&1 | tee logs/rl_training.log

log "RL training done → outputs/vcoder-rl"

# ---- 4. SFT+RL training (SFT → SFT+RL) ----
log "Step 4: SFT+RL training (SFT → SFT+RL)..."

# Find best SFT checkpoint
SFT_CKPT=$(ls -d outputs/vcoder-sft/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "$SFT_CKPT" ]; then
    SFT_CKPT="outputs/vcoder-sft"
fi
log "  Starting from SFT checkpoint: $SFT_CKPT"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 \
$ACCELERATE launch \
    --config_file configs/accelerate_2gpu.yaml \
    vcoder/pipelines/training.py \
    --model_id "$SFT_CKPT" \
    --output_dir outputs/vcoder-sft-rl \
    --max_samples $MAX_SAMPLES \
    --num_train_epochs 1 \
    --batch_size 2 \
    --num_generations 4 \
    --max_completion_length 1024 \
    --vllm_max_model_length 3000 \
2>&1 | tee logs/sft_rl_training.log

log "SFT+RL training done → outputs/vcoder-sft-rl"

# ---- 5. Evaluation ----
log "Step 5: Evaluating all models on ${EVAL_SAMPLES} held-out samples..."

# Find best checkpoints
RL_CKPT=$(ls -d outputs/vcoder-rl/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "$RL_CKPT" ]; then RL_CKPT="outputs/vcoder-rl"; fi

SFT_RL_CKPT=$(ls -d outputs/vcoder-sft-rl/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "$SFT_RL_CKPT" ]; then SFT_RL_CKPT="outputs/vcoder-sft-rl"; fi

log "  Using checkpoints:"
log "    RL:     $RL_CKPT"
log "    SFT+RL: $SFT_RL_CKPT"

CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | cut -d, -f1) \
$PYTHON vcoder/eval/eval_standalone.py \
    --model_ids \
        "$MODEL_ID" \
        "outputs/vcoder-sft" \
        "$RL_CKPT" \
        "$SFT_RL_CKPT" \
    --model_names base sft rl sft_rl \
    --num_samples $EVAL_SAMPLES \
    --max_new_tokens 1024 \
    --output_json outputs/eval_results.json \
2>&1 | tee logs/eval.log

log "All done! Results in outputs/eval_results.json"
