#!/usr/bin/env bash
# Phase 2: GRPO training + eval for 4B models
# Run from vision-coder directory inside Apptainer with /dev/shm/vllm active.
set -eo pipefail
PYTHON=/dev/shm/vllm/bin/python3
ACCELERATE=/dev/shm/vllm/bin/accelerate
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SCRIPT_DIR"

SFT_MODEL="$SCRIPT_DIR/outputs/vcoder-4b-sft"
RL_MODEL="$SCRIPT_DIR/outputs/vcoder-4b-sft-rl"
TESTSET=../Design2Code/testset_final_extracted

echo "=== grpo_and_eval_4b.sh started at $(date) ==="

echo ""
echo "=== Step 1: GRPO 500 steps on 4B SFT checkpoint ==="
rm -rf "$RL_MODEL"
PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 $ACCELERATE launch \
    --config_file configs/accelerate_2gpu.yaml \
    vcoder/pipelines/training.py \
    --model_id "$SFT_MODEL" \
    --output_dir "$RL_MODEL" \
    --max_samples 2000 \
    --max_steps 500 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --num_generations 8 \
    --max_completion_length 2048 \
    --beta 0.0 \
    --optim adamw_bnb_8bit \
    --vllm_gpu_memory_utilization 0.3 \
    --save_steps 100 \
    --save_total_limit 3 \
    2>&1 | tee outputs/grpo_4b_train.log

echo "GRPO done at $(date)"

echo ""
echo "=== Step 2: Re-run inference for vcoder-4b-sft (with correct absolute model path) ==="
rm -rf outputs/benchmark/vcoder-4b-sft
CUDA_VISIBLE_DEVICES=0 $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$SFT_MODEL" \
    --port 8000 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 8192 \
    --trust-remote-code \
    --dtype bfloat16 &
VLLM_PID=$!
until curl -s http://localhost:8000/health > /dev/null 2>&1; do sleep 5; done
echo "Running inference: vcoder-4b-sft"
$PYTHON vcoder/eval/generate_predictions.py \
    --model vcoder-4b-sft \
    --testset_dir $TESTSET \
    --concurrency 16
kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null || true

echo ""
echo "=== Step 3: Inference for vcoder-4b-sft-rl ==="
CUDA_VISIBLE_DEVICES=0 $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$RL_MODEL" \
    --port 8000 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 8192 \
    --trust-remote-code \
    --dtype bfloat16 &
VLLM_PID=$!
until curl -s http://localhost:8000/health > /dev/null 2>&1; do sleep 5; done
echo "Running inference: vcoder-4b-sft-rl"
$PYTHON vcoder/eval/generate_predictions.py \
    --model vcoder-4b-sft-rl \
    --testset_dir $TESTSET \
    --concurrency 16
kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null || true

echo ""
echo "=== Step 4: Design2Code eval for 4B models ==="
cd ../Design2Code/Design2Code
PYTHONPATH=$(pwd) $PYTHON metrics/multi_processing_eval_4b.py 2>&1 | tee metrics/eval_4b_results.log

echo ""
echo "=== All done at $(date) ==="
