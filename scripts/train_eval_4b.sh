#!/usr/bin/env bash
# Full pipeline: wait for 4B SFT → GRPO 500 steps → Eval
# Run from vision-coder directory inside Apptainer with /dev/shm/vllm active.
set -e
PYTHON=/dev/shm/vllm/bin/python3
ACCELERATE=/dev/shm/vllm/bin/accelerate
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SCRIPT_DIR"

echo "=== Step 1: Wait for 4B SFT to finish ==="
until [ -d outputs/vcoder-4b-sft ] && [ -f outputs/vcoder-4b-sft/config.json ]; do
    echo "Waiting for SFT... ($(date '+%H:%M'))"
    sleep 60
done
echo "SFT checkpoint ready at $(date)"

echo ""
echo "=== Step 2: Train 4B GRPO (500 steps, 8-bit optimizer for memory) ==="
rm -rf outputs/vcoder-4b-sft-rl
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 $ACCELERATE launch \
    --config_file configs/accelerate_2gpu.yaml \
    vcoder/pipelines/training.py \
    --model_id outputs/vcoder-4b-sft \
    --output_dir outputs/vcoder-4b-sft-rl \
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
    --vllm_gpu_memory_utilization 0.4 \
    --save_steps 100 \
    --save_total_limit 3 \
    2>&1 | tee outputs/grpo_4b_train.log

echo "GRPO training done at $(date)"

echo ""
echo "=== Step 3: Generate predictions for 4B base + SFT + SFT+RL ==="
TESTSET=../Design2Code/testset_final_extracted

# 4B base model
CUDA_VISIBLE_DEVICES=0 $PYTHON -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 8192 \
    --trust-remote-code \
    --dtype bfloat16 &
VLLM_PID=$!
until curl -s http://localhost:8000/health > /dev/null 2>&1; do sleep 5; done
echo "Running inference: qwen-base-4b"
$PYTHON vcoder/eval/generate_predictions.py \
    --model qwen-base-4b \
    --testset_dir $TESTSET \
    --concurrency 16
kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null || true

# 4B SFT model
CUDA_VISIBLE_DEVICES=0 $PYTHON -m vllm.entrypoints.openai.api_server \
    --model outputs/vcoder-4b-sft \
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

# 4B SFT+RL model
CUDA_VISIBLE_DEVICES=0 $PYTHON -m vllm.entrypoints.openai.api_server \
    --model outputs/vcoder-4b-sft-rl \
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
echo "=== All predictions generated at $(date) ==="
echo "Now run Design2Code eval:"
echo "  cd ../Design2Code/Design2Code"
echo "  python3 metrics/multi_processing_eval.py"
