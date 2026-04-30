#!/usr/bin/env bash
# Post-GRPO: inference for vcoder-4b-sft-rl + Design2Code eval (4×L40S, TP=4)
# vcoder-4b-sft predictions already done — skip that step.
# Run inside Apptainer with /dev/shm/vllm active, from vision-coder directory.
set -eo pipefail
PYTHON=/dev/shm/vllm/bin/python3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SCRIPT_DIR"

SFT_MODEL="$SCRIPT_DIR/outputs/vcoder-4b-sft"
RL_MODEL="$SCRIPT_DIR/outputs/vcoder-4b-sft-rl"
TESTSET=../Design2Code/testset_final_extracted

echo "=== post_grpo_eval_4b.sh started at $(date) ==="

echo ""
echo "=== Step 1: Inference — vcoder-4b-sft-rl (4-GPU TP=4) ==="
rm -rf outputs/benchmark/vcoder-4b-sft-rl
CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$RL_MODEL" \
    --port 8000 \
    --tensor-parallel-size 4 \
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
    --concurrency 32
kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null || true

echo ""
echo "=== Step 2: Design2Code eval for all 4B models ==="
cd ../Design2Code/Design2Code
PYTHONPATH=$(pwd) $PYTHON metrics/multi_processing_eval_4b.py 2>&1 | tee metrics/eval_4b_results.log

echo ""
echo "=== ALL DONE at $(date) ==="
