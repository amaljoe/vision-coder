#!/usr/bin/env bash
set -euo pipefail

# Start a local OpenAI-compatible vLLM server for GRPO training.
#
# Usage examples:
#   bash scripts/vllm_for_training.sh
#   CUDA_VISIBLE_DEVICES=4 bash scripts/vllm_for_training.sh
#   MODEL_ID=Qwen/Qwen3-VL-2B-Instruct PORT=8001 bash scripts/vllm_for_training.sh
#
# Pair with trainer in another terminal:
#   CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
#   NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 \
#   python test_train.py

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

# Default to one GPU if caller didn't set CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[vLLM] MODEL_ID=${MODEL_ID}"
echo "[vLLM] HOST=${HOST} PORT=${PORT}"
echo "[vLLM] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[vLLM] TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"

trl vllm-serve \
  --model "${MODEL_ID}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --host "${HOST}" \
  --port "${PORT}"
