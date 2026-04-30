#!/usr/bin/env bash
# Waits for the current pipeline to finish, then runs grpo_and_eval_4b.sh inside Apptainer.
# This script runs on the HOST (outside Apptainer).
set -e
WORKSPACE=/home/compiling-ganesh/24m0797/workspace/vision-coder
PIPELINE_PID=1071017
APPTAINER=/home/compiling-ganesh/24m0797/spack/linux-nehalem/apptainer-1.4.1-rc4oxxguim3nfahywxqf7cdhkmyowmmo/bin/apptainer
SIF=/home/compiling-ganesh/24m0797/images/cuda-custom-amal_latest.sif

echo "Waiting for pipeline PID $PIPELINE_PID to finish... ($(date))"
while kill -0 $PIPELINE_PID 2>/dev/null; do
    sleep 30
done
echo "Pipeline finished at $(date)"

echo "Killing any lingering vLLM processes..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 5

echo "Launching grpo_and_eval_4b.sh inside Apptainer..."
$APPTAINER exec --nv $SIF bash -c "cd $WORKSPACE && bash scripts/grpo_and_eval_4b.sh" \
    2>&1 | tee $WORKSPACE/outputs/grpo_eval_4b_full.log

echo "All done at $(date)"
