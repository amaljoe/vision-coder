CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m vllm.entrypoints.openai.api_server \
  --model datalab-to/chandra \
  --served-model-name chandra \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 131072 \
  --host 0.0.0.0 \
  --port 8000
