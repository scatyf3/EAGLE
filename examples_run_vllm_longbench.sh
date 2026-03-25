#!/bin/bash

# Example script to run LongBench with vLLM on 3x H100 GPUs with auto-restart

ANSWER_FILE="outputs/qwen3_1.7b_eagle3_vllm_longbench.jsonl"

python scripts/run_longbench_vllm_qwen3.py \
  --base-model-path Qwen/Qwen3-1.7B \
  --tensor-parallel 3 \
  --pipeline-parallel 1 \
  --max-model-len 131072 \
  --vllm-port 8000 \
  --gpu-memory-utilization 0.9 \
  --auto-restart-vllm \
  --vllm-timeout 300 \
  --api-base "http://127.0.0.1:8000" \
  --api-key "EMPTY" \
  --model "qwen3_1.7b" \
  --bench-name longbench_ctx_sampled_1k_128k \
  --answer-file "${ANSWER_FILE}" \
  --max-new-token 256 \
  --temperature 0.0 \
  --request-timeout 600
