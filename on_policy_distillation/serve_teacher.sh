#!/usr/bin/env bash
set -euo pipefail

export USE_HF=1
export ENABLE_AUDIO_OUTPUT=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"

exec swift deploy \
  --model "${TEACHER_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}" \
  --infer_backend vllm \
  --host 127.0.0.1 \
  --port "${TEACHER_PORT:-8001}" \
  --torch_dtype bfloat16 \
  --enable_thinking false \
  --max_logprobs 1 \
  --max_length 8192 \
  --vllm_max_model_len 8192 \
  --vllm_tensor_parallel_size "${TENSOR_PARALLEL_SIZE:-${#devices[@]}}" \
  --vllm_max_num_seqs 1 \
  --vllm_limit_mm_per_prompt "{\"audio\": ${MAX_AUDIOS:-3}}" \
  "$@"
