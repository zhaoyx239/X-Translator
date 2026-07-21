#!/usr/bin/env bash

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Tencent-Hunyuan/Hunyuan-MT-7B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-hunyuan_mt}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
LOG_FILE="${LOG_FILE:-log_hunyuan_server.txt}"

python3 -m vllm.entrypoints.openai.api_server \
    --host "${HOST}" \
    --port "${PORT}" \
    --trust-remote-code \
    --model "${MODEL_PATH}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --dtype "${DTYPE}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    2>&1 | tee "${LOG_FILE}"
