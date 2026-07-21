#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
MODEL_PATH="${MODEL_PATH:-NiuTrans/LMT-60-8B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-lmt}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
LOG_FILE="${LOG_FILE:-log_lmt_server.txt}"

python3 -m vllm.entrypoints.openai.api_server \
    --host "${HOST}" \
    --port "${PORT}" \
    --trust-remote-code \
    --model "${MODEL_PATH}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --dtype "${DTYPE}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    2>&1 | tee "${LOG_FILE}"
