#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-ASR-1.7B}"

python qwen3asr_server.py --model "${MODEL_PATH}" "$@"
