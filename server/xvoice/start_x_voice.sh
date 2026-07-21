#!/usr/bin/env bash

set -euo pipefail

: "${XVOICE_ROOT:?Set XVOICE_ROOT to /PATH/TO/X-Voice}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${XVOICE_ROOT}"
export XVOICE_ROOT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/server.py" \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-11998}" \
    --stage "${XVOICE_STAGE:-2}" \
    --no-auto-detect-lang \
    --ckpt-file "${XVOICE_CKPT_FILE:-ckpts/XVoice_Base_Stage2/model_70000.safetensors}" \
    --srp-ckpt-file "${XVOICE_SRP_CKPT_FILE:-ckpts/SpeedPredictor/model_28000.safetensors}" \
    --device "${XVOICE_DEVICE:-cuda}"
