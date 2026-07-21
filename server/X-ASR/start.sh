#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

MODEL_DIR="${MODEL_DIR:-./96_chunk1s_punctuation}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"
PROVIDER="${PROVIDER:-cpu}"

python sherpa_streaming_server.py \
    --host "${HOST}" \
    --port "${PORT}" \
    --tokens "${MODEL_DIR}/tokens.txt" \
    --encoder "${MODEL_DIR}/encoder.onnx" \
    --decoder "${MODEL_DIR}/decoder.onnx" \
    --joiner "${MODEL_DIR}/joiner.onnx" \
    --provider "${PROVIDER}"
