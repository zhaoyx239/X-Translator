#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy websockets soundfile librosa sherpa-onnx

echo "Place the Zipformer ONNX files under server/X-ASR/96_chunk1s_punctuation/."
