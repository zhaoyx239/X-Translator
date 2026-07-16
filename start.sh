#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

export XTRANSLATE_CONFIG="${XTRANSLATE_CONFIG:-xtranslate/config.json}"

echo "xtranslate starting"
echo "CONFIG: ${XTRANSLATE_CONFIG}"

python xtranslate/main.py
