#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python src/train.py --config configs/cityscapes/diagonal/stage1.yaml "$@"
