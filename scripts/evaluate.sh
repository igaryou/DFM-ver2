#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: $0 CHECKPOINT [extra evaluate.py args]" >&2
  exit 2
fi
checkpoint="$1"
shift
cd "$(dirname "$0")/.."
uv run python src/evaluate.py \
  --config configs/cityscapes/esd/stage2.yaml \
  --checkpoint "$checkpoint" "$@"
