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
  --config configs/stage2_esd_cityscapes.yaml \
  --checkpoint "$checkpoint" "$@"

