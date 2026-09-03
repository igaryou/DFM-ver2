#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python src/train.py --config configs/cityscapes/diagonal/stage1.yaml "$@"


cd /home/igarashi_25/DFM

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2 \
uv run torchrun \
  --standalone \
  --nproc_per_node=1 \
  src/train.py \
  --config configs/cityscapes/diagonal/source_segformer_b1_32k.yaml