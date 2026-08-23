#!/usr/bin/env bash
set -euo pipefail

cd /home/igarashi_25/DFM

CUDA_VISIBLE_DEVICES=0 \
uv run python src/train.py --config configs/debug_diagonal_cityscapes.yaml

for loss in psd csd ecld esd; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0,1 \
  uv run torchrun --standalone --nproc_per_node=2 src/train.py \
    --config "configs/debug_ddp_stage2_${loss}.yaml"
done

for loss in psd csd ecld esd; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0,1 \
  uv run torchrun --standalone --nproc_per_node=2 src/train_joint.py \
    --config "configs/debug_ddp_joint_${loss}.yaml"
done
