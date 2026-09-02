#!/usr/bin/env bash
set -euo pipefail

cd /home/igarashi_25/DFM
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 \
uv run torchrun --standalone --nproc_per_node=2 src/train_joint.py \
  --config configs/cityscapes/esd/joint.yaml



PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/cityscapes/esd/swin_t_core_resume.yaml


cd /home/igarashi_25/DFM

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/cityscapes/esd/swin_t_dfm_recipe_resume.yaml
