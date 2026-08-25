#!/usr/bin/env bash
set -euo pipefail

cd /home/igarashi_25/DFM
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 \
uv run torchrun --standalone --nproc_per_node=2 src/train_joint.py \
  --config configs/joint_psd_cityscapes.yaml

cd /home/igarashi_25/DFM

CUDA_VISIBLE_DEVICES=1,2 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/joint_psd_ade20k.yaml


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
uv run python scripts/diagnose_source_ade20k.py \
  --config configs/joint_psd_ade20k.yaml \
  --checkpoint results/joint_psd_ade20k_ver2/latest.pt \
  --output_dir results/source_diagnostics_final \
  --sigma_values 1.0 0.75 0.5 0.25 0.1 0.0 \
  --step_values 1 2 3 5 \
  --num_visualize 100 \
  --seed 42 \
  --amp \
  --amp_dtype bf16