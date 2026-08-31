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

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1,2 \
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/joint_psd_ade20k_swin_t.yaml

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/joint_psd_ade20k_swin_t.yaml



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


cd /home/igarashi_25/DFM

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/joint_psd_cityscapes_swin_t.yaml


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/joint_psd_cityscapes_swin_t_dfm_recipe.yaml














PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2 \
python scripts/diagnose_psd_gradient_conflict.py \
  --config /home/igarashi_25/DFM/results/joint_psd_cityscapes_swin_t__160k/config_resolved.yaml \
  --checkpoint /rda5/users/igarashi_25/DFM/results/joint_psd_cityscapes_swin_t__160k/step_096000.pt \
  --num-batches 32 \
  --batch-size 8 \
  --psd-weight 0.5 \
  --seed 42


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2 \
python scripts/diagnose_psd_time_maps.py \
  --config /home/igarashi_25/DFM/results/joint_psd_cityscapes_swin_t_adaptive_surgery_resume_160k/config_resolved.yaml \
  --checkpoint /home/igarashi_25/DFM/results/joint_psd_cityscapes_swin_t_adaptive_surgery_resume_160k/step_112000.pt \
  --batch-size 8 \
  --time-bin-batches 8 \
  --eval-batch-size 8 \
  --eval-num-batches 20 \
  --psd-weight 0.5 \
  --teacher-confidence-threshold 0.9 \
  --seed 42