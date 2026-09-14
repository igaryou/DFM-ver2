cd /home/igarashi_25/DFM

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train.py \
  --config configs/lidc/psd/segformer_source_align_joint.yaml