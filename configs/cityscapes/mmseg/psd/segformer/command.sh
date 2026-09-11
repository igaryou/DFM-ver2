cd /home/igarashi_25/DFM

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/cityscapes/mmseg/psd/segformer/joint_ce_160k.yaml \
  --set runtime.debug_first_batch_shapes=true