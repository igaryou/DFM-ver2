PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config /home/igarashi_25/DFM/configs/cityscapes/original/_base_/cityscapes_256×512_b1_learned_sigma_varreg_800ep.yaml


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config /home/igarashi_25/DFM/configs/cityscapes/original/_base_/cityscapes_256x512_800ep_rrdb_ver2.yaml


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config /home/igarashi_25/DFM/configs/cityscapes/original/_base_/cityscapes_256×512_simplex_cfm.yaml

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config /home/igarashi_25/DFM/configs/cityscapes/original/_base_/cityscapes_256×512_800ep_rrdb_no_align.yaml