cd /home/igarashi_25/DFM

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/cityscapes/psd/joint_swin_t_segformer_b1_standard_ce_160k.yaml \
  --set flow.path.type=power \
  --set flow.path.exponent=2.0 \
  --set experiment.name=dfm_joint_psd_cityscapes_swin_t_b1_standard_ce_power2_160k \
  --set experiment.output_dir=/home/igarashi_25/DFM/results/cityscapes/joint_psd_cityscapes_swin_t_b1_standard_ce_power2_160k \
  --set wandb.name=dfm-joint-psd-cityscapes-swin-t-b1-standard-ce-power2-160k


cd /home/igarashi_25/DFM


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/cityscapes/psd/joint_simplex_b1_ce_ignore_void_160k.yaml

cd /home/igarashi_25/DFM


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/cityscapes/psd/joint_simplex_b1_ce_include_void_160k.yaml