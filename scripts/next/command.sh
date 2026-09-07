cd /home/igarashi_25/DFM

#path設計-checkpointあり-freeze
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/cityscapes/psd/joint_bounded_gaussian_b1_ce_160k_exponential_path_frozen_source.yaml


#path & 分散設計-checkpointあり-freeze
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/cityscapes/psd/joint_bounded_gaussian_b1_ce_160k_exponential_path_adaptive_std_frozen_source.yaml

#path設計-joint
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/cityscapes/psd/joint_bounded_gaussian_b1_ce_160k_exponential_path_trainable_source.yaml \
  --set source.checkpoint=null \
  --set experiment.name=dfm_bounded_gaussian_exponential_path_joint_source_160k \
  --set experiment.output_dir=/home/igarashi_25/DFM/results/cityscapes/bounded_gaussian_exponential_path_joint_source_160k \
  --set wandb.name=dfm-bounded-gaussian-exponential-path-joint-source-160k



#path & 分散設計-joint 本命
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/cityscapes/psd/joint_bounded_gaussian_b1_ce_160k_exponential_path_adaptive_std_trainable_source.yaml