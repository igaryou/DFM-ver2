cd /home/igarashi_25/DFM

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/cityscapes/original/psd/joint_bounded_gaussian_b1_exponential_path_adaptive_std_trainable.yaml

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/cityscapes/original/psd/joint_bounded_gaussian_b1_exponential_path_adaptive_std_frozen.yaml.yaml





cd /home/igarashi_25/DFM

CUDA_VISIBLE_DEVICES=0 \
uv run python src/evaluate.py \
  --config configs/cityscapes/original/psd/joint_bounded_gaussian_b1_exponential_path_adaptive_std_trainable.yaml \
  --checkpoint /home/igarashi_25/DFM/results/cityscapes/original/b1_exponential_path_adaptive_std_trainable_800ep/best_flow.pt

cd /home/igarashi_25/DFM

CUDA_VISIBLE_DEVICES=0 \
uv run python src/evaluate.py \
  --config configs/cityscapes/original/psd/joint_bounded_gaussian_b1_exponential_path_adaptive_std_trainable.yaml \
  --checkpoint /home/igarashi_25/DFM/results/cityscapes/original/b1_exponential_path_adaptive_std_trainable_800ep/best_flow.pt \
  --set evaluation.num_samples=8 \
  --set evaluation.aggregation=probability_mean