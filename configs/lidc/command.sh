cd /home/igarashi_25/DFM

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/lidc/psd/segformer_source_align_joint.yaml


cd /home/igarashi_25/DFM

CUDA_VISIBLE_DEVICES=2 \
uv run python scripts/visualize_lidc_source.py \
  --config configs/lidc/psd/segformer_source_align_joint.yaml \
  --checkpoint /home/igarashi_25/DFM/results/lidc/segformer_source_align_psd/epoch_0120.pt \
  --output-dir /home/igarashi_25/DFM/results/lidc/segformer_source_align_psd/source_visualization_epoch0120 \
  --num-visualizations 32 \
  --split val


CUDA_VISIBLE_DEVICES=2 \
uv run python scripts/visualize_lidc_source.py \
  --config configs/lidc/psd/segformer_source_align_joint.yaml \
  --checkpoint /home/igarashi_25/DFM/results/lidc/segformer_source_align_psd/epoch_0480.pt \
  --output-dir /home/igarashi_25/DFM/results/lidc/segformer_source_align_psd/source_visualization_epoch480 \
  --num-visualizations 32 \
  --num-source-samples 16 \
  --split val \
  --seed 42

CUDA_VISIBLE_DEVICES=2 \
uv run python scripts/visualize_lidc_gt_best_predictions.py \
  --config configs/lidc/psd/segformer_source_align_joint.yaml \
  --checkpoint /home/igarashi_25/DFM/results/lidc/segformer_source_align_psd/epoch_0500.pt \
  --output-dir /home/igarashi_25/DFM/results/lidc/segformer_source_align_psd/best_gt_visualization_500pt \
  --num-visualizations 10 \
  --num-samples 16 \
  --num-steps 1 \
  --split val \
  --seed 42