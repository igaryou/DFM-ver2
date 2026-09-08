from __future__ import annotations


SEGFORMER_MODEL_NAMES = {f"b{i}": f"nvidia/mit-b{i}" for i in range(6)}
SEGFORMER_DEPTHS = {
    "b0": [2, 2, 2, 2],
    "b1": [2, 2, 2, 2],
    "b2": [3, 4, 6, 3],
    "b3": [3, 4, 18, 3],
    "b4": [3, 8, 27, 3],
    "b5": [3, 6, 40, 3],
}
SEGFORMER_HIDDEN_SIZES = {
    "b0": [32, 64, 160, 256],
    "b1": [64, 128, 320, 512],
    "b2": [64, 128, 320, 512],
    "b3": [64, 128, 320, 512],
    "b4": [64, 128, 320, 512],
    "b5": [64, 128, 320, 512],
}
SEGFORMER_DECODER_HIDDEN = {
    "b0": 256,
    "b1": 256,
    "b2": 768,
    "b3": 768,
    "b4": 768,
    "b5": 768,
}
SEGFORMER_ATTENTION_HEADS = [1, 2, 5, 8]
SEGFORMER_SR_RATIOS = [8, 4, 2, 1]
SEGFORMER_PATCH_SIZES = [7, 3, 3, 3]
SEGFORMER_STRIDES = [4, 2, 2, 2]


def build_segformer_config(variant: str, in_channels: int):
    """Build the shared MiT architecture used by source and endpoint models."""
    if variant not in SEGFORMER_MODEL_NAMES:
        raise ValueError(f"Unknown SegFormer variant: {variant}")
    try:
        from transformers import SegformerConfig
    except ImportError as exc:
        raise RuntimeError("SegFormer models require transformers") from exc
    return SegformerConfig(
        num_channels=in_channels,
        num_encoder_blocks=4,
        depths=list(SEGFORMER_DEPTHS[variant]),
        sr_ratios=list(SEGFORMER_SR_RATIOS),
        hidden_sizes=list(SEGFORMER_HIDDEN_SIZES[variant]),
        patch_sizes=list(SEGFORMER_PATCH_SIZES),
        strides=list(SEGFORMER_STRIDES),
        num_attention_heads=list(SEGFORMER_ATTENTION_HEADS),
        mlp_ratios=[4, 4, 4, 4],
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
    )
