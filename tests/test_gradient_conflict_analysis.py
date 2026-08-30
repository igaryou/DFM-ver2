from __future__ import annotations

import math

import torch

from gradient_conflict_analysis import (
    _swin_v5_checkpoint_key_to_v4,
    gradient_pair_metrics,
    projection_diagnostics,
)


def _gradient(values: list[float]):
    return (torch.tensor(values, dtype=torch.float32),)


def test_opposite_gradients_have_negative_one_cosine() -> None:
    metrics = gradient_pair_metrics(_gradient([1.0, 0.0]), _gradient([-1.0, 0.0]))
    assert math.isclose(metrics["cosine"], -1.0, abs_tol=1.0e-9)
    assert math.isclose(metrics["angle_degrees"], 180.0, abs_tol=1.0e-4)
    assert metrics["conflict"] is True


def test_orthogonal_gradients_have_zero_cosine() -> None:
    metrics = gradient_pair_metrics(_gradient([1.0, 0.0]), _gradient([0.0, 1.0]))
    assert metrics["cosine"] == 0.0
    assert metrics["angle_degrees"] == 90.0
    assert metrics["conflict"] is False


def test_conflicting_projection_is_orthogonal_to_base() -> None:
    base = _gradient([1.0, 0.0])
    psd = _gradient([-1.0, 1.0])
    metrics = projection_diagnostics(base, psd)
    assert metrics["projection_applied"] is True
    assert abs(metrics["base_vs_projected_psd_cosine"]) < 1.0e-6
    assert math.isclose(metrics["projection_coefficient"], -1.0, abs_tol=1.0e-9)


def test_swin_v5_checkpoint_keys_convert_to_transformers_4_layout() -> None:
    prefix = "image_encoder.backbone.swin.encoder.layers.0.blocks.0"
    assert _swin_v5_checkpoint_key_to_v4(
        f"{prefix}.attention.q_proj.weight"
    ) == (
        "image_encoder.backbone.encoder.layers.0.blocks.0."
        "attention.self.query.weight"
    )
    assert _swin_v5_checkpoint_key_to_v4(
        f"{prefix}.attention.relative_position_bias.relative_position_bias_table"
    ) == (
        "image_encoder.backbone.encoder.layers.0.blocks.0."
        "attention.self.relative_position_bias_table"
    )
    assert _swin_v5_checkpoint_key_to_v4(
        "image_encoder.backbone.swin.layernorm.weight"
    ) is None
