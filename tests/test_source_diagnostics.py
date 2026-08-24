from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from discrete_flow_maps import flow_map, make_time_grid, sample_prior
from inference import sample_segmentation, sample_segmentation_from_x0
from losses import masked_mean
from metrics import SegmentationMetrics
from source_diagnostics import (
    _json_safe,
    _segformer_v5_key_to_v4,
    deterministic_epsilon_like,
    diagnostic_initial_state,
    mu_gt_cosine,
    oracle_alignment_map,
    oracle_state_from_target,
    resolve_diagnostic_checkpoint,
)
import state_space
from state_space import resize_continuous


def test_sigma_zero_mu_zero_and_shared_epsilon_construction():
    mu = torch.randn(2, 5, 3, 4)
    epsilon = torch.randn_like(mu)
    torch.testing.assert_close(
        diagnostic_initial_state(mu, epsilon, 0.0), mu
    )
    torch.testing.assert_close(
        diagnostic_initial_state(mu, epsilon, 0.75, mu_zero=True),
        0.75 * epsilon,
    )
    high = diagnostic_initial_state(mu, epsilon, 1.0)
    low = diagnostic_initial_state(mu, epsilon, 0.25)
    torch.testing.assert_close(high - low, 0.75 * epsilon)


def test_diagnostic_epsilon_is_seed_deterministic_and_global_rng_independent():
    reference = torch.zeros(1, 4, 3, 2)
    first = deterministic_epsilon_like(reference, 123)
    torch.randn(1000)
    second = deterministic_epsilon_like(reference, 123)
    third = deterministic_epsilon_like(reference, 124)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_oracle_uses_nearest_state_one_hot_and_no_full_target_one_hot(monkeypatch):
    target = torch.tensor([[[0, 1, 1, 2], [0, 1, 2, 2],
                            [3, 3, 2, 2], [3, 3, 2, 2]]])
    calls = []
    original = state_space.F.one_hot

    def record_one_hot(tensor, *args, **kwargs):
        calls.append(tuple(tensor.shape))
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(state_space.F, "one_hot", record_one_hot)
    target_state, oracle_state, valid = oracle_state_from_target(
        target, state_size=(2, 2), num_classes=4, ignore_index=0
    )
    expected = F.interpolate(
        target[:, None].float(), size=(2, 2), mode="nearest"
    )[:, 0].long()
    assert torch.equal(target_state, expected)
    assert oracle_state.shape == (1, 4, 2, 2)
    assert calls == [(1, 2, 2)]
    assert valid.shape == target.shape


def test_oracle_upsample_is_bilinear_and_ignore_is_excluded():
    target = torch.tensor([[[0, 1, 1, 1], [1, 1, 1, 1],
                            [2, 2, 2, 2], [2, 2, 2, 2]]])
    align, target_state, valid = oracle_alignment_map(
        target,
        state_size=(2, 2),
        num_classes=3,
        ignore_index=0,
        eps=1.0e-8,
    )
    state_one_hot = F.one_hot(target_state, 3).permute(0, 3, 1, 2).float()
    bilinear = F.interpolate(
        state_one_hot, size=(4, 4), mode="bilinear", align_corners=False
    )
    nearest = F.interpolate(state_one_hot, size=(4, 4), mode="nearest")
    assert not torch.equal(bilinear, nearest)
    old_target = F.one_hot(
        torch.where(valid, target, torch.zeros_like(target)), 3
    ).permute(0, 3, 1, 2).float()
    old_map = (
        F.normalize(bilinear, dim=1) - F.normalize(old_target, dim=1)
    ).square().mean(dim=1)
    torch.testing.assert_close(align, old_map)
    original_scalar = masked_mean(align, valid)
    changed_ignore = align.clone()
    changed_ignore[~valid] = 1.0e6
    torch.testing.assert_close(masked_mean(changed_ignore, valid), original_scalar)


def test_mu_gt_cosine_gather_matches_one_hot_definition():
    torch.manual_seed(7)
    mu = torch.randn(2, 6, 4, 5)
    target = torch.randint(0, 6, (2, 4, 5))
    target[0, 0, 0] = 0
    valid = target != 0
    gathered = mu_gt_cosine(mu, target, eps=1.0e-8, valid_mask=valid)
    one_hot = F.one_hot(target, 6).float()
    expected = (
        F.normalize(mu, dim=1).permute(0, 2, 3, 1) * one_hot
    ).sum(dim=-1)
    torch.testing.assert_close(gathered[valid], expected[valid])


class _TinyEndpoint(nn.Module):
    def forward_logits(self, state, image, s, t):
        image_term = F.adaptive_avg_pool2d(image[:, :1], state.shape[-2:])
        return state + image_term + (s + t)[:, None, None, None]


def _inference_config() -> dict:
    return {
        "dataset": {"num_classes": 3},
        "model": {"state_downsample_factor": 4},
        "source": {
            "prior_type": "gaussian", "prior_noise_std": 1.0,
            "var_weight": 0.0, "align_weight": 0.0,
            "use_loss_align": False,
        },
        "flow": {"time_eps": 1.0e-5},
        "evaluation": {"num_steps": 3},
    }


def _legacy_terminal(model, image, config, steps):
    state, _ = sample_prior(config, image, None, None)
    trajectory = [state.argmax(dim=1)]
    for scalar_s, scalar_t in make_time_grid(steps, image.device):
        s = scalar_s.expand(image.shape[0])
        t = scalar_t.expand(image.shape[0])
        logits = model.forward_logits(state, image, s, t)
        probability = torch.softmax(logits.float(), dim=1).to(state.dtype)
        state = flow_map(state, probability, s, t, config["flow"]["time_eps"])
        trajectory.append(state.argmax(dim=1))
    return state, torch.stack(trajectory, dim=1)


def test_inference_refactor_matches_previous_algorithm_exactly():
    config = _inference_config()
    model = _TinyEndpoint()
    image = torch.randn(2, 3, 8, 12)
    torch.manual_seed(99)
    expected_terminal, expected_trajectory = _legacy_terminal(
        model, image, config, 3
    )
    torch.manual_seed(99)
    terminal = sample_segmentation(
        model, None, image, config, num_steps=3, return_terminal_state=True
    )
    torch.testing.assert_close(terminal, expected_terminal)
    torch.manual_seed(99)
    prediction, trajectory = sample_segmentation(
        model, None, image, config, num_steps=3, return_trajectory=True
    )
    torch.testing.assert_close(trajectory, expected_trajectory)
    assert prediction.shape == (2, 8, 12)


@pytest.mark.parametrize("steps", [1, 3])
def test_from_x0_inference_step_shapes(steps):
    config = _inference_config()
    image = torch.randn(2, 3, 8, 12)
    x0 = torch.randn(2, 3, 2, 3)
    terminal = sample_segmentation_from_x0(
        _TinyEndpoint(), image, x0, config, num_steps=steps,
        return_terminal_state=True,
    )
    prediction, trajectory = sample_segmentation_from_x0(
        _TinyEndpoint(), image, x0, config, num_steps=steps,
        return_trajectory=True,
    )
    assert terminal.shape == x0.shape
    assert prediction.shape == (2, 8, 12)
    assert trajectory.shape == (2, steps + 1, 2, 3)


def test_ade20k_metrics_exclude_class_zero_gt():
    metrics = SegmentationMetrics(
        num_classes=3,
        void_class_index=0,
        evaluated_class_indices=range(1, 3),
        nanmean=True,
    )
    prediction = torch.tensor([[2, 1, 2]])
    target = torch.tensor([[0, 1, 2]])
    metrics.update(prediction, target)
    result = metrics.compute()
    assert sum(map(sum, result["confusion_matrix"])) == 2
    assert result["mIoU"] == pytest.approx(1.0)
    assert result["mAcc"] == pytest.approx(1.0)
    assert result["pixel_acc"] == pytest.approx(1.0)


def test_checkpoint_resolution_prefers_explicit_then_latest(tmp_path):
    output = tmp_path / "training"
    output.mkdir()
    latest = output / "latest.pt"
    latest.touch()
    explicit = tmp_path / "selected.pt"
    explicit.touch()
    config = {
        "evaluation": {"checkpoint": None},
        "experiment": {"output_dir": str(output)},
    }
    assert resolve_diagnostic_checkpoint(config, str(explicit)) == explicit.resolve()
    assert resolve_diagnostic_checkpoint(config, None) == latest.resolve()


@pytest.mark.parametrize(
    "current,legacy",
    [
        (
            "encoder.stages.0.patch_embeddings.proj.weight",
            "encoder.encoder.patch_embeddings.0.proj.weight",
        ),
        (
            "encoder.stages.2.blocks.1.attention.q_proj.weight",
            "encoder.encoder.block.2.1.attention.self.query.weight",
        ),
        (
            "encoder.stages.1.blocks.0.attention.sequence_reduction.layer_norm.bias",
            "encoder.encoder.block.1.0.attention.self.layer_norm.bias",
        ),
        (
            "encoder.stages.3.blocks.1.mlp.fc2.bias",
            "encoder.encoder.block.3.1.mlp.dense2.bias",
        ),
        ("decoder.0.weight", "decoder.0.weight"),
    ],
)
def test_segformer_v5_to_v4_checkpoint_key_translation(current, legacy):
    assert _segformer_v5_key_to_v4(current) == legacy


def test_diagnostics_json_replaces_nonfinite_class_metrics_with_null():
    assert _json_safe({"class_iou": [0.5, float("nan")]}) == {
        "class_iou": [0.5, None]
    }
