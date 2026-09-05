from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import evaluate_source_flow_ablation as ablation
from config import load_config
from inference import terminal_state_to_original_prediction
from metrics import SegmentationMetrics


ROOT = Path(__file__).parents[1]
CONFIG = (
    ROOT
    / "configs/cityscapes/psd/joint_swin_t_segformer_b1_standard_ce_160k.yaml"
)


class _GaussianSource(nn.Module):
    fixed_std = 1.0

    def forward(self, image):
        mu = image.new_full((image.shape[0], 4, 2, 3), 0.25)
        logvar = torch.zeros_like(mu)
        z0 = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return z0, mu, logvar


def test_gaussian_source_formula_and_seed_are_reproducible():
    source = _GaussianSource().eval()
    image = torch.zeros(1, 3, 8, 12)
    first = ablation.source_pair_from_one_forward(source, image, seed=42)
    second = ablation.source_pair_from_one_forward(source, image, seed=42)
    z0, mu, logvar = first
    torch.testing.assert_close(z0, second[0])
    torch.testing.assert_close(mu, second[1])
    torch.testing.assert_close(logvar, second[2])
    epsilon = z0 - mu
    torch.testing.assert_close(z0, mu + torch.exp(0.5 * logvar) * epsilon)
    assert ablation.stable_sample_seed(42, "frankfurt_000001") == (
        ablation.stable_sample_seed(42, "frankfurt_000001")
    )
    assert ablation.stable_sample_seed(42, "frankfurt_000001") != (
        ablation.stable_sample_seed(42, "frankfurt_000002")
    )


def test_flow_conditions_pass_raw_mu_and_z0_to_same_production_helper(monkeypatch):
    calls = []

    def fake_flow(model, image, x0, config, **kwargs):
        del model, image, config
        calls.append((x0, kwargs))
        return x0 + 1.0

    monkeypatch.setattr(ablation, "sample_segmentation_from_x0", fake_flow)
    mu = torch.tensor([[[[-2.0]], [[3.0]]]])
    z0 = torch.tensor([[[[4.0]], [[-1.0]]]])
    mu_terminal, z0_terminal, mu_trajectory, z0_trajectory = (
        ablation.run_flow_conditions(
            object(), torch.zeros(1, 3, 4, 4), mu, z0, {}, num_steps=3
        )
    )
    assert calls[0][0] is mu
    assert calls[1][0] is z0
    assert not torch.allclose(mu, torch.softmax(mu, dim=1))
    torch.testing.assert_close(mu_terminal, mu + 1.0)
    torch.testing.assert_close(z0_terminal, z0 + 1.0)
    assert mu_trajectory is None and z0_trajectory is None
    assert calls[0][1] == calls[1][1]


def test_trajectory_mode_uses_production_run_flow_for_both_states(monkeypatch):
    calls = []

    def fake_run(model, image, initial, config, steps, return_trajectory=False):
        del model, image, config
        calls.append((initial, steps, return_trajectory))
        trajectory = torch.stack((initial.argmax(1), initial.argmax(1)), dim=1)
        return initial + 2.0, trajectory

    monkeypatch.setattr(ablation, "run_flow_from_state", fake_run)
    mu = torch.randn(1, 3, 2, 2)
    z0 = torch.randn(1, 3, 2, 2)
    mu_terminal, z0_terminal, mu_trajectory, z0_trajectory = (
        ablation.run_flow_conditions(
            object(), torch.zeros(1, 3, 8, 8), mu, z0, {},
            num_steps=5, return_trajectory=True,
        )
    )
    assert calls[0][0] is mu and calls[1][0] is z0
    assert calls[0][1:] == calls[1][1:] == (5, True)
    torch.testing.assert_close(mu_terminal, mu + 2.0)
    torch.testing.assert_close(z0_terminal, z0 + 2.0)
    assert mu_trajectory.shape == z0_trajectory.shape == (1, 2, 2, 2)


def test_original_prediction_matches_production_padding_and_void_protocol():
    config = load_config(CONFIG)
    state = torch.zeros(1, 20, 2, 2)
    state[:, 19] = 10.0
    state[:, 3] = 5.0
    sample = {
        "model_shape": (6, 8),
        "padded_shape": (8, 8),
        "original_shape": (3, 4),
    }
    actual = ablation.original_prediction(state, sample, config)
    expected = terminal_state_to_original_prediction(
        state,
        sample["model_shape"],
        sample["original_shape"],
        padded_shape=sample["padded_shape"],
        align_corners=config["evaluation"]["align_corners"],
        void_class_index=19,
        exclude_void=True,
    )
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1, 3, 4)
    assert torch.equal(actual, torch.full_like(actual, 3))


def test_power2_is_required_for_runtime_and_saved_checkpoint():
    config = load_config(CONFIG)
    config["flow"]["path"].update({"type": "power", "exponent": 2.0})
    ablation.assert_power2_path(config, {"config": deepcopy(config)})
    invalid = deepcopy(config)
    invalid["flow"]["path"]["exponent"] = 1.0
    with pytest.raises(ValueError, match="exponent=2.0"):
        ablation.assert_power2_path(invalid)
    with pytest.raises(RuntimeError, match="Checkpoint was not trained"):
        ablation.assert_power2_path(config, {"config": invalid})


def test_checkpoint_directory_priority_and_ambiguity(tmp_path):
    latest = tmp_path / "latest.pt"
    best = tmp_path / "best.pt"
    numbered = tmp_path / "step_160000.pt"
    latest.touch()
    numbered.touch()
    assert ablation.resolve_ablation_checkpoint(None, tmp_path) == latest.resolve()
    best.touch()
    assert ablation.resolve_ablation_checkpoint(None, tmp_path) == best.resolve()
    best.unlink()
    latest.unlink()
    another = tmp_path / "step_144000.pt"
    another.touch()
    with pytest.raises(ValueError, match="multiple candidates"):
        ablation.resolve_ablation_checkpoint(None, tmp_path)


def test_three_metrics_are_independent_and_deltas_are_named():
    target = torch.tensor([[[0, 1], [0, 1]]])
    predictions = (
        torch.tensor([[[0, 1], [0, 1]]]),
        torch.tensor([[[0, 0], [0, 1]]]),
        torch.tensor([[[1, 1], [1, 1]]]),
    )
    metrics = [
        SegmentationMetrics(3, 2, evaluated_class_indices=range(2))
        for _ in range(3)
    ]
    for metric, prediction in zip(metrics, predictions, strict=True):
        metric.update(prediction, target)
    payloads = [ablation._metric_payload(metric) for metric in metrics]
    result = ablation.build_result(*payloads)
    assert len({id(metric.confusion_matrix) for metric in metrics}) == 3
    assert result["source_mu"]["miou"] == 1.0
    assert result["source_mu_miou"] == result["source_mu"]["miou"]
    assert result["flow_from_mu_miou"] == result["flow_from_mu"]["miou"]
    assert result["flow_from_z0_miou"] == result["flow_from_z0"]["miou"]
    assert result["flow_from_mu"]["miou"] != result["flow_from_z0"]["miou"]
    assert result["delta_flow_mu_vs_source"] == pytest.approx(
        payloads[1]["miou"] - payloads[0]["miou"]
    )
    assert result["delta_noise"] == pytest.approx(
        payloads[2]["miou"] - payloads[1]["miou"]
    )


class _TinyFlow(nn.Module):
    def forward_logits(self, state, image, s, t):
        del image, s, t
        return state + 0.1


def test_cpu_synthetic_source_and_both_flow_conditions_smoke():
    config = load_config(CONFIG)
    config["dataset"]["num_classes"] = 4
    config["model"]["num_classes"] = 4
    config["model"]["state_downsample_factor"] = 4
    config["flow"]["path"].update({"type": "power", "exponent": 2.0})
    image = torch.zeros(1, 3, 8, 12)
    z0, mu, _ = ablation.source_pair_from_one_forward(
        _GaussianSource().eval(), image, seed=9
    )
    mu_terminal, z0_terminal, _, _ = ablation.run_flow_conditions(
        _TinyFlow(), image, mu, z0, config, num_steps=3
    )
    assert mu_terminal.shape == z0_terminal.shape == (1, 4, 2, 3)
    assert torch.isfinite(mu_terminal).all()
    assert torch.isfinite(z0_terminal).all()
