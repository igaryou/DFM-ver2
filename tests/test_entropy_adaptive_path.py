from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from adaptive_path import (
    adaptive_lambda,
    adaptive_lambda_derivative,
    normalize_entropy,
    shannon_entropy,
)
from config import load_config
from discrete_flow_maps import (
    linear_path,
    path_coefficient,
    path_derivative,
    sample_prior,
)
from trainer import build_optimizer
from training_objectives import DDPCompatibleTrainingModel


ADAPTIVE = {
    "type": "entropy_adaptive",
    "scheduler": {"type": "mean_preserving_additive", "beta": 0.5},
}


def test_disabled_and_beta_zero_match_linear_exactly() -> None:
    torch.manual_seed(11)
    x0 = torch.randn(2, 3, 3, 4)
    x1 = torch.randn_like(x0)
    time = torch.tensor([0.2, 0.8])
    difficulty = torch.randn(2, 3, 4)
    expected = linear_path(x0, x1, time)
    zero_beta = {
        "type": "entropy_adaptive",
        "scheduler": {"type": "mean_preserving_additive", "beta": 0.0},
    }
    torch.testing.assert_close(
        linear_path(x0, x1, time, zero_beta, difficulty), expected, rtol=0, atol=0
    )


def test_scheduler_invariants_and_direction() -> None:
    difficulty = torch.tensor([[[-1.0, -0.5], [0.5, 1.0]]])
    time = torch.tensor([0.4])
    coefficient = adaptive_lambda(time, difficulty, beta=0.75)
    torch.testing.assert_close(coefficient.mean(), time[0])
    assert bool((coefficient[0, 0] > time[0]).all())
    assert bool((coefficient[0, 1] < time[0]).all())
    torch.testing.assert_close(
        adaptive_lambda(torch.tensor([0.0]), difficulty, beta=1.0),
        torch.zeros_like(difficulty),
    )
    torch.testing.assert_close(
        adaptive_lambda(torch.tensor([1.0]), difficulty, beta=1.0),
        torch.ones_like(difficulty),
    )


def test_lambda_range_for_supported_domain() -> None:
    difficulty = torch.linspace(-1, 1, 1001)[None, None]
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
        for value in torch.linspace(0, 1, 31):
            coefficient = adaptive_lambda(value[None], difficulty, beta=beta)
            assert float(coefficient.min()) >= 0.0
            assert float(coefficient.max()) <= 1.0


@pytest.mark.parametrize("method", ["mean", "zscore", "minmax", "rank"])
@pytest.mark.parametrize("case", ["constant", "few_valid", "ties"])
def test_normalizations_are_finite_mean_zero_and_bounded(method: str, case: str) -> None:
    if case == "constant":
        entropy = torch.ones(2, 3, 4)
        valid = torch.ones_like(entropy, dtype=torch.bool)
    elif case == "few_valid":
        entropy = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        valid = torch.zeros_like(entropy, dtype=torch.bool)
        valid[:, 1, 2] = True
    else:
        entropy = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 2.0, 3.0]]])
        valid = torch.tensor([[[True, True, True, True], [True, False, True, True]]])
    difficulty = normalize_entropy(entropy, method, valid_mask=valid)
    assert bool(torch.isfinite(difficulty).all())
    assert float(difficulty.abs().max()) <= 1.0
    for sample, mask in zip(difficulty, valid):
        if mask.any():
            torch.testing.assert_close(sample[mask].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    assert bool((difficulty[~valid] == 0).all())


def test_entropy_fp32_and_ignore_safe() -> None:
    logits = torch.randn(1, 4, 2, 3, dtype=torch.bfloat16)
    entropy = shannon_entropy(logits)
    assert entropy.dtype == torch.float32
    assert bool(torch.isfinite(entropy).all())


def test_adaptive_derivative_matches_finite_difference() -> None:
    difficulty = torch.tensor([[[-0.8, 0.0, 0.7]]])
    time = torch.tensor([0.37])
    step = 1.0e-3
    finite = (
        adaptive_lambda(time + step, difficulty, beta=0.5)
        - adaptive_lambda(time - step, difficulty, beta=0.5)
    ) / (2 * step)
    analytic = adaptive_lambda_derivative(time, difficulty, beta=0.5)
    torch.testing.assert_close(analytic, finite, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        path_derivative(time, ADAPTIVE, difficulty), analytic
    )
    torch.testing.assert_close(
        path_coefficient(time, ADAPTIVE, difficulty),
        adaptive_lambda(time, difficulty, beta=0.5),
    )


def test_legacy_config_without_explicit_path_loads_linear() -> None:
    config = load_config("configs/debug/diagonal/cityscapes.yaml")
    assert config["flow"]["path"]["type"] == "power"
    assert config["flow"]["path"]["exponent"] == 1.0


def test_example_config_and_cli_switches_load() -> None:
    path = "configs/cityscapes/psd/entropy_adaptive_rank_128k.yaml"
    for method in ("mean", "zscore", "minmax", "rank"):
        for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
            config = load_config(path, [
                f"flow.path.entropy.normalization={method}",
                f"flow.path.scheduler.beta={beta}",
            ])
            assert config["flow"]["path"]["entropy"]["normalization"] == method
            assert config["flow"]["path"]["scheduler"]["beta"] == beta


class _RecordingFrozenSource(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.grad_enabled_during_forward: bool | None = None

    def forward(self, image: torch.Tensor):
        self.grad_enabled_during_forward = torch.is_grad_enabled()
        state = image.new_zeros(image.shape[0], 20, image.shape[-2] // 4, image.shape[-1] // 4)
        state[:, 0] = self.weight
        return state, state, torch.zeros_like(state)


def test_frozen_source_uses_eval_no_grad_and_is_not_optimized() -> None:
    config = load_config("configs/cityscapes/psd/entropy_adaptive_rank_128k.yaml")
    source = _RecordingFrozenSource()
    endpoint = nn.Linear(2, 2)
    adapter = DDPCompatibleTrainingModel(endpoint, source, config)
    adapter.train()
    assert not source.training
    x0, stats = sample_prior(config, torch.zeros(1, 3, 8, 12), None, source)
    assert x0.shape == (1, 20, 2, 3)
    assert source.grad_enabled_during_forward is False
    assert not stats["_path_source_state"].requires_grad
    optimizer = build_optimizer(config, adapter)
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(id(parameter) not in optimized for parameter in source.parameters())
