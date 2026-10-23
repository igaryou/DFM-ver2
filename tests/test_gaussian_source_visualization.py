from __future__ import annotations

import math

import pytest
import torch

from discrete_flow_maps import sample_prior
from inference import (
    sample_lidc_distribution,
    sample_segmentation_probabilities,
)


class _ConstantFlow(torch.nn.Module):
    def forward_logits(self, x, image, s, t):
        return torch.zeros_like(x)


def _gaussian_lidc_config(*, height: int = 8, width: int = 8) -> dict:
    return {
        "dataset": {
            "num_classes": 2,
            "protocol": "lidc",
            "image_size": [height, width],
        },
        "model": {"state_downsample_factor": 1},
        "source": {"prior_type": "gaussian", "prior_noise_std": 1.0},
        "evaluation": {"num_steps": 1},
        "flow": {"time_eps": 1.0e-5, "path": {"type": "linear"}},
    }


def test_gaussian_prior_only_retains_zero_source_mean_when_requested():
    config = _gaussian_lidc_config()
    image = torch.randn(2, 1, 8, 8)

    x0, stats = sample_prior(config, image, None, None)
    assert "_source_mu" not in stats

    retained_x0, retained_stats = sample_prior(
        config, image, None, None, retain_source_tensors=True
    )
    assert "_source_mu" in retained_stats
    assert retained_stats["_source_mu"].shape == retained_x0.shape
    assert torch.all(retained_stats["_source_mu"] == 0)


def test_gaussian_prior_sampling_statistics_are_unchanged():
    torch.manual_seed(0)
    config = _gaussian_lidc_config(height=256, width=256)
    image = torch.zeros(2, 1, 256, 256)

    x0, _ = sample_prior(config, image, None, None, retain_source_tensors=True)

    assert x0.mean().item() == pytest.approx(0.0, abs=0.01)
    assert x0.std().item() == pytest.approx(1.0, abs=0.01)
    assert x0.abs().mean().item() == pytest.approx(math.sqrt(2.0 / math.pi), abs=0.01)


def test_gaussian_probability_inference_returns_source_sample_and_zero_mean():
    config = _gaussian_lidc_config()
    image = torch.randn(2, 1, 8, 8)

    probability, x0, mu = sample_segmentation_probabilities(
        _ConstantFlow(), None, image, config, return_source_sample=True
    )

    assert probability.shape == x0.shape == mu.shape == (2, 2, 8, 8)
    assert torch.all(mu == 0)
    assert torch.allclose(probability.sum(dim=1), torch.ones(2, 8, 8))


def test_gaussian_lidc_distribution_returns_source_sample_and_zero_mean():
    config = _gaussian_lidc_config()
    image = torch.randn(2, 1, 8, 8)

    samples, prediction, x0, mu = sample_lidc_distribution(
        _ConstantFlow(), None, image, config, [1, 3], return_source_sample=True
    )

    assert samples.shape == (2, 3, 8, 8)
    assert prediction.shape == (2, 8, 8)
    assert x0.shape == mu.shape == (2, 2, 8, 8)
    assert torch.all(mu == 0)
