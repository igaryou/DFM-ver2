from __future__ import annotations

import math

import pytest
import torch
import source_noise_diagnostics as diagnostics_module

from discrete_flow_maps import sample_prior
from inference import sample_segmentation
from source_noise_diagnostics import (
    DIAGNOSTIC_RESULT_KEYS,
    SourceNoiseDiagnosticsAccumulator,
    source_noise_diagnostic_maps,
)
from trainer import _wandb_epoch_payload
from visualization import save_source_noise_diagnostics


def test_exact_sampled_noise_and_rms_shapes():
    mu = torch.randn(2, 4, 3, 5)
    sigma = torch.rand(2, 1, 3, 5) + 0.1
    epsilon = torch.randn_like(mu)
    expected_noise = sigma * epsilon
    x0 = mu + expected_noise

    maps = source_noise_diagnostic_maps(x0, mu, sigma)

    torch.testing.assert_close(maps["noise"], expected_noise)
    assert maps["noise_rms"].shape == (2, 3, 5)
    assert maps["mu_rms"].shape == (2, 3, 5)


def test_zero_noise_has_zero_ratio_flip_and_direction_change():
    mu = torch.tensor([[[[2.0]], [[1.0]], [[-1.0]]]])
    maps = source_noise_diagnostic_maps(mu, mu, torch.ones(1, 1, 1, 1))

    assert maps["noise_signal_ratio"].item() == 0.0
    assert maps["flip_map"].sum().item() == 0
    assert maps["direction_change"].item() == pytest.approx(0.0, abs=2.0e-7)

    zero = torch.zeros_like(mu)
    zero_maps = source_noise_diagnostic_maps(zero, zero, torch.ones(1, 1, 1, 1))
    assert zero_maps["direction_change"].item() == 0.0
    assert zero_maps["mu_x0_cosine"].item() == 1.0


def test_intentional_argmax_flip_map_and_rate():
    mu = torch.tensor([[[[2.0, 1.0]], [[1.0, 3.0]]]])
    x0 = torch.tensor([[[[0.0, 1.0]], [[3.0, 3.0]]]])
    maps = source_noise_diagnostic_maps(x0, mu, torch.ones(1, 1, 1, 2))
    expected = torch.tensor([[[True, False]]])
    assert torch.equal(maps["flip_map"], expected)

    accumulator = SourceNoiseDiagnosticsAccumulator()
    accumulator.update(maps)
    result = accumulator.compute(device=torch.device("cpu"))
    assert result["source_argmax_flip_rate"] == pytest.approx(0.5)


def test_distributed_mean_uses_global_sum_and_pixel_count(monkeypatch):
    mu = torch.tensor([[[[1.0]], [[0.0]]]])
    x0 = mu + 1.0
    accumulator = SourceNoiseDiagnosticsAccumulator()
    accumulator.update(source_noise_diagnostic_maps(
        x0, mu, torch.ones(1, 1, 1, 1)
    ))

    def fake_all_reduce(packed, op):
        del op
        # Simulate another rank with nine pixels whose noise RMS is 2.
        packed[0] += 18.0
        packed[1] += 36.0
        packed[2] += 9.0

    def fake_all_gather_object(gathered, samples):
        gathered[0] = samples
        gathered[1] = samples

    monkeypatch.setattr(diagnostics_module.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(diagnostics_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        diagnostics_module.dist, "all_gather_object", fake_all_gather_object
    )
    result = accumulator.compute(device=torch.device("cpu"), distributed=True)
    assert result["source_noise_rms_mean"] == pytest.approx(1.9)


def test_pixel_shared_margin_sigma_ratio_broadcasts_exactly():
    mu = torch.tensor([[[[4.0, 5.0]], [[2.0, 1.0]], [[0.0, 3.0]]]])
    sigma = torch.tensor([[[[2.0, 4.0]]]])
    maps = source_noise_diagnostic_maps(mu, mu, sigma)
    margin = torch.tensor([[[2.0, 2.0]]])
    expected = margin / (math.sqrt(2.0) * sigma[:, 0] + 1.0e-8)
    torch.testing.assert_close(maps["margin"], margin)
    torch.testing.assert_close(maps["margin_sigma_ratio"], expected)


def test_channelwise_margin_uses_top_two_difference_std():
    mu = torch.tensor([[[[4.0]], [[2.0]], [[0.0]]]])
    sigma = torch.tensor([[[[3.0]], [[4.0]], [[20.0]]]])
    maps = source_noise_diagnostic_maps(mu, mu, sigma)
    assert maps["margin_sigma_ratio"].item() == pytest.approx(2.0 / 5.0)


class _ImageGaussianSource:
    fixed_std = None

    def __call__(self, image):
        mu = torch.ones(image.shape[0], 3, *image.shape[-2:])
        logvar = torch.zeros(image.shape[0], 1, *image.shape[-2:])
        epsilon = torch.full_like(mu, 0.25)
        return mu + torch.exp(0.5 * logvar) * epsilon, mu, logvar


def _config() -> dict:
    return {
        "dataset": {"num_classes": 3, "protocol": "lidc"},
        "model": {"state_downsample_factor": 1},
        "source": {
            "prior_type": "image_gaussian",
            "variance_mode": "pixel_shared",
            "freeze": True,
            "type": "trainable_segformer",
            "supervision": {"type": "none"},
            "variance_regularization": {"enabled": False, "target_std": 1.0},
            "var_weight": 0.0,
            "align_weight": 0.0,
            "use_loss_align": False,
        },
        "evaluation": {"num_steps": 1},
        "flow": {"time_eps": 1.0e-5, "path": {"type": "linear"}},
    }


def test_retained_statistics_are_from_the_same_source_sample():
    image = torch.randn(2, 1, 3, 4)
    x0, stats = sample_prior(
        _config(), image, None, _ImageGaussianSource(),
        sampling_mode="inference", retain_source_tensors=True,
    )
    assert stats["_source_mu"].shape == x0.shape == (2, 3, 3, 4)
    assert stats["_source_logvar"].shape == (2, 1, 3, 4)
    assert stats["_source_sigma"].shape == (2, 1, 3, 4)
    torch.testing.assert_close(x0.float() - stats["_source_mu"].float(),
                               torch.full_like(x0, 0.25))


class _ConstantFlow(torch.nn.Module):
    def forward_logits(self, x, image, s, t):
        return torch.zeros_like(x)


def test_inference_legacy_signature_and_diagnostics_off_output_are_unchanged():
    config = {
        "dataset": {"num_classes": 3, "protocol": "lidc"},
        "model": {"state_downsample_factor": 1},
        "source": {"prior_type": "gaussian", "prior_noise_std": 1.0},
        "evaluation": {"num_steps": 1},
        "flow": {"time_eps": 1.0e-5, "path": {"type": "linear"}},
    }
    image = torch.randn(2, 1, 3, 4)
    torch.manual_seed(17)
    legacy = sample_segmentation(
        _ConstantFlow(), None, image, config, return_source_sample=True
    )
    assert isinstance(legacy, tuple) and len(legacy) == 3

    torch.manual_seed(17)
    ordinary = sample_segmentation(_ConstantFlow(), None, image, config)
    torch.manual_seed(17)
    diagnostics_off = sample_segmentation(
        _ConstantFlow(), None, image, config, return_source_diagnostics=False
    )
    assert torch.equal(ordinary, diagnostics_off)


def test_new_inference_diagnostics_payload_uses_returned_x0():
    config = {
        "dataset": {"num_classes": 3, "protocol": "lidc"},
        "model": {"state_downsample_factor": 1},
        "source": {"prior_type": "gaussian", "prior_noise_std": 1.5},
        "evaluation": {"num_steps": 1},
        "flow": {"time_eps": 1.0e-5, "path": {"type": "linear"}},
    }
    image = torch.randn(2, 1, 3, 4)
    prediction, payload = sample_segmentation(
        _ConstantFlow(), None, image, config, return_source_diagnostics=True
    )
    assert prediction.shape == (2, 3, 4)
    assert set(payload) == {"x0", "mu", "sigma", "logvar"}
    assert payload["x0"].shape == payload["mu"].shape == (2, 3, 3, 4)
    assert payload["sigma"].shape == payload["logvar"].shape == (2, 1, 3, 4)
    maps = source_noise_diagnostic_maps(
        payload["x0"], payload["mu"], payload["sigma"]
    )
    torch.testing.assert_close(maps["noise"], payload["x0"].float())


def test_diagnostic_metrics_have_requested_epoch_source_wandb_keys():
    report = {key: float(index) for index, key in enumerate(DIAGNOSTIC_RESULT_KEYS)}
    payload = _wandb_epoch_payload(report, "psd")
    assert {
        f"epoch/source/{key.removeprefix('source_')}"
        for key in DIAGNOSTIC_RESULT_KEYS
    } <= payload.keys()


def test_twelve_panel_visualization_is_saved(tmp_path):
    mu = torch.randn(1, 3, 2, 3)
    sigma = torch.ones(1, 1, 2, 3)
    x0 = mu + torch.randn_like(mu)
    maps = source_noise_diagnostic_maps(x0, mu, sigma)
    destination = tmp_path / "source_diagnostics" / "sample.png"
    save_source_noise_diagnostics(
        torch.rand(3, 8, 12), torch.randint(0, 3, (8, 12)),
        {key: value[0] for key, value in maps.items()},
        destination,
        dataset_name="cityscapes",
    )
    assert destination.is_file()
    assert destination.stat().st_size > 0
