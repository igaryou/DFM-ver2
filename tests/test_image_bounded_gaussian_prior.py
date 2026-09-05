from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from checkpoint import model_signature
from config import load_config, validate_config
from discrete_flow_maps import (
    flow_map,
    linear_path,
    path_coefficient,
    sample_image_bounded_gaussian,
    sample_prior,
)
from state_space import prepare_state_targets, resize_continuous
from training_objectives import DDPCompatibleTrainingModel


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/cityscapes/psd/joint_bounded_gaussian_b1_ce_160k.yaml"
PARENT = ROOT / "configs/cityscapes/psd/joint_swin_t_segformer_b1_standard_ce_160k.yaml"


class RawStatisticsSource(nn.Module):
    fixed_std = 1.0

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = nn.Parameter(logits.clone())
        self.forward_calls = 0

    def forward_statistics(self, image: torch.Tensor):
        shape = (
            image.shape[0], self.logits.numel(),
            image.shape[-2] // 4, image.shape[-1] // 4,
        )
        mu_raw = self.logits[None, :, None, None].expand(shape)
        return mu_raw, torch.zeros_like(mu_raw)

    def forward(self, image: torch.Tensor):
        self.forward_calls += 1
        raise AssertionError("bounded prior must sample from raw statistics explicitly")


def _small_config() -> dict:
    config = load_config(CONFIG)
    config["dataset"]["num_classes"] = 4
    config["model"]["num_classes"] = 4
    config["model"]["state_downsample_factor"] = 4
    config["loss"]["ignore_index"] = 3
    config["evaluation"]["ignore_index"] = 3
    return config


def test_bounded_components_formula_range_order_zero_sigma_and_seed():
    mu_raw = torch.tensor([[[[-8.0]], [[-0.2]], [[0.4]], [[9.0]]]])
    epsilon = torch.tensor([[[[0.5]], [[-1.0]], [[2.0]], [[-0.25]]]])
    mu_state, x0 = sample_image_bounded_gaussian(
        mu_raw, amplitude=1.7, sigma=0.6, epsilon=epsilon
    )
    torch.testing.assert_close(mu_state, 1.7 * torch.tanh(mu_raw))
    torch.testing.assert_close(x0, mu_state + 0.6 * epsilon)
    assert mu_state.min() >= -1.7 and mu_state.max() <= 1.7
    assert torch.equal(mu_raw.argmax(dim=1), mu_state.argmax(dim=1))

    state_zero, x0_zero = sample_image_bounded_gaussian(
        mu_raw, amplitude=1.0, sigma=0.0, epsilon=epsilon
    )
    torch.testing.assert_close(x0_zero, state_zero)

    torch.manual_seed(123)
    first = sample_image_bounded_gaussian(
        mu_raw, amplitude=1.0, sigma=1.0
    )[1]
    torch.manual_seed(123)
    second = sample_image_bounded_gaussian(
        mu_raw, amplitude=1.0, sigma=1.0
    )[1]
    torch.testing.assert_close(first, second)


def test_production_sampling_uses_raw_ce_bounded_state_and_fixed_sigma():
    config = _small_config()
    source = RawStatisticsSource(torch.tensor([3.0, -2.0, 0.5, 1.0]))
    image = torch.zeros(1, 3, 8, 12)
    target = torch.tensor([[i % 4 for i in range(12)] for _ in range(8)])[None]
    targets = prepare_state_targets(
        target, num_classes=4, state_size=(2, 3), ignore_index=3,
        mask_pixel_losses=True,
    )

    torch.manual_seed(19)
    x0, stats = sample_prior(
        config, image, targets.one_hot_state, source,
        target_full=target, valid_mask_full=targets.valid_mask_full,
        sampling_mode="training",
    )
    torch.manual_seed(19)
    mu_raw, _ = source.forward_statistics(image)
    mu_state = torch.tanh(mu_raw)
    expected_x0 = mu_state + torch.randn_like(mu_state)
    torch.testing.assert_close(x0, expected_x0)
    assert source.forward_calls == 0

    raw_full = resize_continuous(mu_raw, target.shape[-2:])
    state_full = resize_continuous(mu_state, target.shape[-2:])
    raw_ce = F.cross_entropy(raw_full, target)
    state_ce = F.cross_entropy(state_full, target)
    torch.testing.assert_close(stats["loss_source_ce"], raw_ce)
    assert not torch.isclose(stats["loss_source_ce"], state_ce)
    assert stats["source_mu_state_min"] >= -1
    assert stats["source_mu_state_max"] <= 1
    assert stats["source_amplitude"] == 1
    assert stats["source_sigma_mean"] == 1
    torch.testing.assert_close(stats["source_mu_raw_abs"], mu_raw.abs().mean())
    torch.testing.assert_close(stats["source_mu_state_abs"], mu_state.abs().mean())


def test_bounded_training_and_inference_draw_the_same_distribution():
    config = _small_config()
    source = RawStatisticsSource(torch.tensor([-2.0, -0.5, 0.25, 3.0]))
    image = torch.zeros(2, 3, 8, 12)
    torch.manual_seed(77)
    training, _ = sample_prior(
        config, image, None, source, sampling_mode="training"
    )
    torch.manual_seed(77)
    inference, _ = sample_prior(
        config, image, None, source, sampling_mode="inference"
    )
    torch.testing.assert_close(training, inference)


@pytest.mark.parametrize(
    "override,match",
    [
        ("source.bounded_gaussian.amplitude=0", "amplitude must be positive"),
        ("source.fixed_std=0", "fixed_std > 0"),
        ("source.learned_logvar=true", "learned_logvar=false"),
    ],
)
def test_bounded_config_validation(override, match):
    with pytest.raises(ValueError, match=match):
        load_config(CONFIG, [override])


def test_bounded_config_inherits_recipe_and_cli_path_overrides():
    config = load_config(CONFIG)
    parent = load_config(PARENT)
    assert config["source"]["prior_type"] == "image_bounded_gaussian"
    assert config["source"]["bounded_gaussian"] == {"amplitude": 1.0}
    assert config["source"]["fixed_std"] == 1.0
    assert config["source"]["learned_logvar"] is False
    assert config["flow"]["target_smoothing"] == {"enabled": False, "p": 0.0}
    for key in ("optimizer", "scheduler"):
        assert config["training"][key] == parent["training"][key]
    assert config["loss"] == parent["loss"]
    assert config["dataset"] == parent["dataset"]
    for exponent in (1, 1.5, 2, 3):
        overridden = load_config(CONFIG, [
            "flow.path.type=power", f"flow.path.exponent={exponent}",
        ])
        assert overridden["flow"]["path"]["exponent"] == exponent


def test_bounded_prior_has_same_source_architecture_signature_as_gaussian():
    bounded = load_config(CONFIG)
    gaussian = deepcopy(bounded)
    gaussian["source"]["prior_type"] = "image_gaussian"
    assert model_signature(bounded) == model_signature(gaussian)


def test_power_path_linear_square_and_flow_map_gamma():
    x0 = torch.randn(2, 4, 2, 3)
    x1 = torch.randn_like(x0)
    time = torch.tensor([0.25, 0.8])
    linear = {"type": "linear", "exponent": 9.0}
    power_one = {"type": "power", "exponent": 1.0}
    torch.testing.assert_close(
        linear_path(x0, x1, time, linear),
        linear_path(x0, x1, time, power_one),
    )
    torch.testing.assert_close(
        path_coefficient(time, {"type": "power", "exponent": 2.0}),
        time.square(),
    )
    s = torch.tensor([0.1, 0.4])
    t = torch.tensor([0.6, 0.9])
    endpoint = torch.randn_like(x0)
    exponent = 1.5
    expected_gamma = (t.pow(exponent) - s.pow(exponent)) / (
        1 - s.pow(exponent)
    )
    expected = x0 + expected_gamma[:, None, None, None] * (endpoint - x0)
    torch.testing.assert_close(
        flow_map(
            x0, endpoint, s, t,
            path_config={"type": "power", "exponent": exponent},
        ),
        expected,
    )


def test_disabled_target_smoothing_is_hard_one_hot_and_cpu_backward_smoke():
    config = _small_config()
    source = RawStatisticsSource(torch.tensor([-1.0, 0.0, 0.5, 2.0]))

    class Endpoint(nn.Module):
        def __init__(self):
            super().__init__()
            self.state = nn.Conv2d(4, 4, 1)
            self.image = nn.Conv2d(3, 4, 1)

        def encode_image(self, image):
            return F.avg_pool2d(self.image(image), 4)

        def forward_logits_with_image_feat(self, x, image_feat, s, t):
            return self.state(x) + image_feat + (t - s)[:, None, None, None]

        def forward_logits(self, x, image, s, t):
            return self.forward_logits_with_image_feat(
                x, self.encode_image(image), s, t
            )

    adapter = DDPCompatibleTrainingModel(Endpoint(), source, config)
    target = torch.randint(0, 4, (1, 8, 12))
    result = adapter(
        operation="joint_objectives", image=torch.randn(1, 3, 8, 12),
        target=target, epoch_index=0, progress_in_epoch=0.0, optimizer_step=0,
    )
    assert torch.isfinite(result["loss"])
    assert result["stats"]["target_smoothing_enabled"] == 0
    assert result["stats"]["x1_state_min"] == 0
    assert result["stats"]["x1_state_max"] == 1
    assert result["stats"]["x1_state_sum_error"] == 0
    result["loss"].backward()
    assert source.logits.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bf16_bounded_forward_backward_smoke():
    device = torch.device("cuda")
    mu_raw = torch.randn(2, 4, 3, 5, device=device, dtype=torch.bfloat16,
                         requires_grad=True)
    mu_state, x0 = sample_image_bounded_gaussian(
        mu_raw, amplitude=1.0, sigma=1.0
    )
    loss = x0.float().square().mean() + mu_state.float().mean()
    loss.backward()
    assert x0.dtype == torch.bfloat16
    assert mu_raw.grad is not None
    assert torch.isfinite(mu_raw.grad.float()).all()
