from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_path import source_entropy_difficulty
from checkpoint import _validate_stage2_init_checkpoint, model_signature
from config import load_config
from discrete_flow_maps import flow_map, linear_path, sample_prior
from inference import sample_segmentation
from training_objectives import DDPCompatibleTrainingModel


CONFIG = "configs/cityscapes/psd/simplex_source_rank_128k.yaml"


class StatisticsOnlySource(nn.Module):
    fixed_std = 1.0

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = nn.Parameter(logits.clone())
        self.gaussian_forward_calls = 0

    def forward_statistics(self, image: torch.Tensor):
        mean = self.logits[None, :, None, None].expand(
            image.shape[0], self.logits.numel(),
            image.shape[-2] // 4, image.shape[-1] // 4,
        )
        return mean, torch.zeros_like(mean)

    def forward(self, image: torch.Tensor):
        self.gaussian_forward_calls += 1
        raise AssertionError("simplex sampling must not call Gaussian forward")


class LegacyGaussianSource(nn.Module):
    fixed_std = None

    def __init__(self, classes: int = 20) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.linspace(-1.0, 1.0, classes))
        self.log_variance = nn.Parameter(torch.full((classes,), -0.4))

    def forward_statistics(self, image: torch.Tensor):
        shape = (image.shape[0], self.mean.numel(), image.shape[-2] // 4, image.shape[-1] // 4)
        return (
            self.mean[None, :, None, None].expand(shape),
            self.log_variance[None, :, None, None].expand(shape),
        )

    def forward(self, image: torch.Tensor):
        mean, log_variance = self.forward_statistics(image)
        return mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean), mean, log_variance


class TinyEndpoint(nn.Module):
    def __init__(self, classes: int = 20) -> None:
        super().__init__()
        self.state = nn.Conv2d(classes, classes, 1)
        self.image = nn.Conv2d(3, classes, 1)

    def encode_image(self, image: torch.Tensor):
        return F.avg_pool2d(self.image(image), 4)

    def forward_logits_with_image_feat(self, x, image_feat, s, t):
        return self.state(x) + image_feat + (t - s)[:, None, None, None]

    def forward_logits(self, x, image, s, t):
        return self.forward_logits_with_image_feat(x, self.encode_image(image), s, t)


class ForbiddenEndpoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def encode_image(self, image):
        raise AssertionError("source-only Stage 1 must not encode the image")


def _config(*overrides: str) -> dict:
    return load_config(CONFIG, list(overrides))


def _sample(config: dict, source: nn.Module, mode: str = "training"):
    image = torch.zeros(2, 3, 8, 12)
    return sample_prior(
        config, image, None, source, sampling_mode=mode
    )


def test_simplex_prior_is_finite_bounded_and_normalized_without_gaussian_forward():
    config = _config()
    source = StatisticsOnlySource(torch.linspace(-2.0, 2.0, 20))
    torch.manual_seed(4)
    x0, stats = _sample(config, source)
    assert source.gaussian_forward_calls == 0
    assert bool(torch.isfinite(x0).all())
    assert x0.min() >= 0
    assert x0.max() <= 1
    torch.testing.assert_close(
        x0.sum(dim=1), torch.ones_like(x0[:, 0]), atol=1e-6, rtol=1e-6
    )
    assert stats["source_x0_sum_error"] <= 1e-6
    assert stats["source_prior_mode"] == 0


def test_lambda_one_is_exact_probability_and_does_not_consume_rng():
    config = _config("source.simplex_prior.training.lambda=1.0", "source.simplex_prior.training.temperature=0.7")
    source = StatisticsOnlySource(torch.linspace(-1.0, 1.0, 20))
    torch.manual_seed(91)
    before = torch.random.get_rng_state().clone()
    x0, _ = _sample(config, source)
    after = torch.random.get_rng_state()
    expected = torch.softmax(source.logits.float() / 0.7, dim=0)
    expected = expected[None, :, None, None].expand_as(x0)
    torch.testing.assert_close(x0, expected)
    assert torch.equal(before, after)


def test_lambda_zero_is_seeded_dirichlet_and_independent_of_source_logits():
    config = _config("source.simplex_prior.training.lambda=0.0", "source.simplex_prior.training.dirichlet_alpha=0.5")
    first_source = StatisticsOnlySource(torch.linspace(-4.0, 4.0, 20))
    second_source = StatisticsOnlySource(torch.linspace(5.0, -5.0, 20))
    torch.manual_seed(12)
    first, _ = _sample(config, first_source)
    torch.manual_seed(12)
    second, _ = _sample(config, second_source)
    torch.testing.assert_close(first, second)


def test_lower_temperature_is_sharper():
    source = StatisticsOnlySource(torch.linspace(-2.0, 2.0, 20))
    sharp, sharp_stats = _sample(_config(
        "source.simplex_prior.training.lambda=1.0",
        "source.simplex_prior.training.temperature=0.5",
    ), source)
    flat, flat_stats = _sample(_config(
        "source.simplex_prior.training.lambda=1.0",
        "source.simplex_prior.training.temperature=2.0",
    ), source)
    assert sharp_stats["source_probability_entropy"] < flat_stats["source_probability_entropy"]
    assert sharp.max() > flat.max()


@pytest.mark.parametrize("mode", ["training", "inference"])
@pytest.mark.parametrize(
    "key,value,match",
    [
        ("lambda", -0.1, "lambda must be in"),
        ("lambda", 1.1, "lambda must be in"),
        ("temperature", 0, "temperature must be positive"),
        ("dirichlet_alpha", 0, "dirichlet_alpha must be positive"),
    ],
)
def test_simplex_parameter_validation(mode, key, value, match):
    with pytest.raises(ValueError, match=match):
        _config(f"source.simplex_prior.{mode}.{key}={value}")


def test_simplex_requires_zero_gaussian_variance_weight():
    with pytest.raises(ValueError, match="requires source.var_weight=0"):
        _config("source.var_weight=0.1")


def test_training_and_inference_use_distinct_parameter_sets():
    config = _config(
        "source.simplex_prior.training.lambda=1.0",
        "source.simplex_prior.training.temperature=2.0",
        "source.simplex_prior.inference.lambda=0.0",
        "source.simplex_prior.inference.dirichlet_alpha=0.2",
    )
    source = StatisticsOnlySource(torch.linspace(-2.0, 2.0, 20))
    training, training_stats = _sample(config, source, "training")
    torch.manual_seed(5)
    inference, inference_stats = _sample(config, source, "inference")
    assert not torch.equal(training, inference)
    assert training_stats["source_prior_lambda"] == 1
    assert training_stats["source_prior_temperature"] == 2
    assert training_stats["source_prior_mode"] == 0
    assert inference_stats["source_prior_lambda"] == 0
    assert inference_stats["source_prior_dirichlet_alpha"] == pytest.approx(0.2)
    assert inference_stats["source_prior_mode"] == 1


def test_legacy_image_gaussian_still_uses_original_formula():
    config = load_config("configs/_base_/cityscapes/swin_t_160k.yaml")
    source = LegacyGaussianSource()
    image = torch.zeros(2, 3, 8, 12)
    torch.manual_seed(33)
    actual, _ = sample_prior(config, image, None, source, sampling_mode="training")
    torch.manual_seed(33)
    mean, log_variance = source.forward_statistics(image)
    expected = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
    torch.testing.assert_close(actual, expected)


def test_simplex_path_and_flow_map_preserve_class_sum():
    config = _config()
    source = StatisticsOnlySource(torch.linspace(-1.0, 1.0, 20))
    x0, stats = _sample(config, source)
    target = F.one_hot(torch.randint(0, 20, (2, 2, 3)), 20).permute(0, 3, 1, 2).float()
    _, difficulty = source_entropy_difficulty(stats["_path_source_state"], config)
    xt = linear_path(x0, target, torch.tensor([0.2, 0.8]), config, difficulty)
    torch.testing.assert_close(xt.sum(dim=1), torch.ones_like(xt[:, 0]), atol=1e-6, rtol=1e-6)
    denoiser = torch.softmax(torch.randn_like(x0), dim=1)
    mapped = flow_map(
        x0, denoiser, torch.tensor([0.1, 0.2]), torch.tensor([0.5, 0.7]),
        path_config=config, difficulty=difficulty,
    )
    torch.testing.assert_close(mapped.sum(dim=1), torch.ones_like(mapped[:, 0]), atol=2e-6, rtol=1e-6)


def test_entropy_difficulty_uses_raw_mu_not_simplex_noise_or_temperature():
    source = StatisticsOnlySource(torch.linspace(-2.0, 2.0, 20))
    first_config = _config("source.simplex_prior.training.temperature=0.5")
    second_config = _config("source.simplex_prior.training.temperature=2.0")
    torch.manual_seed(1)
    first_x0, first_stats = _sample(first_config, source)
    torch.manual_seed(2)
    second_x0, second_stats = _sample(second_config, source)
    assert not torch.equal(first_x0, second_x0)
    torch.testing.assert_close(first_stats["_path_source_state"], second_stats["_path_source_state"])
    _, first_difficulty = source_entropy_difficulty(first_stats["_path_source_state"], first_config)
    _, second_difficulty = source_entropy_difficulty(second_stats["_path_source_state"], second_config)
    torch.testing.assert_close(first_difficulty, second_difficulty)


def test_stage1_source_ce_uses_raw_logits_without_sampling():
    config = load_config(
        "configs/cityscapes/diagonal/source_segformer_b1_32k.yaml",
        ["source.prior_type=image_simplex_mixture", "source.var_weight=0"],
    )
    source = StatisticsOnlySource(torch.linspace(-1.0, 1.0, 20))
    image = torch.zeros(1, 3, 8, 12)
    target = torch.full((1, 8, 12), 19, dtype=torch.long)
    target_state = torch.full((1, 2, 3), 19, dtype=torch.long)
    one_hot = F.one_hot(target_state, 20).permute(0, 3, 1, 2).float()
    _, stats = sample_prior(
        config, image, one_hot, source, target_full=target,
        valid_mask_full=torch.ones_like(target, dtype=torch.bool),
        sample_state=False, sampling_mode="training",
    )
    expected_logits = source.logits[None, :, None, None].expand(1, 20, 8, 12)
    torch.testing.assert_close(stats["loss_source_ce"], F.cross_entropy(expected_logits, target))
    assert source.gaussian_forward_calls == 0

    adapter = DDPCompatibleTrainingModel(ForbiddenEndpoint(), source, config)
    result = adapter(
        operation="stage1_objectives", image=image, target=target,
        epoch_index=0, progress_in_epoch=0.0,
    )
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert source.logits.grad is not None


def test_stage2_psd_backward_and_three_step_inference_smoke():
    config = _config()
    endpoint = TinyEndpoint()
    source = StatisticsOnlySource(torch.linspace(-1.0, 1.0, 20))
    source.requires_grad_(False)
    image = torch.randn(1, 3, 8, 12)
    adapter = DDPCompatibleTrainingModel(endpoint, source, config)
    target = torch.randint(0, 20, (1, 8, 12))
    result = adapter(
        operation="stage2_objectives", image=image, target=target,
        epoch_index=0, progress_in_epoch=0.0, optimizer_step=0,
    )
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert any(parameter.grad is not None for parameter in endpoint.parameters())
    prediction = sample_segmentation(endpoint, source, image, config, num_steps=3)
    assert prediction.shape == (1, 8, 12)


def test_simplex_sampling_does_not_change_architecture_checkpoint_signature():
    simplex = _config()
    gaussian = deepcopy(simplex)
    gaussian["source"]["prior_type"] = "image_gaussian"
    assert model_signature(simplex) == model_signature(gaussian)
    checkpoint = {
        "stage": "diagonal_pretrain",
        "model": {},
        "source_model": {},
        "model_signature": model_signature(gaussian),
    }
    _validate_stage2_init_checkpoint(checkpoint, simplex, "stage1.pt")
