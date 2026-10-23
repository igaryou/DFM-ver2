import math
from pathlib import Path

import pytest
import torch

from config import load_config, validate_config
from discrete_flow_maps import sample_prior, variance_only_kl
from source_model import (
    IndependentGaussianSourceGenerator,
    SegFormerSourceGenerator,
    build_source_model,
)
from trainer import _wandb_epoch_payload


ROOT = Path(__file__).parents[1]
CONFIG = (
    ROOT
    / "configs/cityscapes/original/psd"
    / "joint_image_gaussian_b1_learned_sigma_varreg_800ep.yaml"
)


@pytest.mark.parametrize("target_std", [1.0, 1.5])
def test_variance_only_kl_is_zero_at_target(target_std):
    logvar = torch.full((2, 1, 3, 4), math.log(target_std**2))
    torch.testing.assert_close(
        variance_only_kl(logvar, target_std), torch.tensor(0.0), atol=1e-7, rtol=0
    )


def test_variance_only_kl_is_positive_away_from_target_and_has_no_mu_gradient():
    mu = torch.randn(2, 20, 3, 4, requires_grad=True)
    logvar = torch.full((2, 1, 3, 4), math.log(2.0), requires_grad=True)
    loss = variance_only_kl(logvar, 1.0)
    assert loss > 0
    loss.backward()
    assert logvar.grad is not None
    assert torch.count_nonzero(logvar.grad) > 0
    assert mu.grad is None


def _source(*, target_std=1.0, init_at_target=True):
    return SegFormerSourceGenerator(
        num_classes=20,
        variant="b1",
        pretrained=False,
        decoder_channels=16,
        freeze_encoder=False,
        learned_logvar=True,
        fixed_std=None,
        mu_tanh_scale=0.0,
        input_already_normalized=False,
        state_downsample_factor=4,
        decoder_type="custom",
        variance_mode="pixel_shared",
        variance_target_std=target_std,
        init_logvar_at_target=init_at_target,
    )


def test_pixel_shared_shapes_broadcast_and_target_initialization(monkeypatch):
    target_std = 1.5
    source = _source(target_std=target_std)
    source.eval()
    image = torch.rand(1, 3, 32, 64)
    eps = torch.linspace(0.5, 1.5, 20)[None, :, None, None].expand(1, 20, 8, 16)
    monkeypatch.setattr(torch, "randn_like", lambda tensor: eps.to(tensor))

    with torch.no_grad():
        x0, mu, logvar = source(image)

    assert mu.shape == x0.shape == (1, 20, 8, 16)
    assert logvar.shape == (1, 1, 8, 16)
    recovered_sigma = (x0 - mu) / eps
    torch.testing.assert_close(
        recovered_sigma,
        recovered_sigma[:, :1].expand_as(recovered_sigma),
        atol=1e-6,
        rtol=1e-6,
    )
    assert float(torch.exp(0.5 * logvar.float()).mean()) == pytest.approx(
        target_std, abs=1e-6
    )


def test_new_config_is_joint_variance_only_and_reports_pixel_shared_stats():
    config = load_config(CONFIG)
    source_config = config["source"]
    assert source_config["variance_mode"] == "pixel_shared"
    assert source_config["logvar_model"]["mode"] == "separate"
    assert source_config["learned_logvar"] is True
    assert source_config["fixed_std"] is None
    assert source_config["detach_from_flow"] is False
    assert source_config["supervision"]["type"] == "none"
    assert source_config["variance_regularization"] == {
        "enabled": True,
        "target_std": 1.0,
        "weight": 1.0e-2,
        "init_at_target": True,
    }
    assert config["training"]["stages"]["source_pretrain"]["enabled"] is False
    assert config["training"]["stages"]["flow_training"]["start_epoch"] == 0

    class TinyPixelSharedSource:
        fixed_std = None

        def __call__(self, image):
            mu = torch.randn(image.shape[0], 20, *image.shape[-2:], requires_grad=True)
            logvar = torch.zeros(
                image.shape[0], 1, *image.shape[-2:], requires_grad=True
            )
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu), mu, logvar

    image = torch.randn(2, 3, 3, 4)
    target = torch.nn.functional.one_hot(
        torch.randint(0, 20, (2, 3, 4)), 20
    ).permute(0, 3, 1, 2).float()
    x0, stats = sample_prior(config, image, target, TinyPixelSharedSource())
    assert x0.shape == (2, 20, 3, 4)
    assert float(stats["loss_source_var"].detach()) == pytest.approx(0.0)
    assert float(stats["weighted_var"].detach()) == pytest.approx(0.0)
    assert float(stats["source_variance_ratio"]) == pytest.approx(1.0)
    for key in (
        "source_logvar_mean", "source_logvar_std", "source_logvar_min",
        "source_logvar_max", "source_sigma_mean", "source_sigma_std",
        "source_sigma_min", "source_sigma_max", "source_variance_mean",
        "source_variance_std", "source_variance_min", "source_variance_max",
        "source_target_std", "source_target_variance", "source_variance_ratio",
    ):
        assert key in stats
        assert torch.isfinite(stats[key])


def test_mu_and_logvar_models_can_use_different_segformer_variants():
    config = load_config(CONFIG, [
        "source.pretrained=false",
        "source.logvar_model.pretrained=false",
        "source.logvar_model.segformer_variant=b0",
        "source.logvar_model.decoder_channels=16",
    ])
    model = build_source_model(config)
    assert isinstance(model, IndependentGaussianSourceGenerator)
    assert model.mu_model.encoder.config.hidden_sizes == [64, 128, 320, 512]
    assert model.logvar_model.encoder.config.hidden_sizes == [32, 64, 160, 256]
    assert model.mu_model is not model.logvar_model
    assert model.mu_model.encoder is not model.logvar_model.encoder

    image = torch.randn(1, 3, 32, 64)
    x0, mu, logvar = model(image)
    assert mu.shape == x0.shape == (1, 20, 32, 64)
    assert logvar.shape == (1, 1, 32, 64)
    torch.testing.assert_close(
        torch.exp(0.5 * logvar.float()),
        torch.ones_like(logvar, dtype=torch.float32),
    )
    (x0.float().square().mean() + variance_only_kl(logvar, 1.0)).backward()
    assert model.mu_model.decoder[-1].weight.grad is not None
    assert model.logvar_model.decoder[-1].weight.grad is not None


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"target_std": 0.0}, "target_std must be positive"),
        ({"weight": -0.1}, "weight must be non-negative"),
    ],
)
def test_variance_regularization_rejects_invalid_values(update, message):
    config = load_config(CONFIG)
    config["source"]["variance_regularization"].update(update)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


@pytest.mark.parametrize(
    ("source_update", "message"),
    [
        ({"learned_logvar": False}, "requires learned_logvar=true"),
        ({"fixed_std": 1.0}, "requires fixed_std=null"),
        ({"variance_mode": "unknown"}, "variance_mode must be"),
    ],
)
def test_pixel_shared_rejects_inconsistent_config(source_update, message):
    config = load_config(CONFIG)
    config["source"].update(source_update)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_variance_metrics_have_namespaced_wandb_keys():
    report = {
        "loss_var": 0.1,
        "weighted_var": 0.001,
        "logvar_mean": 0.0,
        "logvar_std": 0.2,
        "logvar_min": -0.5,
        "logvar_max": 0.5,
        "sigma_mean": 1.0,
        "sigma_std": 0.1,
        "sigma_min": 0.8,
        "sigma_max": 1.2,
        "variance_mean": 1.0,
        "variance_std": 0.2,
        "variance_min": 0.64,
        "variance_max": 1.44,
        "target_std": 1.0,
        "target_variance": 1.0,
        "variance_ratio": 1.0,
    }
    payload = _wandb_epoch_payload(report, "psd")
    expected = {
        "epoch/source/logvar_mean", "epoch/source/logvar_std",
        "epoch/source/logvar_min", "epoch/source/logvar_max",
        "epoch/source/std_mean", "epoch/source/std_std",
        "epoch/source/std_min", "epoch/source/std_max",
        "epoch/source/variance_mean", "epoch/source/variance_std",
        "epoch/source/variance_min", "epoch/source/variance_max",
        "epoch/source/target_std", "epoch/source/target_variance",
        "epoch/source/variance_ratio",
        "epoch/loss/variance_regularization",
        "epoch/loss/weighted_variance_regularization",
    }
    assert expected <= payload.keys()
    assert payload["epoch/loss_var"] == report["loss_var"]
    assert payload["epoch/weighted_var"] == report["weighted_var"]
