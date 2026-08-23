from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from discrete_flow_maps import sample_prior
from inference import terminal_state_to_original_prediction
from model import DiscreteFlowMapModel, ImageEncoder
from source_model import UNetSourceGenerator
from state_space import prepare_state_targets, resize_continuous, state_spatial_size
from training_objectives import DDPCompatibleTrainingModel, compute_model_training_objectives


def _tiny_model_config(classes: int) -> dict:
    return {
        "num_classes": classes,
        "state_downsample_factor": 4,
        "fusion_channels": 4,
        "rrdb_blocks": 0,
        "rrdb_growth_channels": 2,
        "unet": {
            "base_channels": 4,
            "channel_mults": [1],
            "num_res_blocks": 1,
            "attention_levels": [],
            "num_heads": 1,
            "dropout": 0.0,
            "time_embedding_dim": 8,
        },
    }


@pytest.mark.parametrize(
    "image_size,classes,expected",
    [((512, 512), 151, (128, 128)), ((256, 512), 20, (64, 128))],
)
def test_production_components_use_quarter_state(image_size, classes, expected):
    image = torch.randn(1, 3, *image_size)
    source = UNetSourceGenerator(classes, 4, False, 1.0, 4)
    x0, mu, logvar = source(image)
    assert x0.shape == mu.shape == logvar.shape == (1, classes, *expected)

    encoder = ImageEncoder(channels=4, blocks=0, growth=2, downsample_factor=4)
    assert encoder(image).shape == (1, 4, *expected)

    endpoint = DiscreteFlowMapModel(_tiny_model_config(classes))
    time = torch.tensor([0.25])
    image_feature = endpoint.encode_image(image)
    logits = endpoint.forward_logits_with_image_feat(x0, image_feature, time, time)
    assert image_feature.shape == (1, 4, *expected)
    assert logits.shape == (1, classes, *expected)
    assert resize_continuous(logits, image_size).shape == (1, classes, *image_size)


def test_variable_width_is_derived_from_input_not_hardcoded():
    image = torch.randn(1, 3, 512, 1024)
    assert state_spatial_size(image, 4) == (128, 256)
    encoder = ImageEncoder(channels=4, blocks=0, growth=2, downsample_factor=4)
    assert encoder(image).shape == (1, 4, 128, 256)
    source = UNetSourceGenerator(5, 4, False, 1.0, 4)
    x0, mu, logvar = source(image)
    assert x0.shape == mu.shape == logvar.shape == (1, 5, 128, 256)


def test_target_full_and_state_are_discrete_and_have_separate_masks():
    target_full = torch.ones(1, 512, 1024, dtype=torch.long)
    target_full[:, :16, :32] = 0
    targets = prepare_state_targets(
        target_full,
        num_classes=151,
        state_size=(128, 256),
        ignore_index=0,
        mask_pixel_losses=True,
    )
    assert targets.target_full.shape == (1, 512, 1024)
    assert targets.target_state.shape == (1, 128, 256)
    assert targets.one_hot_state.shape == (1, 151, 128, 256)
    assert targets.valid_mask_full.shape == targets.target_full.shape
    assert targets.valid_mask_state.shape == targets.target_state.shape
    assert not targets.valid_mask_state[:, :4, :8].any()


class _QuarterEndpoint(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.image_projection = nn.Conv2d(3, classes, 1)
        self.state_projection = nn.Conv2d(classes, classes, 1)
        self.time_scale = nn.Parameter(torch.zeros(classes))

    def encode_image(self, image):
        return F.avg_pool2d(self.image_projection(image), 4, 4)

    def forward_logits_with_image_feat(self, x, image_feat, s, t):
        assert x.shape[-2:] == image_feat.shape[-2:]
        return self.state_projection(x) + image_feat + (
            s + t
        )[:, None, None, None] * self.time_scale[None, :, None, None]

    def forward_logits(self, x, image, s, t):
        return self.forward_logits_with_image_feat(x, self.encode_image(image), s, t)


def _objective_config(loss_type: str) -> dict:
    return {
        "runtime": {"amp": False, "amp_dtype": "bf16"},
        "dataset": {"num_classes": 4},
        "model": {"state_downsample_factor": 4},
        "source": {
            "prior_type": "gaussian",
            "prior_noise_std": 1.0,
            "var_weight": 0.0,
            "align_weight": 0.0,
            "use_loss_align": False,
        },
        "flow": {"time_eps": 1.0e-5, "probability_eps": 1.0e-8},
        "time_sampling": {"min_time": 0.0, "max_time": 1.0, "min_gap": 1e-4},
        "training": {"label_smoothing": 0.0},
        "loss": {
            "ignore_index": 0,
            "mask_pixel_losses": True,
            "primary": {"weight": 1.0},
            "consistency": {
                "enabled": True,
                "type": loss_type,
                "weight": 0.1,
                "start_epoch": 0,
                "warmup_epochs": 0,
                "max_weight": 1.0,
                "precision": {
                    "jvp_dtype": None if loss_type == "psd" else "fp32",
                    "numerical_dtype": "fp32",
                    "debug_assertions": True,
                },
                "ecld": {"ec_weight": 4.0, "td_weight": 2.0, "time_weighting": "none"},
                "adaptive_kl": {
                    "enabled": False, "c": 1e-6, "r": 0.5,
                    "normalize_mean": True, "max_weight": 100.0,
                },
                "invalid_teacher": {
                    "strategy": "mask_pixel", "log_eps": 1e-6,
                    "skip_batch_threshold": None,
                },
            },
        },
    }


@pytest.mark.parametrize("loss_type", ["psd", "csd", "ecld", "esd"])
def test_all_consistency_objectives_run_on_quarter_state(loss_type):
    config = _objective_config(loss_type)
    endpoint = _QuarterEndpoint(4)
    adapter = DDPCompatibleTrainingModel(endpoint, None, config)
    image = torch.randn(2, 3, 32, 48)
    target_full = torch.randint(0, 4, (2, 32, 48))
    one_hot_full = F.one_hot(target_full, 4).permute(0, 3, 1, 2).float()
    result = compute_model_training_objectives(
        adapter,
        operation="joint_objectives",
        image=image,
        one_hot=one_hot_full,
        target=target_full,
        epoch_index=0,
        progress_in_epoch=0.5,
    )
    assert result["stats"]["state_height"] == 8
    assert result["stats"]["state_width"] == 12
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert endpoint.state_projection.weight.grad is not None


class _FixedQuarterSource(nn.Module):
    fixed_std = 1.0

    def forward(self, image):
        mu = torch.tensor(
            [[[[2.0, -1.0], [0.5, 1.0]],
              [[-1.0, 2.0], [0.0, -0.5]],
              [[0.0, 0.0], [1.0, 0.0]]]],
            dtype=image.dtype,
            device=image.device,
        )
        logvar = torch.zeros_like(mu)
        return mu, mu, logvar


@pytest.mark.parametrize("supervision_type", ["align", "cross_entropy"])
def test_source_supervision_upsamples_mu_and_ignores_ade_zero(supervision_type):
    image = torch.zeros(1, 3, 8, 8)
    target_full = torch.zeros(1, 8, 8, dtype=torch.long)
    target_full[:, 4:, 4:] = 2
    targets = prepare_state_targets(
        target_full, num_classes=3, state_size=(2, 2),
        ignore_index=0, mask_pixel_losses=True,
    )
    one_hot_full = F.one_hot(target_full, 3).permute(0, 3, 1, 2).float()
    config = {
        "dataset": {"num_classes": 3},
        "model": {"state_downsample_factor": 4},
        "loss": {"ignore_index": 0},
        "source": {
            "prior_type": "image_gaussian", "var_weight": 0.0,
            "align_weight": 99.0, "align_eps": 1e-8,
            "use_loss_align": True,
            "supervision": {"type": supervision_type, "weight": 0.25},
        },
    }
    _, stats = sample_prior(
        config, image, targets.one_hot_state, _FixedQuarterSource(),
        target_full=target_full, target_one_hot_full=one_hot_full,
        valid_mask_full=targets.valid_mask_full,
    )
    assert torch.isfinite(stats["loss_source_supervision"])
    assert stats["weighted_source_supervision"] == pytest.approx(
        0.25 * stats["loss_source_supervision"]
    )
    if supervision_type == "cross_entropy":
        mu = _FixedQuarterSource()(image)[1]
        expected = F.cross_entropy(
            resize_continuous(mu, (8, 8)), target_full,
            ignore_index=0, reduction="none",
        )[targets.valid_mask_full].mean()
        torch.testing.assert_close(stats["loss_source_ce"], expected)

    all_ignored = torch.zeros_like(target_full)
    ignored_targets = prepare_state_targets(
        all_ignored, num_classes=3, state_size=(2, 2),
        ignore_index=0, mask_pixel_losses=True,
    )
    ignored_one_hot = F.one_hot(all_ignored, 3).permute(0, 3, 1, 2).float()
    _, ignored = sample_prior(
        config, image, ignored_targets.one_hot_state, _FixedQuarterSource(),
        target_full=all_ignored, target_one_hot_full=ignored_one_hot,
        valid_mask_full=ignored_targets.valid_mask_full,
    )
    assert float(ignored["loss_source_supervision"]) == 0.0


def test_low_resolution_terminal_is_upsampled_before_argmax_and_unpadded():
    terminal = torch.zeros(1, 3, 2, 3)
    terminal[:, 0] = 1.0
    terminal[:, 1, :, 1] = 4.0
    terminal[:, 2, :, 2] = 99.0  # padded region must be removed after continuous resize
    prediction = terminal_state_to_original_prediction(
        terminal,
        model_shape=(8, 8),
        padded_shape=(8, 12),
        original_shape=(16, 16),
    )
    full_padded = F.interpolate(
        terminal, (8, 12), mode="bilinear", align_corners=False
    )
    expected = F.interpolate(
        full_padded[..., :8, :8], (16, 16), mode="bilinear", align_corners=False
    ).argmax(1)
    assert torch.equal(prediction, expected)
