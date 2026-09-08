from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from adaptive_path import (
    bounded_gaussian_variance_maps,
    entropy_scheduler_lambda,
    source_entropy_difficulty,
)
from config import load_config
from discrete_flow_maps import linear_path, sample_image_bounded_gaussian
from inference import sample_segmentation_from_x0
from model import DiscreteFlowMapModel, ImageEncoder
from state_space import prepare_state_targets, state_spatial_size


ROOT = Path(__file__).parents[1]
ORIGINAL = ROOT / (
    "configs/cityscapes/original/psd/"
    "joint_bounded_gaussian_b1_exponential_path_adaptive_std_trainable.yaml"
)
MMSEG = ROOT / "configs/cityscapes/mmseg/psd/original/swin_t_linear_160k.yaml"


def test_original_full_resolution_state_fields_have_cfm_shapes():
    batch, height, width, classes = 2, 256, 512, 20
    config = load_config(ORIGINAL)
    image = torch.zeros(batch, 3, height, width)
    target = torch.zeros(batch, height, width, dtype=torch.long)
    targets = prepare_state_targets(
        target,
        num_classes=classes,
        state_size=state_spatial_size(
            image, config["model"]["state_downsample_factor"]
        ),
        ignore_index=19,
        mask_pixel_losses=True,
    )
    mu = torch.randn(batch, classes, height, width)
    logvar = torch.zeros_like(mu)
    entropy, difficulty, variance, sigma = bounded_gaussian_variance_maps(
        mu,
        base_std=config["source"]["fixed_std"],
        variance_type="entropy_adaptive",
        rho=config["source"]["bounded_gaussian"]["variance"]["rho"],
        normalization="rank",
        eps=1.0e-8,
    )
    mu_state, x0 = sample_image_bounded_gaussian(
        mu,
        amplitude=config["source"]["bounded_gaussian"]["amplitude"],
        temperature=config["source"]["bounded_gaussian"]["temperature"],
        sigma=sigma,
        epsilon=torch.zeros_like(mu),
    )
    x1 = targets.one_hot_state
    x_t = linear_path(x0, x1, torch.tensor([0.25, 0.75]))
    path_entropy, path_difficulty = source_entropy_difficulty(
        mu, config, spatial_size=(height, width)
    )
    coefficient = entropy_scheduler_lambda(
        torch.tensor([0.25, 0.75]),
        path_difficulty,
        beta=config["flow"]["path"]["scheduler"]["beta"],
        scheduler_type="exponential",
        difficulty_gamma=config["flow"]["path"]["scheduler"]["difficulty_gamma"],
    )
    # Exercise full-resolution image/state fusion without instantiating the
    # production U-Net's deliberately large activation pyramid in a CPU unit test.
    with torch.no_grad():
        image_feat = nn.Conv2d(3, 128, 1)(image)
        mask_feat = nn.Conv2d(classes, 128, 1)(x_t)
        logits = nn.Conv2d(128, classes, 1)(image_feat + mask_feat)

    assert image.shape == (2, 3, 256, 512)
    for tensor in (mu, mu_state, logvar, x0, x1, x_t, logits):
        assert tensor.shape == (2, 20, 256, 512)
    for field in (
        entropy, difficulty, variance, sigma, path_entropy,
        path_difficulty, coefficient,
    ):
        assert field.shape == (2, 256, 512)
    assert image_feat.shape == mask_feat.shape == (2, 128, 256, 512)


def test_rrdb_factor_one_and_tiny_flow_model_preserve_spatial_resolution():
    encoder = ImageEncoder(channels=4, blocks=0, growth=2, downsample_factor=1)
    image = torch.randn(2, 3, 17, 29)
    assert encoder(image).shape == (2, 4, 17, 29)
    assert encoder.first.stride == (1, 1)
    assert len(encoder.downsample) == 0

    config = load_config(ORIGINAL)["model"]
    tiny = deepcopy(config)
    tiny.update({
        "num_classes": 20,
        "fusion_channels": 4,
        "rrdb_blocks": 0,
        "rrdb_growth_channels": 2,
    })
    tiny["unet"] = {
        "base_channels": 4,
        "channel_mults": [1, 2],
        "num_res_blocks": 1,
        "attention_levels": [],
        "num_heads": 1,
        "dropout": 0.0,
        "time_embedding_dim": 16,
    }
    model = DiscreteFlowMapModel(tiny)
    state = torch.randn(2, 20, 16, 32)
    logits = model.forward_logits(
        state,
        torch.randn(2, 3, 16, 32),
        torch.zeros(2),
        torch.ones(2),
    )
    assert logits.shape == state.shape


class _IdentityLogitFlow(nn.Module):
    def forward_logits(self, state, image, s, t):
        del image, s, t
        return state


def test_original_inference_terminal_and_prediction_are_full_resolution():
    config = load_config(ORIGINAL)
    image = torch.randn(2, 3, 256, 512)
    x0 = torch.randn(2, 20, 256, 512)
    terminal = sample_segmentation_from_x0(
        _IdentityLogitFlow(), image, x0, config,
        num_steps=1, return_terminal_state=True,
        path_difficulty=torch.zeros(2, 256, 512),
    )
    prediction = sample_segmentation_from_x0(
        _IdentityLogitFlow(), image, x0, config, num_steps=1,
        path_difficulty=torch.zeros(2, 256, 512),
    )
    assert terminal.shape == (2, 20, 256, 512)
    assert prediction.shape == (2, 256, 512)


def test_original_is_fullres_while_mmseg_remains_quarter_resolution():
    original = load_config(ORIGINAL)
    mmseg = load_config(MMSEG)
    assert original["dataset"]["image_size"] == [256, 512]
    assert original["model"]["state_downsample_factor"] == 1
    assert state_spatial_size((256, 512), 1) == (256, 512)
    assert mmseg["model"]["state_downsample_factor"] == 4
    assert state_spatial_size((512, 1024), 4) == (128, 256)
