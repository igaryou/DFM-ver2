from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from checkpoint import validate_source_decoder_checkpoint
from config import load_config
from discrete_flow_maps import sample_prior
from source_model import SegFormerSourceGenerator
from state_space import prepare_state_targets, resize_continuous
from training_objectives import (
    DDPCompatibleTrainingModel,
    compute_model_training_objectives,
)


ROOT = Path(__file__).parents[1]
BASE_CONFIG = ROOT / "configs/_base_/cityscapes/swin_t_160k.yaml"
CONFIG = (
    ROOT
    / "configs/cityscapes/psd/joint_swin_t_segformer_b1_standard_ce_160k.yaml"
)


def test_joint_standard_ce_config_resolves_required_contract():
    config = load_config(CONFIG)
    assert config["experiment"]["stage"] == "joint_training"
    assert config["source"]["type"] == "trainable_segformer"
    assert config["source"]["prior_type"] == "image_gaussian"
    assert config["source"]["backbone"] == "segformer"
    assert config["source"]["segformer_variant"] == "b1"
    assert config["source"]["segformer_decoder"] == "standard"
    assert config["source"]["pretrained"] is True
    assert config["source"]["checkpoint"] is None
    assert config["source"]["freeze"] is False
    assert config["source"]["freeze_encoder"] is False
    assert config["source"]["learned_logvar"] is False
    assert config["source"]["fixed_std"] == 1.0
    assert config["source"]["mu_tanh_scale"] == 0.0
    assert config["source"]["input_already_normalized"] is True
    assert config["source"]["use_loss_align"] is False
    assert config["source"]["supervision"] == {
        "type": "cross_entropy",
        "weight": 0.20,
        "include_void": True,
    }
    assert config["training"]["max_optimizer_steps"] == 160000
    assert config["loss"]["consistency"]["type"] == "psd"
    assert config["loss"]["consistency"]["enabled"] is True
    assert config["checkpoint"]["init_from"] is None
    assert config["checkpoint"]["resume"] is None


def test_joint_standard_ce_changes_only_requested_recipe_sections():
    base = load_config(BASE_CONFIG)
    config = load_config(CONFIG)
    assert config["model"] == base["model"]
    assert config["flow"] == base["flow"]
    assert config["training"]["optimizer"] == base["training"]["optimizer"]
    assert config["training"]["scheduler"] == base["training"]["scheduler"]
    assert config["loss"] == base["loss"]
    assert config["dataset"] == base["dataset"]
    assert config["augmentation"] == base["augmentation"]


def test_existing_custom_align_joint_config_is_unchanged():
    config = load_config(BASE_CONFIG)
    assert config["source"]["segformer_decoder"] == "custom"
    assert config["source"]["use_loss_align"] is True
    assert config["source"]["supervision"] == {
        "type": "align", "weight": 0.20,
    }


class _DistinctSampleSource(nn.Module):
    fixed_std = 1.0

    def __init__(self):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor([
            [[[4.0]], [[-1.0]], [[0.5]], [[-2.0]]]
        ]))

    def forward(self, image):
        mu = self.mu.expand(
            image.shape[0], -1, image.shape[-2] // 4, image.shape[-1] // 4
        )
        logvar = torch.zeros_like(mu)
        x0 = -mu
        return x0, mu, logvar


def test_source_ce_uses_mu_not_x0_includes_void_and_backpropagates():
    image = torch.zeros(1, 3, 4, 4)
    target = torch.tensor([[
        [0, 0, 1, 1],
        [0, 2, 2, 1],
        [3, 3, 2, 1],
        [3, 3, 3, 2],
    ]])
    targets = prepare_state_targets(
        target,
        num_classes=4,
        state_size=(1, 1),
        ignore_index=3,
        mask_pixel_losses=True,
    )
    config = {
        "dataset": {"num_classes": 4},
        "model": {"state_downsample_factor": 4},
        "loss": {"ignore_index": 3},
        "source": {
            "prior_type": "image_gaussian",
            "var_weight": 0.0,
            "align_weight": 99.0,
            "align_eps": 1.0e-8,
            "use_loss_align": False,
            "supervision": {
                "type": "cross_entropy", "weight": 0.20, "include_void": True,
            },
        },
        "flow": {"path": {"type": "linear"}},
    }
    source = _DistinctSampleSource()
    x0, stats = sample_prior(
        config,
        image,
        targets.one_hot_state,
        source,
        target_full=target,
        valid_mask_full=targets.valid_mask_full,
    )
    mu_full = resize_continuous(source.mu, target.shape[-2:])
    expected_mu_ce = F.cross_entropy(mu_full, target)
    wrong_x0_ce = F.cross_entropy(resize_continuous(x0, target.shape[-2:]), target)
    torch.testing.assert_close(stats["loss_source_ce"], expected_mu_ce)
    assert not torch.isclose(stats["loss_source_ce"], wrong_x0_ce)
    torch.testing.assert_close(
        stats["weighted_source_supervision"], 0.20 * expected_mu_ce
    )
    stats["weighted_source_supervision"].backward()
    assert source.mu.grad is not None
    assert torch.isfinite(source.mu.grad).all()
    assert source.mu.grad[0, 3].abs().item() > 0


class _TinyEndpoint(nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        self.image_projection = nn.Conv2d(3, classes, 1)
        self.state_projection = nn.Conv2d(classes, classes, 1)

    def encode_image(self, image):
        return F.avg_pool2d(self.image_projection(image), 4)

    def forward_logits_with_image_feat(self, state, image_feat, s, t):
        del s, t
        return self.state_projection(state) + image_feat

    def forward_logits(self, state, image, s, t):
        return self.forward_logits_with_image_feat(
            state, self.encode_image(image), s, t
        )


def test_joint_total_includes_weighted_source_ce():
    config = load_config(CONFIG)
    config["dataset"]["num_classes"] = 4
    config["model"]["num_classes"] = 4
    config["runtime"]["amp"] = False
    config["loss"]["ignore_index"] = 3
    config["evaluation"]["ignore_index"] = 3
    endpoint = _TinyEndpoint()
    source = _DistinctSampleSource()
    adapter = DDPCompatibleTrainingModel(endpoint, source, config)
    result = compute_model_training_objectives(
        adapter,
        operation="joint_objectives",
        image=torch.randn(1, 3, 8, 8),
        target=torch.randint(0, 4, (1, 8, 8)),
        epoch_index=0,
        progress_in_epoch=0.0,
        optimizer_step=0,
    )
    torch.testing.assert_close(
        result["source_objective"],
        0.20 * result["stats"]["loss_source_ce"],
    )
    torch.testing.assert_close(
        result["loss"],
        result["diagonal_objective"]
        + result["psd_objective"]
        + result["source_objective"],
    )


def test_source_ce_alone_reaches_standard_head_classifier():
    source = SegFormerSourceGenerator(
        num_classes=20,
        variant="b1",
        pretrained=False,
        decoder_channels=256,
        freeze_encoder=False,
        learned_logvar=False,
        fixed_std=1.0,
        mu_tanh_scale=0.0,
        input_already_normalized=True,
        state_downsample_factor=4,
        decoder_type="standard",
    )
    image = torch.randn(1, 3, 64, 128)
    target = torch.randint(0, 20, (1, 64, 128))
    target[:, 0, 0] = 19
    targets = prepare_state_targets(
        target,
        num_classes=20,
        state_size=(16, 32),
        ignore_index=19,
        mask_pixel_losses=True,
    )
    config = {
        "dataset": {"num_classes": 20},
        "model": {"state_downsample_factor": 4},
        "loss": {"ignore_index": 19},
        "source": {
            "prior_type": "image_gaussian",
            "var_weight": 0.0,
            "align_weight": 0.20,
            "align_eps": 1.0e-8,
            "use_loss_align": False,
            "supervision": {
                "type": "cross_entropy", "weight": 0.20, "include_void": True,
            },
        },
        "flow": {"path": {"type": "linear"}},
    }
    _, stats = sample_prior(
        config,
        image,
        targets.one_hot_state,
        source,
        target_full=target,
        valid_mask_full=targets.valid_mask_full,
    )
    stats["loss_source_ce"].backward()
    gradient = source.decode_head.classifier.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient[19].abs().sum() > 0


def test_standard_and_custom_joint_checkpoints_cannot_mix():
    standard = load_config(CONFIG)
    custom = deepcopy(standard)
    custom["source"]["segformer_decoder"] = "custom"
    with pytest.raises(RuntimeError, match="source decoder mismatch"):
        validate_source_decoder_checkpoint(
            {"config": custom}, standard, "custom.pt"
        )
