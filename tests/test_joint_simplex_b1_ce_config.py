from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import load_config
from discrete_flow_maps import sample_prior
from state_space import prepare_state_targets
from trainer import build_optimizer
from training_objectives import DDPCompatibleTrainingModel


ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "configs" / "cityscapes" / "psd"
PARENT = CONFIG_DIR / "joint_swin_t_segformer_b1_standard_ce_160k.yaml"
COMMON = CONFIG_DIR / "joint_simplex_b1_ce_160k.yaml"
IGNORE = CONFIG_DIR / "joint_simplex_b1_ce_ignore_void_160k.yaml"
INCLUDE = CONFIG_DIR / "joint_simplex_b1_ce_include_void_160k.yaml"


@pytest.mark.parametrize("path", [IGNORE, INCLUDE])
def test_joint_simplex_b1_ce_contract(path):
    config = load_config(path)
    assert config["experiment"]["stage"] == "joint_training"
    assert config["source"]["type"] == "trainable_segformer"
    assert config["source"]["segformer_variant"] == "b1"
    assert config["source"]["segformer_decoder"] == "standard"
    assert config["source"]["prior_type"] == "image_simplex_mixture"
    assert config["source"]["freeze"] is False
    assert config["source"]["freeze_encoder"] is False
    assert config["source"]["supervision"] == {
        "type": "cross_entropy", "weight": 0.20, "include_void": True,
    }
    assert config["training"]["max_optimizer_steps"] == 160000
    assert config["training"]["optimizer"]["parameter_groups"] == {
        "model": {"lr": 1.0e-4}, "source": {"lr": 5.0e-5},
    }
    assert config["checkpoint"]["init_from"] is None
    assert config["checkpoint"]["resume"] is None
    for mode in ("training", "inference"):
        assert config["source"]["simplex_prior"][mode] == {
            "lambda": 0.8,
            "temperature": 1.0,
            "dirichlet_alpha": 1.0,
        }


def test_joint_simplex_conditions_differ_only_in_identity_and_psd_void_mask():
    ignored = load_config(IGNORE)
    included = load_config(INCLUDE)
    assert ignored["loss"]["consistency"]["psd"]["ignore_void"] is True
    assert included["loss"]["consistency"]["psd"]["ignore_void"] is False

    for config in (ignored, included):
        config["experiment"] = {}
        config["wandb"] = {}
        config["runtime"]["config_path"] = None
        config["loss"]["consistency"]["psd"]["ignore_void"] = True
    assert ignored == included


def test_common_config_changes_only_source_prior_and_metadata_from_parent():
    parent = load_config(PARENT)
    common = load_config(COMMON)
    assert common["source"]["prior_type"] == "image_simplex_mixture"
    assert common["source"]["var_weight"] == 0.0
    assert common["model"] == parent["model"]
    assert common["training"] == parent["training"]
    assert common["loss"] == parent["loss"]
    assert common["dataset"] == parent["dataset"]
    assert common["augmentation"] == parent["augmentation"]
    assert common["checkpoint"] == parent["checkpoint"]
    parent_source = deepcopy(parent["source"])
    common_source = deepcopy(common["source"])
    parent_source["prior_type"] = "image_simplex_mixture"
    parent_source["var_weight"] = 0.0
    parent_source["simplex_prior"] = common_source["simplex_prior"]
    assert common_source == parent_source


class _TinyEndpoint(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.image_projection = nn.Conv2d(3, classes, 1)
        self.state_projection = nn.Conv2d(classes, classes, 1)

    def encode_image(self, image):
        return F.avg_pool2d(self.image_projection(image), 4)

    def forward_logits_with_image_feat(self, state, image_feat, s, t):
        return self.state_projection(state) + image_feat + (
            s + t
        )[:, None, None, None]

    def forward_logits(self, state, image, s, t):
        return self.forward_logits_with_image_feat(
            state, self.encode_image(image), s, t
        )


class _TrainableStatisticsSource(nn.Module):
    fixed_std = 1.0

    def __init__(self, classes: int, state_size: tuple[int, int]):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(1, classes, *state_size))

    def forward_statistics(self, image):
        mu = self.logits.expand(image.shape[0], -1, -1, -1)
        return mu, torch.zeros_like(mu)


def _tiny_joint_config(path: Path) -> dict:
    config = load_config(path)
    config["runtime"]["amp"] = False
    config["dataset"].update({
        "num_classes": 4, "void_class_index": 3,
    })
    config["model"]["num_classes"] = 4
    config["loss"]["ignore_index"] = 3
    config["evaluation"]["ignore_index"] = 3
    config["flow"]["path"] = {"type": "linear"}
    return config


@pytest.mark.parametrize("path", [IGNORE, INCLUDE])
def test_joint_simplex_source_gets_ce_and_dfm_gradients(path):
    config = _tiny_joint_config(path)
    endpoint = _TinyEndpoint(4)
    source = _TrainableStatisticsSource(4, (2, 3))
    adapter = DDPCompatibleTrainingModel(endpoint, source, config)
    optimizer = build_optimizer(config, adapter)
    assert [group["name"] for group in optimizer.param_groups] == ["model", "source"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-4)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(5.0e-5)

    image = torch.randn(1, 3, 8, 12)
    target = torch.tensor([[
        [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1],
        [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1],
        [1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2],
        [1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2],
        [2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
        [2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
        [3, 3, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0],
        [3, 3, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0],
    ]])
    spatial = torch.ones_like(target, dtype=torch.bool)
    spatial[:, :, -2:] = False
    torch.manual_seed(7)
    result = adapter(
        operation="joint_objectives",
        image=image,
        target=target,
        spatial_valid_mask=spatial,
        epoch_index=0,
        progress_in_epoch=0.0,
        optimizer_step=0,
    )
    assert result["stats"]["loss_source_ce"] > 0
    assert torch.isfinite(result["loss"])

    ce_gradient = torch.autograd.grad(
        result["source_objective"], source.parameters(), retain_graph=True
    )[0]
    dfm_gradient = torch.autograd.grad(
        result["diagonal_objective"] + result["psd_objective"],
        source.parameters(),
        retain_graph=True,
    )[0]
    result["loss"].backward()
    assert source.logits.grad is not None
    for gradient in (ce_gradient, dfm_gradient, source.logits.grad):
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in endpoint.parameters()
    )


@pytest.mark.parametrize("path", [IGNORE, INCLUDE])
def test_trainable_source_simplex_prior_preserves_invariant(path):
    config = _tiny_joint_config(path)
    image = torch.randn(1, 3, 8, 12)
    target = torch.randint(0, 4, (1, 8, 12))
    targets = prepare_state_targets(
        target,
        num_classes=4,
        state_size=(2, 3),
        ignore_index=3,
        mask_pixel_losses=True,
    )
    source = _TrainableStatisticsSource(4, (2, 3))
    x0, stats = sample_prior(
        config,
        image,
        targets.one_hot_state,
        source,
        target_full=target,
        valid_mask_full=targets.valid_mask_full,
        sampling_mode="training",
    )
    assert x0.requires_grad
    assert x0.amin() >= 0
    torch.testing.assert_close(
        x0.sum(dim=1), torch.ones_like(x0[:, 0]), atol=2.0e-6, rtol=0
    )
    assert stats["source_x0_sum_error"] < 2.0e-6
