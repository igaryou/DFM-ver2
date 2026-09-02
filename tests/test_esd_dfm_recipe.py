from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import load_config, validate_config
from dfm_stabilization import (
    ESDTimeWeightNetwork,
    uncertainty_weighted_esd_loss,
)
from losses import compute_consistency_loss, esd_loss, masked_mean


ROOT = Path(__file__).parents[1]
FULL_CONFIG = ROOT / "configs" / "cityscapes" / "esd" / "swin_t_dfm_recipe_resume.yaml"
CORE_CONFIG = ROOT / "configs" / "cityscapes" / "esd" / "swin_t_core_resume.yaml"


class AnalyticalModel(nn.Module):
    def __init__(self, classes: int = 4):
        super().__init__()
        self.x_projection = nn.Conv2d(classes, classes, 1, bias=False)
        self.image_projection = nn.Conv2d(3, classes, 1, bias=False)
        self.time_scale = nn.Parameter(torch.linspace(-0.2, 0.2, classes))

    def forward_logits(self, x, image, s, t):
        return (
            self.x_projection(x)
            + self.image_projection(image)
            + (s + 0.5 * t)[:, None, None, None]
            * self.time_scale[None, :, None, None]
        )


class InvalidModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(4))
        self.register_buffer("slopes", torch.tensor([-200.0, -50.0, 50.0, 200.0]))

    def forward_logits(self, x, image, s, t):
        differentiable_zero = s - s.detach()
        values = self.bias[None] + differentiable_zero[:, None] * self.slopes[None]
        return values[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])


def _inputs():
    torch.manual_seed(17)
    x = torch.softmax(torch.randn(2, 4, 2, 3), dim=1)
    image = torch.randn(2, 3, 2, 3)
    s = torch.tensor([0.1, 0.25])
    t = torch.tensor([0.4, 0.7])
    return x, image, s, t


def _esd_config():
    return {
        "flow": {"time_eps": 1.0e-5},
        "loss": {"consistency": {
            "type": "esd",
            "precision": {
                "jvp_dtype": "fp32", "numerical_dtype": "fp32",
                "debug_assertions": True,
            },
            "adaptive_kl": {
                "enabled": True, "c": 1.0e-6, "r": 0.5,
                "normalize_mean": False, "max_weight": None,
            },
            "invalid_teacher": {
                "strategy": "mask_pixel", "log_eps": 1.0e-6,
                "skip_batch_threshold": None,
            },
        }},
    }


def test_stable_teacher_and_jvp_match_explicit_formulas_and_stop_gradient():
    model = AnalyticalModel()
    x, image, s, t = _inputs()
    result = esd_loss(model, x, image, s, t, jvp_dtype="fp32")

    with torch.no_grad():
        logits_ss = model.forward_logits(x, image, s, s)
        probability_ss = logits_ss.softmax(1)
        drift = (probability_ss - x) / (1 - s)[:, None, None, None]
    expected_directional = (
        model.x_projection(drift)
        + model.time_scale[None, :, None, None]
    )
    torch.testing.assert_close(result.directional_logits, expected_directional)

    logits_st = model.forward_logits(x, image, s, t)
    q = logits_st.softmax(1)
    delta = expected_directional - (
        q * expected_directional
    ).sum(1, keepdim=True)
    log_arg = (
        (1 - t)[:, None, None, None]
        - ((1 - s) * (t - s))[:, None, None, None] * delta
    )
    expected_teacher = torch.softmax(
        logits_ss - torch.log(log_arg.clamp_min(1.0e-6)), dim=1
    )
    torch.testing.assert_close(result.teacher_prob, expected_teacher)
    assert not result.teacher_prob.requires_grad

def test_adaptive_kl_exact_equations_and_sample_wise_valid_reduction():
    model = AnalyticalModel()
    x, image, s, t = _inputs()
    valid = torch.tensor([
        [[True, True, False], [True, False, True]],
        [[False, True, True], [True, True, True]],
    ])
    result = compute_consistency_loss(
        "esd", model=model, x_s=x, image=image, s=s, t=t,
        precision=_esd_config()["loss"]["consistency"]["precision"],
        config=_esd_config(), valid_mask=valid,
    )
    kl = F.kl_div(
        result.student_prob.clamp_min(1e-12).log(),
        result.teacher_prob, reduction="none",
    ).sum(1)
    mismatch = (result.student_prob - result.teacher_prob).square().sum(1)
    weight = (mismatch + 1.0e-6).pow(-0.5).detach()
    expected_map = weight * kl
    expected_per_sample = torch.stack([
        expected_map[index][result.valid_pixel[index]].mean()
        for index in range(2)
    ])
    torch.testing.assert_close(result.adaptive_weight, weight)
    torch.testing.assert_close(result.loss, masked_mean(expected_map, result.valid_pixel))
    torch.testing.assert_close(result.loss_per_sample, expected_per_sample)
    torch.testing.assert_close(result.stats["loss_esd_raw_kl"], masked_mean(kl, result.valid_pixel))
    assert not result.adaptive_weight.requires_grad


def test_void_and_numerical_invalid_pixels_control_sample_validity():
    x, image, s, _ = _inputs()
    t = torch.full_like(s, 0.9)
    semantic = torch.ones(2, 2, 3, dtype=torch.bool)
    semantic[0] = False
    result = compute_consistency_loss(
        "esd", model=InvalidModel(), x_s=x, image=image, s=s, t=t,
        precision=_esd_config()["loss"]["consistency"]["precision"],
        config=_esd_config(), valid_mask=semantic,
    )
    assert not result.valid_sample.any()
    assert torch.equal(result.loss_per_sample, torch.zeros_like(result.loss_per_sample))
    assert result.loss == 0


def test_esd_time_weight_network_representation_and_initialization():
    network = ESDTimeWeightNetwork(32, 64, 0.25)
    s = torch.tensor([0.1, 0.1, 0.7])
    t = torch.tensor([0.2, 0.8, 0.8])
    representation = torch.cat((network.embedding(s), network.embedding(t)), dim=1)
    assert not torch.equal(representation[0], representation[1])
    assert not torch.equal(representation[1], representation[2])
    torch.testing.assert_close(
        torch.exp(-network(s, t)), torch.full_like(s, 0.25)
    )


def test_esd_uncertainty_objective_matches_explicit_formula():
    losses = torch.tensor([1.0, 3.0, 100.0], requires_grad=True)
    valid = torch.tensor([True, True, False])
    weight_logit = torch.tensor([0.2, -0.3, 1.0], requires_grad=True)
    objective, stats = uncertainty_weighted_esd_loss(
        losses, valid, weight_logit
    )
    expected = (
        torch.exp(-weight_logit[:2]) * losses[:2] + weight_logit[:2]
    ).mean()
    torch.testing.assert_close(objective, expected)
    objective.backward()
    assert losses.grad[2] == 0 and weight_logit.grad[2] == 0
    torch.testing.assert_close(stats["loss_esd_learnable"], expected.detach())


def test_esd_recipe_config_and_validation_matrix():
    full = load_config(FULL_CONFIG)
    core = load_config(CORE_CONFIG)
    consistency = full["loss"]["consistency"]
    assert consistency["type"] == "esd"
    assert consistency["weight"] == 1.0
    assert consistency["gradient_surgery"]["enabled"] is False
    assert consistency["adaptive_kl"] == {
        "enabled": True, "c": 1.0e-6, "r": 0.5,
        "normalize_mean": False, "max_weight": None,
    }
    assert consistency["learnable_weight"]["dependency"] == "st"
    assert core["loss"]["consistency"]["learnable_weight"]["enabled"] is False
    normalized_core = deepcopy(core)
    normalized_core["experiment"] = deepcopy(full["experiment"])
    normalized_core["wandb"] = deepcopy(full["wandb"])
    normalized_core["runtime"]["config_path"] = full["runtime"]["config_path"]
    normalized_core["loss"]["consistency"]["learnable_weight"]["enabled"] = True
    assert normalized_core == full

    bad = deepcopy(full)
    bad["loss"]["consistency"]["gradient_surgery"]["enabled"] = True
    with pytest.raises(ValueError, match="only for PSD"):
        validate_config(bad)
    bad = deepcopy(full)
    bad["loss"]["consistency"]["learnable_weight"]["dependency"] = "s"
    with pytest.raises(ValueError, match="dependency=st"):
        validate_config(bad)
    psd = deepcopy(full)
    psd["loss"]["consistency"]["type"] = "psd"
    psd["loss"]["consistency"]["precision"]["jvp_dtype"] = None
    with pytest.raises(ValueError, match="dependency=s"):
        validate_config(psd)
    for loss_type in ("csd", "ecld"):
        bad = deepcopy(full)
        bad["loss"]["consistency"]["type"] = loss_type
        bad["loss"]["consistency"]["learnable_weight"]["dependency"] = "st"
        with pytest.raises(ValueError, match="only for PSD and ESD"):
            validate_config(bad)
