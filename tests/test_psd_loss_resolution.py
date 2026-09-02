from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import load_config
from dfm_stabilization import (
    apply_global_gradient_surgery,
    uncertainty_weighted_psd_loss,
)
from losses import compute_consistency_loss, masked_mean
from state_space import resize_continuous


ROOT = Path(__file__).parents[1]
BASE_CONFIG = ROOT / "configs" / "cityscapes" / "psd" / "swin_t_dfm_recipe.yaml"
FULL_CONFIG = (
    ROOT / "configs" / "cityscapes" / "psd"
    / "swin_t_adaptive_surgery_fullres.yaml"
)


class TinyFlowModel(nn.Module):
    def __init__(self, classes: int = 3):
        super().__init__()
        self.x_projection = nn.Conv2d(classes, classes, 1)
        self.image_projection = nn.Conv2d(3, classes, 1)
        self.time_scale = nn.Parameter(torch.linspace(-0.2, 0.2, classes))

    def forward_logits(self, x, image, s, t):
        time = (0.3 * s + 0.7 * t)[:, None, None, None]
        return (
            self.x_projection(x)
            + self.image_projection(image)
            + time * self.time_scale[None, :, None, None]
        )


def _config(loss_type="psd", resolution=None):
    config = {
        "flow": {"time_eps": 1.0e-5, "probability_eps": 1.0e-8},
        "loss": {
            "consistency": {
                "type": loss_type,
                "precision": {
                    "jvp_dtype": None if loss_type == "psd" else "fp32",
                    "numerical_dtype": "fp32",
                    "debug_assertions": True,
                },
                "psd": {},
                "ecld": {"ec_weight": 4.0, "td_weight": 2.0, "time_weighting": "none"},
                "adaptive_kl": {"enabled": False},
                "invalid_teacher": {"strategy": "mask_pixel"},
            }
        },
    }
    if resolution is not None:
        config["loss"]["consistency"]["psd"]["loss_resolution"] = resolution
    return config


def _inputs(batch=2, classes=3):
    torch.manual_seed(23)
    x = torch.softmax(torch.randn(batch, classes, 2, 4), dim=1)
    image = torch.randn(batch, 3, 2, 4)
    s = torch.tensor([0.1, 0.2])[:batch]
    u = torch.tensor([0.3, 0.4])[:batch]
    t = torch.tensor([0.6, 0.8])[:batch]
    return x, image, s, u, t


def _compute(model, config, *, valid_state=None, valid_full=None):
    x, image, s, u, t = _inputs()
    return compute_consistency_loss(
        config["loss"]["consistency"]["type"],
        model=model, x_s=x, image=image, s=s, u=u, t=t,
        precision=config["loss"]["consistency"]["precision"],
        config=config, valid_mask=valid_state,
        full_resolution_size=(8, 16), valid_mask_full=valid_full,
    )


def test_default_config_uses_state_psd_resolution():
    config = load_config(
        ROOT / "configs" / "_base_" / "cityscapes" / "joint_psd_160k.yaml"
    )
    assert config["loss"]["consistency"]["psd"]["loss_resolution"] == "state"


def test_explicit_state_mode_is_numerically_identical_to_legacy_default():
    legacy_model = TinyFlowModel()
    state_model = TinyFlowModel()
    state_model.load_state_dict(legacy_model.state_dict())
    valid = torch.tensor([
        [[True, True, False, True], [True, False, True, True]],
        [[True, True, True, True], [True, True, True, False]],
    ])
    legacy = _compute(legacy_model, _config(), valid_state=valid)
    state = _compute(state_model, _config(resolution="state"), valid_state=valid)
    torch.testing.assert_close(state.loss, legacy.loss, rtol=0, atol=0)
    torch.testing.assert_close(state.loss_per_sample, legacy.loss_per_sample, rtol=0, atol=0)
    assert state.stats["psd_loss_height"] == 2
    assert state.stats["psd_loss_width"] == 4
    assert state.stats["psd_loss_resolution_is_full"] == 0


def test_full_resolution_matches_explicit_formula_and_sample_means():
    model = TinyFlowModel()
    valid_state = torch.ones(2, 2, 4, dtype=torch.bool)
    valid_full = torch.ones(2, 8, 16, dtype=torch.bool)
    valid_full[0, :3, :5] = False
    valid_full[1, :, :8] = False
    config = _config(resolution="full")
    result = _compute(
        model, config, valid_state=valid_state, valid_full=valid_full
    )
    x, image, s, _, t = _inputs()
    student_logits_state = model.forward_logits(x, image, s, t).float()
    student_logits_full = resize_continuous(student_logits_state, (8, 16))
    teacher_full = resize_continuous(result.teacher_prob, (8, 16)).float().clamp_min(1e-8)
    teacher_full = teacher_full / teacher_full.sum(1, keepdim=True).clamp_min(1e-8)
    expected_map = -(teacher_full * F.log_softmax(student_logits_full, dim=1)).sum(1)
    expected_per_sample = torch.stack([
        expected_map[index][valid_full[index]].mean() for index in range(2)
    ])
    torch.testing.assert_close(result.loss, masked_mean(expected_map, valid_full))
    torch.testing.assert_close(result.loss_per_sample, expected_per_sample)
    assert result.teacher_prob.shape[-2:] == (2, 4)
    assert result.student_prob.shape[-2:] == (2, 4)
    assert result.valid_pixel.shape[-2:] == (8, 16)
    assert result.stats["psd_loss_height"] == 8
    assert result.stats["psd_loss_width"] == 16
    assert result.stats["psd_loss_resolution_is_full"] == 1
    result.loss.backward()
    assert model.x_projection.weight.grad is not None


def test_full_resolution_uses_full_mask_not_state_mask():
    model = TinyFlowModel()
    valid_state = torch.zeros(2, 2, 4, dtype=torch.bool)
    valid_full = torch.zeros(2, 8, 16, dtype=torch.bool)
    valid_full[:, 0, 0] = True
    result = _compute(
        model, _config(resolution="full"),
        valid_state=valid_state, valid_full=valid_full,
    )
    assert torch.equal(result.valid_sample, torch.tensor([True, True]))
    assert torch.isfinite(result.loss) and result.loss.item() > 0


def test_full_resolution_backward_is_compatible_with_gradient_surgery():
    model = TinyFlowModel()
    result = _compute(
        model, _config(resolution="full"),
        valid_state=torch.ones(2, 2, 4, dtype=torch.bool),
        valid_full=torch.ones(2, 8, 16, dtype=torch.bool),
    )
    diagonal = model.x_projection.weight.float().square().mean()
    zero = diagonal * 0.0

    class Adapter:
        endpoint_model = model
        source_model = None
        psd_weight_model = None

    stats = apply_global_gradient_surgery(
        adapter=Adapter(),
        objectives={
            "diagonal_objective": diagonal,
            "psd_objective": result.loss,
            "source_objective": zero,
            "loss": diagonal + result.loss,
        },
        scaler=torch.amp.GradScaler("cuda", enabled=False),
    )
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.isfinite(stats["gradient_surgery_cosine"])


def test_full_resolution_sample_losses_support_learnable_weighting():
    model = TinyFlowModel()
    result = _compute(
        model, _config(resolution="full"),
        valid_state=torch.ones(2, 2, 4, dtype=torch.bool),
        valid_full=torch.ones(2, 8, 16, dtype=torch.bool),
    )
    weight_logit = torch.zeros(2, requires_grad=True)
    weighted, stats = uncertainty_weighted_psd_loss(
        result.loss_per_sample, result.valid_sample, weight_logit
    )
    weighted.backward()
    assert weight_logit.grad is not None and torch.isfinite(weight_logit.grad).all()
    assert model.x_projection.weight.grad is not None
    assert torch.isfinite(stats["loss_psd_learnable"])


def test_invalid_psd_resolution_is_rejected():
    with pytest.raises(
        ValueError,
        match="loss.consistency.psd.loss_resolution must be 'state' or 'full'",
    ):
        load_config(
            ROOT / "configs" / "_base_" / "cityscapes" / "joint_psd_160k.yaml",
            ["loss.consistency.psd.loss_resolution=invalid"],
        )


def test_psd_resolution_setting_does_not_affect_non_psd_loss():
    state_config = _config(loss_type="csd", resolution="state")
    full_config = deepcopy(state_config)
    full_config["loss"]["consistency"]["psd"]["loss_resolution"] = "full"
    state_model = TinyFlowModel()
    full_model = TinyFlowModel()
    full_model.load_state_dict(state_model.state_dict())
    state = _compute(state_model, state_config)
    full = _compute(full_model, full_config)
    torch.testing.assert_close(full.loss, state.loss, rtol=0, atol=0)


def test_fullres_experiment_config_changes_only_identity_and_psd_resolution():
    base = load_config(BASE_CONFIG)
    full = load_config(FULL_CONFIG)
    assert full["loss"]["consistency"]["psd"]["loss_resolution"] == "full"
    assert full["experiment"]["name"] != base["experiment"]["name"]
    assert full["experiment"]["output_dir"] != base["experiment"]["output_dir"]
    normalized = deepcopy(full)
    normalized["experiment"] = deepcopy(base["experiment"])
    normalized["runtime"]["config_path"] = base["runtime"]["config_path"]
    normalized["loss"]["consistency"]["psd"]["loss_resolution"] = "state"
    assert normalized == base
