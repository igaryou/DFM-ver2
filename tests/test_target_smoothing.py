from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import losses
import training_objectives
from config import load_config, validate_config
from state_space import smooth_categorical_target, target_state_from_config
from training_objectives import DDPCompatibleTrainingModel, compute_model_training_objectives


ROOT = Path(__file__).parents[1]


def _one_hot(classes=20):
    target = torch.tensor([[[0, 7], [19, 3]]])
    return torch.nn.functional.one_hot(target, classes).permute(0, 3, 1, 2).float(), target


def test_p_zero_is_exact_backward_compatible_identity():
    one_hot, _ = _one_hot()
    result = smooth_categorical_target(one_hot, 0.0)
    assert result is one_hot
    assert torch.equal(result, one_hot)


def test_open_simplex_values_sum_argmax_and_margin_for_cityscapes_p08():
    one_hot, target = _one_hot()
    result = smooth_categorical_target(one_hot, 0.8)
    assert bool((result > 0).all())
    torch.testing.assert_close(result.sum(1), torch.ones_like(result[:, 0]))
    assert torch.equal(result.argmax(1), target)
    gt = result.gather(1, target[:, None]).squeeze(1)
    other = result.clone(); other.scatter_(1, target[:, None], -torch.inf)
    torch.testing.assert_close(gt, torch.full_like(gt, 0.24))
    torch.testing.assert_close(other.amax(1), torch.full_like(gt, 0.04))
    torch.testing.assert_close(gt - other.amax(1), torch.full_like(gt, 0.2))


@pytest.mark.parametrize("p", [-0.01, 1.0, 1.5])
def test_invalid_smoothing_is_rejected(p):
    with pytest.raises(ValueError, match="0 <= p < 1"):
        smooth_categorical_target(_one_hot()[0], p)


def test_dtype_and_device_are_preserved():
    one_hot, _ = _one_hot()
    one_hot = one_hot.to(dtype=torch.bfloat16)
    result = smooth_categorical_target(one_hot, 0.8)
    assert result.dtype == one_hot.dtype
    assert result.device == one_hot.device


def test_config_default_override_and_validation():
    path = ROOT / "configs/cityscapes/psd/joint_simplex_b1_ce_160k.yaml"
    default = load_config(path)
    assert default["flow"]["target_smoothing"] == {"enabled": False, "p": 0.0}
    overridden = load_config(path, [
        "flow.target_smoothing.enabled=true", "flow.target_smoothing.p=0.8"
    ])
    assert overridden["flow"]["target_smoothing"] == {"enabled": True, "p": 0.8}
    for enabled, p in ((True, -0.1), (True, 1.0), (False, 0.2)):
        invalid = deepcopy(default)
        invalid["flow"]["target_smoothing"] = {"enabled": enabled, "p": p}
        with pytest.raises(ValueError, match="target_smoothing"):
            validate_config(invalid)


@pytest.mark.parametrize("suffix,ignore_void", [
    ("", True), ("_ignore_void", True), ("_include_void", False),
])
def test_smoothed_target_yaml_contract(suffix, ignore_void):
    path = ROOT / f"configs/cityscapes/psd/joint_simplex_b1_ce_smoothed_target_p08{suffix}_160k.yaml"
    config = load_config(path)
    assert config["source"]["prior_type"] == "image_simplex_mixture"
    assert config["flow"]["target_smoothing"] == {"enabled": True, "p": 0.8}
    assert config["loss"]["consistency"]["psd"]["ignore_void"] is ignore_void
    assert config["training"]["max_optimizer_steps"] == 160000


class _TinyEndpoint(nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        self.projection = nn.Conv2d(classes, classes, 1)
        self.encoder = nn.Conv2d(3, classes, 1)

    def encode_image(self, image):
        return self.encoder(image)

    def forward_logits_with_image_feat(self, state, image_feat, s, t):
        return self.projection(state) + image_feat


def _training_config(p):
    return {
        "experiment": {"stage": "joint_training"},
        "runtime": {"amp": False, "amp_dtype": "bf16"},
        "dataset": {"num_classes": 4, "void_class_index": 3},
        "model": {"state_downsample_factor": 1},
        "source": {"prior_type": "gaussian", "prior_noise_std": 1.0,
                   "var_weight": 0.0, "align_weight": 0.0, "use_loss_align": False},
        "flow": {"time_eps": 1e-5, "probability_eps": 1e-8,
                 "target_smoothing": {"enabled": p > 0, "p": p}},
        "time_sampling": {"min_time": 0.0, "max_time": 1.0, "min_gap": 1e-4},
        "training": {"label_smoothing": 0.0},
        "loss": {"ignore_index": None, "mask_pixel_losses": False,
                 "primary": {"weight": 1.0},
                 "consistency": {"enabled": True, "type": "psd", "weight": 0.0,
                    "start_epoch": 0, "warmup_epochs": 0, "max_weight": 1.0,
                    "precision": {"jvp_dtype": None, "numerical_dtype": "fp32",
                                  "debug_assertions": False},
                    "psd": {"loss_resolution": "state", "ignore_void": True}}},
    }


def test_diagonal_and_psd_states_share_smoothed_x1_and_ce_stays_hard(monkeypatch):
    config = _training_config(0.8)
    endpoint = _TinyEndpoint()
    adapter = DDPCompatibleTrainingModel(endpoint, None, config)
    image = torch.randn(1, 3, 3, 4)
    target = torch.randint(0, 4, (1, 3, 4))
    captured_x1 = []
    captured_ce_target = []
    original_path = training_objectives.linear_path
    original_ce = losses.diagonal_cross_entropy

    def capture_path(x0, x1, *args, **kwargs):
        captured_x1.append(x1.detach().clone())
        return original_path(x0, x1, *args, **kwargs)

    def capture_ce(logits, integer_target, *args, **kwargs):
        captured_ce_target.append(integer_target.detach().clone())
        return original_ce(logits, integer_target, *args, **kwargs)

    monkeypatch.setattr(training_objectives, "linear_path", capture_path)
    monkeypatch.setattr(losses, "diagonal_cross_entropy", capture_ce)
    result = compute_model_training_objectives(
        adapter, operation="joint_objectives", image=image, target=target,
        epoch_index=0, progress_in_epoch=0.0,
    )
    hard = torch.nn.functional.one_hot(target, 4).permute(0, 3, 1, 2).float()
    expected = smooth_categorical_target(hard, 0.8)
    assert len(captured_x1) == 2  # PSD consistency state and independent diagonal state.
    for actual in captured_x1:
        assert torch.equal(actual, expected)
    assert captured_ce_target[0].dtype == torch.long
    assert torch.equal(captured_ce_target[0], target)
    assert float(result["stats"]["target_smoothing_p"]) == pytest.approx(0.8)
    assert float(result["stats"]["x1_gt_margin"]) == pytest.approx(0.2)


def test_p_zero_training_target_resolution_matches_hard_one_hot():
    one_hot, _ = _one_hot()
    config = {"flow": {"target_smoothing": {"enabled": False, "p": 0.0}}}
    assert target_state_from_config(one_hot, config) is one_hot


def test_p_zero_training_objective_matches_legacy_config_exactly():
    legacy_config = _training_config(0.0)
    legacy_config["flow"].pop("target_smoothing")
    explicit_config = _training_config(0.0)
    legacy_endpoint = _TinyEndpoint()
    explicit_endpoint = _TinyEndpoint()
    explicit_endpoint.load_state_dict(legacy_endpoint.state_dict())
    image = torch.randn(1, 3, 3, 4)
    target = torch.randint(0, 4, (1, 3, 4))
    torch.manual_seed(123)
    legacy = compute_model_training_objectives(
        DDPCompatibleTrainingModel(legacy_endpoint, None, legacy_config),
        operation="joint_objectives", image=image, target=target,
        epoch_index=0, progress_in_epoch=0.0,
    )
    torch.manual_seed(123)
    explicit = compute_model_training_objectives(
        DDPCompatibleTrainingModel(explicit_endpoint, None, explicit_config),
        operation="joint_objectives", image=image, target=target,
        epoch_index=0, progress_in_epoch=0.0,
    )
    assert torch.equal(legacy["loss"], explicit["loss"])
    assert torch.equal(legacy["stats"]["loss_diagonal"], explicit["stats"]["loss_diagonal"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_smoothed_training_objective_smoke():
    config = _training_config(0.8)
    endpoint = _TinyEndpoint().cuda()
    adapter = DDPCompatibleTrainingModel(endpoint, None, config)
    image = torch.randn(1, 3, 8, 12, device="cuda")
    target = torch.randint(0, 4, (1, 8, 12), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = compute_model_training_objectives(
            adapter, operation="joint_objectives", image=image, target=target,
            epoch_index=0, progress_in_epoch=0.0,
        )
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert endpoint.projection.weight.grad is not None
    assert float(result["stats"]["x1_state_sum_error"]) < 1.0e-5
