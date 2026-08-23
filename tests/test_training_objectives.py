from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import losses
from config import load_config, validate_config
from training_objectives import (
    DDPCompatibleTrainingModel,
    compute_model_training_objectives,
)


ROOT = Path(__file__).parents[1]


class TinyEndpoint(nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        self.projection = nn.Conv2d(classes, classes, 1)
        self.image_encoder = nn.Conv2d(3, classes, 1)
        self.time_scale = nn.Parameter(torch.zeros(classes))

    def encode_image(self, image):
        return self.image_encoder(image)

    def forward_logits_with_image_feat(self, x, image_feat, s, t):
        return self.projection(x) + image_feat + (
            s + t
        )[:, None, None, None] * self.time_scale[None, :, None, None]

    def forward_logits(self, x, image, s, t):
        return self.forward_logits_with_image_feat(
            x, self.encode_image(image), s, t
        )


def _config(loss_type: str, stage: str = "joint_training"):
    jvp_dtype = None if loss_type == "psd" else "fp32"
    return {
        "experiment": {"stage": stage},
        "runtime": {"amp": False, "amp_dtype": "bf16"},
        "dataset": {"num_classes": 4},
        "source": {
            "prior_type": "gaussian",
            "prior_noise_std": 1.0,
            "var_weight": 0.0,
            "align_weight": 0.0,
            "use_loss_align": False,
        },
        "flow": {"time_eps": 1.0e-5, "probability_eps": 1.0e-8},
        "time_sampling": {
            "min_time": 0.0,
            "max_time": 1.0,
            "min_gap": 1.0e-4,
        },
        "training": {"label_smoothing": 0.0},
        "loss": {
            "primary": {"weight": 1.0},
            "consistency": {
                "enabled": stage != "diagonal_pretrain",
                "type": loss_type,
                "weight": 0.1,
                "start_epoch": 0,
                "warmup_epochs": 0,
                "max_weight": 1.0,
                "precision": {
                    "jvp_dtype": jvp_dtype,
                    "numerical_dtype": "fp32",
                    "debug_assertions": True,
                },
                "ecld": {
                    "ec_weight": 4.0,
                    "td_weight": 2.0,
                    "time_weighting": "none",
                },
                "adaptive_kl": {
                    "enabled": False,
                    "c": 1.0e-6,
                    "r": 0.5,
                    "normalize_mean": True,
                    "max_weight": 100.0,
                },
                "invalid_teacher": {
                    "strategy": "mask_pixel",
                    "log_eps": 1.0e-6,
                    "skip_batch_threshold": None,
                },
            },
        },
    }


def _batch(classes=4):
    image = torch.randn(2, 3, 4, 5)
    target = torch.randint(0, classes, (2, 4, 5))
    return image, target


def test_stage1_composite_forward_never_calls_consistency(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Stage 1 must not call consistency/JVP")

    monkeypatch.setattr(losses, "compute_consistency_loss", forbidden)
    config = _config("esd", "diagonal_pretrain")
    adapter = DDPCompatibleTrainingModel(TinyEndpoint(), None, config)
    image, target = _batch()
    result = compute_model_training_objectives(
        adapter, operation="stage1_objectives", image=image, target=target,
        epoch_index=0, progress_in_epoch=0.0,
    )
    assert float(result["stats"]["loss_consistency"]) == 0.0
    result["loss"].backward()
    assert adapter.endpoint_model.projection.weight.grad is not None


@pytest.mark.parametrize("loss_type", ["psd", "csd", "ecld", "esd"])
def test_stage2_and_joint_select_each_consistency_loss(loss_type):
    for operation in ("stage2_objectives", "joint_objectives"):
        config = _config(loss_type)
        adapter = DDPCompatibleTrainingModel(TinyEndpoint(), None, config)
        image, target = _batch()
        result = compute_model_training_objectives(
            adapter, operation=operation, image=image, target=target,
            epoch_index=0, progress_in_epoch=0.5,
        )
        assert result["consistency_type"] == loss_type
        assert torch.isfinite(result["loss"])
        result["loss"].backward()
        assert adapter.endpoint_model.projection.weight.grad is not None


def test_joint_samples_diagonal_and_consistency_times_independently(monkeypatch):
    monkeypatch.setattr(
        "training_objectives.sample_stage1_times",
        lambda batch_size, device, *args: torch.full(
            (batch_size,), 0.8, device=device
        ),
    )
    monkeypatch.setattr(
        "training_objectives.sample_consistency_times",
        lambda loss_type, batch_size, device, *args: (
            torch.full((batch_size,), 0.1, device=device),
            None,
            torch.full((batch_size,), 0.4, device=device),
        ),
    )
    config = _config("csd")
    adapter = DDPCompatibleTrainingModel(TinyEndpoint(), None, config)
    image, target = _batch()
    result = compute_model_training_objectives(
        adapter, operation="joint_objectives", image=image, target=target,
        epoch_index=0, progress_in_epoch=0.0,
    )
    assert float(result["stats"]["diagonal_time_mean"]) == pytest.approx(0.8)
    assert float(result["stats"]["consistency_s_mean"]) == pytest.approx(0.1)


def test_joint_config_forbids_init_from_and_stage2_requires_checkpoint():
    config = load_config(ROOT / "configs" / "joint_ecld_cityscapes.yaml")
    bad_joint = deepcopy(config)
    bad_joint["checkpoint"]["init_from"] = "stage1.pt"
    with pytest.raises(ValueError, match="forbids checkpoint.init_from"):
        validate_config(bad_joint)

    bad_stage2 = load_config(ROOT / "configs" / "stage2_ecld_cityscapes.yaml")
    bad_stage2["checkpoint"]["init_from"] = None
    bad_stage2["checkpoint"]["resume"] = None
    with pytest.raises(ValueError, match="requires checkpoint"):
        validate_config(bad_stage2)
