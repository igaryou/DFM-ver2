from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import source_model as source_module
from adaptive_path import (
    entropy_scheduler_lambda,
    entropy_scheduler_lambda_derivative,
)
from config import load_config
from discrete_flow_maps import (
    path_coefficient,
    path_derivative,
    sample_prior,
    source_supervision_schedule,
)
from source_model import build_source_model
from state_space import prepare_state_targets
from trainer import _wandb_epoch_payload, build_optimizer


ROOT = Path(__file__).parents[1]
BASE_CONFIG = (
    ROOT / "configs/cityscapes/psd/joint_bounded_gaussian_b1_ce_160k.yaml"
)
CONFIG_DIR = ROOT / "configs/cityscapes/psd"
LINEAR_SCHEDULE = {
    "type": "linear",
    "initial": 3.0,
    "final": 0.2,
    "steps": 16000,
}


class _TinySource(nn.Module):
    fixed_std = 1.0

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([1.5, -0.5, 0.25, 0.75]))

    def forward_statistics(self, image: torch.Tensor):
        mean = self.logits[None, :, None, None].expand(
            image.shape[0], 4, image.shape[-2] // 4, image.shape[-1] // 4
        )
        return mean, torch.zeros_like(mean)

    def forward(self, image: torch.Tensor):
        mean, logvar = self.forward_statistics(image)
        return mean + torch.randn_like(mean), mean, logvar


def _factory_config(*, checkpoint: str | None, freeze: bool) -> dict:
    config = load_config(BASE_CONFIG)
    config["dataset"]["num_classes"] = 4
    config["model"]["num_classes"] = 4
    config["source"]["checkpoint"] = checkpoint
    config["source"]["freeze"] = freeze
    config["source"]["pretrained"] = False
    return config


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 3.0), (4000, 2.3), (8000, 1.6), (12000, 0.9),
     (16000, 0.2), (20000, 0.2)],
)
def test_linear_source_ce_schedule_uses_optimizer_step(step: int, expected: float):
    supervision = {"weight": 0.2, "weight_schedule": LINEAR_SCHEDULE}
    weight, progress = source_supervision_schedule(supervision, step)
    assert weight == pytest.approx(expected)
    assert progress == pytest.approx(min(step / 16000, 1.0))


def test_fixed_schedule_is_legacy_weight_and_microbatches_do_not_advance():
    legacy = {"type": "cross_entropy", "weight": 0.2}
    explicit = {
        "type": "cross_entropy",
        "weight": 0.2,
        "weight_schedule": {"type": "fixed"},
    }
    assert source_supervision_schedule(legacy, 8000) == (0.2, 0.0)
    assert source_supervision_schedule(explicit, 8000) == (0.2, 0.0)

    scheduled = {"weight": 0.2, "weight_schedule": LINEAR_SCHEDULE}
    # Every accumulated microbatch sees the same completed optimizer update count.
    repeated = [source_supervision_schedule(scheduled, 4000)[0] for _ in range(4)]
    assert repeated == pytest.approx([2.3] * 4)
    # Resume needs no private scheduler state: restored global_step is sufficient.
    assert source_supervision_schedule(scheduled, 12000)[0] == pytest.approx(0.9)


def test_linear_source_ce_schedule_can_be_set_entirely_by_cli():
    config = load_config(BASE_CONFIG, [
        "source.supervision.weight_schedule.type=linear",
        "source.supervision.weight_schedule.initial=3.0",
        "source.supervision.weight_schedule.final=0.2",
        "source.supervision.weight_schedule.steps=16000",
    ])
    assert config["source"]["supervision"]["weight_schedule"] == LINEAR_SCHEDULE


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("source.supervision.weight_schedule.type=unknown", "type must be fixed or linear"),
        ("source.supervision.weight_schedule.initial=-1", "initial must be non-negative"),
        ("source.supervision.weight_schedule.final=-1", "final must be non-negative"),
        ("source.supervision.weight_schedule.steps=0", "steps must be a positive integer"),
    ],
)
def test_source_ce_schedule_validation(override: str, match: str):
    overrides = [
        "source.supervision.weight_schedule.type=linear",
        "source.supervision.weight_schedule.initial=3.0",
        "source.supervision.weight_schedule.final=0.2",
        "source.supervision.weight_schedule.steps=16000",
        override,
    ]
    with pytest.raises(ValueError, match=match):
        load_config(BASE_CONFIG, overrides)


def test_checkpoint_and_freeze_are_independent(monkeypatch, tmp_path):
    monkeypatch.setattr(
        source_module, "SegFormerSourceGenerator", lambda *args, **kwargs: _TinySource()
    )
    trained = _TinySource()
    with torch.no_grad():
        trained.logits.copy_(torch.tensor([8.0, 7.0, 6.0, 5.0]))
    checkpoint = tmp_path / "source.pt"
    torch.save({"source_model": trained.state_dict()}, checkpoint)

    random_trainable = build_source_model(
        _factory_config(checkpoint=None, freeze=False)
    )
    assert random_trainable._source_checkpoint_loaded is False
    assert all(parameter.requires_grad for parameter in random_trainable.parameters())

    loaded_trainable = build_source_model(
        _factory_config(checkpoint=str(checkpoint), freeze=False)
    )
    torch.testing.assert_close(loaded_trainable.logits, trained.logits)
    assert loaded_trainable._source_checkpoint_loaded is True
    assert all(parameter.requires_grad for parameter in loaded_trainable.parameters())
    loaded_trainable.train()
    assert loaded_trainable.training
    trainable_optimizer = build_optimizer(
        _factory_config(checkpoint=str(checkpoint), freeze=False),
        SimpleNamespace(
            endpoint_model=nn.Linear(2, 2),
            source_model=loaded_trainable,
            consistency_weight_model=None,
        ),
    )
    assert "source" in [
        group["name"] for group in trainable_optimizer.param_groups
    ]

    loaded_frozen = build_source_model(
        _factory_config(checkpoint=str(checkpoint), freeze=True)
    )
    torch.testing.assert_close(loaded_frozen.logits, trained.logits)
    assert loaded_frozen._source_checkpoint_loaded is True
    assert not any(parameter.requires_grad for parameter in loaded_frozen.parameters())
    loaded_frozen.train()
    assert not loaded_frozen.training

    random_frozen = build_source_model(
        _factory_config(checkpoint=None, freeze=True)
    )
    assert random_frozen._source_checkpoint_loaded is False
    assert not any(parameter.requires_grad for parameter in random_frozen.parameters())
    random_frozen.train(True)
    assert not random_frozen.training


def test_frozen_source_has_no_optimizer_group(monkeypatch):
    monkeypatch.setattr(
        source_module, "SegFormerSourceGenerator", lambda *args, **kwargs: _TinySource()
    )
    config = _factory_config(checkpoint=None, freeze=True)
    source = build_source_model(config)
    adapter = SimpleNamespace(
        endpoint_model=nn.Linear(2, 2),
        source_model=source,
        consistency_weight_model=None,
    )
    optimizer = build_optimizer(config, adapter)
    assert [group["name"] for group in optimizer.param_groups] == ["model"]


def _sampling_config(*, freeze: bool) -> dict:
    config = _factory_config(checkpoint=None, freeze=freeze)
    config["source"]["supervision"] = {
        "type": "cross_entropy",
        "weight": 0.2,
        "include_void": True,
        "weight_schedule": deepcopy(LINEAR_SCHEDULE),
    }
    return config


def test_trainable_ce_schedule_uses_raw_logits_and_frozen_ce_is_zero():
    image = torch.zeros(1, 3, 8, 12)
    target = torch.tensor([[[0, 1, 2, 3] * 3] * 8])
    targets = prepare_state_targets(
        target,
        num_classes=4,
        state_size=(2, 3),
        ignore_index=3,
        mask_pixel_losses=True,
    )

    trainable_config = _sampling_config(freeze=False)
    trainable = _TinySource()
    _, stats = sample_prior(
        trainable_config,
        image,
        targets.one_hot_state,
        trainable,
        target_full=target,
        valid_mask_full=targets.valid_mask_full,
        optimizer_step=8000,
    )
    raw, _ = trainable.forward_statistics(image)
    expected_ce = F.cross_entropy(
        F.interpolate(raw, target.shape[-2:], mode="bilinear", align_corners=False),
        target,
    )
    torch.testing.assert_close(stats["source_ce_raw"], expected_ce)
    assert stats["source_ce_effective_weight"] == pytest.approx(1.6)
    torch.testing.assert_close(stats["source_ce_weighted"], 1.6 * expected_ce)
    torch.testing.assert_close(
        stats["weighted_source_supervision"], 1.6 * expected_ce
    )

    frozen_config = _sampling_config(freeze=True)
    frozen = _TinySource()
    frozen.requires_grad_(False)
    _, frozen_stats = sample_prior(
        frozen_config,
        image,
        targets.one_hot_state,
        frozen,
        target_full=target,
        valid_mask_full=targets.valid_mask_full,
        optimizer_step=8000,
    )
    assert frozen_stats["source_frozen"] == 1
    assert frozen_stats["source_ce_raw"] == 0
    assert frozen_stats["source_ce_effective_weight"] == 0
    assert frozen_stats["source_ce_weighted"] == 0
    assert frozen_stats["weighted_source_supervision"] == 0


def test_exponential_entropy_scheduler_production_formula_and_derivative():
    time = torch.tensor([0.37])
    difficulty = torch.tensor([[[-0.8, 0.0, 0.7]]])
    config = {
        "type": "entropy_adaptive",
        "scheduler": {
            "type": "exponential",
            "beta": 2.0,
            "difficulty_gamma": 1.0,
        },
    }
    exponent = torch.exp(2.0 * difficulty)
    expected = time[:, None, None].pow(exponent)
    torch.testing.assert_close(path_coefficient(time, config, difficulty), expected)
    torch.testing.assert_close(
        entropy_scheduler_lambda(
            time,
            difficulty,
            beta=2.0,
            scheduler_type="exponential",
            difficulty_gamma=1.0,
        ),
        expected,
    )
    expected_derivative = exponent * time[:, None, None].pow(exponent - 1.0)
    torch.testing.assert_close(
        path_derivative(time, config, difficulty), expected_derivative
    )
    torch.testing.assert_close(
        entropy_scheduler_lambda_derivative(
            time,
            difficulty,
            beta=2.0,
            scheduler_type="exponential",
            difficulty_gamma=1.0,
        ),
        expected_derivative,
    )


@pytest.mark.parametrize(
    ("filename", "freeze", "checkpoint", "variance_type", "scheduled"),
    [
        (
            "joint_bounded_gaussian_b1_ce_160k_exponential_path_frozen_source.yaml",
            True, True, "fixed", False,
        ),
        (
            "joint_bounded_gaussian_b1_ce_160k_exponential_path_trainable_source.yaml",
            False, True, "fixed", True,
        ),
        (
            "joint_bounded_gaussian_b1_ce_160k_exponential_path_adaptive_std_frozen_source.yaml",
            True, True, "entropy_adaptive", False,
        ),
        (
            "joint_bounded_gaussian_b1_ce_160k_exponential_path_adaptive_std_trainable_source.yaml",
            False, False, "entropy_adaptive", True,
        ),
    ],
)
def test_experiment_configs_resolve(
    filename: str,
    freeze: bool,
    checkpoint: bool,
    variance_type: str,
    scheduled: bool,
):
    config = load_config(CONFIG_DIR / filename)
    source = config["source"]
    assert source["checkpoint"] is not None if checkpoint else source["checkpoint"] is None
    assert source["freeze"] is freeze
    assert source["fixed_std"] == 1.0
    assert source["bounded_gaussian"]["amplitude"] == 0.5
    assert source["bounded_gaussian"]["temperature"] == 4.0
    assert source["bounded_gaussian"]["variance"]["type"] == variance_type
    scheduler = config["flow"]["path"]["scheduler"]
    assert scheduler == {
        "type": "exponential", "beta": 2.0, "difficulty_gamma": 1.0
    }
    weight_schedule = source["supervision"].get(
        "weight_schedule", {"type": "fixed"}
    )
    assert weight_schedule["type"] == ("linear" if scheduled else "fixed")


def test_wandb_payload_contains_source_schedule_diagnostics():
    report = {
        "loss_avg": 1.0,
        "source_frozen": 0.0,
        "source_checkpoint_loaded": 1.0,
        "source_ce_raw": 2.0,
        "source_ce_effective_weight": 1.6,
        "source_ce_weighted": 3.2,
        "source_ce_schedule_progress": 0.5,
        "optimizer_step": 8000,
    }
    payload = _wandb_epoch_payload(report, "psd")
    for key in (
        "source_frozen",
        "source_checkpoint_loaded",
        "source_ce_raw",
        "source_ce_effective_weight",
        "source_ce_weighted",
        "source_ce_schedule_progress",
        "optimizer_step",
    ):
        assert payload[f"epoch/{key}"] == report[key]
