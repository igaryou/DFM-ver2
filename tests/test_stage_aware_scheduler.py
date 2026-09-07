from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from checkpoint import _resume_best_metrics
from config import load_config
from stage_scheduler import StageAwareGroupLRScheduler
from trainer import build_scheduler
from training_stages import update_staged_best_metrics


ROOT = Path(__file__).parents[1]
REPRESENTATIVE = ROOT / (
    "configs/cityscapes/original/psd/"
    "joint_bounded_gaussian_b1_exponential_path_adaptive_std_trainable.yaml"
)


def _small_config(*, source_in_flow: bool = True) -> dict:
    config = load_config(REPRESENTATIVE)
    config["training"]["epochs"] = 5
    config["training"]["stages"] = {
        "enabled": True,
        "source_pretrain": {
            "enabled": True, "start_epoch": 0, "end_epoch": 2,
            "train_source": True, "train_flow": False,
        },
        "flow_training": {
            "enabled": True, "start_epoch": 2, "end_epoch": 5,
            "train_source": source_in_flow, "train_flow": True,
        },
    }
    config["training"]["scheduler"].update({
        "name": "cosine", "step_unit": "epoch", "stage_aware": True,
        "warmup_epochs": 2, "warmup_start_factor": 0.1,
        "eta_min": 5.0e-7,
    })
    return config


def _optimizer():
    model = nn.Linear(2, 2)
    source = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": 1.0e-4, "name": "model"},
        {"params": source.parameters(), "lr": 1.0e-5, "name": "source"},
    ])
    return optimizer


def _lrs(optimizer):
    return {
        group["name"]: float(group["lr"])
        for group in optimizer.param_groups
    }


def test_flow_clock_stops_in_source_stage_and_starts_local_zero():
    optimizer = _optimizer()
    scheduler = StageAwareGroupLRScheduler(optimizer, _small_config())
    initial_flow_lr = _lrs(optimizer)["model"]
    assert scheduler.progress == {"flow": 0, "source": 0}

    scheduler.step(train_flow=False, train_source=True)
    assert scheduler.progress == {"flow": 0, "source": 1}
    assert _lrs(optimizer)["model"] == pytest.approx(initial_flow_lr)

    scheduler.step(train_flow=False, train_source=True)
    assert scheduler.progress == {"flow": 0, "source": 2}
    assert _lrs(optimizer)["model"] == pytest.approx(initial_flow_lr)

    # Beginning of global epoch 2: flow local epoch is still exactly zero.
    assert scheduler.progress_for("flow") == 0
    scheduler.step(train_flow=True, train_source=True)
    assert scheduler.progress == {"flow": 1, "source": 3}
    assert _lrs(optimizer)["model"] > initial_flow_lr


def test_source_clock_advances_only_while_source_is_trainable():
    optimizer = _optimizer()
    scheduler = StageAwareGroupLRScheduler(
        optimizer, _small_config(source_in_flow=False)
    )
    scheduler.step(train_flow=False, train_source=True)
    scheduler.step(train_flow=False, train_source=True)
    source_lr = _lrs(optimizer)["source"]
    assert scheduler.progress_for("source") == 2

    for _ in range(3):
        scheduler.step(train_flow=True, train_source=False)
    assert scheduler.progress == {"flow": 3, "source": 2}
    assert _lrs(optimizer)["source"] == pytest.approx(source_lr)


def test_legacy_scheduler_trajectory_is_unchanged_when_disabled():
    config = load_config(
        ROOT / "configs/cityscapes/psd/swin_t_linear_160k.yaml"
    )
    assert config["training"]["scheduler"]["stage_aware"] is False
    first_optimizer = _optimizer()
    second_optimizer = _optimizer()
    first = build_scheduler(config, first_optimizer)
    second = build_scheduler(deepcopy(config), second_optimizer)
    assert not getattr(first, "stage_aware", False)
    trajectory_a = []
    trajectory_b = []
    for _ in range(5):
        trajectory_a.append(first.get_last_lr())
        trajectory_b.append(second.get_last_lr())
        first.step()
        second.step()
    assert trajectory_a == trajectory_b


@pytest.mark.parametrize("completed", [1, 4])
def test_scheduler_resume_matches_continuous_training(completed: int):
    config = _small_config()
    continuous_optimizer = _optimizer()
    continuous = StageAwareGroupLRScheduler(
        continuous_optimizer, config
    )
    for epoch in range(completed):
        continuous.step(
            train_flow=epoch >= 2,
            train_source=True,
        )
    state = continuous.state_dict()

    resumed_optimizer = _optimizer()
    resumed = StageAwareGroupLRScheduler(resumed_optimizer, config)
    resumed.load_state_dict(state)
    assert resumed.progress == continuous.progress
    assert _lrs(resumed_optimizer) == pytest.approx(
        _lrs(continuous_optimizer)
    )

    for epoch in range(completed, 5):
        kwargs = {"train_flow": epoch >= 2, "train_source": True}
        continuous.step(**kwargs)
        resumed.step(**kwargs)
        assert resumed.progress == continuous.progress
        assert _lrs(resumed_optimizer) == pytest.approx(
            _lrs(continuous_optimizer)
        )


def test_legacy_scheduler_state_migrates_from_completed_epoch():
    config = _small_config()
    scheduler = StageAwareGroupLRScheduler(_optimizer(), config)
    scheduler.set_progress_from_completed_epochs(4)
    assert scheduler.progress == {"flow": 2, "source": 4}


def test_source_and_flow_best_checkpoints_are_independent():
    state = SimpleNamespace(
        best_miou=float("-inf"),
        best_source_miou=float("-inf"),
        best_flow_miou=float("-inf"),
    )
    assert update_staged_best_metrics(
        state, {"source_mIoU": 0.5}, flow_enabled=False
    ) == ["best_source.pt"]
    assert update_staged_best_metrics(
        state, {"source_mIoU": 0.6}, flow_enabled=False
    ) == ["best_source.pt"]
    assert state.best_source_miou == pytest.approx(0.6)
    assert state.best_flow_miou == float("-inf")

    names = update_staged_best_metrics(
        state, {"source_mIoU": 0.75, "flow_mIoU": 0.5},
        flow_enabled=True,
    )
    assert names == ["best_source.pt", "best_flow.pt"]
    assert update_staged_best_metrics(
        state, {"source_mIoU": 0.7, "flow_mIoU": 0.6},
        flow_enabled=True,
    ) == ["best_flow.pt"]
    assert update_staged_best_metrics(
        state, {"source_mIoU": 0.7, "flow_mIoU": 0.55},
        flow_enabled=True,
    ) == []
    assert state.best_source_miou == pytest.approx(0.75)
    assert state.best_flow_miou == pytest.approx(0.6)
    assert state.best_miou == pytest.approx(0.6)


def test_old_checkpoint_best_miou_migration_is_stage_aware():
    legacy = {"metrics": {"best_mIoU": 0.7}}
    assert _resume_best_metrics(
        {"current_stage": "source_pretrain"}, legacy["metrics"]
    ) == (0.7, 0.7, float("-inf"))
    assert _resume_best_metrics(
        {"current_stage": "flow_training"}, legacy["metrics"]
    ) == (0.7, float("-inf"), 0.7)
    assert _resume_best_metrics({}, legacy["metrics"]) == (
        0.7, float("-inf"), 0.7
    )


def _ddp_scheduler_worker(rank: int, rendezvous: str) -> None:
    os.environ.update({
        "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "2",
        "GLOO_SOCKET_IFNAME": "lo",
    })
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        optimizer = _optimizer()
        scheduler = StageAwareGroupLRScheduler(
            optimizer, _small_config()
        )
        for epoch in range(5):
            scheduler.step(
                train_flow=epoch >= 2,
                train_source=True,
            )
            local = torch.tensor([
                scheduler.progress_for("flow"),
                scheduler.progress_for("source"),
                _lrs(optimizer)["model"],
                _lrs(optimizer)["source"],
            ], dtype=torch.float64)
            gathered = [torch.zeros_like(local) for _ in range(2)]
            dist.all_gather(gathered, local)
            assert torch.equal(gathered[0], gathered[1])
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_scheduler_progress_matches(tmp_path):
    rendezvous = tmp_path / "scheduler-rendezvous"
    mp.spawn(
        _ddp_scheduler_worker,
        args=(str(rendezvous),),
        nprocs=2,
        join=True,
    )
