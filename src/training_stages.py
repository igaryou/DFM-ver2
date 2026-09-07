from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class TrainingStage:
    name: str
    start_epoch: int
    end_epoch: int
    train_source: bool
    train_flow: bool

    @property
    def source_only(self) -> bool:
        return self.train_source and not self.train_flow

    def local_epoch(self, epoch_index: int) -> int:
        return int(epoch_index) - self.start_epoch


def staged_training_enabled(config: dict) -> bool:
    return bool(config.get("training", {}).get("stages", {}).get("enabled", False))


def training_stage_for_epoch(config: dict, epoch_index: int) -> TrainingStage:
    """Resolve the active 0-based epoch interval without mutating config."""
    epoch_index = int(epoch_index)
    if epoch_index < 0:
        raise ValueError("epoch_index must be non-negative")
    stages = config["training"].get("stages", {})
    if not stages.get("enabled", False):
        experiment_stage = config["experiment"]["stage"]
        source_only = (
            experiment_stage == "diagonal_pretrain"
            and not config["training"].get("train_endpoint", True)
            and float(config["loss"]["primary"]["weight"]) == 0.0
        )
        return TrainingStage(
            name="source_pretrain" if source_only else "legacy",
            start_epoch=0,
            end_epoch=int(config["training"].get("epochs", epoch_index + 1)),
            train_source=not bool(config["source"].get("freeze", False)),
            train_flow=not source_only,
        )

    configured = []
    for name in ("source_pretrain", "flow_training"):
        stage = stages[name]
        if not stage["enabled"]:
            continue
        configured.append(TrainingStage(
            name=name,
            start_epoch=int(stage["start_epoch"]),
            end_epoch=int(stage["end_epoch"]),
            train_source=bool(stage["train_source"]),
            train_flow=bool(stage["train_flow"]),
        ))
    matching = [
        stage for stage in configured
        if stage.start_epoch <= epoch_index < stage.end_epoch
    ]
    if len(matching) != 1:
        raise ValueError(
            f"epoch {epoch_index} must belong to exactly one enabled training stage"
        )
    return matching[0]


def schedule_value(
    schedule: dict,
    *,
    epoch_index: int,
    optimizer_step: int,
    fallback: float,
) -> tuple[float, float]:
    """Resolve a fixed/linear scalar schedule in epoch or optimizer-step units."""
    schedule_type = schedule.get("type", "fixed")
    if schedule_type == "fixed":
        value = schedule.get("value", fallback)
        return float(fallback if value is None else value), 0.0
    if schedule_type != "linear":
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    unit = schedule.get("unit", "optimizer_step")
    current = int(epoch_index) if unit == "epoch" else int(optimizer_step)
    configured_start_epoch = schedule.get("start_epoch")
    start = int(
        configured_start_epoch
        if configured_start_epoch is not None
        else schedule.get("start", 0)
    )
    duration = schedule.get("duration")
    if duration is None:
        duration = schedule.get("steps")
    duration = int(duration)
    progress = min(max((current - start) / duration, 0.0), 1.0)
    initial = float(schedule["initial"])
    final = float(schedule["final"])
    return initial + (final - initial) * progress, progress


def remember_base_trainability(module: nn.Module | None) -> None:
    if module is None or hasattr(module, "_stage_base_requires_grad"):
        return
    module._stage_base_requires_grad = {
        name: parameter.requires_grad
        for name, parameter in module.named_parameters()
    }


def set_stage_trainability(
    endpoint: nn.Module,
    source: nn.Module | None,
    stage: TrainingStage,
) -> None:
    """Apply stage modes while preserving intentionally frozen submodules."""
    remember_base_trainability(endpoint)
    remember_base_trainability(source)
    for name, parameter in endpoint.named_parameters():
        parameter.requires_grad_(
            bool(endpoint._stage_base_requires_grad[name] and stage.train_flow)
        )
    endpoint.train(stage.train_flow)
    if source is None:
        if stage.train_source:
            raise RuntimeError("active stage requires a source model")
        return
    if stage.train_source and bool(getattr(source, "_source_frozen", False)):
        raise RuntimeError(
            "source.freeze=true cannot be dynamically unfrozen; configure freeze=false"
        )
    for name, parameter in source.named_parameters():
        parameter.requires_grad_(
            bool(source._stage_base_requires_grad[name] and stage.train_source)
        )
    source.train(stage.train_source)
    if not stage.train_source:
        source.eval()


def stage_operation(config: dict, stage: TrainingStage) -> str:
    if stage.source_only:
        return "source_pretrain_objectives"
    experiment_stage = config["experiment"]["stage"]
    if experiment_stage == "joint_training":
        return "joint_objectives"
    if experiment_stage in {"consistency_distillation", "esd_distillation"}:
        return "stage2_objectives"
    return "stage1_objectives"
