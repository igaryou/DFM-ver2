from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel


SCHEDULER_STEP_UNIT = "epoch"
SCHEDULER_VERSION = 2
OPTIMIZER_STEP_SCHEDULER_VERSION = 3


@dataclass
class TrainingState:
    start_epoch: int = 0
    global_step: int = 0
    micro_step: int = 0
    best_miou: float = float("-inf")


def model_signature(config: dict) -> dict[str, Any]:
    return {
        "num_classes": config["dataset"]["num_classes"],
        "model": copy.deepcopy(config["model"]),
        "source": {
            key: copy.deepcopy(config["source"][key])
            for key in (
                "prior_type", "prior_noise_std", "backbone", "segformer_variant",
                "pretrained", "freeze_encoder", "decoder_channels",
                "learned_logvar", "fixed_std", "mu_tanh_scale",
                "supervision",
            )
        },
    }


def checkpoint_payload(
    *,
    config: dict,
    epoch: int,
    global_step: int,
    model,
    source_model,
    optimizer,
    scheduler,
    scaler,
    metrics: dict,
    distributed: dict | None = None,
    micro_step: int = 0,
) -> dict:
    raw_model = model
    while isinstance(raw_model, DistributedDataParallel):
        raw_model = raw_model.module
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    raw_source = source_model
    while isinstance(raw_source, DistributedDataParallel):
        raw_source = raw_source.module
    raw_source = getattr(raw_source, "_orig_mod", raw_source)
    return {
        "stage": config["experiment"]["stage"],
        "epoch": epoch,
        "global_step": global_step,
        "model": raw_model.state_dict(),
        "source_model": raw_source.state_dict() if raw_source is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "micro_step": micro_step,
        "scheduler_step_unit": config["training"]["scheduler"]["step_unit"],
        "scheduler_version": (
            SCHEDULER_VERSION
            if config["training"]["scheduler"]["step_unit"] == "epoch"
            else OPTIMIZER_STEP_SCHEDULER_VERSION
        ),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": copy.deepcopy(config),
        "model_signature": model_signature(config),
        "metrics": copy.deepcopy(metrics),
        "distributed": copy.deepcopy(distributed or {
            "world_size": 1,
            "global_batch_size": config["training"]["batch_size"],
            "local_batch_size": config["training"]["batch_size"],
        }),
    }


def save_checkpoint(payload: dict, output_dir: str | Path, filename: str) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / filename
    temporary = output / f".{filename}.tmp"
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def _validate_joint_stage1_boundary(
    checkpoint: dict, path: str | Path
) -> None:
    saved_config = checkpoint.get("config")
    try:
        consistency = saved_config["loss"]["consistency"]
        consistency_enabled = consistency["enabled"]
        saved_start_epoch = consistency["start_epoch"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "Joint Stage 2 initialization checkpoint is missing "
            f"config.loss.consistency metadata: {path}"
        ) from exc

    saved_epoch = checkpoint.get("epoch")
    if (
        isinstance(saved_epoch, bool)
        or not isinstance(saved_epoch, int)
        or isinstance(saved_start_epoch, bool)
        or not isinstance(saved_start_epoch, int)
    ):
        raise RuntimeError(
            "Joint Stage 2 initialization checkpoint must contain integer epoch "
            f"and consistency.start_epoch values: {path}"
        )
    if consistency_enabled is not True:
        raise RuntimeError(
            "Joint Stage 2 initialization checkpoint must have consistency loss "
            f"enabled in its saved config: {path}"
        )

    # The checkpoint epoch is the 1-indexed number of completed epochs, while
    # the schedule receives the loop's 0-indexed epoch. Thus start_epoch=N first
    # permits a consistency update in displayed epoch N+1; epoch N is safe.
    if saved_epoch > saved_start_epoch:
        raise RuntimeError(
            "Joint checkpoint may already contain consistency-loss updates and "
            "cannot initialize Stage 2: "
            f"completed epoch={saved_epoch}, last safe epoch={saved_start_epoch}, "
            f"path={path}"
        )


def _validate_stage2_init_checkpoint(
    checkpoint: dict, config: dict, path: str | Path
) -> None:
    saved_stage = checkpoint.get("stage")
    if saved_stage == "joint_training":
        _validate_joint_stage1_boundary(checkpoint, path)
    elif saved_stage != "diagonal_pretrain":
        raise RuntimeError(
            "Stage 2 init_from requires a diagonal_pretrain checkpoint or a "
            "joint_training checkpoint saved no later than its Stage 1 boundary: "
            f"stage={saved_stage!r}, path={path}"
        )

    if checkpoint.get("model") is None:
        raise RuntimeError(
            f"Stage 2 initialization checkpoint has no model state: {path}"
        )
    if "source_model" not in checkpoint:
        raise RuntimeError(
            f"Stage 2 initialization checkpoint has no source_model state: {path}"
        )

    saved_signature = checkpoint.get("model_signature")
    current_signature = model_signature(config)
    if saved_signature is None:
        saved_config = checkpoint.get("config", {})
        try:
            saved_signature = model_signature(saved_config)
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Stage 2 initialization checkpoint has no usable model signature: {path}"
            ) from exc
    if saved_signature != current_signature:
        raise RuntimeError(
            "Stage 2 initialization checkpoint is incompatible with the current "
            "model/source configuration.\n"
            f"saved={saved_signature}\ncurrent={current_signature}"
        )


def _without_module_prefix(state_dict: dict) -> dict:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def _resume_stage_compatible(checkpoint: dict, config: dict) -> bool:
    saved_stage = checkpoint.get("stage")
    current_stage = config["experiment"]["stage"]
    if saved_stage == current_stage:
        return True
    consistency_type = config["loss"]["consistency"]["type"]
    stage2_names = {"consistency_distillation", "esd_distillation"}
    return (
        saved_stage in stage2_names
        and current_stage in stage2_names
        and consistency_type == "esd"
    )


def _validate_resume_scheduler(
    checkpoint: dict, config: dict, path: str | Path
) -> None:
    step_unit = checkpoint.get("scheduler_step_unit")
    version = checkpoint.get("scheduler_version")
    expected_unit = config["training"]["scheduler"]["step_unit"]
    expected_version = (
        SCHEDULER_VERSION
        if expected_unit == "epoch" else OPTIMIZER_STEP_SCHEDULER_VERSION
    )
    if step_unit != expected_unit or version != expected_version:
        raise RuntimeError(
            "Resume checkpoint uses an incompatible scheduler format: "
            f"{path} has scheduler_step_unit={step_unit!r}, "
            f"scheduler_version={version!r}; expected "
            f"{expected_unit!r}, version {expected_version}. "
            "Legacy optimizer-step scheduler checkpoints cannot be resumed "
            "without matching scheduler metadata."
        )


def initialize_or_resume(
    config: dict,
    model,
    source_model,
    optimizer,
    scheduler,
    scaler,
    logger=None,
) -> TrainingState:
    checkpoint_config = config["checkpoint"]
    init_from = checkpoint_config["init_from"]
    resume = checkpoint_config["resume"]
    if init_from and resume:
        raise ValueError("checkpoint.init_from and checkpoint.resume are mutually exclusive")
    strict = checkpoint_config["strict_model"]
    if init_from:
        checkpoint = torch.load(init_from, map_location="cpu", weights_only=False)
        _validate_stage2_init_checkpoint(checkpoint, config, init_from)
        saved_source = checkpoint.get("source_model")
        if source_model is not None:
            if saved_source is None:
                raise RuntimeError(
                    "Stage 2 initialization checkpoint has no source_model state: "
                    f"{init_from}"
                )
        model.load_state_dict(
            _without_module_prefix(checkpoint["model"]), strict=strict
        )
        if source_model is not None:
            source_model.load_state_dict(
                _without_module_prefix(saved_source), strict=strict
            )
        lines = (
            f"Loaded Stage 2 initialization checkpoint: {init_from}",
            f"Checkpoint original stage: {checkpoint.get('stage')}",
            f"Checkpoint completed epoch: {checkpoint.get('epoch', 'unknown')}",
            "Loaded states: model, source_model",
            "Optimizer state: newly initialized",
            "Scheduler state: newly initialized",
            "Scaler state: newly initialized",
            "Stage 2 start epoch: 1",
            "Global step reset to: 0",
            "Best mIoU reset to: -inf",
            f"Consistency loss: {config['loss']['consistency']['type']}",
        )
        if logger is not None:
            for line in lines:
                logger.info(line)
        return TrainingState(
            start_epoch=0,
            global_step=0,
            best_miou=float("-inf"),
        )
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if not _resume_stage_compatible(checkpoint, config):
            raise RuntimeError(
                f"Resume stage mismatch: checkpoint={checkpoint.get('stage')} "
                f"config={config['experiment']['stage']}"
            )
        _validate_resume_scheduler(checkpoint, config, resume)
        model.load_state_dict(_without_module_prefix(checkpoint["model"]), strict=strict)
        if source_model is not None:
            if checkpoint.get("source_model") is None:
                raise RuntimeError("Resume checkpoint has no source_model state")
            source_model.load_state_dict(
                _without_module_prefix(checkpoint["source_model"]), strict=strict
            )
        # A resume is deliberately a complete continuation. The load_* fields are
        # relevant to legacy/import workflows, but may not weaken resume semantics.
        if checkpoint.get("optimizer") is None or checkpoint.get("scheduler") is None:
            raise RuntimeError("Resume checkpoint lacks optimizer or scheduler state")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        saved_distributed = checkpoint.get("distributed", {})
        saved_global_batch = saved_distributed.get("global_batch_size")
        current_global_batch = config["training"]["batch_size"]
        if (
            saved_global_batch is not None
            and saved_global_batch != current_global_batch
        ):
            warnings.warn(
                "Resuming with a changed global batch size: "
                f"checkpoint={saved_global_batch}, current={current_global_batch}",
                RuntimeWarning,
            )
        metrics = checkpoint.get("metrics", {})
        if logger is not None:
            logger.info(
                "Resumed %s at completed epoch %s, global_step=%s",
                resume, checkpoint["epoch"], checkpoint["global_step"],
            )
        return TrainingState(
            start_epoch=int(checkpoint["epoch"]),
            global_step=int(checkpoint["global_step"]),
            micro_step=int(checkpoint.get("micro_step", 0)),
            best_miou=float(metrics.get("best_mIoU", metrics.get("mIoU", float("-inf")))),
        )
    return TrainingState()
