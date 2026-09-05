from __future__ import annotations

import copy
import re
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
    model_config = copy.deepcopy(config["model"])
    image_encoder = model_config.get("image_encoder")
    # Old signatures predate the image_encoder block. The resolved legacy RRDB
    # defaults are omitted so old Stage-1 checkpoints still compare identically.
    if image_encoder is not None and image_encoder.get("type") == "rrdb":
        model_config.pop("image_encoder")
    source_keys = [
        "prior_type", "prior_noise_std", "backbone", "segformer_variant",
        "pretrained", "freeze_encoder", "decoder_channels",
        "learned_logvar", "fixed_std", "mu_tanh_scale", "supervision",
    ]
    if config["source"].get("type") == "task_finetuned_segformer":
        source_keys.extend((
            "type", "model_id", "representation", "void_channel_value",
        ))
    source_signature = {
        key: copy.deepcopy(config["source"][key])
        for key in source_keys
    }
    # Sampling on the simplex uses the exact same source network as
    # image_gaussian. Normalize the sampling-only choice so Stage-1 Gaussian
    # checkpoints remain architecture-compatible with simplex Stage 2.
    if source_signature["prior_type"] in {
        "image_bounded_gaussian", "image_simplex_mixture"
    }:
        source_signature["prior_type"] = "image_gaussian"
    decoder_type = config["source"].get("segformer_decoder", "custom")
    if decoder_type != "custom":
        source_signature["segformer_decoder"] = decoder_type
    # Omit the new false default so checkpoints created before include_void
    # existed retain exactly the same architecture signature.
    supervision = source_signature.get("supervision")
    if isinstance(supervision, dict) and not supervision.get("include_void", False):
        supervision.pop("include_void", None)
    return {
        "num_classes": config["dataset"]["num_classes"],
        "model": model_config,
        "source": source_signature,
    }


def _configured_source_decoder(config: dict) -> str | None:
    source = config.get("source", {})
    if (
        source.get("type", "trainable_segformer") == "trainable_segformer"
        and source.get("backbone") == "segformer"
    ):
        return source.get("segformer_decoder", "custom")
    return None


def validate_source_decoder_checkpoint(
    checkpoint: dict, config: dict, path: str | Path
) -> None:
    """Reject cross-decoder loads before state-dict shape/key failures."""
    current = _configured_source_decoder(config)
    saved_config = checkpoint.get("config")
    if current is None or not isinstance(saved_config, dict):
        return
    saved = _configured_source_decoder(saved_config)
    if saved is not None and saved != current:
        raise RuntimeError(
            "SegFormer source decoder mismatch: "
            f"checkpoint={saved!r}, config={current!r}, path={path}. "
            "Custom and standard decoder checkpoints are not interchangeable."
        )


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
    psd_weight_model=None,
    consistency_weight_model=None,
) -> dict:
    raw_model = model
    while isinstance(raw_model, DistributedDataParallel):
        raw_model = raw_model.module
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    raw_source = source_model
    while isinstance(raw_source, DistributedDataParallel):
        raw_source = raw_source.module
    raw_source = getattr(raw_source, "_orig_mod", raw_source)
    payload = {
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
    if consistency_weight_model is not None:
        payload["consistency_weight_model"] = consistency_weight_model.state_dict()
        if config["loss"]["consistency"]["type"] == "psd":
            # Keep emitting the legacy key so older PSD-only readers still work.
            payload["psd_weight_model"] = consistency_weight_model.state_dict()
    elif psd_weight_model is not None:
        payload["psd_weight_model"] = psd_weight_model.state_dict()
    return payload


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
    validate_source_decoder_checkpoint(checkpoint, config, path)
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


def _swin_v5_key_to_v4(key: str) -> str | None:
    prefix = "image_encoder.backbone.swin."
    if not key.startswith(prefix):
        return key
    suffix = key.removeprefix(prefix)
    if suffix in {"layernorm.weight", "layernorm.bias"}:
        return None
    key = "image_encoder.backbone." + suffix
    for current, legacy in (
        (".attention.q_proj.", ".attention.self.query."),
        (".attention.k_proj.", ".attention.self.key."),
        (".attention.v_proj.", ".attention.self.value."),
        (".attention.o_proj.", ".attention.output.dense."),
        (".attention.relative_position_bias.relative_position_bias_table", ".attention.self.relative_position_bias_table"),
        (".mlp.fc1.", ".intermediate.dense."),
        (".mlp.fc2.", ".output.dense."),
    ):
        key = key.replace(current, legacy)
    return key


def _segformer_v5_key_to_v4(key: str) -> str:
    match = re.match(r"^encoder\.stages\.(\d+)\.(.+)$", key)
    if match is None:
        return key
    stage, suffix = match.groups()
    if suffix.startswith("patch_embeddings."):
        return f"encoder.encoder.patch_embeddings.{stage}." + suffix.removeprefix("patch_embeddings.")
    if suffix.startswith("layer_norm."):
        return f"encoder.encoder.layer_norm.{stage}." + suffix.removeprefix("layer_norm.")
    block = re.match(r"^blocks\.(\d+)\.(.+)$", suffix)
    if block is None:
        return key
    block_index, block_suffix = block.groups()
    for current, legacy in (
        ("layernorm_before.", "layer_norm_1."),
        ("attention.q_proj.", "attention.self.query."),
        ("attention.k_proj.", "attention.self.key."),
        ("attention.v_proj.", "attention.self.value."),
        ("attention.o_proj.", "attention.output.dense."),
        ("attention.sequence_reduction.sequence_reduction.", "attention.self.sr."),
        ("attention.sequence_reduction.layer_norm.", "attention.self.layer_norm."),
        ("layernorm_after.", "layer_norm_2."),
        ("mlp.fc1.", "mlp.dense1."),
        ("mlp.fc2.", "mlp.dense2."),
    ):
        if block_suffix.startswith(current):
            block_suffix = legacy + block_suffix.removeprefix(current)
            break
    return f"encoder.encoder.block.{stage}.{block_index}.{block_suffix}"


def _audit_compatible_state(
    state: dict, module, *, component: str, converter
) -> dict:
    expected = module.state_dict()
    converted = {}
    for key, value in state.items():
        converted_key = converter(key)
        if converted_key is None:
            continue
        if converted_key in converted:
            raise RuntimeError(f"{component} checkpoint conversion collision: {converted_key}")
        converted[converted_key] = value
    expected_buffers = dict(module.named_buffers())
    for key, value in expected.items():
        if (
            key not in converted
            and key.endswith("attention.self.relative_position_index")
            and key in expected_buffers
        ):
            converted[key] = value
    missing = sorted(set(expected) - set(converted))
    unexpected = sorted(set(converted) - set(expected))
    shape_mismatches = sorted(
        (key, tuple(converted[key].shape), tuple(expected[key].shape))
        for key in set(converted) & set(expected)
        if converted[key].shape != expected[key].shape
    )
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            f"{component} checkpoint compatibility audit failed before strict load: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatches={shape_mismatches}"
        )
    return converted


def _model_state_for_current_transformers(state: dict, model) -> dict:
    expected = model.state_dict()
    if (
        any(key.startswith("image_encoder.backbone.swin.") for key in state)
        and any(key.startswith("image_encoder.backbone.embeddings.") for key in expected)
    ):
        return _audit_compatible_state(
            state, model, component="Endpoint", converter=_swin_v5_key_to_v4
        )
    return state


def _source_state_for_current_transformers(state: dict, source_model) -> dict:
    expected = source_model.state_dict()
    if (
        any(key.startswith("encoder.stages.") for key in state)
        and any(key.startswith("encoder.encoder.patch_embeddings.") for key in expected)
    ):
        return _audit_compatible_state(
            state, source_model, component="Source", converter=_segformer_v5_key_to_v4
        )
    return state


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
    psd_weight_model=None,
    consistency_weight_model=None,
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
        model_state = _model_state_for_current_transformers(
            _without_module_prefix(checkpoint["model"]), model
        )
        model.load_state_dict(model_state, strict=strict)
        if source_model is not None:
            source_state = _source_state_for_current_transformers(
                _without_module_prefix(saved_source), source_model
            )
            source_model.load_state_dict(source_state, strict=strict)
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
        validate_source_decoder_checkpoint(checkpoint, config, resume)
        if not _resume_stage_compatible(checkpoint, config):
            raise RuntimeError(
                f"Resume stage mismatch: checkpoint={checkpoint.get('stage')} "
                f"config={config['experiment']['stage']}"
            )
        _validate_resume_scheduler(checkpoint, config, resume)
        model_state = _model_state_for_current_transformers(
            _without_module_prefix(checkpoint["model"]), model
        )
        model.load_state_dict(model_state, strict=strict)
        if source_model is not None:
            if checkpoint.get("source_model") is None:
                raise RuntimeError("Resume checkpoint has no source_model state")
            source_state = _source_state_for_current_transformers(
                _without_module_prefix(checkpoint["source_model"]), source_model
            )
            source_model.load_state_dict(source_state, strict=strict)
        weight_model = (
            consistency_weight_model
            if consistency_weight_model is not None else psd_weight_model
        )
        saved_weight = checkpoint.get(
            "consistency_weight_model", checkpoint.get("psd_weight_model")
        )
        consistency_type = config["loss"]["consistency"]["type"]
        if weight_model is not None:
            if saved_weight is None:
                if logger is not None:
                    if consistency_type == "psd":
                        logger.info(
                            "Initializing new PSDTimeWeightNetwork because checkpoint "
                            "predates learnable PSD weighting."
                        )
                    else:
                        logger.info(
                            "Initializing new %s learnable weight network because "
                            "the checkpoint predates that network.",
                            consistency_type.upper(),
                        )
            else:
                weight_model.load_state_dict(
                    _without_module_prefix(saved_weight), strict=True
                )
        elif saved_weight is not None:
            raise RuntimeError(
                "Resume checkpoint contains a consistency weight network, but "
                "the current configuration disables learnable weighting"
            )
        # A resume is deliberately a complete continuation. The load_* fields are
        # relevant to legacy/import workflows, but may not weaken resume semantics.
        if checkpoint.get("optimizer") is None or checkpoint.get("scheduler") is None:
            raise RuntimeError("Resume checkpoint lacks optimizer or scheduler state")
        saved_optimizer = checkpoint["optimizer"]
        current_optimizer = optimizer.state_dict()
        if len(saved_optimizer["param_groups"]) == len(current_optimizer["param_groups"]):
            optimizer.load_state_dict(saved_optimizer)
            scheduler.load_state_dict(checkpoint["scheduler"])
        elif weight_model is not None and saved_weight is None:
            weight_group_name = (
                "psd_weight" if consistency_type == "psd" else "esd_weight"
            )
            _load_optimizer_with_new_weight_group(
                optimizer,
                saved_optimizer,
                weight_group_name=weight_group_name,
                model=model,
                source_model=source_model,
                saved_model_state=_without_module_prefix(checkpoint["model"]),
                saved_source_state=_without_module_prefix(checkpoint["source_model"]),
            )
            _load_scheduler_with_new_weight_group(
                scheduler, checkpoint["scheduler"], optimizer
            )
            if logger is not None:
                logger.info(
                    "Restored existing optimizer/scheduler state and initialized "
                    "the new %s parameter group with fresh optimizer state.",
                    weight_group_name,
                )
        else:
            raise RuntimeError(
                "Resume optimizer parameter groups do not match the current model"
            )
        _validate_optimizer_state_shapes(optimizer)
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


def _group_by_name(groups: list[dict], name: str) -> dict | None:
    return next((group for group in groups if group.get("name") == name), None)


def _validate_optimizer_state_shapes(optimizer) -> None:
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            for key, value in optimizer.state.get(parameter, {}).items():
                if (
                    torch.is_tensor(value)
                    and value.numel() != 1
                    and value.shape != parameter.shape
                ):
                    raise RuntimeError(
                        "Optimizer state shape mismatch after resume: "
                        f"group={group.get('name')!r}, state={key!r}, "
                        f"parameter={tuple(parameter.shape)}, state={tuple(value.shape)}"
                    )


def _saved_parameter_names(
    state: dict,
    current_named_parameters: dict[str, torch.nn.Parameter],
    converter,
) -> list[str]:
    names = []
    for key in state:
        converted = converter(key)
        if converted is not None and converted in current_named_parameters:
            names.append(converted)
    if len(names) != len(set(names)):
        raise RuntimeError("Checkpoint parameter-name conversion produced duplicates")
    return names


def _load_optimizer_with_new_weight_group(
    optimizer,
    saved_state: dict,
    *,
    weight_group_name: str,
    model,
    source_model,
    saved_model_state: dict,
    saved_source_state: dict,
) -> None:
    """Restore old named groups while leaving the new weight group fresh."""
    current = optimizer.state_dict()
    merged = copy.deepcopy(current)
    merged["state"] = {}
    saved_groups = saved_state["param_groups"]
    group_modules = {"model": model, "source": source_model}
    group_saved_states = {
        "model": saved_model_state,
        "source": saved_source_state,
    }
    for group_index, current_group in enumerate(merged["param_groups"]):
        name = current_group.get("name")
        if name == weight_group_name:
            continue
        saved_group = _group_by_name(saved_groups, name)
        if saved_group is None or len(saved_group["params"]) != len(current_group["params"]):
            raise RuntimeError(f"Cannot migrate optimizer group {name!r}")
        module = group_modules.get(name)
        if module is None:
            raise RuntimeError(f"Cannot identify module for optimizer group {name!r}")
        current_named = {
            parameter_name: parameter
            for parameter_name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        if name == "model" and any(
            key.startswith("image_encoder.backbone.embeddings.")
            for key in current_named
        ):
            converter = _swin_v5_key_to_v4
        elif name == "source" and any(
            key.startswith("encoder.encoder.patch_embeddings.")
            for key in current_named
        ):
            converter = _segformer_v5_key_to_v4
        else:
            converter = lambda key: key
        saved_names = _saved_parameter_names(
            group_saved_states[name], current_named, converter
        )
        saved_ids = list(saved_group["params"])
        if len(saved_names) != len(saved_ids):
            raise RuntimeError(
                f"Optimizer checkpoint parameter names do not match group {name!r}: "
                f"names={len(saved_names)}, optimizer={len(saved_ids)}"
            )
        saved_id_by_name = dict(zip(saved_names, saved_ids, strict=True))
        current_actual_group = optimizer.param_groups[group_index]
        current_name_by_object = {
            id(parameter): parameter_name
            for parameter_name, parameter in current_named.items()
        }
        for key, value in saved_group.items():
            if key != "params":
                current_group[key] = copy.deepcopy(value)
        for current_id, parameter in zip(
            current_group["params"], current_actual_group["params"], strict=True
        ):
            parameter_name = current_name_by_object.get(id(parameter))
            saved_id = saved_id_by_name.get(parameter_name)
            if saved_id is None:
                raise RuntimeError(
                    f"No saved optimizer state mapping for {name}.{parameter_name}"
                )
            if saved_id in saved_state["state"]:
                merged["state"][current_id] = copy.deepcopy(saved_state["state"][saved_id])
    model_group = _group_by_name(merged["param_groups"], "model")
    weight_group = _group_by_name(merged["param_groups"], weight_group_name)
    if model_group is None or weight_group is None:
        raise RuntimeError(
            f"Optimizer migration requires model and {weight_group_name} groups"
        )
    model_initial = float(model_group.get("initial_lr", model_group["lr"]))
    schedule_factor = float(model_group["lr"]) / max(model_initial, 1.0e-30)
    weight_initial = float(weight_group.get("initial_lr", weight_group["lr"]))
    weight_group["lr"] = weight_initial * schedule_factor
    optimizer.load_state_dict(merged)


def _load_scheduler_with_new_weight_group(scheduler, saved_state: dict, optimizer) -> None:
    current = scheduler.state_dict()
    migrated = copy.deepcopy(saved_state)
    old_count = len(saved_state.get("base_lrs", []))
    new_count = len(optimizer.param_groups)
    for key, current_value in current.items():
        saved_value = migrated.get(key)
        if (
            isinstance(saved_value, list)
            and isinstance(current_value, list)
            and len(saved_value) == old_count
            and len(current_value) == new_count
        ):
            migrated[key] = copy.deepcopy(saved_value) + copy.deepcopy(current_value[old_count:])
    if "_last_lr" in migrated and len(migrated["_last_lr"]) == new_count:
        migrated["_last_lr"][-1] = optimizer.param_groups[-1]["lr"]
    scheduler.load_state_dict(migrated)


def _load_optimizer_with_new_psd_group(optimizer, saved_state: dict, **kwargs) -> None:
    """Backward-compatible private wrapper."""
    _load_optimizer_with_new_weight_group(
        optimizer, saved_state, weight_group_name="psd_weight", **kwargs
    )


def _load_scheduler_with_new_psd_group(scheduler, saved_state: dict, optimizer) -> None:
    """Backward-compatible private wrapper."""
    _load_scheduler_with_new_weight_group(scheduler, saved_state, optimizer)
