from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "",
        "seed": 42,
        "output_dir": "",
        "stage": "diagonal_pretrain",
    },
    "runtime": {
        "device": "auto",
        "amp": True,
        "amp_dtype": "bf16",
        "compile": False,
        "deterministic": False,
        "config_path": None,
    },
    "distributed": {
        "enabled": "auto",
        "backend": "nccl",
        "init_method": "env://",
        "find_unused_parameters": False,
        "broadcast_buffers": False,
        "gradient_as_bucket_view": True,
    },
    "dataset": {
        "name": "cityscapes",
        "root": "",
        "num_classes": 20,
        "eval_num_classes": 19,
        "void_class_index": 19,
        "background_index": None,
        "ignore_index": 19,
        "reduce_zero_label": False,
        "train_split": "train",
        "val_split": "val",
        "image_size": [256, 512],
        "crop_size": None,
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": False,
    },
    "augmentation": {
        "enabled": True,
        "horizontal_flip": {"enabled": True, "probability": 0.5},
        "color_jitter": {
            "enabled": True,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
        },
        "imagenet_normalize": False,
        "random_resize": {
            "enabled": False,
            "base_scale": {"width": 2048, "height": 512},
            "ratio_range": [0.5, 2.0],
            "keep_ratio": True,
        },
        "random_crop": {
            "enabled": False,
            "size": [512, 512],
            "cat_max_ratio": 0.75,
            "ignore_index": 0,
            "max_attempts": 10,
        },
        "photometric_distortion": {
            "enabled": False,
            "brightness_delta": 32.0,
            "contrast_range": [0.5, 1.5],
            "saturation_range": [0.5, 1.5],
            "hue_delta": 18.0,
        },
        "normalize": {
            "enabled": False,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "pad": {
            "enabled": False,
            "size": [512, 512],
            "image_value": 0.0,
            "mask_value": 0,
        },
    },
    "model": {
        "backbone": "unet",
        "num_classes": 20,
        "state_downsample_factor": 4,
        "fusion_channels": 128,
        "image_encoder": {
            "type": "rrdb",
            "variant": "tiny",
            "pretrained": False,
            "freeze": False,
            "input_already_normalized": False,
            "neck": {
                "type": "none",
                "channels": 256,
            },
        },
        "rrdb_blocks": 5,
        "rrdb_growth_channels": 32,
        "unet": {
            "base_channels": 64,
            "channel_mults": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attention_levels": [3],
            "num_heads": 4,
            "dropout": 0.0,
            "time_embedding_dim": 256,
        },
    },
    "source": {
        "type": "trainable_segformer",
        "prior_type": "image_gaussian",
        "prior_noise_std": 1.0,
        "backbone": "segformer",
        "segformer_variant": "b0",
        "pretrained": True,
        "checkpoint": None,
        "freeze": False,
        "freeze_encoder": False,
        "decoder_channels": 128,
        "learned_logvar": False,
        "fixed_std": None,
        "mu_tanh_scale": 0.0,
        "use_loss_align": True,
        "align_weight": 0.15,
        "var_weight": 0.0,
        "align_eps": 1.0e-8,
        "input_already_normalized": False,
        "model_id": None,
        "representation": "probability",
        "void_channel_value": 0.0,
        "supervision": {
            "type": None,
            "weight": None,
        },
    },
    "flow": {
        "time_eps": 1.0e-5,
        "probability_eps": 1.0e-8,
        "start_time": 0.0,
        "path": {"type": "power", "exponent": 1.0},
    },
    "time_sampling": {
        "distribution": "uniform_sorted",
        "min_time": 0.0,
        "max_time": 1.0,
        "min_gap": 1.0e-5,
    },
    "training": {
        "epochs": 150,
        "max_iterations": None,
        "max_optimizer_steps": None,
        "max_batches_per_epoch": None,
        "batch_size": 4,
        "grad_accum_steps": 1,
        "optimizer": {
            "name": "adamw",
            "lr": 5.0e-5,
            "weight_decay": 1.0e-4,
            "betas": [0.9, 0.999],
            "parameter_groups": {
                "model": {"lr": None},
                "source": {"lr": None},
            },
        },
        "scheduler": {
            "name": "cosine",
            "warmup_epochs": 20,
            "warmup_start_factor": 0.1,
            "eta_min": 1.0e-6,
            "warmup_steps": 0,
            "power": 1.0,
            "min_lr": 0.0,
            "step_unit": "epoch",
        },
        "grad_clip": 1.0,
        "label_smoothing": 0.1,
        "log_interval": 50,
        "checkpoint_interval_epochs": 10,
        "checkpoint_interval_steps": None,
        "validation_epochs": [50, 100, 150],
    },
    "loss": {
        "ignore_index": None,
        "mask_pixel_losses": False,
        "primary": {
            "type": "diagonal_ce",
            "weight": 1.0,
            "adaptive_weighting": {"enabled": False, "r": 0.5, "c": 0.01},
        },
        "consistency": {
            "enabled": False,
            "type": "esd",
            "weight": 0.0,
            "start_epoch": 0,
            "start": {"unit": "epoch", "value": 0},
            "warmup_epochs": 0,
            "warmup_steps": 0,
            "schedule": "linear",
            "max_weight": 1.0,
            "precision": {
                "jvp_dtype": "bf16",
                "numerical_dtype": "fp32",
                "debug_assertions": False,
            },
            "psd": {"loss_resolution": "state"},
            "gradient_surgery": {
                "enabled": False,
                "priority": "diagonal",
                "scope": "endpoint_model",
                "eps": 1.0e-12,
            },
            "learnable_weight": {
                "enabled": False,
                "dependency": "s",
                "type": "uncertainty",
                "time_embedding_dim": 32,
                "hidden_dim": 64,
                "init_effective_weight": 0.5,
                "lr": None,
                "weight_decay": 0.0,
            },
            "csd": {},
            "ecld": {
                "ec_weight": 4.0,
                "td_weight": 2.0,
                "time_weighting": "none",
            },
            "esd": {
                "formulation": "stabilized_logit_space",
                "source": "discrete_flow_maps",
                "additional_numerical_safeguards": True,
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
    "checkpoint": {
        "resume": None,
        "init_from": None,
        "load_optimizer": True,
        "load_scheduler": True,
        "strict_model": True,
    },
    "evaluation": {
        "split": "val",
        "batch_size": 4,
        "num_steps": 15,
        "sampler": "flow_map",
        "save_predictions": True,
        "max_visualizations": 16,
        "max_batches": None,
        "checkpoint": None,
        "resize": {"width": 2048, "height": 512, "keep_ratio": True},
        "size_divisor": None,
        "original_resolution": False,
        "interpolation": "bilinear",
        "align_corners": False,
        "eval_class_indices": None,
        "ignore_index": None,
        "nanmean": False,
        "exclude_void_from_prediction": True,
        "interval": {"unit": "epoch", "value": None},
        "test_time_augmentation": {"enabled": False, "flip": False},
    },
    "wandb": {
        "enabled": True,
        "project": "DFM",
        "name": None,
        "entity": None,
        "mode": "online",
        "tags": [],
    },
}

# `distributed` was added after the original YAML format. Let older single-GPU
# configs inherit its safe defaults instead of breaking config compatibility.
REQUIRED_SECTIONS = tuple(
    section for section in DEFAULT_CONFIG if section != "distributed"
)
REQUIRED_KEYS = (
    "experiment.name",
    "experiment.output_dir",
    "experiment.stage",
    "dataset.root",
    "dataset.num_classes",
    "model.num_classes",
    "training.epochs",
    "loss.primary.type",
)


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    return value


def _merge(defaults: dict[str, Any], supplied: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(defaults)
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _check_unknown(supplied: dict[str, Any], schema: dict[str, Any], prefix: str = "") -> None:
    for key, value in supplied.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if key not in schema:
            raise ValueError(f"Unknown config key: {dotted}")
        if isinstance(value, dict):
            if not isinstance(schema[key], dict):
                raise ValueError(f"Config key must not be a mapping: {dotted}")
            _check_unknown(value, schema[key], dotted)


def select(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _validate_required(raw: dict[str, Any]) -> None:
    for section in REQUIRED_SECTIONS:
        if section not in raw:
            raise ValueError(f"Missing required config section: {section}")
    for dotted in REQUIRED_KEYS:
        if select(raw, dotted, None) in (None, ""):
            raise ValueError(f"Missing required config key: {dotted}")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    stage = select(config, "experiment.stage")
    valid_stages = {
        "diagonal_pretrain",
        "consistency_distillation",
        "esd_distillation",
        "joint_training",
    }
    if stage not in valid_stages:
        raise ValueError(f"experiment.stage must be one of {sorted(valid_stages)}")
    dataset = config["dataset"]
    if dataset["name"] not in {"cityscapes", "ade20k"}:
        raise ValueError("dataset.name must be cityscapes or ade20k")
    if dataset["num_classes"] != config["model"]["num_classes"]:
        raise ValueError("dataset.num_classes and model.num_classes must match")
    exclude_void = config["evaluation"]["exclude_void_from_prediction"]
    if not isinstance(exclude_void, bool):
        raise ValueError("evaluation.exclude_void_from_prediction must be a boolean")
    void_class_index = dataset["void_class_index"]
    if exclude_void and (
        isinstance(void_class_index, bool)
        or not isinstance(void_class_index, int)
        or not 0 <= void_class_index < dataset["num_classes"]
    ):
        raise ValueError(
            "dataset.void_class_index must be a valid class index when "
            "evaluation.exclude_void_from_prediction=true"
        )
    if dataset["name"] == "cityscapes":
        if dataset["num_classes"] != 20:
            raise ValueError("DFM Cityscapes training requires exactly 20 classes")
        if dataset["eval_num_classes"] != 19 or dataset["void_class_index"] != 19:
            raise ValueError("Cityscapes requires 19 eval classes and void index 19")
        augmentation = config["augmentation"]
        if (
            augmentation["random_crop"]["enabled"]
            and augmentation["random_crop"]["ignore_index"] != 19
        ):
            raise ValueError(
                "Cityscapes random_crop.ignore_index must be void index 19"
            )
        if (
            augmentation["pad"]["enabled"]
            and augmentation["pad"]["mask_value"] != 19
        ):
            raise ValueError("Cityscapes pad.mask_value must be void index 19")
        if (
            augmentation["normalize"]["enabled"]
            and augmentation["imagenet_normalize"]
        ):
            raise ValueError(
                "Cityscapes normalize and legacy imagenet_normalize must not both be enabled"
            )
        if (
            augmentation["normalize"]["enabled"]
            and config["source"]["backbone"] == "segformer"
            and not config["source"]["input_already_normalized"]
        ):
            raise ValueError(
                "Normalized Cityscapes input requires "
                "source.input_already_normalized=true for SegFormer"
            )
    else:
        if dataset["num_classes"] != 151 or dataset["eval_num_classes"] != 150:
            raise ValueError("ADE20K 151-state protocol requires 151 model and 150 eval classes")
        if dataset["background_index"] != 0 or dataset["ignore_index"] != 0:
            raise ValueError("ADE20K requires background_index=ignore_index=0")
        if dataset["reduce_zero_label"] is not False:
            raise ValueError("ADE20K 151-state protocol requires reduce_zero_label=false")
        if config["loss"]["ignore_index"] != 0 or not config["loss"]["mask_pixel_losses"]:
            raise ValueError("ADE20K requires loss.ignore_index=0 and mask_pixel_losses=true")
        if dataset["void_class_index"] != 0:
            raise ValueError("ADE20K dataset.void_class_index must be 0")
        if config["source"]["input_already_normalized"] != config["augmentation"]["normalize"]["enabled"]:
            raise ValueError(
                "ADE20K source.input_already_normalized must match dataset normalization"
            )
        if config["evaluation"]["eval_class_indices"] != [1, 150]:
            raise ValueError("ADE20K evaluation.eval_class_indices must be [1, 150]")
        if config["evaluation"]["ignore_index"] != 0:
            raise ValueError("ADE20K evaluation.ignore_index must be 0")
        evaluation = config["evaluation"]
        if not evaluation["original_resolution"]:
            raise ValueError("ADE20K requires original_resolution evaluation")
        if evaluation["interpolation"] != "bilinear" or evaluation["align_corners"]:
            raise ValueError("ADE20K evaluation requires bilinear and align_corners=false")
        if evaluation["test_time_augmentation"]["enabled"]:
            raise ValueError("ADE20K main protocol requires TTA disabled")
        crop = config["augmentation"]["random_crop"]
        if crop["ignore_index"] != 0:
            raise ValueError("ADE20K random_crop.ignore_index must be 0")
        if config["augmentation"]["pad"]["mask_value"] != 0:
            raise ValueError("ADE20K padding mask value must be 0")
    if len(config["dataset"]["image_size"]) != 2:
        raise ValueError("dataset.image_size must be [height, width]")
    if config["dataset"]["crop_size"] is not None and len(config["dataset"]["crop_size"]) != 2:
        raise ValueError("dataset.crop_size must be null or [height, width]")
    if config["runtime"]["amp_dtype"] not in {"bf16", "fp16"}:
        raise ValueError("runtime.amp_dtype must be bf16 or fp16")
    training = config["training"]
    if (
        isinstance(training["grad_accum_steps"], bool)
        or not isinstance(training["grad_accum_steps"], int)
        or training["grad_accum_steps"] <= 0
    ):
        raise ValueError("training.grad_accum_steps must be a positive integer")
    if (
        training["max_batches_per_epoch"] is not None
        and training["max_batches_per_epoch"] <= 0
    ):
        raise ValueError("training.max_batches_per_epoch must be null or positive")
    if training["max_optimizer_steps"] is not None and training["max_optimizer_steps"] <= 0:
        raise ValueError("training.max_optimizer_steps must be null or positive")
    checkpoint_interval_steps = training["checkpoint_interval_steps"]
    if checkpoint_interval_steps is not None and (
        isinstance(checkpoint_interval_steps, bool)
        or not isinstance(checkpoint_interval_steps, int)
        or checkpoint_interval_steps <= 0
    ):
        raise ValueError(
            "training.checkpoint_interval_steps must be null or a positive integer"
        )
    scheduler = training["scheduler"]
    if scheduler["step_unit"] not in {"epoch", "optimizer_step"}:
        raise ValueError("training.scheduler.step_unit must be epoch or optimizer_step")
    if scheduler["name"] not in {"constant", "cosine", "poly"}:
        raise ValueError("training.scheduler.name must be constant, cosine, or poly")
    if scheduler["name"] == "poly":
        if scheduler["step_unit"] != "optimizer_step":
            raise ValueError("poly scheduler requires step_unit=optimizer_step")
        if training["max_optimizer_steps"] is None:
            raise ValueError("poly scheduler requires training.max_optimizer_steps")
        if scheduler["warmup_steps"] < 0 or scheduler["power"] <= 0:
            raise ValueError("poly warmup_steps must be non-negative and power positive")
    if scheduler["name"] == "cosine" and scheduler["step_unit"] == "optimizer_step":
        maximum = training["max_optimizer_steps"]
        if maximum is None:
            raise ValueError(
                "optimizer-step cosine scheduler requires training.max_optimizer_steps"
            )
        if scheduler["warmup_steps"] < 0 or scheduler["warmup_steps"] >= maximum:
            raise ValueError(
                "optimizer-step cosine warmup_steps must satisfy "
                "0 <= warmup_steps < max_optimizer_steps"
            )
    if not (0.0 < scheduler["warmup_start_factor"] <= 1.0):
        raise ValueError(
            "training.scheduler.warmup_start_factor must satisfy 0 < factor <= 1"
        )
    evaluation_interval = config["evaluation"]["interval"]
    if evaluation_interval["unit"] not in {"epoch", "optimizer_step"}:
        raise ValueError("evaluation.interval.unit must be epoch or optimizer_step")
    interval_value = evaluation_interval["value"]
    if interval_value is not None and (
        isinstance(interval_value, bool)
        or not isinstance(interval_value, int)
        or interval_value <= 0
    ):
        raise ValueError("evaluation.interval.value must be null or a positive integer")
    distributed = config["distributed"]
    if distributed["enabled"] not in {"auto", True, False}:
        raise ValueError("distributed.enabled must be auto, true, or false")
    if distributed["backend"] not in {"nccl", "gloo"}:
        raise ValueError("distributed.backend must be nccl or gloo")
    if distributed["init_method"] != "env://":
        raise ValueError("distributed.init_method currently supports only env://")
    if config["model"]["backbone"] != "unet":
        raise ValueError("This DFM implementation currently supports model.backbone=unet")
    image_encoder = config["model"]["image_encoder"]
    if image_encoder["type"] not in {"rrdb", "swin", "convnext"}:
        raise ValueError("model.image_encoder.type must be rrdb, swin, or convnext")
    if image_encoder["variant"] not in {"tiny", "small", "base", "large"}:
        raise ValueError(
            "model.image_encoder.variant must be tiny, small, base, or large"
        )
    if image_encoder["neck"]["type"] not in {"none", "ddp_fpn_merge"}:
        raise ValueError(
            "model.image_encoder.neck.type must be none or ddp_fpn_merge"
        )
    if (
        isinstance(image_encoder["neck"]["channels"], bool)
        or not isinstance(image_encoder["neck"]["channels"], int)
        or image_encoder["neck"]["channels"] <= 0
    ):
        raise ValueError("model.image_encoder.neck.channels must be a positive integer")
    if image_encoder["type"] == "rrdb" and image_encoder["pretrained"]:
        raise ValueError("RRDB does not provide pretrained weights")
    if image_encoder["type"] == "rrdb" and image_encoder["freeze"]:
        raise ValueError("RRDB has no separate pretrained backbone to freeze")
    if image_encoder["type"] in {"swin", "convnext"}:
        dataset_input_normalized = bool(
            config["augmentation"]["normalize"]["enabled"]
            or config["augmentation"]["imagenet_normalize"]
        )
        encoder_expects_normalized = image_encoder["input_already_normalized"]
        if dataset_input_normalized and not encoder_expects_normalized:
            raise ValueError(
                "Normalized dataset input requires "
                "model.image_encoder.input_already_normalized=true "
                "for swin/convnext"
            )
        if not dataset_input_normalized and encoder_expects_normalized:
            raise ValueError(
                "Unnormalized dataset input requires "
                "model.image_encoder.input_already_normalized=false "
                "for swin/convnext"
            )
    state_factor = config["model"]["state_downsample_factor"]
    if (
        isinstance(state_factor, bool)
        or not isinstance(state_factor, int)
        or state_factor < 1
        or state_factor & (state_factor - 1)
    ):
        raise ValueError("model.state_downsample_factor must be a positive power of two")
    if config["source"]["prior_type"] not in {"gaussian", "dirichlet", "image_gaussian"}:
        raise ValueError("source.prior_type must be gaussian, dirichlet, or image_gaussian")
    if config["source"]["type"] not in {
        "trainable_segformer", "task_finetuned_segformer"
    }:
        raise ValueError(
            "source.type must be trainable_segformer or task_finetuned_segformer"
        )
    if config["source"]["backbone"] not in {"segformer", "unet"}:
        raise ValueError("source.backbone must be segformer or unet")
    if config["source"]["representation"] not in {"probability", "logits"}:
        raise ValueError("source.representation must be probability or logits")
    supervision = config["source"]["supervision"]
    if supervision["type"] not in {None, "none", "align", "cross_entropy"}:
        raise ValueError(
            "source.supervision.type must be none, align, or cross_entropy"
        )
    if supervision["weight"] is not None and supervision["weight"] < 0:
        raise ValueError("source.supervision.weight must be non-negative")
    if config["source"]["type"] == "task_finetuned_segformer":
        if config["source"]["prior_type"] != "image_gaussian":
            raise ValueError(
                "task_finetuned_segformer requires source.prior_type=image_gaussian"
            )
        if config["source"]["backbone"] != "segformer":
            raise ValueError(
                "task_finetuned_segformer requires source.backbone=segformer"
            )
        if not config["source"]["model_id"]:
            raise ValueError("task_finetuned_segformer requires source.model_id")
        if config["source"]["learned_logvar"]:
            raise ValueError(
                "task_finetuned_segformer does not support learned_logvar=true"
            )
        if config["source"]["freeze"] and supervision["type"] not in {None, "none"}:
            raise ValueError(
                "Frozen task_finetuned_segformer requires source.supervision.type=none"
            )
    if config["flow"]["time_eps"] <= 0:
        raise ValueError("flow.time_eps must be positive")
    path = config["flow"]["path"]
    if path["type"] != "power":
        raise ValueError("flow.path.type must be power")
    if (
        isinstance(path["exponent"], bool)
        or not isinstance(path["exponent"], (int, float))
        or path["exponent"] <= 0
    ):
        raise ValueError("flow.path.exponent must be positive")
    ts = config["time_sampling"]
    if ts["distribution"] != "uniform_sorted":
        raise ValueError("time_sampling.distribution must be uniform_sorted")
    if not (0.0 <= ts["min_time"] < ts["max_time"] <= 1.0):
        raise ValueError("time range must satisfy 0 <= min_time < max_time <= 1")
    if not (0.0 < ts["min_gap"] <= ts["max_time"] - ts["min_time"]):
        raise ValueError("time_sampling.min_gap is outside the configured range")
    invalid = config["loss"]["consistency"]["invalid_teacher"]
    if invalid["strategy"] not in {"clamp", "mask_pixel", "skip_batch"}:
        raise ValueError("invalid_teacher.strategy must be clamp, mask_pixel, or skip_batch")
    if config["checkpoint"]["init_from"] and config["checkpoint"]["resume"]:
        raise ValueError("checkpoint.init_from and checkpoint.resume are mutually exclusive")
    consistency = config["loss"]["consistency"]
    adaptive_diagonal = config["loss"]["primary"]["adaptive_weighting"]
    if not isinstance(adaptive_diagonal["enabled"], bool):
        raise ValueError("loss.primary.adaptive_weighting.enabled must be boolean")
    if adaptive_diagonal["r"] < 0 or adaptive_diagonal["c"] <= 0:
        raise ValueError("adaptive diagonal weighting requires r >= 0 and c > 0")
    surgery = consistency["gradient_surgery"]
    if surgery["priority"] != "diagonal" or surgery["scope"] != "endpoint_model":
        raise ValueError("gradient surgery supports diagonal/endpoint_model only")
    if surgery["eps"] <= 0:
        raise ValueError("gradient surgery eps must be positive")
    learnable = consistency["learnable_weight"]
    if learnable["type"] != "uncertainty":
        raise ValueError("learnable consistency weighting supports uncertainty only")
    if learnable["time_embedding_dim"] <= 0 or learnable["hidden_dim"] <= 0:
        raise ValueError("learnable consistency weight dimensions must be positive")
    if learnable["init_effective_weight"] <= 0:
        raise ValueError("learnable consistency initial effective weight must be positive")
    if surgery["enabled"] and consistency["type"] != "psd":
        raise ValueError("gradient surgery is supported only for PSD")
    if learnable["enabled"]:
        expected_dependency = {"psd": "s", "esd": "st"}.get(consistency["type"])
        if expected_dependency is None:
            raise ValueError("learnable weighting is supported only for PSD and ESD")
        if learnable["dependency"] != expected_dependency:
            raise ValueError(
                f"{consistency['type'].upper()} learnable weighting requires "
                f"dependency={expected_dependency}"
            )
    start = consistency["start"]
    if start["unit"] not in {"epoch", "optimizer_step"}:
        raise ValueError("loss.consistency.start.unit must be epoch or optimizer_step")
    if start["value"] < 0:
        raise ValueError("loss.consistency.start.value must be non-negative")
    if consistency["warmup_epochs"] < 0:
        raise ValueError("loss.consistency.warmup_epochs must be non-negative")
    if consistency["warmup_steps"] < 0:
        raise ValueError("loss.consistency.warmup_steps must be non-negative")
    if consistency["type"] not in {"psd", "csd", "ecld", "esd"}:
        raise ValueError("loss.consistency.type must be psd, csd, ecld, or esd")
    if consistency["schedule"] != "linear":
        raise ValueError("loss.consistency.schedule currently supports only linear")
    precision = consistency["precision"]
    if precision["numerical_dtype"] != "fp32":
        raise ValueError("loss.consistency.precision.numerical_dtype must be fp32")
    if consistency["type"] == "psd":
        if consistency["psd"]["loss_resolution"] not in {"state", "full"}:
            raise ValueError(
                "loss.consistency.psd.loss_resolution must be 'state' or 'full'"
            )
        if precision["jvp_dtype"] is not None:
            raise ValueError("PSD does not use JVP; precision.jvp_dtype must be null")
    elif precision["jvp_dtype"] not in {"bf16", "fp32"}:
        raise ValueError("precision.jvp_dtype must be bf16 or fp32")
    if (
        consistency["enabled"]
        and precision["jvp_dtype"] == "bf16"
        and (
            not config["runtime"]["amp"]
            or config["runtime"]["amp_dtype"] != "bf16"
        )
    ):
        raise ValueError(
            "bf16 JVP requires runtime.amp=true and runtime.amp_dtype=bf16"
        )
    if consistency["ecld"]["time_weighting"] not in {"none", "inverse_square"}:
        raise ValueError("ECLD time_weighting must be none or inverse_square")
    if consistency["type"] == "esd":
        esd = consistency["esd"]
        if esd["formulation"] != "stabilized_logit_space":
            raise ValueError(
                "ESD formulation must be stabilized_logit_space"
            )
        if esd["source"] != "discrete_flow_maps":
            raise ValueError("ESD source must be discrete_flow_maps")
        if not isinstance(esd["additional_numerical_safeguards"], bool):
            raise ValueError(
                "ESD additional_numerical_safeguards must be a boolean"
            )
    if stage == "diagonal_pretrain" and consistency["enabled"]:
        raise ValueError("Stage 1 must not enable a consistency loss")
    if stage in {"consistency_distillation", "esd_distillation"}:
        if not consistency["enabled"]:
            raise ValueError("Stage 2 requires loss.consistency.enabled=true")
        if stage == "esd_distillation" and consistency["type"] != "esd":
            raise ValueError("Legacy esd_distillation stage requires consistency.type=esd")
        if not config["checkpoint"]["init_from"] and not config["checkpoint"]["resume"]:
            raise ValueError("Stage 2 requires checkpoint.init_from or checkpoint.resume")
    if stage == "joint_training":
        if not consistency["enabled"]:
            raise ValueError("joint_training requires consistency.enabled=true")
        if config["checkpoint"]["init_from"]:
            raise ValueError("joint_training forbids checkpoint.init_from")
    return config


def apply_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    override_keys: set[str] = set()
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must have KEY=VALUE form: {override}")
        dotted, raw_value = override.split("=", 1)
        override_keys.add(dotted)
        parts = dotted.split(".")
        cursor: Any = result
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise ValueError(f"Unknown override key: {dotted}")
            cursor = cursor[part]
        if not isinstance(cursor, dict) or parts[-1] not in cursor:
            raise ValueError(f"Unknown override key: {dotted}")
        cursor[parts[-1]] = yaml.safe_load(raw_value)
    if (
        "loss.consistency.start_epoch" in override_keys
        and not any(key.startswith("loss.consistency.start.") for key in override_keys)
        and result["loss"]["consistency"]["start"]["unit"] == "epoch"
    ):
        result["loss"]["consistency"]["start"]["value"] = result["loss"]["consistency"]["start_epoch"]
    if (
        {"source.use_loss_align", "source.align_weight"} & override_keys
        and not any(key.startswith("source.supervision.") for key in override_keys)
    ):
        result["source"]["supervision"] = {
            "type": "align" if result["source"]["use_loss_align"] else "none",
            "weight": result["source"]["align_weight"],
        }
    if {
        "model.image_encoder.type", "model.image_encoder.variant"
    } & override_keys:
        aliases = {"t": "tiny", "s": "small", "b": "base", "l": "large"}
        result["model"]["image_encoder"]["variant"] = aliases.get(
            result["model"]["image_encoder"]["variant"],
            result["model"]["image_encoder"]["variant"],
        )
        if (
            "model.image_encoder.type" in override_keys
            and result["model"]["image_encoder"]["type"] in {"swin", "convnext"}
            and "model.image_encoder.pretrained" not in override_keys
        ):
            result["model"]["image_encoder"]["pretrained"] = True
    normalization_context_changed = bool({
        "model.image_encoder.type",
        "augmentation.normalize.enabled",
        "augmentation.imagenet_normalize",
    } & override_keys)
    if (
        normalization_context_changed
        and "model.image_encoder.input_already_normalized" not in override_keys
        and result["model"]["image_encoder"]["type"] in {"swin", "convnext"}
    ):
        result["model"]["image_encoder"]["input_already_normalized"] = bool(
            result["augmentation"]["normalize"]["enabled"]
            or result["augmentation"]["imagenet_normalize"]
        )
    return validate_config(result)


def _load_raw_config(
    path: Path,
    stack: tuple[Path, ...] = (),
) -> tuple[dict[str, Any], list[str]]:
    path = path.expanduser().resolve()
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Recursive config extends detected: {cycle}")
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    extends = raw.pop("extends", None)
    if extends is None:
        parents: list[str] = []
    elif isinstance(extends, str):
        parents = [extends]
    elif (
        isinstance(extends, list)
        and all(isinstance(parent, str) and parent for parent in extends)
    ):
        parents = extends
    else:
        raise ValueError(f"extends must be a string or list of strings: {path}")
    merged: dict[str, Any] = {}
    chain: list[str] = []
    for parent in parents:
        base_path = Path(parent)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        parent_raw, parent_chain = _load_raw_config(
            base_path, (*stack, path)
        )
        merged = _merge(merged, parent_raw)
        chain.extend(parent_chain)
    merged = _merge(merged, raw)
    chain.append(str(path))
    return merged, chain


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    raw, _config_chain = _load_raw_config(config_path)
    _validate_required(raw)
    _check_unknown(raw, DEFAULT_CONFIG)
    config = _merge(DEFAULT_CONFIG, _expand(raw))
    raw_image_encoder = select(raw, "model.image_encoder", None)
    if raw_image_encoder is not None:
        aliases = {"t": "tiny", "s": "small", "b": "base", "l": "large"}
        config["model"]["image_encoder"]["variant"] = aliases.get(
            config["model"]["image_encoder"]["variant"],
            config["model"]["image_encoder"]["variant"],
        )
        if (
            config["model"]["image_encoder"]["type"] in {"swin", "convnext"}
            and "pretrained" not in raw_image_encoder
        ):
            config["model"]["image_encoder"]["pretrained"] = True
    # Translate legacy source flags only when the new block is absent.
    if select(raw, "source.supervision", None) is None:
        config["source"]["supervision"] = {
            "type": "align" if config["source"]["use_loss_align"] else "none",
            "weight": config["source"]["align_weight"],
        }
    if select(raw, "loss.consistency.start", None) is None:
        config["loss"]["consistency"]["start"] = {
            "unit": "epoch",
            "value": config["loss"]["consistency"]["start_epoch"],
        }
    config["runtime"]["config_path"] = str(config_path)
    return apply_overrides(validate_config(config), overrides)


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
