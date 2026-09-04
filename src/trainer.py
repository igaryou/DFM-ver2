from __future__ import annotations

import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from checkpoint import (
    checkpoint_payload,
    initialize_or_resume,
    save_checkpoint,
)
from adaptive_path import shannon_entropy
from config import save_resolved_config
from dataset import ade20k_eval_collate, build_dataset
from dfm_stabilization import GradientSurgeryAccumulator
from distributed import (
    DistributedContext,
    DistributedEvalSampler,
    EpochMetricMeter,
    all_reduce_confusion_matrix,
    assert_config_equal_across_ranks,
    barrier,
    cleanup_distributed,
    parameter_checksum,
    reduce_epoch_metric_meter,
    reduce_max_values,
    reduce_scalar,
    seed_data_loader_worker,
    setup_distributed,
    unwrap_model,
    validate_global_batch_size,
    wrap_ddp,
)
from inference import (
    sample_segmentation,
    state_to_original_continuous,
    terminal_state_to_original_prediction,
)
from metrics import SegmentationMetrics
from model_factory import build_models
from source_model import source_statistics
from training_objectives import (
    DDPCompatibleTrainingModel,
    run_model_training_objectives,
)
from utils import (
    append_jsonl,
    autocast_context,
    build_grad_scaler,
    init_wandb,
    seed_everything,
    setup_logger,
)
from visualization import (
    save_adaptive_path_debug,
    save_prediction,
    save_source_diagnostics,
)


MAX_REDUCTION_KEYS = {
    "esd_delta_abs_max",
    "esd_adaptive_weight_max",
    "esd_max_sample_invalid_ratio",
    "esd_jvp_output_abs_max",
    "csd_jvp_output_abs_max",
    "ecld_jvp_output_abs_max",
    "esd_teacher_max",
    "source_mu_max",
    "s_max",
    "t_max",
    "path_entropy_max",
    "path_difficulty_max",
    "path_lambda_max",
}

MIN_REDUCTION_KEYS = {
    "esd_log_arg_min",
    "esd_teacher_min",
    "source_mu_min",
    "s_min",
    "t_min",
    "diagonal_adaptive_weight_min",
    "psd_weight_logit_min",
    "psd_effective_multiplier_min",
    "esd_adaptive_weight_min",
    "esd_weight_logit_min",
    "esd_effective_multiplier_min",
    "path_entropy_min",
    "path_difficulty_min",
    "path_lambda_min",
}

MAX_REDUCTION_KEYS.update({
    "diagonal_adaptive_weight_max",
    "psd_weight_logit_max",
    "psd_effective_multiplier_max",
    "esd_weight_logit_max",
    "esd_effective_multiplier_max",
})

DFM_RECIPE_SUMMARY_KEYS = (
    "loss_diagonal_raw",
    "loss_diagonal_adaptive",
    "diagonal_adaptive_weight_mean",
    "psd_weight_logit_mean",
    "psd_effective_multiplier_mean",
    "gradient_surgery_cosine",
    "gradient_surgery_conflict_fraction",
    "gradient_surgery_removed_fraction",
    "gradient_surgery_accumulated_microbatches",
    "gradient_surgery_accumulation_enabled",
    "gradient_surgery_partial_accumulation",
    "psd_loss_height",
    "psd_loss_width",
    "psd_loss_resolution_is_full",
    "loss_esd_raw_kl",
    "loss_esd_adaptive_kl",
    "esd_log_arg_min",
    "esd_log_arg_mean",
    "esd_nonfinite_ratio",
    "esd_valid_pixel_ratio",
    "esd_adaptive_weight_mean",
    "esd_adaptive_weight_std",
    "esd_adaptive_weight_min",
    "esd_adaptive_weight_max",
    "esd_mismatch_l2_sq_mean",
    "esd_weight_logit_mean",
    "esd_weight_logit_std",
    "esd_weight_logit_min",
    "esd_weight_logit_max",
    "esd_effective_multiplier_mean",
    "esd_effective_multiplier_std",
    "esd_effective_multiplier_min",
    "esd_effective_multiplier_max",
    "loss_esd_raw",
    "loss_esd_learnable",
    "loss_esd_uncertainty_weighted",
    "loss_esd_uncertainty_regularizer",
    "esd_jvp_output_abs_mean",
    "esd_jvp_output_abs_max",
    "esd_effective_multiplier_s_0_0_1",
    "esd_effective_multiplier_s_0_1_0_25",
    "esd_effective_multiplier_s_0_25_0_5",
    "esd_effective_multiplier_s_0_5_0_75",
    "esd_effective_multiplier_s_0_75_1",
    "esd_effective_multiplier_t_0_0_1",
    "esd_effective_multiplier_t_0_1_0_25",
    "esd_effective_multiplier_t_0_25_0_5",
    "esd_effective_multiplier_t_0_5_0_75",
    "esd_effective_multiplier_t_0_75_1",
    "esd_effective_multiplier_delta_0_0_1",
    "esd_effective_multiplier_delta_0_1_0_25",
    "esd_effective_multiplier_delta_0_25_0_5",
    "esd_effective_multiplier_delta_0_5_1",
)

CONSISTENCY_SUMMARY_KEYS = {
    "psd": ("loss_psd", "psd_teacher_entropy"),
    "csd": ("loss_csd", "csd_residual_norm"),
    "ecld": (
        "loss_ecld",
        "loss_ecld_ec",
        "loss_ecld_td",
        "ecld_dt_prob_norm",
    ),
    "esd": (
        "loss_esd",
        "esd_teacher_entropy",
        "esd_clamp_ratio",
        "esd_pixel_invalid_ratio",
        "esd_sample_invalid_ratio",
    ),
}

SOURCE_SUMMARY_ALIASES = (
    ("loss_var", "loss_source_var"),
    ("loss_align", "loss_source_align"),
    ("weighted_var", "weighted_var"),
    ("weighted_align", "weighted_align"),
    ("mu_abs", "source_mu_abs"),
    ("mu_min", "source_mu_min"),
    ("mu_max", "source_mu_max"),
    ("logvar_mean", "source_logvar_mean"),
    ("sigma_mean", "source_sigma_mean"),
    ("x0_abs", "source_x0_abs"),
    ("x1_abs", "target_x1_abs"),
)


def _epoch_total_iterations(
    loader_length: int,
    max_iterations: int | None,
    total_iterations: int,
    max_batches_per_epoch: int | None = None,
    max_optimizer_steps: int | None = None,
    optimizer_step: int = 0,
    grad_accum_steps: int = 1,
) -> int:
    total = loader_length
    if max_iterations is not None:
        remaining_iterations = max(max_iterations - total_iterations, 0)
        total = min(total, remaining_iterations)
    if max_batches_per_epoch is not None:
        total = min(total, max_batches_per_epoch)
    if max_optimizer_steps is not None:
        remaining_updates = max(max_optimizer_steps - optimizer_step, 0)
        total = min(total, remaining_updates * grad_accum_steps)
    return total


def _optimizer_step_validation_trigger(
    config: dict, global_step: int
) -> str | None:
    max_optimizer_steps = config["training"]["max_optimizer_steps"]
    if (
        max_optimizer_steps is not None
        and global_step >= max_optimizer_steps
    ):
        return "final_optimizer_step"
    interval = config["evaluation"]["interval"]
    if (
        interval["unit"] == "optimizer_step"
        and interval["value"] is not None
        and global_step > 0
        and global_step % interval["value"] == 0
    ):
        return "optimizer_step_interval"
    return None


def _optimizer_step_checkpoint_due(config: dict, global_step: int) -> bool:
    interval = config["training"]["checkpoint_interval_steps"]
    return (
        interval is not None
        and global_step > 0
        and global_step % interval == 0
    )


def _numbered_checkpoint_epochs(
    *,
    total_epochs: int,
    checkpoint_interval: int,
    joint_entrypoint: bool,
    consistency_enabled: bool,
    consistency_start_epoch: int,
) -> set[int]:
    """Return 1-indexed epochs that receive an ``epoch_NNNN.pt`` file."""
    epochs = (
        set(range(checkpoint_interval, total_epochs + 1, checkpoint_interval))
        if checkpoint_interval > 0
        else set()
    )

    # The training loop passes its 0-indexed epoch_index directly to the
    # consistency schedule. Therefore start_epoch=N first contributes during
    # displayed epoch N+1, and displayed epoch N is the last pure Stage 1 epoch.
    first_consistency_epoch = consistency_start_epoch + 1
    stage1_end_epoch = first_consistency_epoch - 1
    if (
        joint_entrypoint
        and consistency_enabled
        and 1 <= stage1_end_epoch < total_epochs
    ):
        epochs.add(stage1_end_epoch)
    return epochs


def _create_epoch_progress(
    *,
    epoch_index: int,
    total: int,
    is_main_process: bool,
):
    if not is_main_process:
        return None
    return tqdm(
        total=total,
        desc=f"epoch {epoch_index + 1}",
        dynamic_ncols=True,
        leave=True,
        unit="batch",
        mininterval=0.5,
    )


def _parameter_group_lr(optimizer, name: str) -> float | None:
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return float(group["lr"])
    return None


def _build_epoch_report(
    *,
    epoch: int,
    reduced_epoch: dict[str, torch.Tensor | float],
    consistency_type: str,
    primary_weight: float,
    local_batch_size: int,
    global_batch_size: int,
    grad_accum_steps: int,
    optimizer_step: int,
    processed_batches: int,
    optimizer_updates: int,
    elapsed_seconds: float,
    rank0_peak_allocated_mb: float,
    max_peak_allocated_mb: float,
    rank0_peak_reserved_mb: float,
    max_peak_reserved_mb: float,
    epoch_lr: float,
    epoch_source_lr: float | None,
    epoch_consistency_weight_lr: float | None = None,
) -> dict[str, float | int | str]:
    required = (
        "loss_total",
        "loss_diagonal",
        "loss_consistency",
        "consistency_effective_weight",
    )
    missing = [key for key in required if key not in reduced_epoch]
    if missing:
        raise KeyError(f"Missing required epoch summary metrics: {missing}")

    loss_total = float(reduced_epoch["loss_total"])
    loss_primary = float(reduced_epoch["loss_diagonal"])
    loss_consistency = float(reduced_epoch["loss_consistency"])
    consistency_weight = float(reduced_epoch["consistency_effective_weight"])
    loss_base = (
        primary_weight * loss_primary
        + consistency_weight * loss_consistency
    )
    elapsed = max(elapsed_seconds, 1.0e-12)
    report: dict[str, float | int | str] = {
        "epoch": epoch,
        "loss_avg": loss_total,
        "loss_base": loss_base,
        "inf": loss_primary,
        "distill": loss_consistency,
        "loss_total": loss_total,
        "loss_primary": loss_primary,
        "loss_consistency": loss_consistency,
        "distill_type": consistency_type,
        "consistency_loss_type": consistency_type,
        "consistency_weight": consistency_weight,
    }
    if consistency_type == "ecld":
        if "loss_ecld_ec" in reduced_epoch:
            report["ce_ec"] = float(reduced_epoch["loss_ecld_ec"])
        if "loss_ecld_td" in reduced_epoch:
            report["td"] = float(reduced_epoch["loss_ecld_td"])
    for output_key, metric_key in SOURCE_SUMMARY_ALIASES:
        if metric_key in reduced_epoch:
            report[output_key] = float(reduced_epoch[metric_key])
    report.update({
        "local_batch_size": local_batch_size,
        "global_batch_size": global_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": global_batch_size * grad_accum_steps,
        "optimizer_step": optimizer_step,
    })
    if "grad_norm" in reduced_epoch:
        report["grad_norm"] = float(reduced_epoch["grad_norm"])
    report.update({
        "rank0_peak_allocated_mb": rank0_peak_allocated_mb,
        "max_peak_allocated_mb": max_peak_allocated_mb,
        "rank0_peak_reserved_mb": rank0_peak_reserved_mb,
        "max_peak_reserved_mb": max_peak_reserved_mb,
        "images_per_second": (
            processed_batches * global_batch_size / elapsed
        ),
        "optimizer_updates_per_second": optimizer_updates / elapsed,
        "sec_per_batch": elapsed / max(processed_batches, 1),
        "lr": epoch_lr,
    })
    if epoch_source_lr is not None:
        report["source_lr"] = epoch_source_lr
    if epoch_consistency_weight_lr is not None:
        report["consistency_weight_lr"] = epoch_consistency_weight_lr
    for key in CONSISTENCY_SUMMARY_KEYS.get(consistency_type, ()):
        if key in reduced_epoch:
            report[key] = float(reduced_epoch[key])
    for key in DFM_RECIPE_SUMMARY_KEYS:
        source_key = (
            "gradient_surgery_conflict"
            if key == "gradient_surgery_conflict_fraction" else key
        )
        if source_key in reduced_epoch:
            report[key] = float(reduced_epoch[source_key])
    return report


def _format_epoch_summary(
    report: dict[str, float | int | str],
) -> str:
    fields = []
    integer_keys = {
        "epoch",
        "local_batch_size",
        "global_batch_size",
        "grad_accum_steps",
        "effective_batch_size",
        "optimizer_step",
    }
    memory_keys = {
        "rank0_peak_allocated_mb",
        "max_peak_allocated_mb",
        "rank0_peak_reserved_mb",
        "max_peak_reserved_mb",
    }
    rate_keys = {"images_per_second", "optimizer_updates_per_second"}
    for key, value in report.items():
        if isinstance(value, str):
            rendered = value
        elif key in integer_keys:
            rendered = str(int(value))
        elif key in {"lr", "source_lr", "consistency_weight_lr"}:
            rendered = f"{float(value):.8e}"
        elif key in memory_keys:
            rendered = f"{float(value):.1f}"
        elif key in rate_keys:
            rendered = f"{float(value):.3f}"
        else:
            rendered = f"{float(value):.6f}"
        fields.append(f"{key}:{rendered}")
    return " ".join(fields)


def _wandb_epoch_payload(
    report: dict[str, float | int | str],
    consistency_type: str,
) -> dict[str, float | int]:
    aliases = {
        "loss": "loss_avg",
        "loss_base": "loss_base",
        "loss_primary": "loss_primary",
        "loss_consistency": "loss_consistency",
        "loss_align": "loss_align",
        "weighted_align": "weighted_align",
        "lr": "lr",
        "source_lr": "source_lr",
        "consistency_weight_lr": "consistency_weight_lr",
        "images_per_second": "images_per_second",
        "optimizer_updates_per_second": "optimizer_updates_per_second",
        "grad_norm": "grad_norm",
        "epoch": "epoch",
    }
    payload = {
        f"epoch/{output_key}": report[report_key]
        for output_key, report_key in aliases.items()
        if report_key in report
    }
    for key in CONSISTENCY_SUMMARY_KEYS.get(consistency_type, ()):
        if key in report:
            payload[f"epoch/{key}"] = report[key]
    for key in DFM_RECIPE_SUMMARY_KEYS:
        if key in report:
            payload[f"epoch/{key}"] = report[key]
    return payload


class NullLogger:
    def info(self, *args, **kwargs) -> None:
        del args, kwargs

    def warning(self, *args, **kwargs) -> None:
        del args, kwargs


def log_esd_experiment_metadata(config: dict, logger) -> None:
    consistency = config["loss"]["consistency"]
    esd = consistency["esd"]
    precision = consistency["precision"]
    invalid = consistency["invalid_teacher"]
    logger.info("ESD formulation: %s", esd["formulation"])
    logger.info("ESD source: %s", esd["source"])
    logger.info(
        "ESD additional numerical safeguards: %s",
        str(esd["additional_numerical_safeguards"]).lower(),
    )
    logger.info("ESD invalid teacher strategy: %s", invalid["strategy"])
    logger.info("ESD JVP dtype: %s", precision["jvp_dtype"])
    logger.info("ESD numerical dtype: %s", precision["numerical_dtype"])


def build_optimizer(config: dict, adapter: DDPCompatibleTrainingModel):
    optimizer_config = config["training"]["optimizer"]
    model_lr = (
        optimizer_config["parameter_groups"]["model"]["lr"]
        or optimizer_config["lr"]
    )
    groups = [{
        "params": [
            parameter
            for parameter in adapter.endpoint_model.parameters()
            if parameter.requires_grad
        ],
        "lr": model_lr,
        "name": "model",
    }]
    if adapter.source_model is not None and not config["source"]["freeze"]:
        source_parameters = [
            parameter
            for parameter in adapter.source_model.parameters()
            if parameter.requires_grad
        ]
        if source_parameters:
            source_lr = (
                optimizer_config["parameter_groups"]["source"]["lr"]
                or optimizer_config["lr"]
            )
            groups.append({
                "params": source_parameters,
                "lr": source_lr,
                "name": "source",
            })
    if adapter.consistency_weight_model is not None:
        weight_group = config["loss"]["consistency"]["learnable_weight"]
        consistency_type = config["loss"]["consistency"]["type"]
        groups.append({
            "params": [
                parameter for parameter in adapter.consistency_weight_model.parameters()
                if parameter.requires_grad
            ],
            "lr": weight_group["lr"] or model_lr,
            "weight_decay": weight_group["weight_decay"],
            "name": "psd_weight" if consistency_type == "psd" else "esd_weight",
        })
    optimizer_class = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }.get(optimizer_config["name"])
    if optimizer_class is None:
        raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")
    return optimizer_class(
        groups,
        lr=optimizer_config["lr"],
        weight_decay=optimizer_config["weight_decay"],
        betas=tuple(optimizer_config["betas"]),
    )


class _RatioPreservingCosineAnnealingLR(
    torch.optim.lr_scheduler.CosineAnnealingLR
):
    """CFM cosine schedule with a shared multiplicative floor across groups."""

    def __init__(self, optimizer, T_max: int, eta_min: float):
        reference_base_lr = float(
            optimizer.param_groups[0].get(
                "initial_lr", optimizer.param_groups[0]["lr"]
            )
        )
        eta_ratio = eta_min / reference_base_lr
        self.eta_mins = [
            float(group.get("initial_lr", group["lr"])) * eta_ratio
            for group in optimizer.param_groups
        ]
        super().__init__(optimizer, T_max=T_max, eta_min=eta_min)

    def get_lr(self):
        if self._is_initial:
            return [group["lr"] for group in self.optimizer.param_groups]
        if self._step_count == 1 and self.last_epoch > 0:
            return self._get_closed_form_lr()
        if (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"]
                + (base_lr - eta_min)
                * (1 - math.cos(math.pi / self.T_max))
                / 2
                for base_lr, eta_min, group in zip(
                    self.base_lrs,
                    self.eta_mins,
                    self.optimizer.param_groups,
                    strict=True,
                )
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (group["lr"] - eta_min)
            + eta_min
            for eta_min, group in zip(
                self.eta_mins, self.optimizer.param_groups, strict=True
            )
        ]

    def _get_closed_form_lr(self):
        return [
            eta_min
            + (base_lr - eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for base_lr, eta_min in zip(
                self.base_lrs, self.eta_mins, strict=True
            )
        ]


def build_scheduler(config: dict, optimizer):
    scheduler_config = config["training"]["scheduler"]

    if scheduler_config["name"] == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if scheduler_config["name"] == "poly":
        maximum = config["training"]["max_optimizer_steps"]
        warmup = scheduler_config["warmup_steps"]
        start_factor = scheduler_config["warmup_start_factor"]
        power = scheduler_config["power"]
        reference_lr = float(optimizer.param_groups[0]["lr"])
        minimum_factor = scheduler_config["min_lr"] / reference_lr

        def factor(step: int) -> float:
            if warmup > 0 and step < warmup:
                return start_factor + (1.0 - start_factor) * step / warmup
            denominator = max(maximum - warmup, 1)
            progress = min(max((step - warmup) / denominator, 0.0), 1.0)
            return max((1.0 - progress) ** power, minimum_factor)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    if scheduler_config["name"] != "cosine":
        raise ValueError(f"Unknown scheduler: {scheduler_config['name']}")
    if scheduler_config["step_unit"] == "optimizer_step":
        total_units = config["training"]["max_optimizer_steps"]
        warmup_units = scheduler_config["warmup_steps"]
    else:
        total_units = config["training"]["epochs"]
        warmup_units = scheduler_config["warmup_epochs"]
    if warmup_units <= 0:
        return _RatioPreservingCosineAnnealingLR(
            optimizer,
            T_max=max(total_units, 1),
            eta_min=scheduler_config["eta_min"],
        )
    linear = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=scheduler_config["warmup_start_factor"],
        end_factor=1.0,
        total_iters=warmup_units,
    )
    cosine = _RatioPreservingCosineAnnealingLR(
        optimizer,
        T_max=max(total_units - warmup_units, 1),
        eta_min=scheduler_config["eta_min"],
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[linear, cosine],
        milestones=[warmup_units],
    )


def _build_loaders(
    config: dict,
    context: DistributedContext,
    local_batch_size: int,
):
    train_dataset = build_dataset(
        config, config["dataset"]["train_split"], augment=True
    )
    val_dataset = build_dataset(config, config["evaluation"]["split"], augment=False)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            drop_last=True,
        )
        if context.distributed else None
    )
    val_sampler = (
        DistributedEvalSampler(
            val_dataset, rank=context.rank, world_size=context.world_size
        )
        if context.distributed else None
    )
    workers = config["dataset"]["num_workers"]
    generator = torch.Generator()
    generator.manual_seed(config["experiment"]["seed"] + context.rank)
    common = {
        "num_workers": workers,
        "pin_memory": config["dataset"]["pin_memory"],
        "persistent_workers": (
            config["dataset"]["persistent_workers"] and workers > 0
        ),
        "worker_init_fn": seed_data_loader_worker,
        "generator": generator,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        **common,
    )
    evaluation_global_batch = config["evaluation"]["batch_size"]
    evaluation_local_batch = validate_global_batch_size(
        evaluation_global_batch, context.world_size
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=evaluation_local_batch,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
        collate_fn=(
            ade20k_eval_collate
            if (
                config["dataset"]["name"] == "ade20k"
                or config["evaluation"]["original_resolution"]
            ) else None
        ),
        **common,
    )
    return train_loader, val_loader, train_sampler


def _accumulate_entropy_percentiles(
    entropy: torch.Tensor,
    correct: torch.Tensor,
    valid: torch.Tensor,
    counts: torch.Tensor,
    correct_counts: torch.Tensor,
    entropy_sums: torch.Tensor,
) -> None:
    """Accumulate image-wise entropy percentile diagnostics on non-void GT."""
    bins = counts.numel()
    for sample_entropy, sample_correct, sample_valid in zip(
        entropy, correct, valid, strict=True
    ):
        values = sample_entropy[sample_valid].double()
        correctness = sample_correct[sample_valid].double()
        if not values.numel():
            continue
        order = torch.argsort(values, stable=True)
        bin_indices = torch.div(
            torch.arange(values.numel(), device=values.device) * bins,
            values.numel(), rounding_mode="floor",
        ).clamp_max(bins - 1)
        ones = torch.ones_like(values, dtype=counts.dtype)
        counts.scatter_add_(0, bin_indices, ones)
        correct_counts.scatter_add_(0, bin_indices, correctness.to(counts.dtype)[order])
        entropy_sums.scatter_add_(0, bin_indices, values[order])


@torch.no_grad()
def validate_source_only(
    config: dict,
    training_model,
    loader,
    context: DistributedContext,
    output_dir: Path,
) -> dict:
    """Validate the source mean directly, without endpoint or Flow Map inference."""
    adapter = unwrap_model(training_model)
    endpoint, source = adapter.endpoint_model, adapter.source_model
    if source is None:
        raise RuntimeError("source-only validation requires a source model")
    endpoint.eval()
    source.eval()
    classes = config["dataset"]["num_classes"]
    void_index = config["dataset"]["void_class_index"]
    semantic_metrics = SegmentationMetrics(
        classes, void_index, device=context.device,
        evaluated_class_indices=[
            index for index in range(classes) if index != void_index
        ],
        prediction_void_retained=True,
    )
    full_confusion = torch.zeros(
        classes, classes, dtype=torch.int64, device=context.device
    )
    diagnostic_config = config["source"]["diagnostics"]
    bins = int(diagnostic_config["entropy_bins"])
    bin_counts = torch.zeros(bins, dtype=torch.float64, device=context.device)
    bin_correct = torch.zeros_like(bin_counts)
    bin_entropy = torch.zeros_like(bin_counts)
    correct_entropy = torch.zeros(2, dtype=torch.float64, device=context.device)
    incorrect_entropy = torch.zeros(2, dtype=torch.float64, device=context.device)
    visualized = 0
    maximum_batches = config["evaluation"]["max_batches"]
    representation = (
        config["source"]["representation"]
        if config["source"].get("type") == "task_finetuned_segformer"
        else "logits"
    )

    for batch_index, batch in enumerate(loader):
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
        samples = batch if isinstance(batch, list) else [
            {"image": image, "target": target}
            for image, target in zip(*batch, strict=True)
        ]
        for sample in samples:
            image = sample["image"].unsqueeze(0).to(context.device, non_blocking=True)
            target = sample["target"].unsqueeze(0).to(context.device, non_blocking=True)
            with autocast_context(config, context.device):
                mean, _ = source_statistics(source, image)
            entropy_state = shannon_entropy(
                mean, representation=representation,
                eps=config["flow"]["probability_eps"],
            )
            if "original_shape" in sample:
                mean_full = state_to_original_continuous(
                    mean.float(), sample["model_shape"], sample["original_shape"],
                    padded_shape=sample["padded_shape"],
                    align_corners=config["evaluation"]["align_corners"],
                )
                entropy_full = state_to_original_continuous(
                    entropy_state[:, None], sample["model_shape"],
                    sample["original_shape"], padded_shape=sample["padded_shape"],
                    align_corners=config["evaluation"]["align_corners"],
                )[:, 0]
                model_height, model_width = (int(v) for v in sample["model_shape"])
                display_image = F.interpolate(
                    image[..., :model_height, :model_width].float(),
                    size=target.shape[-2:], mode="bilinear", align_corners=False,
                )[0]
            else:
                mean_full = F.interpolate(
                    mean.float(), target.shape[-2:], mode="bilinear", align_corners=False
                )
                entropy_full = F.interpolate(
                    entropy_state[:, None], target.shape[-2:],
                    mode="bilinear", align_corners=False,
                )[:, 0]
                display_image = image[0]
            prediction = mean_full.argmax(dim=1)
            semantic_metrics.update(prediction, target)
            valid_all = (
                (target >= 0) & (target < classes)
                & (prediction >= 0) & (prediction < classes)
            )
            indices = target[valid_all] * classes + prediction[valid_all]
            full_confusion += torch.bincount(
                indices, minlength=classes**2
            ).reshape(classes, classes)
            semantic_valid = valid_all & (target != void_index)
            correct = prediction == target
            if diagnostic_config["enabled"]:
                _accumulate_entropy_percentiles(
                    entropy_full, correct, semantic_valid,
                    bin_counts, bin_correct, bin_entropy,
                )
                correct_values = entropy_full[semantic_valid & correct].double()
                incorrect_values = entropy_full[semantic_valid & ~correct].double()
                correct_entropy[0] += correct_values.sum()
                correct_entropy[1] += correct_values.numel()
                incorrect_entropy[0] += incorrect_values.sum()
                incorrect_entropy[1] += incorrect_values.numel()
            if (
                context.is_main_process
                and diagnostic_config["enabled"]
                and diagnostic_config["visualization"]
                and visualized < diagnostic_config["max_visualizations"]
            ):
                save_source_diagnostics(
                    display_image, target[0], prediction[0], entropy_full[0],
                    output_dir / "source_diagnostics"
                    / f"source_val_{visualized:04d}.png",
                    num_classes=classes,
                    imagenet_normalize=(
                        config["augmentation"]["imagenet_normalize"]
                        or config["augmentation"]["normalize"]["enabled"]
                    ),
                    dataset_name=config["dataset"]["name"],
                )
                visualized += 1

    semantic_metrics.confusion_matrix = all_reduce_confusion_matrix(
        semantic_metrics.confusion_matrix, context
    )
    full_confusion = all_reduce_confusion_matrix(full_confusion, context)
    if context.distributed:
        for tensor in (
            bin_counts, bin_correct, bin_entropy,
            correct_entropy, incorrect_entropy,
        ):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    semantic = semantic_metrics.compute()
    true_positive = full_confusion.diag().double()
    gt_count = full_confusion.sum(dim=1).double()
    predicted_count = full_confusion.sum(dim=0).double()
    total = full_confusion.sum().double().clamp_min(1.0)
    void_union = gt_count[void_index] + predicted_count[void_index] - true_positive[void_index]
    result = {
        **semantic,
        "source_mIoU": semantic["mIoU"],
        "source_pixel_acc": semantic["pixel_acc"],
        "source_mAcc": semantic["mAcc"],
        "source_void_iou": float(true_positive[void_index] / void_union.clamp_min(1.0)),
        "source_predicted_void_ratio": float(predicted_count[void_index] / total),
        "source_gt_void_ratio": float(gt_count[void_index] / total),
        "source_void_precision": float(
            true_positive[void_index] / predicted_count[void_index].clamp_min(1.0)
        ),
        "source_void_recall": float(
            true_positive[void_index] / gt_count[void_index].clamp_min(1.0)
        ),
        "source_entropy_correct_mean": float(
            correct_entropy[0] / correct_entropy[1].clamp_min(1.0)
        ),
        "source_entropy_incorrect_mean": float(
            incorrect_entropy[0] / incorrect_entropy[1].clamp_min(1.0)
        ),
    }
    for index in range(bins):
        count = bin_counts[index].clamp_min(1.0)
        accuracy = bin_correct[index] / count
        has_values = bool(bin_counts[index] > 0)
        result[f"source_entropy_bin_{index}_count"] = int(bin_counts[index])
        result[f"source_entropy_bin_{index}_accuracy"] = float(accuracy)
        result[f"source_entropy_bin_{index}_error_rate"] = float(
            1.0 - accuracy if has_values else 0.0
        )
        result[f"source_entropy_bin_{index}_mean"] = float(bin_entropy[index] / count)
    endpoint.train(config["training"].get("train_endpoint", True))
    source.train(not config["source"]["freeze"])
    return result


@torch.no_grad()
def validate(
    config: dict,
    training_model,
    loader,
    context: DistributedContext,
    output_dir: Path,
) -> dict:
    if config["evaluation"].get("source_only", False):
        return validate_source_only(
            config, training_model, loader, context, output_dir
        )
    adapter = unwrap_model(training_model)
    endpoint = adapter.endpoint_model
    source = adapter.source_model
    endpoint.eval()
    if source is not None:
        source.eval()
    eval_range = config["evaluation"]["eval_class_indices"]
    evaluated = (
        range(eval_range[0], eval_range[1] + 1) if eval_range is not None else None
    )
    metrics = SegmentationMetrics(
        config["dataset"]["num_classes"],
        config["dataset"]["void_class_index"],
        device=context.device,
        evaluated_class_indices=evaluated,
        nanmean=config["evaluation"]["nanmean"],
        prediction_void_retained=not config["evaluation"][
            "exclude_void_from_prediction"
        ],
    )
    visualized = 0
    maximum_batches = config["evaluation"]["max_batches"]
    for batch_index, batch in enumerate(loader):
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
        if (
            config["dataset"]["name"] == "ade20k"
            or config["evaluation"]["original_resolution"]
        ):
            visualization_items = []
            for sample in batch:
                image = sample["image"].unsqueeze(0).to(
                    context.device, non_blocking=True
                )
                target = sample["target"].unsqueeze(0).to(
                    context.device, non_blocking=True
                )
                with autocast_context(config, context.device):
                    terminal = sample_segmentation(
                        endpoint, source, image, config, return_terminal_state=True
                    )
                prediction = terminal_state_to_original_prediction(
                    terminal,
                    sample["model_shape"],
                    sample["original_shape"],
                    padded_shape=sample["padded_shape"],
                    align_corners=config["evaluation"]["align_corners"],
                    void_class_index=config["dataset"]["void_class_index"],
                    exclude_void=config["evaluation"][
                        "exclude_void_from_prediction"
                    ],
                )
                metrics.update(prediction, target)
                visualization_items.append((image, target, prediction))
        else:
            image, target = batch
            image = image.to(context.device, non_blocking=True)
            target = target.to(context.device, non_blocking=True)
            with autocast_context(config, context.device):
                prediction = sample_segmentation(endpoint, source, image, config)
            metrics.update(prediction, target)
            visualization_items = [(image, target, prediction)]
        if context.is_main_process:
            remaining = config["evaluation"]["max_visualizations"] - visualized
            for item_index, (images, targets, predictions) in enumerate(visualization_items):
                for sample_index in range(min(images.shape[0], max(remaining, 0))):
                    visualization_image = images[sample_index]
                    if visualization_image.shape[-2:] != targets[sample_index].shape:
                        visualization_image = torch.nn.functional.interpolate(
                            visualization_image[None].float(),
                            size=targets[sample_index].shape,
                            mode="bilinear",
                            align_corners=False,
                        )[0]
                    save_prediction(
                        visualization_image, targets[sample_index], predictions[sample_index],
                        output_dir / "visualizations"
                        / f"val_{batch_index:04d}_{item_index:02d}_{sample_index:02d}.png",
                        (
                            config["augmentation"]["imagenet_normalize"]
                            or config["augmentation"]["normalize"]["enabled"]
                        ),
                        dataset_name=config["dataset"]["name"],
                    )
                    visualized += 1
                    remaining -= 1
    metrics.confusion_matrix = all_reduce_confusion_matrix(
        metrics.confusion_matrix, context
    )
    result = metrics.compute()
    endpoint.train()
    if source is not None:
        source.train(not config["source"]["freeze"])
    return result


def _operation_for_stage(stage: str) -> str:
    if stage == "diagonal_pretrain":
        return "stage1_objectives"
    if stage in {"consistency_distillation", "esd_distillation"}:
        return "stage2_objectives"
    if stage == "joint_training":
        return "joint_objectives"
    raise ValueError(f"Unknown training stage: {stage}")


def _distributed_metadata(
    context: DistributedContext,
    global_batch_size: int,
    local_batch_size: int,
) -> dict:
    return {
        "world_size": context.world_size,
        "global_batch_size": global_batch_size,
        "local_batch_size": local_batch_size,
    }


def _save_training_checkpoint(
    *,
    config: dict,
    training_model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    micro_step: int,
    metrics: dict,
    context: DistributedContext,
    output_dir: Path,
    filenames: list[str],
    global_batch_size: int,
    local_batch_size: int,
) -> None:
    barrier(context)
    if context.is_main_process:
        adapter = unwrap_model(training_model)
        payload = checkpoint_payload(
            config=config,
            epoch=epoch,
            global_step=global_step,
            micro_step=micro_step,
            model=adapter.endpoint_model,
            source_model=adapter.source_model,
            consistency_weight_model=adapter.consistency_weight_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metrics=metrics,
            distributed=_distributed_metadata(
                context, global_batch_size, local_batch_size
            ),
        )
        for filename in filenames:
            save_checkpoint(payload, output_dir, filename)
    barrier(context)


def run_training(config: dict, *, joint_entrypoint: bool = False) -> dict:
    stage = config["experiment"]["stage"]
    if joint_entrypoint and stage != "joint_training":
        raise ValueError("train_joint.py requires experiment.stage=joint_training")
    if not joint_entrypoint and stage == "joint_training":
        raise ValueError("joint_training must use src/train_joint.py")
    if config["runtime"]["compile"]:
        raise ValueError("runtime.compile is not supported by the DDP/JVP composite trainer")

    context = setup_distributed(config)
    output_dir = Path(config["experiment"]["output_dir"])
    logger = NullLogger()
    wandb_run = None
    try:
        assert_config_equal_across_ranks(config, context)
        global_batch_size = config["training"]["batch_size"]
        local_batch_size = validate_global_batch_size(
            global_batch_size, context.world_size
        )
        effective_global_batch_size = (
            global_batch_size * config["training"]["grad_accum_steps"]
        )
        if (
            config["loss"]["consistency"]["enabled"]
            and config["loss"]["consistency"]["precision"]["jvp_dtype"] == "bf16"
            and context.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("bf16 JVP was requested but this CUDA device lacks bf16")

        if context.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_resolved_config(config, output_dir / "config_resolved.yaml")
            logger = setup_logger(output_dir)
        barrier(context)
        if (
            config["loss"]["primary"]["adaptive_weighting"]["enabled"]
            and config["training"]["label_smoothing"] != 0.0
        ):
            logger.warning(
                "Adaptive diagonal weighting uses hard one-hot targets while CE "
                "label_smoothing=%s; label_smoothing=0 is the closest paper recipe.",
                config["training"]["label_smoothing"],
            )
        logger.info(
            "world_size=%d rank=%d local_rank=%d global_batch_size=%d "
            "local_batch_size=%d grad_accum_steps=%d effective_global_batch_size=%d",
            context.world_size, context.rank, context.local_rank,
            global_batch_size, local_batch_size,
            config["training"]["grad_accum_steps"], effective_global_batch_size,
        )
        logger.info("Local batch size: %d", local_batch_size)
        logger.info("World size: %d", context.world_size)
        logger.info("Global physical batch size: %d", global_batch_size)
        logger.info("Gradient accumulation: %d", config["training"]["grad_accum_steps"])
        logger.info("Effective batch size: %d", effective_global_batch_size)
        logger.info(
            "Training schedule: max_optimizer_steps=%s scheduler=%s/%s "
            "lr_warmup_steps=%s validation_interval=%s/%s "
            "checkpoint_interval_steps=%s consistency_start=%s/%s "
            "consistency_warmup_steps=%s",
            config["training"]["max_optimizer_steps"],
            config["training"]["scheduler"]["name"],
            config["training"]["scheduler"]["step_unit"],
            config["training"]["scheduler"]["warmup_steps"],
            config["evaluation"]["interval"]["unit"],
            config["evaluation"]["interval"]["value"],
            config["training"]["checkpoint_interval_steps"],
            config["loss"]["consistency"]["start"]["unit"],
            config["loss"]["consistency"]["start"]["value"],
            config["loss"]["consistency"]["warmup_steps"],
        )
        if config["source"]["prior_type"] == "image_simplex_mixture":
            simplex = config["source"]["simplex_prior"]["training"]
            logger.info(
                "Simplex source prior: mode=training lambda=%s temperature=%s "
                "dirichlet_alpha=%s",
                simplex["lambda"], simplex["temperature"],
                simplex["dirichlet_alpha"],
            )
        if (
            context.is_main_process
            and stage in {
                "consistency_distillation",
                "esd_distillation",
                "joint_training",
            }
            and config["loss"]["consistency"]["type"] == "esd"
        ):
            log_esd_experiment_metadata(config, logger)

        seed_everything(
            config["experiment"]["seed"], config["runtime"]["deterministic"]
        )
        train_loader, val_loader, train_sampler = _build_loaders(
            config, context, local_batch_size
        )
        endpoint, source = build_models(config, context.device)
        if context.is_main_process and source is not None:
            from model_inspection import inspect_source_parameters
            parameter_report = inspect_source_parameters(source)
            source_decoder = parameter_report["source_decoder"]
            source_projections = parameter_report["source_projections"]
            logger.info(
                "Source parameters: total=%d trainable=%d encoder=%d/%d "
                "decoder_or_head=%d/%d",
                parameter_report["source"]["total"],
                parameter_report["source"]["trainable"],
                parameter_report["source_encoder"]["trainable"],
                parameter_report["source_encoder"]["total"],
                source_decoder["trainable"] + source_projections["trainable"],
                source_decoder["total"] + source_projections["total"],
            )
        adapter = DDPCompatibleTrainingModel(endpoint, source, config).to(context.device)
        optimizer = build_optimizer(config, adapter)
        max_iterations = config["training"]["max_iterations"]
        max_optimizer_steps = config["training"]["max_optimizer_steps"]
        scheduler = build_scheduler(config, optimizer)
        scaler = build_grad_scaler(config, context.device)
        state = initialize_or_resume(
            config, endpoint, source, optimizer, scheduler, scaler,
            logger if context.is_main_process else None,
            consistency_weight_model=adapter.consistency_weight_model,
        )
        training_model = wrap_ddp(adapter, context, config)
        # Model initialization/checkpoint loading used the same seed. From here on,
        # rank-local stochastic paths intentionally differ.
        torch.manual_seed(config["experiment"]["seed"] + context.rank)
        if context.device.type == "cuda":
            torch.cuda.manual_seed_all(config["experiment"]["seed"] + context.rank)
            torch.cuda.reset_peak_memory_stats(context.device)
        if context.is_main_process:
            wandb_run = init_wandb(config)

        training = config["training"]
        consistency_config = config["loss"]["consistency"]
        surgery_config = consistency_config["gradient_surgery"]
        surgery_enabled = surgery_config["enabled"]
        surgery_accumulator = (
            GradientSurgeryAccumulator() if surgery_enabled else None
        )
        numbered_checkpoint_epochs = _numbered_checkpoint_epochs(
            total_epochs=training["epochs"],
            checkpoint_interval=training["checkpoint_interval_epochs"],
            joint_entrypoint=joint_entrypoint,
            consistency_enabled=consistency_config["enabled"],
            consistency_start_epoch=consistency_config["start_epoch"],
        )
        operation = _operation_for_stage(stage)
        metrics_path = output_dir / "metrics.jsonl"
        total_iterations = getattr(state, "micro_step", 0)
        last_metrics: dict = {"best_mIoU": state.best_miou}
        last_epoch_report: dict[str, float | int | str] = {}
        validated_optimizer_steps: set[int] = set()
        best_checkpoint_steps: set[int] = set()
        latest_checkpoint_steps: set[int] = set()

        def assert_checkpointable_surgery_state() -> None:
            if surgery_accumulator is not None and not surgery_accumulator.is_empty:
                raise RuntimeError(
                    "refusing to checkpoint a partially accumulated gradient surgery window"
                )

        def run_validation(
            displayed_epoch: int,
            trigger: str,
            *,
            checkpoint_epoch: int,
            checkpoint_micro_step: int,
        ) -> dict:
            previous_best = state.best_miou
            result = validate(
                config, training_model, val_loader, context, output_dir
            )
            is_new_best = result["mIoU"] > previous_best
            if is_new_best:
                state.best_miou = result["mIoU"]
            validated_optimizer_steps.add(state.global_step)
            if context.is_main_process:
                append_jsonl(metrics_path, {
                    "scope": "validation",
                    "epoch": displayed_epoch,
                    "optimizer_step": state.global_step,
                    "trigger": trigger,
                    **result,
                })
                logger.info(
                    "validation epoch=%d optimizer_step=%d trigger=%s "
                    "mIoU=%.6g pixel_acc=%.6g mAcc=%.6g",
                    displayed_epoch, state.global_step, trigger,
                    result["mIoU"], result["pixel_acc"], result["mAcc"],
                )
                if wandb_run is not None:
                    wandb_payload = {
                        "validation/mIoU": result["mIoU"],
                        "validation/pixel_acc": result["pixel_acc"],
                        "validation/mAcc": result["mAcc"],
                    }
                    if config["evaluation"].get("source_only", False):
                        wandb_payload.update({
                            f"validation/{key}": value
                            for key, value in result.items()
                            if key.startswith("source_")
                            and isinstance(value, (int, float))
                        })
                    wandb_run.log(wandb_payload, step=state.global_step)
            if is_new_best and trigger in {
                "optimizer_step_interval", "final_optimizer_step"
            }:
                assert_checkpointable_surgery_state()
                checkpoint_metrics = {
                    **result,
                    "best_mIoU": state.best_miou,
                    "optimizer_step": state.global_step,
                }
                _save_training_checkpoint(
                    config=config,
                    training_model=training_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=checkpoint_epoch,
                    global_step=state.global_step,
                    micro_step=checkpoint_micro_step,
                    metrics=checkpoint_metrics,
                    context=context,
                    output_dir=output_dir,
                    filenames=["best.pt"],
                    global_batch_size=global_batch_size,
                    local_batch_size=local_batch_size,
                )
                best_checkpoint_steps.add(state.global_step)
            return result

        optimizer.zero_grad(set_to_none=True)

        for epoch_index in range(state.start_epoch, training["epochs"]):
            if (
                max_optimizer_steps is not None
                and state.global_step >= max_optimizer_steps
            ):
                break
            training_model.train()
            adapter = unwrap_model(training_model)
            if config["source"]["freeze"] and adapter.source_model is not None:
                adapter.source_model.eval()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch_index)
            epoch_meter = EpochMetricMeter(
                min_keys=MIN_REDUCTION_KEYS,
                max_keys=MAX_REDUCTION_KEYS,
            )
            epoch_lr = float(optimizer.param_groups[0]["lr"])
            epoch_source_lr = _parameter_group_lr(optimizer, "source")
            epoch_consistency_weight_lr = _parameter_group_lr(
                optimizer, "psd_weight"
            )
            if epoch_consistency_weight_lr is None:
                epoch_consistency_weight_lr = _parameter_group_lr(
                    optimizer, "esd_weight"
                )
            epoch_total_iterations = _epoch_total_iterations(
                len(train_loader),
                max_iterations,
                total_iterations,
                training["max_batches_per_epoch"],
                max_optimizer_steps,
                state.global_step,
                training["grad_accum_steps"],
            )
            if epoch_total_iterations <= 0:
                break
            processed_batches = 0
            optimizer_updates_in_epoch = 0
            validation_metrics = None
            validation_step = None
            epoch_start_time = time.perf_counter()
            progress = _create_epoch_progress(
                epoch_index=epoch_index,
                total=epoch_total_iterations,
                is_main_process=context.is_main_process,
            )
            try:
                for batch_index, (image, target) in enumerate(train_loader):
                    if batch_index >= epoch_total_iterations:
                        break
                    image = image.to(context.device, non_blocking=True)
                    target = target.to(context.device, non_blocking=True)
                    reaches_limit = (
                        max_iterations is not None
                        and total_iterations + 1 >= max_iterations
                    )
                    should_step = (
                        (batch_index + 1) % training["grad_accum_steps"] == 0
                        or batch_index + 1 == epoch_total_iterations
                        or reaches_limit
                    )
                    sync_context = (
                        training_model.no_sync()
                        if context.distributed and not should_step and not surgery_enabled
                        else nullcontext()
                    )
                    with sync_context:
                        with autocast_context(config, context.device):
                            objectives = run_model_training_objectives(
                                adapter if surgery_enabled else training_model,
                                operation=operation,
                                image=image,
                                target=target,
                                epoch_index=epoch_index,
                                progress_in_epoch=(
                                    batch_index / max(len(train_loader), 1)
                                ),
                                optimizer_step=state.global_step,
                            )
                            scaled_loss = (
                                objectives["loss"] / training["grad_accum_steps"]
                            )
                        if surgery_enabled:
                            assert surgery_accumulator is not None
                            surgery_accumulator.accumulate(
                                adapter=adapter,
                                objectives=objectives,
                                scaler=scaler,
                            )
                        else:
                            scaler.scale(scaled_loss).backward()

                    if (
                        context.is_main_process
                        and batch_index == 0
                        and "path_debug" in objectives
                    ):
                        save_adaptive_path_debug(
                            image[0],
                            target[0],
                            objectives["path_debug"],
                            output_dir / "adaptive_path"
                            / f"epoch_{epoch_index + 1:04d}.png",
                            imagenet_normalize=(
                                config["augmentation"]["imagenet_normalize"]
                                or config["augmentation"]["normalize"]["enabled"]
                            ),
                            dataset_name=config["dataset"]["name"],
                            num_classes=config["dataset"]["num_classes"],
                        )

                    grad_norm = None
                    if should_step:
                        if surgery_enabled:
                            assert surgery_accumulator is not None
                            accumulated_microbatches = (
                                surgery_accumulator.microbatch_count
                            )
                            surgery_stats = surgery_accumulator.finalize(
                                adapter=adapter,
                                scaler=scaler,
                                world_size=context.world_size,
                                eps=surgery_config["eps"],
                            )
                            surgery_stats["gradient_surgery_partial_accumulation"] = (
                                torch.tensor(
                                    float(
                                        accumulated_microbatches
                                        < training["grad_accum_steps"]
                                    ),
                                    device=context.device,
                                )
                            )
                            objectives["stats"].update(surgery_stats)
                        scaler.unscale_(optimizer)
                        if training["grad_clip"] is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                adapter.parameters(), training["grad_clip"]
                            )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        state.global_step += 1
                        optimizer_updates_in_epoch += 1
                        if training["scheduler"]["step_unit"] == "optimizer_step":
                            scheduler.step()
                        numbered_step_due = _optimizer_step_checkpoint_due(
                            config, state.global_step
                        )
                        validation_trigger = _optimizer_step_validation_trigger(
                            config, state.global_step
                        )
                        if (
                            validation_trigger is not None
                            and state.global_step not in validated_optimizer_steps
                        ):
                            validation_metrics = run_validation(
                                epoch_index + 1,
                                validation_trigger,
                                checkpoint_epoch=epoch_index,
                                checkpoint_micro_step=total_iterations + 1,
                            )
                            validation_step = state.global_step
                        if validation_trigger is not None or numbered_step_due:
                            assert_checkpointable_surgery_state()
                            checkpoint_metrics = {
                                **(
                                    validation_metrics
                                    if validation_step == state.global_step
                                    else {}
                                ),
                                "best_mIoU": state.best_miou,
                                "optimizer_step": state.global_step,
                            }
                            filenames = ["latest.pt"]
                            if numbered_step_due:
                                filenames.append(
                                    f"step_{state.global_step:06d}.pt"
                                )
                            _save_training_checkpoint(
                                config=config,
                                training_model=training_model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=scaler,
                                epoch=epoch_index,
                                global_step=state.global_step,
                                micro_step=total_iterations + 1,
                                metrics=checkpoint_metrics,
                                context=context,
                                output_dir=output_dir,
                                filenames=filenames,
                                global_batch_size=global_batch_size,
                                local_batch_size=local_batch_size,
                            )
                            latest_checkpoint_steps.add(state.global_step)
                    total_iterations += 1
                    processed_batches += 1
                    batch_stats = dict(objectives["stats"])
                    if grad_norm is not None:
                        batch_stats["grad_norm"] = grad_norm.detach()
                    epoch_meter.update(batch_stats)

                    if context.is_main_process:
                        progress.update(1)
            finally:
                if progress is not None:
                    progress.close()

            if context.device.type == "cuda":
                torch.cuda.synchronize(context.device)
                local_peak_allocated_mb = (
                    torch.cuda.max_memory_allocated(context.device) / 1024**2
                )
                local_peak_reserved_mb = (
                    torch.cuda.max_memory_reserved(context.device) / 1024**2
                )
            else:
                local_peak_allocated_mb = 0.0
                local_peak_reserved_mb = 0.0
            local_elapsed_seconds = time.perf_counter() - epoch_start_time
            runtime_maxima = reduce_max_values(
                [
                    local_elapsed_seconds,
                    local_peak_allocated_mb,
                    local_peak_reserved_mb,
                ],
                context,
            )
            max_elapsed_seconds = float(runtime_maxima[0].cpu())
            max_peak_allocated_mb = float(runtime_maxima[1].cpu())
            max_peak_reserved_mb = float(runtime_maxima[2].cpu())
            reduced_epoch = reduce_epoch_metric_meter(epoch_meter, context)
            consistency_type = (
                "none"
                if stage == "diagonal_pretrain"
                else config["loss"]["consistency"]["type"]
            )
            epoch_report = _build_epoch_report(
                epoch=epoch_index + 1,
                reduced_epoch=reduced_epoch,
                consistency_type=consistency_type,
                primary_weight=config["loss"]["primary"]["weight"],
                local_batch_size=local_batch_size,
                global_batch_size=global_batch_size,
                grad_accum_steps=training["grad_accum_steps"],
                optimizer_step=state.global_step,
                processed_batches=processed_batches,
                optimizer_updates=optimizer_updates_in_epoch,
                elapsed_seconds=max_elapsed_seconds,
                rank0_peak_allocated_mb=local_peak_allocated_mb,
                max_peak_allocated_mb=max_peak_allocated_mb,
                rank0_peak_reserved_mb=local_peak_reserved_mb,
                max_peak_reserved_mb=max_peak_reserved_mb,
                epoch_lr=epoch_lr,
                epoch_source_lr=epoch_source_lr,
                epoch_consistency_weight_lr=epoch_consistency_weight_lr,
            )
            raw_epoch_metrics = {
                key: float(value) for key, value in reduced_epoch.items()
            }
            if context.is_main_process:
                append_jsonl(metrics_path, {
                    "scope": "epoch",
                    "stage": stage,
                    "consistency_type": consistency_type,
                    **epoch_report,
                    **{
                        key: value
                        for key, value in raw_epoch_metrics.items()
                        if key not in epoch_report
                    },
                })
                summary = _format_epoch_summary(epoch_report)
                tqdm.write(summary)
                logger.info(summary, extra={"console": False})
                if wandb_run is not None:
                    wandb_run.log(
                        _wandb_epoch_payload(epoch_report, consistency_type),
                        step=state.global_step,
                    )
            if training["scheduler"]["step_unit"] == "epoch":
                scheduler.step()
            last_epoch_report = epoch_report

            displayed_epoch = epoch_index + 1
            evaluation_interval = config["evaluation"]["interval"]
            interval_epoch_due = (
                evaluation_interval["unit"] == "epoch"
                and evaluation_interval["value"] is not None
                and displayed_epoch % evaluation_interval["value"] == 0
            )
            configured_epoch_due = (
                displayed_epoch in set(training["validation_epochs"])
                or interval_epoch_due
            )
            reached_final_optimizer_step = (
                max_optimizer_steps is not None
                and state.global_step >= max_optimizer_steps
            )
            if (
                (configured_epoch_due or reached_final_optimizer_step)
                and state.global_step not in validated_optimizer_steps
            ):
                trigger = (
                    "final_optimizer_step"
                    if reached_final_optimizer_step else "epoch"
                )
                validation_metrics = run_validation(
                    displayed_epoch,
                    trigger,
                    checkpoint_epoch=displayed_epoch,
                    checkpoint_micro_step=total_iterations,
                )
                validation_step = state.global_step

            last_metrics = {
                **raw_epoch_metrics,
                **epoch_report,
                **(validation_metrics or {}),
                "best_mIoU": state.best_miou,
            }
            filenames = []
            if state.global_step not in latest_checkpoint_steps:
                filenames.append("latest.pt")
            if displayed_epoch in numbered_checkpoint_epochs:
                filenames.append(f"epoch_{epoch_index + 1:04d}.pt")
            if (
                validation_metrics is not None
                and validation_step == state.global_step
                and state.global_step not in best_checkpoint_steps
                and validation_metrics["mIoU"] >= state.best_miou
            ):
                filenames.append("best.pt")
            if filenames:
                assert_checkpointable_surgery_state()
                _save_training_checkpoint(
                    config=config,
                    training_model=training_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch_index + 1,
                    global_step=state.global_step,
                    micro_step=total_iterations,
                    metrics=last_metrics,
                    context=context,
                    output_dir=output_dir,
                    filenames=filenames,
                    global_batch_size=global_batch_size,
                    local_batch_size=local_batch_size,
                )
                if "latest.pt" in filenames:
                    latest_checkpoint_steps.add(state.global_step)
            if max_iterations is not None and total_iterations >= max_iterations:
                break
            if (
                max_optimizer_steps is not None
                and state.global_step >= max_optimizer_steps
            ):
                break

        checksum_stats = parameter_checksum(unwrap_model(training_model), context)
        local_peak = (
            torch.cuda.max_memory_allocated(context.device) / 1024**2
            if context.device.type == "cuda" else 0.0
        )
        mean_peak = float(reduce_scalar(local_peak, context, "mean").cpu())
        max_peak = float(reduce_scalar(local_peak, context, "max").cpu())
        rank_peaks: list[float] = [local_peak]
        if context.distributed:
            gathered = [None] * context.world_size
            dist.all_gather_object(gathered, local_peak)
            rank_peaks = [float(value) for value in gathered]
        runtime_stats = {
            "runtime/world_size": context.world_size,
            "runtime/global_batch_size": global_batch_size,
            "runtime/local_batch_size": local_batch_size,
            "runtime/effective_global_batch_size": effective_global_batch_size,
            "runtime/iteration_time": float(
                last_epoch_report.get("sec_per_batch", 0.0)
            ),
            "runtime/samples_per_second": float(
                last_epoch_report.get("images_per_second", 0.0)
            ),
            "runtime/peak_gpu_memory_mb": mean_peak,
            "runtime/max_peak_gpu_memory_mb_across_ranks": max_peak,
            "runtime/peak_gpu_memory_mb_by_rank": rank_peaks,
            **checksum_stats,
        }
        if context.is_main_process:
            append_jsonl(metrics_path, {"scope": "runtime", **runtime_stats})
            logger.info(
                "Peak GPU allocated memory by rank=%s; max=%.2f MiB; checksum_diff=%.3g",
                rank_peaks, max_peak, checksum_stats["checksum_max_diff"],
            )
        last_metrics.update(runtime_stats)
        return last_metrics
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        cleanup_distributed(context)
