from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

import losses
from adaptive_path import (
    adaptive_path_stats,
    entropy_adaptive_enabled,
    source_entropy_difficulty,
    source_predicted_semantic_mask,
)
from dfm_stabilization import (
    ESDTimeWeightNetwork,
    PSDTimeWeightNetwork,
    esd_multiplier_bucket_stats,
    psd_multiplier_bucket_stats,
    uncertainty_weighted_esd_loss,
    uncertainty_weighted_psd_loss,
)
from discrete_flow_maps import (
    linear_path,
    path_coefficient,
    sample_consistency_times,
    sample_prior,
    sample_stage1_times,
)
from state_space import (
    prepare_state_targets,
    resize_continuous,
    state_spatial_size,
    target_state_from_config,
)


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.float().sum() * 0.0


@dataclass(frozen=True)
class PSDValidMasks:
    semantic_full: torch.Tensor
    semantic_state: torch.Tensor
    spatial_full: torch.Tensor
    spatial_state: torch.Tensor
    psd_full: torch.Tensor
    psd_state: torch.Tensor


def build_psd_valid_masks(
    targets,
    *,
    void_class_index: int | None,
    ignore_void: bool,
) -> PSDValidMasks:
    """Separate semantic void masking from artificial-padding masking."""
    if not isinstance(ignore_void, bool):
        raise ValueError("PSD ignore_void must be a boolean")
    if void_class_index is None:
        semantic_full = (
            targets.valid_mask_full
            if targets.valid_mask_full is not None
            else torch.ones_like(targets.target_full, dtype=torch.bool)
        )
        semantic_state = (
            targets.valid_mask_state
            if targets.valid_mask_state is not None
            else torch.ones_like(targets.target_state, dtype=torch.bool)
        )
    else:
        semantic_full = targets.target_full != void_class_index
        semantic_state = targets.target_state != void_class_index
    spatial_full = targets.spatial_valid_mask_full
    spatial_state = targets.spatial_valid_mask_state
    if spatial_full is None or spatial_state is None:
        if not ignore_void:
            raise ValueError(
                "PSD ignore_void=false requires an explicit spatial valid mask "
                "so real void and artificial padding remain distinguishable"
            )
        spatial_full = torch.ones_like(targets.target_full, dtype=torch.bool)
        spatial_state = torch.ones_like(targets.target_state, dtype=torch.bool)
    psd_full = spatial_full & semantic_full if ignore_void else spatial_full
    psd_state = spatial_state & semantic_state if ignore_void else spatial_state
    return PSDValidMasks(
        semantic_full=semantic_full,
        semantic_state=semantic_state,
        spatial_full=spatial_full,
        spatial_state=spatial_state,
        psd_full=psd_full,
        psd_state=psd_state,
    )


def _safe_mask_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return (
        numerator.float().sum()
        / denominator.float().sum().clamp_min(1.0)
    )


def consistency_schedule_weight(
    consistency_config: dict, *, epoch_index: int,
    progress_in_epoch: float, optimizer_step: int,
) -> float:
    start = consistency_config.get("start", {
        "unit": "epoch", "value": consistency_config["start_epoch"]
    })
    if start["unit"] == "optimizer_step":
        warmup_steps = consistency_config.get("warmup_steps", 0)
        if optimizer_step < start["value"]:
            weight = 0.0
        elif warmup_steps <= 0:
            weight = 1.0
        else:
            weight = (optimizer_step - start["value"]) / warmup_steps
    else:
        weight = losses.esd_schedule_weight(
            epoch_index, progress_in_epoch,
            start["value"], consistency_config["warmup_epochs"],
        )
    return min(max(float(weight), 0.0), 1.0)


def compute_model_training_objectives(
    adapter: "DDPCompatibleTrainingModel",
    *,
    operation: str,
    image: torch.Tensor,
    target: torch.Tensor,
    spatial_valid_mask: torch.Tensor | None = None,
    epoch_index: int,
    progress_in_epoch: float,
    optimizer_step: int = 0,
) -> dict[str, Any]:
    """Build the complete endpoint/source graph inside one composite forward."""
    if operation not in {"stage1_objectives", "stage2_objectives", "joint_objectives"}:
        raise ValueError(f"Unknown training operation: {operation}")
    config = adapter.config
    endpoint = adapter.endpoint_model
    source = adapter.source_model
    training = config["training"]
    consistency_config = config["loss"]["consistency"]
    time_config = config["time_sampling"]
    batch_size = image.shape[0]

    target_full = target
    factor = config.get("model", {}).get("state_downsample_factor", 1)
    state_size = state_spatial_size(image, factor)
    ignore_index = config["loss"].get("ignore_index")
    targets = prepare_state_targets(
        target_full,
        num_classes=config["dataset"]["num_classes"],
        state_size=state_size,
        ignore_index=ignore_index,
        mask_pixel_losses=config["loss"].get("mask_pixel_losses", False),
        spatial_valid_mask_full=spatial_valid_mask,
    )
    psd_masks = None
    if consistency_config["type"] == "psd":
        psd_masks = build_psd_valid_masks(
            targets,
            void_class_index=config["dataset"].get(
                "void_class_index", ignore_index
            ),
            ignore_void=consistency_config.get("psd", {}).get(
                "ignore_void", True
            ),
        )
    source_only_stage1 = (
        operation == "stage1_objectives"
        and not training.get("train_endpoint", True)
        and float(config["loss"]["primary"]["weight"]) == 0.0
    )
    x0, source_stats = sample_prior(
        config,
        image,
        targets.one_hot_state,
        source,
        target_full=targets.target_full,
        valid_mask_full=targets.valid_mask_full,
        sample_state=not source_only_stage1,
        sampling_mode="training",
    )
    x1_state = target_state_from_config(targets.one_hot_state, config)
    smoothing = config.get("flow", {}).get("target_smoothing", {})
    smoothing_enabled = bool(smoothing.get("enabled", False))
    smoothing_p = float(smoothing.get("p", 0.0)) if smoothing_enabled else 0.0
    x1_gt = x1_state.gather(1, targets.target_state[:, None]).squeeze(1)
    x1_competitors = x1_state.clone()
    x1_competitors.scatter_(1, targets.target_state[:, None], -torch.inf)
    source_stats.update({
        "target_smoothing_enabled": x1_state.new_tensor(float(smoothing_enabled)),
        "target_smoothing_p": x1_state.new_tensor(smoothing_p),
        "x1_state_min": x1_state.detach().amin(),
        "x1_state_max": x1_state.detach().amax(),
        "x1_state_abs": x1_state.detach().abs().mean(),
        "x1_state_sum_error": (
            x1_state.detach().sum(dim=1) - 1.0
        ).abs().amax(),
        "x1_gt_margin": (
            x1_gt - x1_competitors.amax(dim=1)
        ).detach().mean(),
        "x1_hard_abs": targets.one_hot_state.detach().abs().mean(),
        "x1_smoothed_abs": x1_state.detach().abs().mean(),
        # Preserve the legacy dashboard name while reporting the trajectory x1.
        "target_x1_abs": x1_state.detach().abs().mean(),
    })
    if source_only_stage1:
        zero = _zero(image)
        source_objective = source_stats.get(
            "weighted_source_supervision", source_stats["weighted_align"]
        ).float()
        stats = {
            "loss_total": source_objective.detach(),
            "loss_diagonal": zero.detach(),
            "loss_diagonal_raw": zero.detach(),
            "loss_consistency": zero.detach(),
            "consistency_base_weight": zero.detach(),
            "consistency_schedule_weight": zero.detach(),
            "consistency_effective_weight": zero.detach(),
            "diagonal_time_mean": zero.detach(),
            "valid_pixel_ratio": targets.valid_mask_full.float().mean().detach()
            if targets.valid_mask_full is not None else zero.new_tensor(1.0),
            "valid_state_pixel_ratio": targets.valid_mask_state.float().mean().detach()
            if targets.valid_mask_state is not None else zero.new_tensor(1.0),
            "state_height": zero.new_tensor(state_size[0]),
            "state_width": zero.new_tensor(state_size[1]),
        }
        stats.update({
            key: value.detach()
            for key, value in source_stats.items()
            if torch.is_tensor(value) and value.numel() == 1
        })
        return {
            "loss": source_objective,
            "diagonal_objective": zero,
            "psd_objective": zero,
            "source_objective": source_objective,
            "stats": stats,
            "operation": operation,
            "consistency_type": "none",
        }
    path_entropy = None
    path_difficulty = None
    if entropy_adaptive_enabled(config):
        source_state = source_stats.get("_path_source_state")
        if source_state is None:
            raise RuntimeError("adaptive path requires source model output")
        path_entropy, path_difficulty = source_entropy_difficulty(
            source_state,
            config,
            valid_mask=(
                targets.valid_mask_state
                if config["flow"]["path"]["entropy"]["exclude_ignore"]
                else None
            ),
            spatial_size=state_size,
        )
    image_feat = endpoint.encode_image(image)
    assert x0.shape == x1_state.shape == targets.one_hot_state.shape
    assert image_feat.shape[-2:] == state_size
    zero = _zero(image)
    consistency_result = None
    u = None
    consistency_s = None
    consistency_t = None

    if operation == "stage1_objectives":
        diagonal_time = sample_stage1_times(
            batch_size, image.device,
            time_config["min_time"], time_config["max_time"],
        )
        diagonal_state = linear_path(
            x0, x1_state, diagonal_time, config, path_difficulty
        )
        schedule_weight = 0.0
        effective_weight = 0.0
    else:
        consistency_s, u, consistency_t = sample_consistency_times(
            consistency_config["type"], batch_size, image.device,
            time_config["min_time"], time_config["max_time"],
            time_config["min_gap"],
        )
        consistency_state = linear_path(
            x0, x1_state, consistency_s, config, path_difficulty
        )
        if operation == "joint_objectives":
            # Joint training intentionally samples an independent diagonal time.
            diagonal_time = sample_stage1_times(
                batch_size, image.device,
                time_config["min_time"], time_config["max_time"],
            )
            diagonal_state = linear_path(
                x0, x1_state, diagonal_time, config, path_difficulty
            )
        else:
            # Preserve the original Stage 2 diagonal-at-s behavior.
            diagonal_time = consistency_s
            diagonal_state = consistency_state
        schedule_weight = consistency_schedule_weight(
            consistency_config,
            epoch_index=epoch_index,
            progress_in_epoch=progress_in_epoch,
            optimizer_step=optimizer_step,
        )
        effective_weight = (
            consistency_config["weight"]
            * consistency_config["max_weight"]
            * schedule_weight
        )

    diagonal_logits = endpoint.forward_logits_with_image_feat(
        diagonal_state, image_feat, diagonal_time, diagonal_time
    )
    assert diagonal_logits.shape == targets.one_hot_state.shape
    diagonal_logits_full = resize_continuous(
        diagonal_logits, targets.target_full.shape[-2:]
    )
    diagonal_config = config["loss"]["primary"].get(
        "adaptive_weighting", {"enabled": False, "r": 0.5, "c": 0.01}
    )
    diagonal_ignore = ignore_index if targets.valid_mask_full is not None else None
    if diagonal_config["enabled"]:
        diagonal_result = losses.adaptive_diagonal_cross_entropy(
            diagonal_logits_full,
            targets.target_full,
            r=diagonal_config["r"],
            c=diagonal_config["c"],
            label_smoothing=training["label_smoothing"],
            ignore_index=diagonal_ignore,
        )
        diagonal_loss = diagonal_result.loss
        diagonal_raw_loss = diagonal_result.raw_loss
        diagonal_stats = diagonal_result.stats
    else:
        diagonal_loss = losses.diagonal_cross_entropy(
            diagonal_logits_full,
            targets.target_full,
            training["label_smoothing"],
            ignore_index=diagonal_ignore,
        ).float()
        diagonal_raw_loss = diagonal_loss
        diagonal_stats = {}

    if operation != "stage1_objectives" and effective_weight > 0.0 :
        consistency_image_feat = image_feat
        if (
            consistency_config["precision"].get("jvp_dtype") == "fp32"
            and image_feat.dtype != torch.float32
        ):
            with torch.autocast(device_type=image.device.type, enabled=False):
                consistency_image_feat = endpoint.encode_image(image.float())
        consistency_valid_mask = (
            psd_masks.psd_state if psd_masks is not None
            else targets.valid_mask_state
        )
        consistency_valid_mask_full = (
            psd_masks.psd_full if psd_masks is not None
            else targets.valid_mask_full
        )
        consistency_result = losses.compute_consistency_loss(
            consistency_config["type"],
            model=endpoint,
            x_s=consistency_state,
            image=image,
            image_feat=consistency_image_feat,
            s=consistency_s,
            u=u,
            t=consistency_t,
            precision=consistency_config["precision"],
            config=config,
            valid_mask=consistency_valid_mask,
            full_resolution_size=tuple(targets.target_full.shape[-2:]),
            valid_mask_full=consistency_valid_mask_full,
            path_difficulty=path_difficulty,
        )
        consistency_loss = consistency_result.loss
    else:
        consistency_result = None
        consistency_loss = zero

    learnable_config = consistency_config.get("learnable_weight", {"enabled": False})
    learnable_stats = {}
    if consistency_result is not None and learnable_config["enabled"]:
        if consistency_result.loss_per_sample is None or consistency_result.valid_sample is None:
            raise RuntimeError("learnable weighting requires sample-wise consistency losses")
        if consistency_config["type"] == "psd":
            weight_logit = adapter.consistency_weight_model(consistency_s)
            consistency_loss, learnable_stats = uncertainty_weighted_psd_loss(
                consistency_result.loss_per_sample,
                consistency_result.valid_sample,
                weight_logit,
            )
            learnable_stats.update(psd_multiplier_bucket_stats(
                consistency_s.detach(), torch.exp(-weight_logit.detach().float()),
                consistency_result.valid_sample,
            ))
        elif consistency_config["type"] == "esd":
            weight_logit = adapter.consistency_weight_model(
                consistency_s, consistency_t
            )
            consistency_loss, learnable_stats = uncertainty_weighted_esd_loss(
                consistency_result.loss_per_sample,
                consistency_result.valid_sample,
                weight_logit,
            )
            learnable_stats.update(esd_multiplier_bucket_stats(
                consistency_s.detach(), consistency_t.detach(),
                torch.exp(-weight_logit.detach().float()),
                consistency_result.valid_sample,
            ))
        else:  # Config validation rejects this before training.
            raise RuntimeError("learnable weighting supports PSD and ESD only")
    primary_objective = (
        config["loss"]["primary"]["weight"] * diagonal_loss
    ).float()
    psd_objective = (effective_weight * consistency_loss).float()
    if adapter.consistency_weight_model is not None and consistency_result is None:
        # Keep every DDP-managed weight-network parameter in the graph before
        # the consistency schedule opens, while producing exactly zero gradients.
        psd_objective = psd_objective + sum(
            (parameter.float().sum() * 0.0)
            for parameter in adapter.consistency_weight_model.parameters()
        )
    source_objective = (
        source_stats["weighted_var"]
        + source_stats.get(
            "weighted_source_supervision", source_stats["weighted_align"]
        )
    ).float()
    total = (primary_objective + psd_objective + source_objective).float()
    stats = {
        "loss_total": total.detach(),
        "loss_diagonal": diagonal_loss.detach(),
        "loss_diagonal_raw": diagonal_raw_loss.detach(),
        "loss_consistency": consistency_loss.detach(),
        "loss_source_var": source_stats["loss_source_var"].detach(),
        "loss_source_align": source_stats["loss_source_align"].detach(),
        "loss_source_ce": source_stats.get("loss_source_ce", zero).detach(),
        "loss_source_supervision": source_stats.get(
            "loss_source_supervision", source_stats["loss_source_align"]
        ).detach(),
        "consistency_base_weight": total.new_tensor(
            consistency_config["weight"] if operation != "stage1_objectives" else 0.0
        ),
        "consistency_schedule_weight": total.new_tensor(schedule_weight),
        "consistency_effective_weight": total.new_tensor(effective_weight),
        # Legacy ESD log names are retained for existing dashboards.
        "esd_base_weight": total.new_tensor(
            consistency_config["weight"]
            if consistency_config["type"] == "esd" and operation != "stage1_objectives"
            else 0.0
        ),
        "esd_schedule_weight": total.new_tensor(
            schedule_weight if consistency_config["type"] == "esd" else 0.0
        ),
        "esd_effective_weight": total.new_tensor(
            effective_weight if consistency_config["type"] == "esd" else 0.0
        ),
        "diagonal_time_mean": diagonal_time.detach().float().mean(),
        "valid_pixel_ratio": (
            targets.valid_mask_full.float().mean().detach()
            if targets.valid_mask_full is not None else total.new_tensor(1.0)
        ),
        "valid_state_pixel_ratio": (
            targets.valid_mask_state.float().mean().detach()
            if targets.valid_mask_state is not None else total.new_tensor(1.0)
        ),
        "state_height": total.new_tensor(state_size[0]),
        "state_width": total.new_tensor(state_size[1]),
    }
    if psd_masks is not None:
        psd_config = consistency_config.get("psd", {})
        use_full = psd_config.get("loss_resolution", "state") == "full"
        semantic_mask = (
            psd_masks.semantic_full if use_full else psd_masks.semantic_state
        )
        spatial_mask = (
            psd_masks.spatial_full if use_full else psd_masks.spatial_state
        )
        psd_mask = psd_masks.psd_full if use_full else psd_masks.psd_state
        real_void = spatial_mask & ~semantic_mask
        stats.update({
            "psd_ignore_void": total.new_tensor(
                float(psd_config.get("ignore_void", True))
            ),
            "psd_valid_pixel_ratio": psd_mask.float().mean().detach(),
            "psd_spatial_valid_pixel_ratio": spatial_mask.float().mean().detach(),
            "psd_semantic_valid_pixel_ratio": _safe_mask_ratio(
                spatial_mask & semantic_mask, spatial_mask
            ).detach(),
            "psd_real_void_included_ratio": _safe_mask_ratio(
                psd_mask & real_void, psd_mask
            ).detach(),
        })
    stats.update(diagonal_stats)
    if path_difficulty is not None and config["flow"]["path"]["diagnostics"]["enabled"]:
        coefficient = path_coefficient(
            diagonal_time, config, path_difficulty
        )
        stats.update(adaptive_path_stats(
            path_entropy,
            path_difficulty,
            diagonal_time,
            coefficient,
            targets.valid_mask_state
            if config["flow"]["path"]["entropy"]["exclude_ignore"] else None,
        ))
    for key, value in source_stats.items():
        if key not in stats and torch.is_tensor(value) and value.numel() == 1:
            stats[key] = value.detach()
    if consistency_result is not None:
        stats.update(consistency_result.stats)
        stats["consistency_s_mean"] = consistency_s.detach().float().mean()
        stats["consistency_t_mean"] = consistency_t.detach().float().mean()
        if u is not None:
            stats["consistency_u_mean"] = u.detach().float().mean()
    # The raw consistency result is merged first; learned-objective metrics
    # then deliberately define the optimizer-facing loss_consistency value.
    stats.update(learnable_stats)
    stats["loss_consistency"] = consistency_loss.detach()
    result = {
        "loss": total,
        "diagonal_objective": primary_objective,
        "psd_objective": psd_objective,
        "source_objective": source_objective,
        "stats": stats,
        "operation": operation,
        "consistency_type": (
            consistency_config["type"] if operation != "stage1_objectives" else "none"
        ),
    }
    if path_difficulty is not None and config["flow"]["path"]["diagnostics"][
        "visualization"
    ]:
        selected_times = config["flow"]["path"]["diagnostics"]["times"]
        result["path_debug"] = {
            "source_mean": source_stats["_path_source_state"].detach(),
            "entropy": path_entropy.detach(),
            "difficulty": path_difficulty.detach(),
            "source_semantic_mask": source_predicted_semantic_mask(
                source_stats["_path_source_state"], config
            ).detach(),
            "lambdas": torch.stack([
                path_coefficient(
                    diagonal_time.new_full(diagonal_time.shape, float(value)),
                    config,
                    path_difficulty,
                ).detach()
                for value in selected_times
            ], dim=1),
            "times": tuple(float(value) for value in selected_times),
        }
    return result


class DDPCompatibleTrainingModel(nn.Module):
    """Composite endpoint/source adapter whose complete graph is one DDP forward."""

    _is_dfm_ddp_adapter = True

    def __init__(
        self,
        endpoint_model: nn.Module,
        source_model: nn.Module | None,
        config: dict,
    ) -> None:
        super().__init__()
        self.endpoint_model = endpoint_model
        self.source_model = source_model
        self.config = config
        learnable = config["loss"]["consistency"].get(
            "learnable_weight", {"enabled": False}
        )
        consistency_type = config["loss"]["consistency"]["type"]
        if learnable["enabled"] and consistency_type == "psd":
            self.consistency_weight_model = PSDTimeWeightNetwork(
                time_embedding_dim=learnable["time_embedding_dim"],
                hidden_dim=learnable["hidden_dim"],
                init_effective_weight=learnable["init_effective_weight"],
            )
        elif learnable["enabled"] and consistency_type == "esd":
            self.consistency_weight_model = ESDTimeWeightNetwork(
                time_embedding_dim=learnable["time_embedding_dim"],
                hidden_dim=learnable["hidden_dim"],
                init_effective_weight=learnable["init_effective_weight"],
            )
        else:
            self.consistency_weight_model = None

    @property
    def psd_weight_model(self):
        """Backward-compatible alias used by existing PSD surgery/tests."""
        return self.consistency_weight_model

    def forward(self, *, operation: str, **kwargs):
        return compute_model_training_objectives(
            self, operation=operation, **kwargs
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.source_model is not None and self.config["source"]["freeze"]:
            self.source_model.eval()
        if not self.config.get("training", {}).get("train_endpoint", True):
            self.endpoint_model.eval()
        return self


def run_model_training_objectives(model: nn.Module, *, operation: str, **kwargs):
    """Never unwrap DDP here: the training graph must enter through DDP.forward."""
    return model(operation=operation, **kwargs)
