from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalScalarEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 1:
            raise ValueError(f"expected scalar batch [B], got {tuple(value.shape)}")
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequency = torch.exp(
            torch.arange(half, device=value.device, dtype=torch.float32) * -scale
        )
        phase = value.float()[:, None] * frequency[None]
        return F.pad(torch.cat((phase.sin(), phase.cos()), dim=1), (0, self.dim - 2 * half))


class PSDTimeWeightNetwork(nn.Module):
    """Original-like reconstruction of the paper's unspecified w(s) network."""

    def __init__(
        self,
        time_embedding_dim: int = 32,
        hidden_dim: int = 64,
        init_effective_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if time_embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("PSD weight embedding and hidden dimensions must be positive")
        if not 0.0 < init_effective_weight:
            raise ValueError("init_effective_weight must be positive")
        self.embedding = SinusoidalScalarEmbedding(time_embedding_dim)
        self.hidden = nn.Linear(time_embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, -math.log(init_effective_weight))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.output(F.silu(self.hidden(self.embedding(s)))).squeeze(-1)


def uncertainty_weighted_psd_loss(
    loss_per_sample: torch.Tensor,
    valid_sample: torch.Tensor,
    weight_logit: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_per_sample.shape != weight_logit.shape or valid_sample.shape != weight_logit.shape:
        raise ValueError("sample PSD loss, validity, and w(s) must all have shape [B]")
    valid = valid_sample.to(dtype=torch.bool, device=weight_logit.device)
    multiplier = torch.exp(-weight_logit.float())
    weighted = multiplier * loss_per_sample.float()
    objective_per_sample = weighted + weight_logit.float()
    local_valid_count = valid.sum().to(dtype=objective_per_sample.dtype)
    global_valid_count = local_valid_count.detach().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_valid_count, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    if valid.any():
        # Both DDP and the explicit surgery reducer average rank gradients.
        # This scale makes that average equal the mean over all valid samples,
        # even when valid-sample counts differ between ranks.
        objective = (
            objective_per_sample[valid].sum()
            * world_size
            / global_valid_count.clamp_min(1.0)
        )
        raw = loss_per_sample.float()[valid].mean()
        learned = weighted[valid].mean()
        regularizer = weight_logit.float()[valid].mean()
        w_values = weight_logit.float()[valid]
        multipliers = multiplier[valid]
    else:
        objective = objective_per_sample.sum() * 0.0
        raw = learned = regularizer = objective.detach()
        w_values = multipliers = None
    zero = objective.detach() * 0.0
    stats = {
        "loss_psd_raw": raw.detach(),
        "loss_psd_learnable": objective.detach(),
        "loss_psd_uncertainty_weighted": learned.detach(),
        "loss_psd_uncertainty_regularizer": regularizer.detach(),
        "psd_weight_logit_mean": w_values.mean().detach() if w_values is not None else zero,
        "psd_weight_logit_std": w_values.std(unbiased=False).detach() if w_values is not None else zero,
        "psd_weight_logit_min": w_values.min().detach() if w_values is not None else zero,
        "psd_weight_logit_max": w_values.max().detach() if w_values is not None else zero,
        "psd_effective_multiplier_mean": multipliers.mean().detach() if multipliers is not None else zero,
        "psd_effective_multiplier_std": multipliers.std(unbiased=False).detach() if multipliers is not None else zero,
        "psd_effective_multiplier_min": multipliers.min().detach() if multipliers is not None else zero,
        "psd_effective_multiplier_max": multipliers.max().detach() if multipliers is not None else zero,
    }
    return objective.float(), stats


def psd_multiplier_bucket_stats(
    s: torch.Tensor,
    multiplier: torch.Tensor,
    valid_sample: torch.Tensor,
) -> dict[str, torch.Tensor]:
    buckets = (
        ("psd_effective_multiplier_s_0_0_1", s < 0.1),
        ("psd_effective_multiplier_s_0_1_0_25", (s >= 0.1) & (s < 0.25)),
        ("psd_effective_multiplier_s_0_25_0_5", (s >= 0.25) & (s < 0.5)),
        ("psd_effective_multiplier_s_0_5_0_75", (s >= 0.5) & (s < 0.75)),
        ("psd_effective_multiplier_s_0_75_1", s >= 0.75),
    )
    valid = valid_sample.bool()
    empty = multiplier.detach().new_tensor(float("nan"))

    return {
        name: (
            multiplier[valid & mask].mean().detach()
            if (valid & mask).any()
            else empty
        )
        for name, mask in buckets
    }


@dataclass
class SurgeryResult:
    projected: list[torch.Tensor | None]
    stats: dict[str, torch.Tensor]


def project_conflicting_gradient(
    diagonal: Iterable[torch.Tensor | None],
    psd: Iterable[torch.Tensor | None],
    *,
    eps: float = 1.0e-12,
) -> SurgeryResult:
    diagonal = list(diagonal)
    psd = list(psd)
    if len(diagonal) != len(psd):
        raise ValueError("diagonal and PSD gradient lists differ in length")
    reference = next((x for x in diagonal + psd if x is not None), None)
    if reference is None:
        reference = torch.zeros(())
    dot = reference.new_zeros(())
    diagonal_sq = reference.new_zeros(())
    psd_sq = reference.new_zeros(())
    for gd, gp in zip(diagonal, psd, strict=True):
        if gd is not None:
            diagonal_sq = diagonal_sq + gd.float().square().sum()
        if gp is not None:
            psd_sq = psd_sq + gp.float().square().sum()
        if gd is not None and gp is not None:
            dot = dot + (gd.float() * gp.float()).sum()
    conflict = dot < 0
    coefficient = torch.where(conflict, dot / (diagonal_sq + eps), dot.new_zeros(()))
    projected: list[torch.Tensor | None] = []
    removed_sq = reference.new_zeros(())
    after_sq = reference.new_zeros(())
    for gd, gp in zip(diagonal, psd, strict=True):
        if gp is None:
            projected.append(None)
            continue
        value = gp if gd is None else gp - coefficient.to(gp.dtype) * gd
        projected.append(value)
        removed_sq = removed_sq + (gp.float() - value.float()).square().sum()
        after_sq = after_sq + value.float().square().sum()
    gd_norm = diagonal_sq.sqrt()
    gp_norm = psd_sq.sqrt()
    after_norm = after_sq.sqrt()
    removed_norm = removed_sq.sqrt()
    cosine = dot / (gd_norm * gp_norm + eps)
    stats = {
        "gradient_surgery_dot": dot.detach(),
        "gradient_surgery_cosine": cosine.detach(),
        "gradient_surgery_conflict": conflict.float().detach(),
        "gradient_surgery_projection_applied": conflict.float().detach(),
        "gradient_surgery_diagonal_norm": gd_norm.detach(),
        "gradient_surgery_psd_norm_before": gp_norm.detach(),
        "gradient_surgery_psd_norm_after": after_norm.detach(),
        "gradient_surgery_removed_component_norm": removed_norm.detach(),
        "gradient_surgery_removed_fraction": (removed_norm / (gp_norm + eps)).detach(),
    }
    return SurgeryResult(projected=projected, stats=stats)


def _global_gradients(
    objective: torch.Tensor,
    parameters: list[nn.Parameter],
    *,
    world_size: int,
) -> list[torch.Tensor | None]:
    if not parameters:
        return []
    if not objective.requires_grad:
        return [None] * len(parameters)
    local = torch.autograd.grad(
        objective, parameters, retain_graph=True, allow_unused=True
    )
    global_gradients: list[torch.Tensor | None] = [None] * len(parameters)
    used = torch.tensor(
        [gradient is not None for gradient in local],
        device=parameters[0].device,
        dtype=torch.int32,
    )
    if world_size > 1:
        dist.all_reduce(used, op=dist.ReduceOp.MAX)
    # A model can contain mixed parameter dtypes. Pack each dtype into one
    # collective rather than issuing one all-reduce per parameter.
    buckets: dict[tuple[torch.device, torch.dtype], list[int]] = {}
    for index, (parameter, globally_used) in enumerate(zip(parameters, used, strict=True)):
        if globally_used.item():
            buckets.setdefault((parameter.device, parameter.dtype), []).append(index)
    for indices in buckets.values():
        values = [
            torch.zeros_like(parameters[index])
            if local[index] is None else local[index].contiguous()
            for index in indices
        ]
        flat = torch.cat([value.reshape(-1) for value in values])
        if world_size > 1:
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(world_size)
        offset = 0
        for index, value in zip(indices, values, strict=True):
            count = value.numel()
            global_gradients[index] = flat[offset:offset + count].view_as(value)
            offset += count
    return global_gradients


def apply_global_gradient_surgery(
    *,
    adapter: nn.Module,
    objectives: dict,
    scaler,
    world_size: int = 1,
    eps: float = 1.0e-12,
) -> dict[str, torch.Tensor]:
    """Assign globally averaged, projected scaled gradients to adapter parameters."""
    scaler_enabled = scaler is not None and scaler.is_enabled()
    if scaler_enabled:
        # GradScaler lazily creates its device-side scale on scale().  The
        # explicit-gradient path still needs that state for unscale_/step().
        scaler.scale(objectives["loss"])
    scale = float(scaler.get_scale()) if scaler_enabled else 1.0
    endpoint = [p for p in adapter.endpoint_model.parameters() if p.requires_grad]
    source = (
        [p for p in adapter.source_model.parameters() if p.requires_grad]
        if adapter.source_model is not None else []
    )
    weight = (
        [p for p in adapter.psd_weight_model.parameters() if p.requires_grad]
        if adapter.psd_weight_model is not None else []
    )
    gd = _global_gradients(objectives["diagonal_objective"] * scale, endpoint, world_size=world_size)
    gp = _global_gradients(objectives["psd_objective"] * scale, endpoint, world_size=world_size)
    other = _global_gradients(objectives["source_objective"] * scale, endpoint, world_size=world_size)
    surgery = project_conflicting_gradient(gd, gp, eps=eps)
    for parameter, d_value, p_value, other_value in zip(
        endpoint, gd, surgery.projected, other, strict=True
    ):
        values = [value for value in (d_value, p_value, other_value) if value is not None]
        parameter.grad = sum(values[1:], values[0].clone()) if values else None
    for parameters, objective in ((source, objectives["loss"]), (weight, objectives["loss"])):
        gradients = _global_gradients(objective * scale, parameters, world_size=world_size)
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.grad = gradient
    if scale == 1.0:
        return surgery.stats
    stats = dict(surgery.stats)
    stats["gradient_surgery_dot"] = stats["gradient_surgery_dot"] / (scale * scale)
    for key in (
        "gradient_surgery_diagonal_norm",
        "gradient_surgery_psd_norm_before",
        "gradient_surgery_psd_norm_after",
        "gradient_surgery_removed_component_norm",
    ):
        stats[key] = stats[key] / scale
    return stats
