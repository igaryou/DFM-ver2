from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.func import jvp

from discrete_flow_maps import flow_map
from state_space import resize_continuous


@dataclass
class ConsistencyResult:
    loss: torch.Tensor
    stats: dict[str, torch.Tensor]
    teacher_prob: torch.Tensor | None = None
    student_prob: torch.Tensor | None = None
    valid_pixel: torch.Tensor | None = None
    adaptive_weight: torch.Tensor | None = None
    directional_output: torch.Tensor | None = None
    dtypes: dict[str, torch.dtype] = field(default_factory=dict)
    loss_per_sample: torch.Tensor | None = None
    valid_sample: torch.Tensor | None = None


@dataclass
class AdaptiveDiagonalResult:
    loss: torch.Tensor
    raw_loss: torch.Tensor
    adaptive_weight: torch.Tensor
    mismatch_l2_sq: torch.Tensor
    valid_mask: torch.Tensor
    stats: dict[str, torch.Tensor]


@dataclass
class ESDResult:
    """Backward-compatible ESD return object."""

    loss: torch.Tensor
    stats: dict[str, torch.Tensor]
    teacher_prob: torch.Tensor
    student_prob: torch.Tensor
    valid_pixel: torch.Tensor
    adaptive_weight: torch.Tensor
    directional_logits: torch.Tensor
    dtypes: dict[str, torch.dtype] = field(default_factory=dict)
    loss_per_sample: torch.Tensor | None = None
    valid_sample: torch.Tensor | None = None


def masked_mean(loss_map: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    """Average over valid pixels only, with a differentiable all-invalid zero."""
    if valid_mask is None:
        return loss_map.mean()
    valid_mask = valid_mask.to(device=loss_map.device, dtype=torch.bool)
    if valid_mask.shape != loss_map.shape:
        raise ValueError(
            f"valid_mask shape {tuple(valid_mask.shape)} != loss map {tuple(loss_map.shape)}"
        )
    weights = valid_mask.to(loss_map.dtype)
    return (loss_map * weights).sum() / weights.sum().clamp_min(1.0)


def diagonal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    ignore_index: int | None = None,
) -> torch.Tensor:
    assert logits.shape[0] == target.shape[0]
    assert logits.shape[-2:] == target.shape[-2:], (
        f"supervised logits {logits.shape[-2:]} != target {target.shape[-2:]}"
    )
    loss_map = F.cross_entropy(
        logits,
        target,
        label_smoothing=label_smoothing,
        reduction="none",
        ignore_index=-100 if ignore_index is None else ignore_index,
    )
    valid_mask = None if ignore_index is None else target != ignore_index
    return masked_mean(loss_map, valid_mask)


def adaptive_diagonal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    r: float = 0.5,
    c: float = 0.01,
    label_smoothing: float = 0.0,
    ignore_index: int | None = None,
) -> AdaptiveDiagonalResult:
    """DFM Eq. (45)-(46) for a hard diagonal target.

    The mismatch uses ``||softmax(logits) - one_hot(target)||^2`` in closed
    form.  Its derived weight is detached, and the weighted CE is divided by
    the valid-pixel count (not by the sum of weights).
    """
    if logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
        raise ValueError("adaptive diagonal logits and target shapes do not match")
    if r < 0.0 or c <= 0.0:
        raise ValueError("adaptive diagonal requires r >= 0 and c > 0")
    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0)
    ce_map = F.cross_entropy(
        logits,
        target,
        label_smoothing=label_smoothing,
        reduction="none",
        ignore_index=-100 if ignore_index is None else ignore_index,
    )
    probability = torch.softmax(logits.float(), dim=1)
    target_probability = probability.gather(1, safe_target[:, None]).squeeze(1)
    mismatch = probability.square().sum(dim=1) - 2.0 * target_probability + 1.0
    mismatch = mismatch.clamp_min(0.0)
    adaptive_weight = (mismatch + c).pow(-r).detach()
    valid_float = valid.to(ce_map.dtype)
    count = valid_float.sum()
    raw_loss = (ce_map * valid_float).sum() / count.clamp_min(1.0)
    loss = (ce_map * adaptive_weight.to(ce_map.dtype) * valid_float).sum()
    loss = loss / count.clamp_min(1.0)
    valid_weight = adaptive_weight[valid]
    valid_mismatch = mismatch[valid]
    zero = loss.detach() * 0.0
    stats = {
        "loss_diagonal_raw": raw_loss.detach(),
        "loss_diagonal_adaptive": loss.detach(),
        "diagonal_adaptive_weight_mean": valid_weight.mean() if valid_weight.numel() else zero,
        "diagonal_adaptive_weight_std": valid_weight.std(unbiased=False) if valid_weight.numel() else zero,
        "diagonal_adaptive_weight_min": valid_weight.min() if valid_weight.numel() else zero,
        "diagonal_adaptive_weight_max": valid_weight.max() if valid_weight.numel() else zero,
        "diagonal_mismatch_l2_sq_mean": valid_mismatch.mean() if valid_mismatch.numel() else zero,
    }
    return AdaptiveDiagonalResult(
        loss=loss.float(), raw_loss=raw_loss.float(),
        adaptive_weight=adaptive_weight, mismatch_l2_sq=mismatch,
        valid_mask=valid, stats=_detached(stats),
    )


def esd_schedule_weight(
    epoch: int, progress_in_epoch: float, start_epoch: int, warmup_epochs: int
) -> float:
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, (epoch - start_epoch + progress_in_epoch) / warmup_epochs)


def _disabled_autocast(tensor: torch.Tensor):
    if tensor.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


def _jvp_autocast(tensor: torch.Tensor, jvp_dtype: str):
    if jvp_dtype == "fp32":
        return _disabled_autocast(tensor)
    if jvp_dtype != "bf16":
        raise ValueError("jvp_dtype must be bf16 or fp32")
    if tensor.device.type not in {"cpu", "cuda"}:
        raise RuntimeError(f"bf16 JVP is unsupported on {tensor.device.type}")
    return torch.autocast(
        device_type=tensor.device.type, dtype=torch.bfloat16, enabled=True
    )


def _torch_jvp_dtype(jvp_dtype: str) -> torch.dtype:
    return torch.bfloat16 if jvp_dtype == "bf16" else torch.float32


def _precision_image(image: torch.Tensor, jvp_dtype: str) -> torch.Tensor:
    return image if jvp_dtype == "bf16" else image.float()


def _dtype_code(dtype: torch.dtype) -> float:
    if dtype == torch.float32:
        return 0.0
    if dtype == torch.bfloat16:
        return 1.0
    return -1.0


def _normalize_probability(probability: torch.Tensor, eps: float) -> torch.Tensor:
    probability = probability.float().clamp_min(eps)
    return probability / probability.sum(dim=1, keepdim=True).clamp_min(eps)


def _forward_logits(
    model,
    x: torch.Tensor,
    image: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    image_feat: torch.Tensor | None = None,
) -> torch.Tensor:
    if image_feat is not None:
        return model.forward_logits_with_image_feat(x, image_feat, s, t)
    return model.forward_logits(x, image, s, t)


def _finite_loss(name: str, loss: torch.Tensor) -> None:
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError(f"{name} consistency loss is NaN or Inf")


def _detached(stats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach() for key, value in stats.items()}


def _bucket_ratios(
    invalid_class: torch.Tensor, t: torch.Tensor
) -> dict[str, torch.Tensor]:
    buckets = (
        ("esd_clamp_ratio_t_0_0_5", t < 0.5),
        ("esd_clamp_ratio_t_0_5_0_9", (t >= 0.5) & (t < 0.9)),
        ("esd_clamp_ratio_t_0_9_0_99", (t >= 0.9) & (t < 0.99)),
        ("esd_clamp_ratio_t_0_99_1_0", t >= 0.99),
    )
    zero = invalid_class.float().sum() * 0.0
    return {
        name: invalid_class[mask].float().mean() if mask.any() else zero
        for name, mask in buckets
    }


def _psd_loss(
    *,
    model,
    x_s: torch.Tensor,
    image: torch.Tensor,
    image_feat: torch.Tensor | None,
    s: torch.Tensor,
    u: torch.Tensor,
    t: torch.Tensor,
    time_eps: float,
    probability_eps: float,
    flow: Callable = flow_map,
    valid_mask: torch.Tensor | None = None,
    loss_resolution: str = "state",
    full_resolution_size: tuple[int, int] | None = None,
    valid_mask_full: torch.Tensor | None = None,
) -> ConsistencyResult:
    if not bool(((s < u) & (u < t)).all()):
        raise ValueError("PSD requires s < u < t")
    teacher_image_feat = image_feat.detach() if image_feat is not None else None
    with torch.no_grad():
        logits_su = _forward_logits(
            model, x_s, image, s, u, image_feat=teacher_image_feat
        )
        probability_su = _normalize_probability(
            torch.softmax(logits_su.float(), dim=1), probability_eps
        )
        x_su = flow(x_s.float(), probability_su, s, u, time_eps)
        logits_ut = _forward_logits(
            model, x_su, image, u, t, image_feat=teacher_image_feat
        )
        probability_ut = _normalize_probability(
            torch.softmax(logits_ut.float(), dim=1), probability_eps
        )
        numerator = (1.0 - t) * (u - s)
        denominator = (1.0 - u).clamp_min(time_eps) * (t - s).clamp_min(time_eps)
        mixture = (numerator / denominator).clamp(0.0, 1.0)
        teacher = (
            mixture[:, None, None, None] * probability_su
            + (1.0 - mixture[:, None, None, None]) * probability_ut
        )
        teacher = _normalize_probability(teacher, probability_eps).detach()

    student_logits = _forward_logits(
        model, x_s, image, s, t, image_feat=image_feat
    ).float()
    if loss_resolution == "state":
        # Keep this path identical to the original state-resolution PSD.
        student_log_probability = F.log_softmax(student_logits, dim=1)
        student_probability = student_log_probability.exp()
        loss_teacher = teacher
        loss_map = -(loss_teacher * student_log_probability).sum(dim=1)
        loss_valid_mask = valid_mask
    elif loss_resolution == "full":
        if full_resolution_size is None:
            raise ValueError("full-resolution PSD requires full_resolution_size")
        full_resolution_size = tuple(int(value) for value in full_resolution_size)
        student_logits_full = resize_continuous(
            student_logits, full_resolution_size
        )
        student_log_probability = F.log_softmax(student_logits_full, dim=1)
        with torch.no_grad():
            loss_teacher = _normalize_probability(
                resize_continuous(teacher, full_resolution_size), probability_eps
            ).detach()
        loss_map = -(loss_teacher * student_log_probability).sum(dim=1)
        loss_valid_mask = valid_mask_full
        # Returning a full [B,C,H,W] probability would retain another very large
        # tensor. Diagnostics keep the state-resolution probability instead.
        student_probability = torch.softmax(student_logits, dim=1)
    else:
        raise ValueError("PSD loss_resolution must be state or full")
    loss = masked_mean(loss_map, loss_valid_mask).float()
    if loss_valid_mask is None:
        valid_pixel = torch.ones_like(loss_map, dtype=torch.bool)
    else:
        valid_pixel = loss_valid_mask.to(device=loss_map.device, dtype=torch.bool)
    valid_count = valid_pixel.flatten(1).sum(dim=1)
    loss_per_sample = (
        (loss_map * valid_pixel.to(loss_map.dtype)).flatten(1).sum(dim=1)
        / valid_count.clamp_min(1).to(loss_map.dtype)
    ).float()
    valid_sample = valid_count > 0
    _finite_loss("PSD", loss)
    return ConsistencyResult(
        loss=loss,
        stats=_detached({
            "loss_consistency": loss,
            "loss_psd": loss,
            "psd_teacher_entropy": -(
                teacher * teacher.clamp_min(probability_eps).log()
            ).sum(dim=1).mean(),
            "psd_direct_probability_sum_error": (
                student_probability.sum(dim=1) - 1.0
            ).abs().max(),
            "psd_loss_height": loss.new_tensor(loss_map.shape[-2]),
            "psd_loss_width": loss.new_tensor(loss_map.shape[-1]),
            "psd_loss_resolution_is_full": loss.new_tensor(
                float(loss_resolution == "full")
            ),
        }),
        teacher_prob=teacher,
        student_prob=student_probability,
        valid_pixel=valid_pixel,
        loss_per_sample=loss_per_sample,
        valid_sample=valid_sample,
        dtypes={
            "student_logits_before_cast": student_logits.dtype,
            "student_logits_after_cast": student_logits.dtype,
            "student_prob": student_probability.dtype,
            "teacher_prob": teacher.dtype,
            "loss": loss.dtype,
        },
    )


def _csd_loss(
    *,
    model,
    x_s: torch.Tensor,
    image: torch.Tensor,
    image_feat: torch.Tensor | None,
    s: torch.Tensor,
    t: torch.Tensor,
    time_eps: float,
    probability_eps: float,
    jvp_dtype: str,
    flow: Callable = flow_map,
    valid_mask: torch.Tensor | None = None,
) -> ConsistencyResult:
    target_dtype = _torch_jvp_dtype(jvp_dtype)
    s32, t32 = s.float(), t.float()
    x_primal = x_s.to(dtype=target_dtype)
    image_primal = _precision_image(image, jvp_dtype)
    image_feat_primal = (
        image_feat.to(dtype=target_dtype) if image_feat is not None else None
    )
    with _jvp_autocast(x_s, jvp_dtype):
        def transported_at(t_in: torch.Tensor) -> torch.Tensor:
            logits = _forward_logits(
                model, x_primal, image_primal, s32, t_in,
                image_feat=image_feat_primal,
            )
            logits = logits.to(dtype=target_dtype)
            probability = torch.softmax(logits, dim=1).to(dtype=target_dtype)
            return flow(x_primal, probability, s32, t_in, time_eps).to(target_dtype)

        transported_before, derivative_before = jvp(
            transported_at, (t32,), (torch.ones_like(t32),)
        )

    with _disabled_autocast(x_s):
        transported = transported_before.float()
        derivative = derivative_before.float()
    with torch.no_grad():
        teacher_input = transported.detach().to(dtype=target_dtype)
        teacher_image_feat = (
            image_feat_primal.detach() if image_feat_primal is not None else None
        )
        with _jvp_autocast(x_s, jvp_dtype):
            teacher_logits_before = _forward_logits(
                model, teacher_input, image_primal, t32, t32,
                image_feat=teacher_image_feat,
            ).to(dtype=target_dtype)
        with _disabled_autocast(x_s):
            teacher = _normalize_probability(
                torch.softmax(teacher_logits_before.float(), dim=1), probability_eps
            ).detach()
    with _disabled_autocast(x_s):
        residual = (
            (1.0 - t32)[:, None, None, None] * derivative
            - teacher
            + transported
        )
        residual_map = residual.square().sum(dim=1)
        loss = masked_mean(residual_map, valid_mask).float()
        residual_norm = masked_mean(residual_map.sqrt(), valid_mask)
    _finite_loss("CSD", loss)
    return ConsistencyResult(
        loss=loss,
        stats=_detached({
            "loss_consistency": loss,
            "loss_csd": loss,
            "csd_residual_norm": residual_norm,
            "csd_jvp_output_abs_mean": derivative_before.float().abs().mean(),
            "csd_jvp_output_abs_max": derivative_before.float().abs().max(),
            "csd_jvp_dtype_code": loss.new_tensor(_dtype_code(derivative_before.dtype)),
        }),
        teacher_prob=teacher,
        directional_output=derivative,
        dtypes={
            "student_logits_before_cast": transported_before.dtype,
            "directional_logits_before_cast": derivative_before.dtype,
            "student_logits_after_cast": transported.dtype,
            "directional_logits_after_cast": derivative.dtype,
            "teacher_prob": teacher.dtype,
            "loss": loss.dtype,
        },
    )


def _ecld_loss(
    *,
    model,
    x_s: torch.Tensor,
    image: torch.Tensor,
    image_feat: torch.Tensor | None,
    s: torch.Tensor,
    t: torch.Tensor,
    time_eps: float,
    probability_eps: float,
    jvp_dtype: str,
    ec_weight: float,
    td_weight: float,
    time_weighting: str,
    flow: Callable = flow_map,
    valid_mask: torch.Tensor | None = None,
) -> ConsistencyResult:
    target_dtype = _torch_jvp_dtype(jvp_dtype)
    s32, t32 = s.float(), t.float()
    x_primal = x_s.to(dtype=target_dtype)
    image_primal = _precision_image(image, jvp_dtype)
    image_feat_primal = (
        image_feat.to(dtype=target_dtype) if image_feat is not None else None
    )
    with _jvp_autocast(x_s, jvp_dtype):
        def logits_at(t_in: torch.Tensor) -> torch.Tensor:
            return _forward_logits(
                model, x_primal, image_primal, s32, t_in,
                image_feat=image_feat_primal,
            ).to(target_dtype)

        logits_before, derivative_before = jvp(
            logits_at, (t32,), (torch.ones_like(t32),)
        )
    with _disabled_autocast(x_s):
        logits = logits_before.float()
        derivative = derivative_before.float()
        student_log_probability = F.log_softmax(logits, dim=1)
        student_probability = student_log_probability.exp()
        center = (student_probability * derivative).sum(dim=1, keepdim=True)
        probability_derivative = student_probability * (derivative - center)
        transported = flow(
            x_s.float(), student_probability, s32, t32, time_eps
        )
    with torch.no_grad():
        teacher_input = transported.detach().to(dtype=target_dtype)
        teacher_image_feat = (
            image_feat_primal.detach() if image_feat_primal is not None else None
        )
        with _jvp_autocast(x_s, jvp_dtype):
            teacher_logits_before = _forward_logits(
                model, teacher_input, image_primal, t32, t32,
                image_feat=teacher_image_feat,
            ).to(target_dtype)
        with _disabled_autocast(x_s):
            teacher = _normalize_probability(
                torch.softmax(teacher_logits_before.float(), dim=1), probability_eps
            ).detach()
    with _disabled_autocast(x_s):
        endpoint_ce_map = -(teacher * student_log_probability).sum(dim=1)
        if time_weighting == "none":
            temporal_weight = torch.ones_like(t32)
        elif time_weighting == "inverse_square":
            temporal_weight = (1.0 - t32).clamp_min(time_eps).pow(-2)
        else:
            raise ValueError(f"Unknown ECLD time weighting: {time_weighting}")
        endpoint_ce = masked_mean(
            endpoint_ce_map * temporal_weight[:, None, None], valid_mask
        )
        gamma = (t32 - s32) / (1.0 - s32).clamp_min(time_eps)
        temporal_derivative_map = (
            gamma.square()[:, None, None]
            * probability_derivative.square().sum(dim=1)
        )
        temporal_derivative = masked_mean(temporal_derivative_map, valid_mask)
        loss = (ec_weight * endpoint_ce + td_weight * temporal_derivative).float()
        derivative_norm = masked_mean(
            probability_derivative.square().sum(dim=1).sqrt(), valid_mask
        )
    _finite_loss("ECLD", loss)
    return ConsistencyResult(
        loss=loss,
        stats=_detached({
            "loss_consistency": loss,
            "loss_ecld": loss,
            "loss_ecld_ec": endpoint_ce,
            "loss_ecld_td": temporal_derivative,
            "ecld_dt_prob_norm": derivative_norm,
            "ecld_jvp_output_abs_mean": derivative_before.float().abs().mean(),
            "ecld_jvp_output_abs_max": derivative_before.float().abs().max(),
            "ecld_jvp_dtype_code": loss.new_tensor(_dtype_code(derivative_before.dtype)),
        }),
        teacher_prob=teacher,
        student_prob=student_probability,
        directional_output=probability_derivative,
        dtypes={
            "student_logits_before_cast": logits_before.dtype,
            "directional_logits_before_cast": derivative_before.dtype,
            "student_logits_after_cast": logits.dtype,
            "directional_logits_after_cast": derivative.dtype,
            "student_prob": student_probability.dtype,
            "teacher_prob": teacher.dtype,
            "loss": loss.dtype,
        },
    )


def _esd_loss(
    *,
    model,
    x_s: torch.Tensor,
    image: torch.Tensor,
    image_feat: torch.Tensor | None,
    s: torch.Tensor,
    t: torch.Tensor,
    time_eps: float,
    log_eps: float,
    invalid_strategy: str,
    skip_batch_threshold: float | None,
    adaptive_enabled: bool,
    adaptive_c: float,
    adaptive_r: float,
    adaptive_normalize_mean: bool,
    adaptive_max_weight: float | None,
    jvp_dtype: str,
    valid_mask: torch.Tensor | None = None,
) -> ConsistencyResult:
    if invalid_strategy not in {"clamp", "mask_pixel", "skip_batch"}:
        raise ValueError(f"Unknown invalid teacher strategy: {invalid_strategy}")
    target_dtype = _torch_jvp_dtype(jvp_dtype)
    s32, t32 = s.float(), t.float()
    x32 = x_s.float()
    image_primal = _precision_image(image, jvp_dtype)
    image_feat_primal = (
        image_feat.to(dtype=target_dtype) if image_feat is not None else None
    )
    teacher_image_feat = (
        image_feat_primal.detach() if image_feat_primal is not None else None
    )

    with torch.no_grad():
        with _jvp_autocast(x_s, jvp_dtype):
            logits_ss_before = _forward_logits(
                model, x32.to(target_dtype), image_primal, s32, s32,
                image_feat=teacher_image_feat,
            ).to(target_dtype)
        with _disabled_autocast(x_s):
            logits_ss = logits_ss_before.float()
            probability_ss = torch.softmax(logits_ss, dim=1)
            denominator_s = (1.0 - s32).clamp_min(time_eps)
            drift = (
                probability_ss - x32
            ) / denominator_s[:, None, None, None]

    x_primal = x32.to(target_dtype)
    drift_primal = drift.to(target_dtype)
    with _jvp_autocast(x_s, jvp_dtype):
        def logits_along_flow(x_in: torch.Tensor, s_in: torch.Tensor) -> torch.Tensor:
            return _forward_logits(
                model, x_in, image_primal, s_in, t32,
                image_feat=image_feat_primal,
            ).to(target_dtype)

        logits_before, directional_before = jvp(
            logits_along_flow,
            (x_primal, s32),
            (drift_primal, torch.ones_like(s32)),
        )

    with _disabled_autocast(x_s):
        logits = logits_before.float()
        directional = directional_before.float()
        student_log_probability = F.log_softmax(logits, dim=1)
        student_probability = student_log_probability.exp()
        center = (student_probability * directional).sum(dim=1, keepdim=True)
        delta = directional - center
        one_minus_s = 1.0 - s32
        one_minus_t = 1.0 - t32
        delta_time = t32 - s32
        log_arg_raw = (
            one_minus_t[:, None, None, None]
            - (one_minus_s * delta_time)[:, None, None, None] * delta
        )

        invalid_class = ~torch.isfinite(log_arg_raw) | (log_arg_raw <= log_eps)
        numerical_valid_pixel = torch.isfinite(log_arg_raw).all(dim=1)
        numerical_valid_pixel &= (log_arg_raw > log_eps).all(dim=1)
        semantic_valid_pixel = (
            torch.ones_like(numerical_valid_pixel)
            if valid_mask is None
            else valid_mask.to(device=numerical_valid_pixel.device, dtype=torch.bool)
        )
        valid_pixel = numerical_valid_pixel & semantic_valid_pixel
        invalid_pixel = invalid_class.any(dim=1)
        invalid_sample = invalid_pixel.flatten(1).any(dim=1)
        per_sample_ratio = invalid_class.float().flatten(1).mean(dim=1)
        finite_safe = torch.nan_to_num(
            log_arg_raw, nan=log_eps, neginf=log_eps,
            posinf=torch.finfo(torch.float32).max,
        )
        log_arg_safe = finite_safe.clamp_min(log_eps)
        teacher_logits = logits_ss.detach() - torch.log(log_arg_safe)
        teacher_probability = torch.softmax(teacher_logits, dim=1).detach()
        kl_per_pixel = F.kl_div(
            student_log_probability, teacher_probability, reduction="none"
        ).sum(dim=1)

        mismatch = (
            teacher_probability - student_probability
        ).square().sum(dim=1)
        if adaptive_enabled:
            adaptive_weight = (mismatch.detach() + adaptive_c).pow(-adaptive_r)
            if adaptive_max_weight is not None:
                adaptive_weight = adaptive_weight.clamp_max(adaptive_max_weight)
            normalization_mask = (
                valid_pixel if invalid_strategy == "mask_pixel"
                else semantic_valid_pixel
            )
            if adaptive_normalize_mean and normalization_mask.any():
                adaptive_weight = adaptive_weight / adaptive_weight[
                    normalization_mask
                ].mean().clamp_min(1.0e-8)
        else:
            adaptive_weight = torch.ones_like(kl_per_pixel)
        adaptive_weight = adaptive_weight.detach()
        loss_map = adaptive_weight * kl_per_pixel
        zero_with_graph = student_log_probability.sum() * 0.0
        invalid_ratio = invalid_class.float().mean()
        skip_for_threshold = (
            skip_batch_threshold is not None
            and float(invalid_ratio.detach()) > skip_batch_threshold
        )
        skip_for_strategy = (
            invalid_strategy == "skip_batch"
            and float(invalid_ratio.detach()) > (
                skip_batch_threshold if skip_batch_threshold is not None else 0.0
            )
        )
        reduction_mask = (
            valid_pixel if invalid_strategy == "mask_pixel"
            else semantic_valid_pixel
        )
        valid_count = reduction_mask.flatten(1).sum(dim=1)
        loss_per_sample = (
            (loss_map * reduction_mask.to(loss_map.dtype)).flatten(1).sum(dim=1)
            / valid_count.clamp_min(1).to(loss_map.dtype)
        ).float()
        valid_sample = valid_count > 0
        if skip_for_threshold or skip_for_strategy:
            loss = zero_with_graph
            loss_per_sample = loss_per_sample * 0.0
            valid_sample = torch.zeros_like(valid_sample)
        else:
            loss = masked_mean(loss_map, reduction_mask)
        loss = loss.float()
        entropy = -(
            teacher_probability
            * teacher_probability.clamp_min(1.0e-12).log()
        ).sum(dim=1)
        valid_adaptive = adaptive_weight[reduction_mask]
        valid_mismatch = mismatch.detach()[reduction_mask]
        raw_kl = masked_mean(kl_per_pixel, reduction_mask).float()
        zero_stat = loss.detach() * 0.0
        stats = {
            "loss_consistency": loss,
            "loss_esd": loss,
            "loss_esd_raw_kl": raw_kl,
            "loss_esd_adaptive_kl": loss,
            "esd_log_arg_min": torch.nan_to_num(
                log_arg_raw.detach(), nan=0.0,
                posinf=torch.finfo(torch.float32).max,
                neginf=torch.finfo(torch.float32).min,
            ).min(),
            "esd_log_arg_mean": torch.nan_to_num(
                log_arg_raw.detach(), nan=0.0, posinf=0.0, neginf=0.0
            ).mean(),
            "esd_clamp_ratio": invalid_ratio,
            "esd_pixel_invalid_ratio": invalid_pixel.float().mean(),
            "esd_sample_invalid_ratio": invalid_sample.float().mean(),
            "esd_max_sample_invalid_ratio": per_sample_ratio.max(),
            "esd_nonfinite_ratio": (~torch.isfinite(log_arg_raw)).float().mean(),
            "esd_teacher_entropy": entropy.mean(),
            "esd_teacher_min": teacher_probability.min(),
            "esd_teacher_max": teacher_probability.max(),
            "esd_delta_abs_mean": delta.detach().abs().mean(),
            "esd_delta_abs_max": delta.detach().abs().max(),
            "esd_valid_pixel_ratio": valid_pixel.float().mean(),
            "esd_adaptive_weight_mean": (
                valid_adaptive.mean() if valid_adaptive.numel() else zero_stat
            ),
            "esd_adaptive_weight_std": (
                valid_adaptive.std(unbiased=False) if valid_adaptive.numel() else zero_stat
            ),
            "esd_adaptive_weight_min": (
                valid_adaptive.min() if valid_adaptive.numel() else zero_stat
            ),
            "esd_adaptive_weight_max": (
                valid_adaptive.max() if valid_adaptive.numel() else zero_stat
            ),
            "esd_mismatch_l2_sq_mean": (
                valid_mismatch.mean() if valid_mismatch.numel() else zero_stat
            ),
            "esd_skipped_batch": loss.new_tensor(
                float(skip_for_threshold or skip_for_strategy)
            ),
            "esd_jvp_output_abs_mean": directional_before.float().abs().mean(),
            "esd_jvp_output_abs_max": directional_before.float().abs().max(),
            "esd_jvp_dtype_code": loss.new_tensor(
                _dtype_code(directional_before.dtype)
            ),
            "s_mean": s32.mean(),
            "s_min": s32.min(),
            "s_max": s32.max(),
            "t_mean": t32.mean(),
            "t_min": t32.min(),
            "t_max": t32.max(),
        }
        stats.update(_bucket_ratios(invalid_class.detach(), t32.detach()))
    _finite_loss("ESD", loss)
    return ConsistencyResult(
        loss=loss,
        stats=_detached(stats),
        teacher_prob=teacher_probability,
        student_prob=student_probability,
        valid_pixel=valid_pixel.detach(),
        adaptive_weight=adaptive_weight,
        directional_output=directional,
        loss_per_sample=loss_per_sample,
        valid_sample=valid_sample,
        dtypes={
            "student_logits_before_cast": logits_before.dtype,
            "directional_logits_before_cast": directional_before.dtype,
            "student_logits_after_cast": logits.dtype,
            "directional_logits_after_cast": directional.dtype,
            "student_prob": student_probability.dtype,
            "teacher_prob": teacher_probability.dtype,
            "loss": loss.dtype,
        },
    )


def compute_consistency_loss(
    loss_type: str,
    *,
    model,
    x_s: torch.Tensor,
    image: torch.Tensor,
    image_feat: torch.Tensor | None = None,
    s: torch.Tensor,
    t: torch.Tensor,
    u: torch.Tensor | None = None,
    flow: Callable = flow_map,
    precision: dict | None = None,
    config: dict | None = None,
    valid_mask: torch.Tensor | None = None,
    full_resolution_size: tuple[int, int] | None = None,
    valid_mask_full: torch.Tensor | None = None,
) -> ConsistencyResult:
    if loss_type not in {"psd", "csd", "ecld", "esd"}:
        raise ValueError(f"Unknown consistency loss: {loss_type}")
    if image_feat is not None:
        assert image_feat.shape[0] == x_s.shape[0]
        assert image_feat.shape[-2:] == x_s.shape[-2:]
    if valid_mask is not None:
        assert valid_mask.shape == x_s.shape[:1] + x_s.shape[-2:], (
            f"consistency mask {valid_mask.shape} != state pixels "
            f"{x_s.shape[:1] + x_s.shape[-2:]}"
        )
    if loss_type == "psd" and valid_mask_full is not None:
        if full_resolution_size is None:
            full_resolution_size = tuple(valid_mask_full.shape[-2:])
        assert valid_mask_full.shape == (
            x_s.shape[0], *tuple(full_resolution_size)
        ), (
            f"full consistency mask {valid_mask_full.shape} != full pixels "
            f"{(x_s.shape[0], *tuple(full_resolution_size))}"
        )
    config = config or {}
    consistency = config.get("loss", {}).get("consistency", {})
    precision = precision or consistency.get(
        "precision", {"jvp_dtype": "fp32", "numerical_dtype": "fp32"}
    )
    if precision.get("numerical_dtype") != "fp32":
        raise ValueError("Consistency numerical_dtype must be fp32")
    jvp_dtype = precision.get("jvp_dtype")
    if loss_type == "psd":
        if jvp_dtype is not None:
            raise ValueError("PSD does not use JVP; jvp_dtype must be null")
    elif jvp_dtype not in {"bf16", "fp32"}:
        raise ValueError("JVP loss requires jvp_dtype=bf16 or fp32")
    flow_config = config.get("flow", {})
    time_eps = float(flow_config.get("time_eps", 1.0e-5))
    probability_eps = float(flow_config.get("probability_eps", 1.0e-8))

    if loss_type == "psd":
        if u is None:
            raise ValueError("PSD requires intermediate time u")
        psd = consistency.get("psd", {})
        result = _psd_loss(
            model=model, x_s=x_s, image=image, image_feat=image_feat,
            s=s, u=u, t=t,
            time_eps=time_eps, probability_eps=probability_eps, flow=flow,
            valid_mask=valid_mask,
            loss_resolution=psd.get("loss_resolution", "state"),
            full_resolution_size=full_resolution_size,
            valid_mask_full=valid_mask_full,
        )
    elif loss_type == "csd":
        result = _csd_loss(
            model=model, x_s=x_s, image=image, image_feat=image_feat, s=s, t=t,
            time_eps=time_eps, probability_eps=probability_eps,
            jvp_dtype=jvp_dtype, flow=flow, valid_mask=valid_mask,
        )
    elif loss_type == "ecld":
        ecld = consistency.get("ecld", {})
        result = _ecld_loss(
            model=model, x_s=x_s, image=image, image_feat=image_feat, s=s, t=t,
            time_eps=time_eps, probability_eps=probability_eps,
            jvp_dtype=jvp_dtype,
            ec_weight=float(ecld.get("ec_weight", 4.0)),
            td_weight=float(ecld.get("td_weight", 2.0)),
            time_weighting=ecld.get("time_weighting", "none"),
            flow=flow, valid_mask=valid_mask,
        )
    else:
        invalid = consistency.get("invalid_teacher", {})
        adaptive = consistency.get("adaptive_kl", {})
        result = _esd_loss(
            model=model, x_s=x_s, image=image, image_feat=image_feat, s=s, t=t,
            time_eps=time_eps,
            log_eps=float(invalid.get("log_eps", 1.0e-6)),
            invalid_strategy=invalid.get("strategy", "mask_pixel"),
            skip_batch_threshold=invalid.get("skip_batch_threshold"),
            adaptive_enabled=bool(adaptive.get("enabled", False)),
            adaptive_c=float(adaptive.get("c", 1.0e-6)),
            adaptive_r=float(adaptive.get("r", 0.5)),
            adaptive_normalize_mean=bool(adaptive.get("normalize_mean", True)),
            adaptive_max_weight=adaptive.get("max_weight", 100.0),
            jvp_dtype=jvp_dtype, valid_mask=valid_mask,
        )
    if precision.get("debug_assertions", False):
        expected = None if loss_type == "psd" else _torch_jvp_dtype(jvp_dtype)
        if expected is not None:
            assert result.dtypes["student_logits_before_cast"] == expected
            assert result.dtypes["directional_logits_before_cast"] == expected
            assert result.dtypes["student_logits_after_cast"] == torch.float32
            assert result.dtypes["directional_logits_after_cast"] == torch.float32
        if result.student_prob is not None:
            assert result.student_prob.dtype == torch.float32
        if result.teacher_prob is not None:
            assert result.teacher_prob.dtype == torch.float32
        assert result.loss.dtype == torch.float32
    return result


def esd_loss(
    model,
    x_s: torch.Tensor,
    image: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    *,
    time_eps: float = 1.0e-5,
    log_eps: float = 1.0e-6,
    invalid_strategy: str = "mask_pixel",
    skip_batch_threshold: float | None = None,
    adaptive_enabled: bool = False,
    adaptive_c: float = 1.0e-6,
    adaptive_r: float = 0.5,
    adaptive_normalize_mean: bool = True,
    adaptive_max_weight: float | None = 100.0,
    jvp_dtype: str = "fp32",
    image_feat: torch.Tensor | None = None,
) -> ESDResult:
    result = _esd_loss(
        model=model, x_s=x_s, image=image, image_feat=image_feat, s=s, t=t,
        time_eps=time_eps, log_eps=log_eps,
        invalid_strategy=invalid_strategy,
        skip_batch_threshold=skip_batch_threshold,
        adaptive_enabled=adaptive_enabled,
        adaptive_c=adaptive_c,
        adaptive_r=adaptive_r,
        adaptive_normalize_mean=adaptive_normalize_mean,
        adaptive_max_weight=adaptive_max_weight,
        jvp_dtype=jvp_dtype,
    )
    return ESDResult(
        loss=result.loss,
        stats=result.stats,
        teacher_prob=result.teacher_prob,
        student_prob=result.student_prob,
        valid_pixel=result.valid_pixel,
        adaptive_weight=result.adaptive_weight,
        directional_logits=result.directional_output,
        dtypes=result.dtypes,
        loss_per_sample=result.loss_per_sample,
        valid_sample=result.valid_sample,
    )


def compute_training_losses(
    *,
    stage: str,
    model,
    x_path: torch.Tensor,
    image: torch.Tensor,
    target: torch.Tensor,
    diagonal_time: torch.Tensor,
    label_smoothing: float,
    primary_weight: float,
    source_stats: dict[str, torch.Tensor],
    esd_kwargs: dict[str, Any] | None = None,
    esd_effective_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], ESDResult | None]:
    """Legacy stage helper retained for checkpoint/tests compatibility."""
    diagonal_logits = model.forward_logits(
        x_path, image, diagonal_time, diagonal_time
    )
    diagonal_logits = resize_continuous(diagonal_logits, target.shape[-2:])
    diagonal = diagonal_cross_entropy(diagonal_logits, target, label_smoothing)
    total = (
        primary_weight * diagonal
        + source_stats["weighted_var"]
        + source_stats["weighted_align"]
    )
    stats = {
        "loss_diagonal": diagonal.detach(),
        "loss_source_var": source_stats["loss_source_var"].detach(),
        "loss_source_align": source_stats["loss_source_align"].detach(),
    }
    result = None
    if stage in {"esd_distillation", "consistency_distillation"}:
        if esd_kwargs is None:
            raise ValueError("Stage 2 requires esd_kwargs")
        result = esd_loss(model, x_path, image, **esd_kwargs)
        total = total + esd_effective_weight * result.loss
        stats["loss_esd"] = result.loss.detach()
        stats.update(result.stats)
    elif stage != "diagonal_pretrain":
        raise ValueError(f"Unknown training stage: {stage}")
    stats["loss_total"] = total.detach()
    return total, stats, result
