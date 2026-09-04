from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from adaptive_path import (
    adaptive_lambda,
    adaptive_lambda_derivative,
    entropy_adaptive_enabled,
    path_config as _path_config,
)
from state_space import resize_continuous, state_spatial_size
from source_model import source_statistics


def _time_view(time: torch.Tensor, ndim: int = 4) -> torch.Tensor:
    if time.ndim != 1:
        raise ValueError(f"time must have shape [B], got {tuple(time.shape)}")
    return time.reshape(time.shape[0], *([1] * (ndim - 1)))


def path_coefficient(
    time: torch.Tensor,
    config: dict | None = None,
    difficulty: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return alpha(t) while keeping raw model time inputs unchanged."""
    path = _path_config(config)
    path_type = path.get("type", "power")
    if path_type == "power":
        return time.pow(float(path.get("exponent", 1.0)))
    if path_type == "linear":
        return time
    if path_type == "entropy_adaptive":
        if difficulty is None:
            raise ValueError("entropy_adaptive path requires pixel difficulty")
        return adaptive_lambda(
            time, difficulty, beta=float(path["scheduler"]["beta"])
        )
    raise ValueError(f"Unknown flow path type: {path_type}")


def path_derivative(
    time: torch.Tensor,
    config: dict | None = None,
    difficulty: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return d alpha(t) / dt for the configured path."""
    path = _path_config(config)
    path_type = path.get("type", "power")
    if path_type == "power":
        exponent = float(path.get("exponent", 1.0))
        return exponent * time.pow(exponent - 1.0)
    if path_type == "linear":
        return torch.ones_like(time)
    if path_type == "entropy_adaptive":
        if difficulty is None:
            raise ValueError("entropy_adaptive path requires pixel difficulty")
        return adaptive_lambda_derivative(
            time, difficulty, beta=float(path["scheduler"]["beta"])
        )
    raise ValueError(f"Unknown flow path type: {path_type}")


def _state_view(value: torch.Tensor, ndim: int) -> torch.Tensor:
    if value.ndim == 1:
        return _time_view(value, ndim)
    if value.ndim == ndim - 1:
        return value.unsqueeze(1)
    raise ValueError(f"path coefficient has incompatible shape {tuple(value.shape)}")


def linear_path(
    x0: torch.Tensor,
    x1: torch.Tensor,
    time: torch.Tensor,
    config: dict | None = None,
    difficulty: torch.Tensor | None = None,
) -> torch.Tensor:
    """Backward-compatible name for the configurable interpolation path."""
    if x0.shape != x1.shape:
        raise ValueError("x0 and x1 must have the same shape")
    alpha = _state_view(
        path_coefficient(time, config, difficulty), x0.ndim
    ).to(dtype=x0.dtype)
    return (1.0 - alpha) * x0 + alpha * x1


def flow_map(
    x_s: torch.Tensor,
    mean_denoiser: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    time_eps: float = 1.0e-5,
    path_config: dict | None = None,
    difficulty: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generalized X_{s,t} using progress alpha(t), with raw s/t inputs."""
    if x_s.shape != mean_denoiser.shape:
        raise ValueError("x_s and mean_denoiser must have identical shapes")
    if s.shape != t.shape or s.ndim != 1:
        raise ValueError("s and t must both have shape [B]")
    alpha_s = path_coefficient(s, path_config, difficulty)
    alpha_t = path_coefficient(t, path_config, difficulty)
    denominator = (1.0 - alpha_s).clamp_min(time_eps)
    gamma = _state_view(
        (alpha_t - alpha_s) / denominator, x_s.ndim
    ).to(dtype=x_s.dtype)
    return x_s + gamma * (mean_denoiser - x_s)


def sample_stage1_times(
    batch_size: int,
    device: torch.device,
    min_time: float = 0.0,
    max_time: float = 1.0,
) -> torch.Tensor:
    return min_time + (max_time - min_time) * torch.rand(batch_size, device=device)


def sample_sorted_times(
    batch_size: int,
    device: torch.device,
    min_time: float = 0.0,
    max_time: float = 1.0,
    min_gap: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample sorted uniform times and enforce a positive minimum gap safely."""
    times = min_time + (max_time - min_time) * torch.rand(batch_size, 2, device=device)
    times, _ = torch.sort(times, dim=1)
    s, t = times.unbind(dim=1)
    too_close = (t - s) < min_gap
    if too_close.any():
        # Preserve t when possible, otherwise move s left from max_time.
        proposed_t = s + min_gap
        t = torch.where(too_close, proposed_t.clamp_max(max_time), t)
        s = torch.where(too_close & (t - s < min_gap), (t - min_gap).clamp_min(min_time), s)
    if not torch.all(t - s >= min_gap * (1.0 - 1.0e-4)):
        raise RuntimeError("Failed to enforce time_sampling.min_gap")
    return s, t


def sample_ordered_times(
    batch_size: int,
    count: int,
    device: torch.device,
    min_time: float = 0.0,
    max_time: float = 1.0,
    min_gap: float = 1.0e-5,
) -> tuple[torch.Tensor, ...]:
    """Sorted uniform times with rejection for every adjacent minimum gap."""
    if count < 2:
        raise ValueError("count must be at least 2")
    if (count - 1) * min_gap > max_time - min_time:
        raise ValueError("Configured time interval cannot fit all minimum gaps")
    times = min_time + (max_time - min_time) * torch.rand(
        batch_size, count, device=device
    )
    times = torch.sort(times, dim=1).values
    invalid = (times[:, 1:] - times[:, :-1] < min_gap).any(dim=1)
    attempts = 0
    while invalid.any() and attempts < 100:
        replacement = min_time + (max_time - min_time) * torch.rand(
            int(invalid.sum()), count, device=device
        )
        times[invalid] = torch.sort(replacement, dim=1).values
        invalid = (times[:, 1:] - times[:, :-1] < min_gap).any(dim=1)
        attempts += 1
    if invalid.any():
        fallback = torch.linspace(
            min_time, max_time, count, device=device, dtype=times.dtype
        )
        times[invalid] = fallback
    return tuple(times.unbind(dim=1))


def sample_consistency_times(
    loss_type: str,
    batch_size: int,
    device: torch.device,
    min_time: float = 0.0,
    max_time: float = 1.0,
    min_gap: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if loss_type == "psd":
        s, u, t = sample_ordered_times(
            batch_size, 3, device, min_time, max_time, min_gap
        )
        return s, u, t
    if loss_type in {"csd", "ecld", "esd"}:
        s, t = sample_ordered_times(
            batch_size, 2, device, min_time, max_time, min_gap
        )
        return s, None, t
    raise ValueError(f"Unknown consistency loss: {loss_type}")


def source_alignment_map_from_indices(
    mu_full: torch.Tensor,
    target_full: torch.Tensor,
    *,
    num_classes: int,
    eps: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """One-hot-free equivalent of mean((normalize(mu) - normalize(e_y))**2)."""
    if mu_full.shape != (
        target_full.shape[0], num_classes, *target_full.shape[-2:]
    ):
        raise ValueError(
            f"mu_full shape {tuple(mu_full.shape)} is incompatible with "
            f"target_full {tuple(target_full.shape)} and {num_classes} classes"
        )
    in_range = (target_full >= 0) & (target_full < num_classes)
    if valid_mask is None:
        if not bool(in_range.all()):
            raise ValueError("target_full contains an out-of-range class index")
        gather_mask = in_range
    else:
        if valid_mask.shape != target_full.shape:
            raise ValueError("valid_mask and target_full must have identical shapes")
        valid_mask = valid_mask.to(device=target_full.device, dtype=torch.bool)
        if bool((valid_mask & ~in_range).any()):
            raise ValueError("a valid target pixel contains an out-of-range class index")
        gather_mask = valid_mask & in_range
    safe_target = torch.where(
        gather_mask, target_full, torch.zeros_like(target_full)
    )
    mu_norm = F.normalize(mu_full, dim=1, eps=eps)
    target_component = mu_norm.gather(
        1, safe_target.unsqueeze(1)
    ).squeeze(1)
    # F.normalize(one_hot, eps) has magnitude 1/max(1, eps).
    target_scale = 1.0 / max(1.0, float(eps))
    return (
        mu_norm.square().sum(dim=1)
        + target_scale**2
        - 2.0 * target_scale * target_component
    ) / num_classes


def _sample_image_simplex_mixture(
    mu: torch.Tensor,
    parameters: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sample lambda*softmax(mu/T) + (1-lambda)*Dirichlet(alpha)."""
    lambda_value = float(parameters["lambda"])
    temperature = float(parameters["temperature"])
    dirichlet_alpha = float(parameters["dirichlet_alpha"])
    probability = torch.softmax(mu.float() / temperature, dim=1)
    if lambda_value == 1.0:
        x0_fp32 = probability
    else:
        batch, classes, height, width = mu.shape
        concentration = torch.full(
            (classes,), dirichlet_alpha, device=mu.device, dtype=torch.float32
        )
        noise = torch.distributions.Dirichlet(concentration).sample(
            (batch, height, width)
        ).permute(0, 3, 1, 2)
        x0_fp32 = lambda_value * probability + (1.0 - lambda_value) * noise
    entropy = -(
        probability * probability.clamp_min(torch.finfo(torch.float32).tiny).log()
    ).sum(dim=1)
    stats = {
        "source_prior_lambda": mu.new_tensor(lambda_value),
        "source_prior_temperature": mu.new_tensor(temperature),
        "source_prior_dirichlet_alpha": mu.new_tensor(dirichlet_alpha),
        "source_x0_sum_error": (x0_fp32.sum(dim=1) - 1.0).abs().amax(),
        "source_x0_min": x0_fp32.amin(),
        "source_x0_max": x0_fp32.amax(),
        "source_probability_entropy": entropy.mean(),
    }
    return x0_fp32.to(dtype=mu.dtype), stats


def sample_prior(
    config: dict,
    image: torch.Tensor,
    target_one_hot_state: torch.Tensor | None,
    source_model,
    *,
    target_full: torch.Tensor | None = None,
    valid_mask_full: torch.Tensor | None = None,
    sample_state: bool = True,
    sampling_mode: str = "training",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sample state-resolution x0 and compute optional full-resolution supervision."""
    if sampling_mode not in {"training", "inference"}:
        raise ValueError("sampling_mode must be training or inference")
    factor = config.get("model", {}).get("state_downsample_factor", 1)
    expected_size = state_spatial_size(image, factor)
    batch, _, height, width = (
        target_one_hot_state.shape
        if target_one_hot_state is not None
        else (
            image.shape[0], config["dataset"]["num_classes"], *expected_size
        )
    )
    if (height, width) != expected_size:
        raise AssertionError(
            f"target state {(height, width)} != image-derived state {expected_size}"
        )
    classes = config["dataset"]["num_classes"]
    source = config["source"]
    dtype = image.dtype
    zero = image.sum() * 0.0
    if source["prior_type"] == "gaussian":
        x0 = torch.randn(
            batch, classes, height, width, device=image.device, dtype=dtype
        ) * source["prior_noise_std"]
        stats = {
            "loss_source_var": zero, "loss_source_align": zero,
            "loss_source_ce": zero, "loss_source_supervision": zero,
            "weighted_var": zero, "weighted_align": zero,
            "weighted_source_supervision": zero,
            "source_x0_abs": x0.detach().abs().mean(),
        }
        if target_one_hot_state is not None:
            stats["target_x1_abs"] = target_one_hot_state.detach().abs().mean()
        return x0, stats
    if source["prior_type"] == "dirichlet":
        concentration = torch.ones(classes, device=image.device, dtype=torch.float32)
        x0 = torch.distributions.Dirichlet(concentration).sample(
            (batch, height, width)
        ).permute(0, 3, 1, 2).to(dtype=dtype)
        stats = {
            "loss_source_var": zero, "loss_source_align": zero,
            "loss_source_ce": zero, "loss_source_supervision": zero,
            "weighted_var": zero, "weighted_align": zero,
            "weighted_source_supervision": zero,
            "source_x0_abs": x0.detach().abs().mean(),
        }
        if target_one_hot_state is not None:
            stats["target_x1_abs"] = target_one_hot_state.detach().abs().mean()
        return x0, stats
    if source_model is None:
        raise RuntimeError(
            f"source.prior_type={source['prior_type']} requires a source model"
        )

    adaptive_frozen = entropy_adaptive_enabled(config) and source.get("freeze", False)
    source_context = torch.no_grad() if adaptive_frozen else nullcontext()
    simplex_stats: dict[str, torch.Tensor] = {}
    with source_context:
        if sample_state and source["prior_type"] == "image_gaussian":
            x0, mu, logvar = source_model(image)
        else:
            if (
                source["prior_type"] == "image_simplex_mixture"
                and getattr(source_model, "forward_statistics", None) is None
            ):
                raise RuntimeError(
                    "image_simplex_mixture requires source_model.forward_statistics"
                )
            mu, logvar = source_statistics(source_model, image)
            x0 = mu
        if sample_state and source["prior_type"] == "image_simplex_mixture":
            x0, simplex_stats = _sample_image_simplex_mixture(
                mu, source["simplex_prior"][sampling_mode]
            )
    assert x0.shape == mu.shape == logvar.shape
    assert x0.shape[:2] == (batch, classes)
    assert x0.shape[-2:] == (height, width), (
        f"source state {x0.shape[-2:]} != target state {(height, width)}"
    )
    fixed_std = getattr(source_model, "fixed_std", None)
    frozen_task_without_supervision = (
        source.get("type") == "task_finetuned_segformer"
        and source.get("freeze", False)
        and source.get("supervision", {}).get("type") in {None, "none"}
    )
    source_zero = image.new_zeros(()) if frozen_task_without_supervision else zero
    loss_var = (
        source_zero
        if source["prior_type"] == "image_simplex_mixture"
        or fixed_std is not None
        or adaptive_frozen
        else 0.5 * torch.mean(torch.exp(logvar) - logvar - 1.0)
    )
    supervision = source.get("supervision")
    if adaptive_frozen:
        # Stage 2 uses the checkpointed source only as x0/entropy conditioning.
        supervision_type = "none"
        supervision_weight = 0.0
    elif supervision is not None and supervision.get("type") is not None:
        supervision_type = supervision["type"]
        supervision_weight = (
            source["align_weight"]
            if supervision.get("weight") is None
            else supervision["weight"]
        )
    else:
        supervision_type = "align" if source.get("use_loss_align", False) else "none"
        supervision_weight = source.get("align_weight", 0.0)
    if (
        supervision_type in {"align", "cross_entropy"}
        and target_one_hot_state is not None
        and target_full is None
    ):
        raise ValueError(
            f"source supervision {supervision_type!r} requires integer target_full"
        )

    loss_align = source_zero
    loss_ce = source_zero
    if supervision_type == "align" and target_full is not None:
        mu_full = resize_continuous(mu, target_full.shape[-2:])
        alignment_map = source_alignment_map_from_indices(
            mu_full,
            target_full,
            num_classes=classes,
            eps=source["align_eps"],
            valid_mask=valid_mask_full,
        )
        if valid_mask_full is None:
            loss_align = alignment_map.mean()
        else:
            assert valid_mask_full.shape == target_full.shape
            weights = valid_mask_full.to(
                device=mu.device, dtype=alignment_map.dtype
            )
            loss_align = (
                (alignment_map * weights).sum() / weights.sum().clamp_min(1.0)
            )
    elif supervision_type == "cross_entropy" and target_full is not None:
        mu_full = resize_continuous(mu, target_full.shape[-2:])
        include_void = bool(supervision.get("include_void", False))
        ignore_index = (
            None if include_void else config.get("loss", {}).get("ignore_index")
        )
        loss_map = F.cross_entropy(
            mu_full,
            target_full,
            reduction="none",
            ignore_index=-100 if ignore_index is None else ignore_index,
        )
        source_valid_mask = None if include_void else valid_mask_full
        if source_valid_mask is None:
            loss_ce = loss_map.mean()
        else:
            assert source_valid_mask.shape == target_full.shape
            weights = source_valid_mask.to(loss_map)
            loss_ce = (loss_map * weights).sum() / weights.sum().clamp_min(1.0)
    loss_supervision = loss_align if supervision_type == "align" else loss_ce
    weighted_supervision = supervision_weight * loss_supervision
    sigma = torch.exp(0.5 * logvar.detach())
    stats = {
        "loss_source_var": loss_var,
        "loss_source_align": loss_align,
        "loss_source_ce": loss_ce,
        "loss_source_supervision": loss_supervision,
        "weighted_var": source["var_weight"] * loss_var,
        # weighted_align is a legacy dashboard/checkpoint-stat alias.
        "weighted_align": weighted_supervision,
        "weighted_source_supervision": weighted_supervision,
        "source_mu_abs": mu.detach().abs().mean(),
        "source_mu_min": mu.detach().amin(),
        "source_mu_max": mu.detach().amax(),
        "source_logvar_mean": logvar.detach().mean(),
        "source_sigma_mean": sigma.mean(),
        "source_x0_abs": x0.detach().abs().mean(),
    }
    stats.update({key: value.detach() for key, value in simplex_stats.items()})
    if simplex_stats:
        stats["source_prior_mode"] = mu.new_tensor(
            0.0 if sampling_mode == "training" else 1.0
        )
    if target_one_hot_state is not None:
        stats["target_x1_abs"] = target_one_hot_state.detach().abs().mean()
        assert x0.shape == target_one_hot_state.shape
    if entropy_adaptive_enabled(config):
        # Internal non-scalar payload consumed by training/inference in this batch.
        stats["_path_source_state"] = mu.detach()
    return x0, stats


def make_time_grid(num_steps: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if num_steps <= 0:
        raise ValueError("evaluation.num_steps must be positive")
    grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    return [(grid[index], grid[index + 1]) for index in range(num_steps)]
