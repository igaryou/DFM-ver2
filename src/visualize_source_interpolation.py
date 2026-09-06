from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from adaptive_path import (
    adaptive_lambda,
    bounded_gaussian_variance_maps,
    normalize_entropy,
    shannon_entropy,
)
from config import DEFAULT_CONFIG, load_config
from dataset import build_dataset
from discrete_flow_maps import sample_image_simplex_components
from source_model import source_statistics
from state_space import prepare_state_targets, smooth_categorical_target
from utils import resolve_device
from visualization import colorize
from visualize_simplex_source import (
    _inverse_normalized_image,
    _state_to_display,
    load_source_checkpoint,
    resolve_checkpoint,
)


DEFAULT_TIMES = (0.0, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95)
MODES = ("simplex", "bounded_gaussian")
RAW_GAUSSIAN_MODE = "raw_gaussian"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose linear interpolation from a learned source to GT"
    )
    parser.add_argument("--config", required=True)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint")
    checkpoint.add_argument("--checkpoint-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode", choices=(*MODES, RAW_GAUSSIAN_MODE, "both"), default="both"
    )
    parser.add_argument("--times", type=float, nargs="+", default=DEFAULT_TIMES)
    parser.add_argument(
        "--path-type", choices=("power", "entropy_adaptive"), default="power"
    )
    parser.add_argument("--path-exponent", type=float, default=1.0)
    parser.add_argument("--entropy-beta", type=float, default=None)
    parser.add_argument(
        "--entropy-scheduler", choices=("additive", "exponential"),
        default="additive",
    )
    parser.add_argument("--difficulty-gamma", type=float, default=1.0)
    parser.add_argument(
        "--entropy-normalization",
        choices=("rank", "mean", "zscore", "minmax"), default=None,
    )
    parser.add_argument("--entropy-eps", type=float, default=None)
    parser.add_argument("--entropy-zscore-clip", type=float, default=None)
    ignore_group = parser.add_mutually_exclusive_group()
    ignore_group.add_argument(
        "--entropy-exclude-ignore", dest="entropy_exclude_ignore",
        action="store_true",
    )
    ignore_group.add_argument(
        "--entropy-include-ignore", dest="entropy_exclude_ignore",
        action="store_false",
    )
    parser.set_defaults(entropy_exclude_ignore=None)
    parser.add_argument("--lambda", dest="lambda_value", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=6.0)
    parser.add_argument("--dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--tanh-temperature", type=float, default=5.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument(
        "--variance-type", choices=("fixed", "entropy_adaptive"),
        default="fixed",
    )
    parser.add_argument("--variance-rho", type=float, default=0.8)
    parser.add_argument("--target-smoothing-p", type=float, default=0.0)
    parser.add_argument("--compare-hard-target", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def validate_entropy_beta(beta: float, scheduler: str, *, label: str) -> None:
    """Validate beta using the selected entropy scheduler's domain."""
    if scheduler == "additive":
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"{label} must be in [0,1] for additive scheduler")
    elif scheduler == "exponential":
        if beta < 0.0:
            raise ValueError(f"{label} must be non-negative for exponential scheduler")
    else:
        raise ValueError(f"Unknown entropy scheduler: {scheduler}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.indices is not None and any(index < 0 for index in args.indices):
        raise ValueError("--indices must be non-negative")
    if args.path_exponent <= 0:
        raise ValueError("--path-exponent must be positive")
    if args.entropy_beta is not None:
        validate_entropy_beta(
            args.entropy_beta, args.entropy_scheduler, label="--entropy-beta"
        )
    if args.difficulty_gamma <= 0:
        raise ValueError("--difficulty-gamma must be positive")
    if args.entropy_eps is not None and args.entropy_eps <= 0:
        raise ValueError("--entropy-eps must be positive")
    if (
        args.entropy_zscore_clip is not None
        and args.entropy_zscore_clip <= 0
    ):
        raise ValueError("--entropy-zscore-clip must be positive")
    if not args.times or any(not 0.0 <= time <= 1.0 for time in args.times):
        raise ValueError("--times must contain values in [0,1]")
    if any(right <= left for left, right in zip(args.times, args.times[1:])):
        raise ValueError("--times must be strictly increasing")
    if not 0.0 <= args.lambda_value <= 1.0:
        raise ValueError("--lambda must be in [0,1]")
    for name in ("temperature", "dirichlet_alpha", "amplitude", "tanh_temperature"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.sigma < 0:
        raise ValueError("--sigma must be non-negative")
    if not 0.0 <= args.variance_rho < 1.0:
        raise ValueError("--variance-rho must satisfy 0 <= rho < 1")
    if (
        args.variance_type == "entropy_adaptive"
        and args.mode not in {"bounded_gaussian", "both"}
    ):
        raise ValueError(
            "--variance-type entropy_adaptive requires bounded_gaussian or both mode"
        )
    if not 0.0 <= args.target_smoothing_p < 1.0:
        raise ValueError("--target-smoothing-p must satisfy 0 <= p < 1")
    if args.compare_hard_target and args.target_smoothing_p == 0.0:
        raise ValueError("--compare-hard-target requires --target-smoothing-p > 0")
    if args.compare_hard_target and args.mode not in ("simplex", "both"):
        raise ValueError("--compare-hard-target requires simplex or both mode")
    return args


def resolve_entropy_scheduler_settings(
    args: argparse.Namespace, config: dict
) -> dict[str, Any]:
    """Resolve CLI-over-config adaptive scheduler settings."""
    default_path = DEFAULT_CONFIG["flow"]["path"]
    path = config.get("flow", {}).get("path", {})
    entropy_config = path.get("entropy", default_path["entropy"])
    scheduler_config = path.get("scheduler", default_path["scheduler"])
    exclude_ignore = (
        bool(entropy_config.get("exclude_ignore", True))
        if args.entropy_exclude_ignore is None
        else bool(args.entropy_exclude_ignore)
    )
    settings = {
        "beta": float(
            scheduler_config["beta"]
            if args.entropy_beta is None else args.entropy_beta
        ),
        "normalization": (
            entropy_config["normalization"]
            if args.entropy_normalization is None
            else args.entropy_normalization
        ),
        "eps": float(
            entropy_config["eps"]
            if args.entropy_eps is None else args.entropy_eps
        ),
        "zscore_clip": float(
            entropy_config["zscore_clip"]
            if args.entropy_zscore_clip is None
            else args.entropy_zscore_clip
        ),
        "exclude_ignore": exclude_ignore,
        "scheduler": args.entropy_scheduler,
        "difficulty_gamma": float(args.difficulty_gamma),
    }
    validate_entropy_beta(
        settings["beta"], settings["scheduler"], label="entropy beta"
    )
    if settings["difficulty_gamma"] <= 0:
        raise ValueError("difficulty gamma must be positive")
    if settings["eps"] <= 0:
        raise ValueError("entropy eps must be positive")
    if settings["zscore_clip"] <= 0:
        raise ValueError("entropy zscore_clip must be positive")
    return settings


def source_entropy_difficulty_from_raw_logits(
    mu_raw: torch.Tensor,
    *,
    target: torch.Tensor,
    void_index: int,
    settings: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``H(softmax(mu_raw))`` and production-normalized difficulty.

    ``representation="logits"`` makes :func:`shannon_entropy` apply softmax
    over the class dimension.  Only the original source logits are passed
    here: neither bounded ``a*tanh(mu_raw/T)`` nor Simplex
    ``softmax(mu_raw/T)`` participates in scheduler entropy.
    """
    entropy = shannon_entropy(
        mu_raw, representation="logits", eps=float(settings["eps"])
    )
    valid_mask = target != void_index
    normalization_mask = valid_mask if settings["exclude_ignore"] else None
    difficulty = normalize_entropy(
        entropy,
        settings["normalization"],
        valid_mask=normalization_mask,
        eps=float(settings["eps"]),
        zscore_clip=float(settings["zscore_clip"]),
        num_classes=mu_raw.shape[1],
    )
    return entropy, difficulty, valid_mask


def emphasize_difficulty(
    difficulty: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Return ``sign(d) * abs(d)**gamma`` without changing its sign/range."""
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError("difficulty gamma must be positive")
    return difficulty.sign() * difficulty.abs().pow(gamma)


def entropy_scheduler_lambda(
    time: torch.Tensor,
    difficulty: torch.Tensor,
    *,
    beta: float,
    scheduler: str,
) -> torch.Tensor:
    """Compute additive or exponential entropy-adaptive progress maps."""
    if scheduler == "additive":
        return adaptive_lambda(time, difficulty, beta=beta)
    if scheduler != "exponential":
        raise ValueError(f"Unknown entropy scheduler: {scheduler}")
    if time.ndim != 1 or difficulty.ndim != 3:
        raise ValueError("time must be [B] and difficulty must be [B,H,W]")
    if time.shape[0] != difficulty.shape[0]:
        raise ValueError("time and difficulty batch sizes must match")
    t = time.float()[:, None, None]
    exponent = torch.exp(float(beta) * difficulty.float())
    return t.pow(exponent)


def interpolation_path(
    x0: torch.Tensor,
    x1: torch.Tensor,
    time: float,
    *,
    path_type: str,
    path_exponent: float,
    difficulty: torch.Tensor | None = None,
    entropy_beta: float | None = None,
    entropy_scheduler: str = "additive",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return x_t and its optional per-pixel adaptive lambda map."""
    if path_type == "power":
        return linear_interpolation(x0, x1, time, path_exponent), None
    if path_type != "entropy_adaptive":
        raise ValueError(f"Unknown path type: {path_type}")
    if difficulty is None or entropy_beta is None:
        raise ValueError("entropy_adaptive requires difficulty and entropy_beta")
    batch_time = x0.new_full((x0.shape[0],), float(time), dtype=torch.float32)
    coefficient = entropy_scheduler_lambda(
        batch_time, difficulty, beta=float(entropy_beta),
        scheduler=entropy_scheduler,
    )
    coefficient_state = coefficient[:, None].to(dtype=x0.dtype)
    state = coefficient_state * x1 + (1.0 - coefficient_state) * x0
    return state, coefficient


def adaptive_scheduler_statistics(
    entropy: torch.Tensor,
    difficulty: torch.Tensor,
    coefficient: torch.Tensor,
    valid_mask: torch.Tensor,
) -> list[dict[str, float | int]]:
    """Return per-sample scheduler statistics plus pooled-moment payloads."""
    results: list[dict[str, float | int]] = []
    for sample_index in range(entropy.shape[0]):
        valid = valid_mask[sample_index]
        result: dict[str, float | int] = {}

        def add_moments(name: str, tensor: torch.Tensor, mask: torch.Tensor) -> None:
            values = tensor[sample_index][mask].float()
            count = values.numel()
            total = float(values.sum()) if count else 0.0
            square_total = float(values.square().sum()) if count else 0.0
            mean = total / count if count else float("nan")
            variance = max(square_total / count - mean * mean, 0.0) if count else 0.0
            result[name + "_mean"] = mean
            result[name + "_std"] = variance ** 0.5 if count else float("nan")
            result["_" + name + "_count"] = count
            result["_" + name + "_sum"] = total
            result["_" + name + "_sum_sq"] = square_total

        add_moments("lambda", coefficient, valid)
        add_moments("difficulty", difficulty, valid)
        add_moments("entropy", entropy, valid)
        lambda_values = coefficient[sample_index][valid].float()
        result["lambda_min"] = (
            float(lambda_values.min()) if lambda_values.numel() else float("nan")
        )
        result["lambda_max"] = (
            float(lambda_values.max()) if lambda_values.numel() else float("nan")
        )
        for label, mask in (
            ("easy", valid & (difficulty[sample_index] < 0)),
            ("hard", valid & (difficulty[sample_index] > 0)),
        ):
            values = coefficient[sample_index][mask].float()
            result[f"lambda_{label}_mean"] = (
                float(values.mean()) if values.numel() else float("nan")
            )
            result[f"_lambda_{label}_count"] = values.numel()
            result[f"_lambda_{label}_sum"] = (
                float(values.sum()) if values.numel() else 0.0
            )
        results.append(result)
    return results


def raw_gaussian_components(
    mu: torch.Tensor,
    *,
    sigma: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample x0 = mu + sigma * epsilon without transforming raw source logits."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    devices = [] if mu.device.type != "cuda" else [mu.device.index or 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        epsilon = torch.randn_like(mu)
    return epsilon, mu + float(sigma) * epsilon


def bounded_gaussian_components(
    mu: torch.Tensor,
    *,
    amplitude: float,
    tanh_temperature: float,
    sigma: float,
    seed: int,
    sigma_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the diagnostic bounded Gaussian from raw source logits."""
    if amplitude <= 0 or tanh_temperature <= 0 or sigma < 0:
        raise ValueError("amplitude/tanh_temperature must be positive and sigma non-negative")
    mu_new = float(amplitude) * torch.tanh(mu.float() / float(tanh_temperature))
    devices = [] if mu.device.type != "cuda" else [mu.device.index or 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        noise = torch.randn(mu.shape, device=mu.device, dtype=torch.float32)
    if sigma_map is None:
        # Preserve the exact legacy fixed-variance operation.
        x0 = mu_new + float(sigma) * noise
    else:
        if sigma_map.shape != (mu.shape[0], *mu.shape[-2:]):
            raise ValueError("sigma_map must have shape [B,H,W]")
        x0 = mu_new + sigma_map.float()[:, None] * noise
    if not torch.equal(mu.argmax(dim=1), mu_new.argmax(dim=1)):
        raise AssertionError("Positive tanh transform changed raw-logit argmax")
    return mu_new, noise, x0


def linear_interpolation(
    x0: torch.Tensor,
    x1: torch.Tensor,
    time: float,
    path_exponent: float = 1.0,
) -> torch.Tensor:
    """Interpolate with alpha(t)=t**path_exponent (legacy p=1 by default)."""
    path_exponent = float(path_exponent)
    if path_exponent <= 0:
        raise ValueError("path_exponent must be positive")
    alpha = float(time) ** path_exponent
    return alpha * x1 + (1.0 - alpha) * x0


def gt_margin(state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gt = state.gather(1, target[:, None]).squeeze(1)
    competitors = state.clone()
    competitors.scatter_(1, target[:, None], -torch.inf)
    return gt - competitors.amax(dim=1)


def interpolation_statistics(
    states: list[torch.Tensor],
    times: list[float],
    target: torch.Tensor,
    source_prediction: torch.Tensor,
    void_index: int,
) -> tuple[list[dict[str, Any]], torch.Tensor, dict[str, list[float]]]:
    valid = target != void_index
    source_correct = valid & (source_prediction == target)
    source_incorrect = valid & ~source_correct
    first = torch.full(target.shape, -1, dtype=torch.long, device=target.device)
    rows: list[dict[str, Any]] = []
    margins: dict[str, list[float]] = {}

    def ratio(mask: torch.Tensor, hits: torch.Tensor) -> tuple[int, int, float]:
        denominator = int(mask.sum())
        numerator = int((hits & mask).sum())
        return numerator, denominator, numerator / denominator if denominator else float("nan")

    for time_index, (time, state) in enumerate(zip(times, states, strict=True)):
        prediction = state.argmax(dim=1)
        hits = prediction == target
        newly_correct = valid & hits & (first < 0)
        first[newly_correct] = time_index
        margin = gt_margin(state.float(), target)
        row: dict[str, Any] = {"t": float(time)}
        for label, mask in (
            ("", valid),
            ("_source_correct", source_correct),
            ("_source_incorrect", source_incorrect),
        ):
            numerator, denominator, value = ratio(mask, hits)
            values = margin[mask].detach().float().cpu()
            prefix = "gt_argmax_ratio" + label
            row[prefix] = value
            row[prefix + "_numerator"] = numerator
            row[prefix + "_denominator"] = denominator
            margin_prefix = label[1:] + "_" if label else ""
            row[margin_prefix + "mean_gt_margin"] = (
                float(values.mean()) if values.numel() else float("nan")
            )
            row[margin_prefix + "median_gt_margin"] = (
                float(values.median()) if values.numel() else float("nan")
            )
            margins[f"{time_index}:{label or 'all'}"] = values.tolist()
        rows.append(row)
    return rows, first[valid].detach().cpu(), margins


def batched_interpolation_statistics(
    states: list[torch.Tensor],
    times: list[float],
    target: torch.Tensor,
    source_prediction: torch.Tensor,
    void_index: int,
) -> list[tuple[list[dict[str, Any]], torch.Tensor, dict[str, list[float]]]]:
    """Compute trajectory predictions/margins in batch, retaining sample rows."""
    batch_size = target.shape[0]
    valid = target != void_index
    source_correct = valid & (source_prediction == target)
    source_incorrect = valid & ~source_correct
    first = torch.full_like(target, -1, dtype=torch.long)
    rows: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]
    margins: list[dict[str, list[float]]] = [
        {} for _ in range(batch_size)
    ]

    for time_index, (time, state) in enumerate(zip(times, states, strict=True)):
        prediction = state.argmax(dim=1)
        hits = prediction == target
        newly_correct = valid & hits & (first < 0)
        first[newly_correct] = time_index
        margin = gt_margin(state.float(), target)
        for sample_index in range(batch_size):
            row: dict[str, Any] = {"t": float(time)}
            for label, batch_mask in (
                ("", valid),
                ("_source_correct", source_correct),
                ("_source_incorrect", source_incorrect),
            ):
                mask = batch_mask[sample_index]
                sample_hits = hits[sample_index]
                denominator = int(mask.sum())
                numerator = int((sample_hits & mask).sum())
                prefix = "gt_argmax_ratio" + label
                row[prefix] = (
                    numerator / denominator if denominator else float("nan")
                )
                row[prefix + "_numerator"] = numerator
                row[prefix + "_denominator"] = denominator
                values = margin[sample_index][mask].detach().float().cpu()
                margin_prefix = label[1:] + "_" if label else ""
                row[margin_prefix + "mean_gt_margin"] = (
                    float(values.mean()) if values.numel() else float("nan")
                )
                row[margin_prefix + "median_gt_margin"] = (
                    float(values.median()) if values.numel() else float("nan")
                )
                margins[sample_index][
                    f"{time_index}:{label or 'all'}"
                ] = values.tolist()
            rows[sample_index].append(row)

    return [
        (
            rows[sample_index],
            first[sample_index][valid[sample_index]].detach().cpu(),
            margins[sample_index],
        )
        for sample_index in range(batch_size)
    ]


def _semantic_display(state: torch.Tensor, sample: dict) -> torch.Tensor:
    return _state_to_display(state, sample).argmax(dim=1)[0].cpu()


def _save_mode_figure(
    path: Path,
    image: torch.Tensor,
    target: torch.Tensor,
    source: torch.Tensor,
    x0: torch.Tensor,
    states: list[torch.Tensor],
    times: list[float],
    sample: dict,
    mode: str,
    mu_new: torch.Tensor | None = None,
    path_description: str | None = None,
    lambda_means: list[float] | None = None,
) -> None:
    panels: list[tuple[str, Any, bool]] = [
        ("Input", image.permute(1, 2, 0), True),
        ("GT", target, False),
        ("Source argmax(mu)", source, False),
    ]
    if mu_new is not None:
        panels.append(("argmax(mu_new)", _semantic_display(mu_new, sample), False))
    x0_title = "x0 = mu + sigma*eps" if mode == RAW_GAUSSIAN_MODE else "x0"
    panels.append((x0_title, _semantic_display(x0, sample), False))
    panels.extend(
        (
            f"t={time:g}" + (
                f"\nmean lambda={lambda_means[index]:.3f}"
                if lambda_means is not None else ""
            ),
            _semantic_display(state, sample), False,
        )
        for index, (time, state) in enumerate(zip(times, states, strict=True))
        if time != 0.0
    )
    columns = 4
    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    axes = np.asarray(axes).reshape(-1)
    for axis, (title, values, rgb) in zip(axes, panels, strict=False):
        axis.imshow(values if rgb else colorize(values, "cityscapes"))
        axis.set_title(title)
        axis.axis("off")
    for axis in axes[len(panels):]:
        axis.set_visible(False)
    title = mode.replace("_", " ").title()
    if path_description is not None:
        title += " | " + path_description
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=130, bbox_inches="tight"); plt.close(figure)


def _save_scheduler_figure(
    path: Path,
    image: torch.Tensor,
    source: torch.Tensor,
    entropy: torch.Tensor,
    difficulty: torch.Tensor,
    lambda_maps: list[torch.Tensor],
    times: list[float],
    title: str,
) -> None:
    panels = 4 + len(lambda_maps)
    columns = 4
    rows = math.ceil(panels / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    axes = np.asarray(axes).reshape(-1)
    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title("Input")
    axes[1].imshow(colorize(source, "cityscapes"))
    axes[1].set_title("Source argmax(mu_raw)")
    entropy_plot = axes[2].imshow(entropy, cmap="viridis")
    axes[2].set_title("Source entropy H")
    figure.colorbar(entropy_plot, ax=axes[2], fraction=0.046, pad=0.04)
    difficulty_plot = axes[3].imshow(
        difficulty, cmap="coolwarm", vmin=-1.0, vmax=1.0
    )
    axes[3].set_title("Effective difficulty d′")
    figure.colorbar(difficulty_plot, ax=axes[3], fraction=0.046, pad=0.04)
    for index, (time, coefficient) in enumerate(
        zip(times, lambda_maps, strict=True), start=4
    ):
        coefficient_plot = axes[index].imshow(
            coefficient, cmap="viridis", vmin=0.0, vmax=1.0
        )
        axes[index].set_title(
            f"lambda(t={time:g})\nmean={float(coefficient.mean()):.3f}"
        )
        figure.colorbar(
            coefficient_plot, ax=axes[index], fraction=0.046, pad=0.04
        )
    for axis in axes:
        axis.axis("off")
    for axis in axes[panels:]:
        axis.set_visible(False)
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def _save_variance_figure(
    path: Path,
    image: torch.Tensor,
    entropy: torch.Tensor,
    difficulty: torch.Tensor,
    variance: torch.Tensor,
    std: torch.Tensor,
    x0: torch.Tensor,
    sample: dict,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = (
        ("Input", image.permute(1, 2, 0), None),
        ("Entropy H(softmax(mu_raw))", entropy, "viridis"),
        ("Difficulty d (GT-independent)", difficulty, "coolwarm"),
        ("Variance sigma_i^2", variance, "viridis"),
        ("Std sigma_i", std, "viridis"),
    )
    for axis, (title, values, cmap) in zip(axes.reshape(-1), panels):
        limits = {"vmin": -1.0, "vmax": 1.0} if "Difficulty" in title else {}
        plot = axis.imshow(values, cmap=cmap, **limits)
        axis.set_title(title)
        axis.axis("off")
        if cmap is not None:
            figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04)
    axes[1, 2].imshow(colorize(_semantic_display(x0, sample), "cityscapes"))
    axes[1, 2].set_title("argmax(x0)")
    axes[1, 2].axis("off")
    figure.suptitle("Bounded Gaussian Entropy-Adaptive Variance")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def _save_comparison(
    path: Path, image: torch.Tensor, target: torch.Tensor,
    mode_states: dict[str, list[torch.Tensor]], mode_x0: dict[str, torch.Tensor],
    times: list[float], sample: dict,
) -> None:
    displayed = 1 + sum(time != 0.0 for time in times)
    columns = displayed + 2
    figure, axes = plt.subplots(2, columns, figsize=(3.2 * columns, 6.5))
    for row, mode in enumerate(MODES):
        axes[row, 0].imshow(image.permute(1, 2, 0))
        axes[row, 0].set_title(f"{mode.replace('_', ' ').title()} / Input")
        axes[row, 1].imshow(colorize(target, "cityscapes")); axes[row, 1].set_title("GT")
        trajectory = [("x0", mode_x0[mode])] + [
            (f"t={time:g}", state)
            for time, state in zip(times, mode_states[mode], strict=True)
            if time != 0.0
        ]
        for column, (label, state) in enumerate(trajectory, 2):
            axes[row, column].imshow(colorize(_semantic_display(state, sample), "cityscapes"))
            axes[row, column].set_title(label)
        axes[row, 0].set_ylabel(mode.replace("_", " "))
        for axis in axes[row]: axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=120, bbox_inches="tight"); plt.close(figure)


def _save_hard_target_comparison(
    path: Path,
    image: torch.Tensor,
    target: torch.Tensor,
    states: dict[str, list[torch.Tensor]],
    x0: torch.Tensor,
    times: list[float],
    sample: dict,
) -> None:
    labels = ("simplex_hard_target", "simplex")
    displayed = 1 + sum(time != 0.0 for time in times)
    figure, axes = plt.subplots(2, displayed + 2, figsize=(3.2 * (displayed + 2), 6.5))
    for row, (key, title) in enumerate(zip(labels, ("Hard target", "Smoothed target"), strict=True)):
        axes[row, 0].imshow(image.permute(1, 2, 0)); axes[row, 0].set_title(f"{title} / Input")
        axes[row, 1].imshow(colorize(target, "cityscapes")); axes[row, 1].set_title("GT")
        trajectory = [("x0", x0)] + [
            (f"t={time:g}", state)
            for time, state in zip(times, states[key], strict=True)
            if time != 0.0
        ]
        for column, (label, state) in enumerate(trajectory, 2):
            axes[row, column].imshow(colorize(_semantic_display(state, sample), "cityscapes"))
            axes[row, column].set_title(label)
        for axis in axes[row]: axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=120, bbox_inches="tight"); plt.close(figure)


def _aggregate_rows(rows: list[dict[str, Any]], times: list[float]) -> list[dict[str, Any]]:
    result = []
    for mode in dict.fromkeys(row["mode"] for row in rows):
        selected_mode = [row for row in rows if row["mode"] == mode]
        if not selected_mode:
            continue
        for time in times:
            selected = [row for row in selected_mode if row["t"] == time]
            aggregate: dict[str, Any] = {"mode": mode, "t": time}
            for suffix in ("", "_source_correct", "_source_incorrect"):
                key = "gt_argmax_ratio" + suffix
                numerator = sum(row[key + "_numerator"] for row in selected)
                denominator = sum(row[key + "_denominator"] for row in selected)
                aggregate[key] = numerator / denominator if denominator else float("nan")
                aggregate[key + "_numerator"] = numerator
                aggregate[key + "_denominator"] = denominator
            for key in (
                "mean_gt_margin", "source_correct_mean_gt_margin",
                "source_incorrect_mean_gt_margin",
            ):
                weight_key = "gt_argmax_ratio" + (
                    "_source_correct" if key.startswith("source_correct") else
                    "_source_incorrect" if key.startswith("source_incorrect") else ""
                ) + "_denominator"
                total_weight = sum(row[weight_key] for row in selected)
                aggregate[key] = (
                    sum(row[key] * row[weight_key] for row in selected if math.isfinite(row[key])) / total_weight
                    if total_weight else float("nan")
                )
            if selected and "_lambda_count" in selected[0]:
                for name in ("lambda", "difficulty", "entropy"):
                    count = sum(row[f"_{name}_count"] for row in selected)
                    total = sum(row[f"_{name}_sum"] for row in selected)
                    square_total = sum(
                        row[f"_{name}_sum_sq"] for row in selected
                    )
                    mean = total / count if count else float("nan")
                    variance = (
                        max(square_total / count - mean * mean, 0.0)
                        if count else float("nan")
                    )
                    aggregate[f"{name}_mean"] = mean
                    aggregate[f"{name}_std"] = variance ** 0.5
                finite_minima = [
                    row["lambda_min"] for row in selected
                    if math.isfinite(row["lambda_min"])
                ]
                finite_maxima = [
                    row["lambda_max"] for row in selected
                    if math.isfinite(row["lambda_max"])
                ]
                aggregate["lambda_min"] = (
                    min(finite_minima) if finite_minima else float("nan")
                )
                aggregate["lambda_max"] = (
                    max(finite_maxima) if finite_maxima else float("nan")
                )
                for label in ("easy", "hard"):
                    count = sum(
                        row[f"_lambda_{label}_count"] for row in selected
                    )
                    total = sum(
                        row[f"_lambda_{label}_sum"] for row in selected
                    )
                    aggregate[f"lambda_{label}_mean"] = (
                        total / count if count else float("nan")
                    )
            result.append(aggregate)
    return result


def _save_summary_plots(rows: list[dict[str, Any]], output: Path) -> None:
    for filename, keys, ylabel in (
        ("gt_argmax_ratio.png", ("gt_argmax_ratio",), "P[argmax(x_t) = y]"),
        ("gt_argmax_ratio_by_source_correctness.png", (
            "gt_argmax_ratio_source_correct", "gt_argmax_ratio_source_incorrect"
        ), "Conditional GT argmax ratio"),
        ("gt_margin.png", ("mean_gt_margin",), "Mean GT margin"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        for mode in dict.fromkeys(row["mode"] for row in rows):
            selected = [row for row in rows if row["mode"] == mode]
            for key in keys:
                if selected:
                    label = mode.replace("_", " ") + (" / " + key.removeprefix("gt_argmax_ratio_") if len(keys) > 1 else "")
                    axis.plot([row["t"] for row in selected], [row[key] for row in selected], marker="o", label=label)
        axis.set_xlabel("t"); axis.set_ylabel(ylabel); axis.grid(alpha=0.3); axis.legend()
        figure.tight_layout(); figure.savefig(output / filename, dpi=150); plt.close(figure)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config, args.set)
    scheduler_settings = (
        resolve_entropy_scheduler_settings(args, config)
        if args.path_type == "entropy_adaptive" else None
    )
    if config["dataset"]["name"] != "cityscapes":
        raise ValueError("This diagnostic currently supports Cityscapes")
    if config["source"]["type"] != "trainable_segformer" or config["source"]["segformer_variant"] != "b1":
        raise ValueError("A trainable SegFormer-B1 source config is required")
    device = resolve_device(args.device or config["runtime"]["device"])
    checkpoint_path = resolve_checkpoint(args.checkpoint, args.checkpoint_dir)
    checkpoint, source_model = load_source_checkpoint(config, checkpoint_path, device)
    source_model.eval().requires_grad_(False)
    dataset = build_dataset(config, args.split, augment=False)
    indices = args.indices or list(range(min(args.num_images, len(dataset))))
    if any(index >= len(dataset) for index in indices):
        raise IndexError(f"Dataset index exceeds split size {len(dataset)}")
    output = Path(args.output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    modes = MODES if args.mode == "both" else (args.mode,)
    times = [float(time) for time in args.times]
    rows: list[dict[str, Any]] = []
    first_times: dict[str, list[float | None]] = {mode: [] for mode in modes}
    margin_values: dict[tuple[str, int, str], list[float]] = {}
    gaussian_totals = {"pixels": 0, "raw_abs": 0.0, "new_abs": 0.0, "noise_abs": 0.0, "x0_abs": 0.0,
                       "raw_min": math.inf, "raw_max": -math.inf, "new_min": math.inf, "new_max": -math.inf, "flips": 0}
    raw_gaussian_totals = {
        "elements": 0, "semantic_pixels": 0, "mu_abs": 0.0,
        "mu_min": math.inf, "mu_max": -math.inf, "noise_abs": 0.0,
        "x0_abs": 0.0, "x0_min": math.inf, "x0_max": -math.inf, "flips": 0,
    }
    simplex_flips = {"pixels": 0, "source_q": 0, "q_x0": 0}
    shapes: dict[str, Any] = {}

    for batch_start in range(0, len(indices), args.batch_size):
        batch_indices = indices[batch_start:batch_start + args.batch_size]
        samples = [dataset[dataset_index] for dataset_index in batch_indices]
        image = torch.stack([sample["image"] for sample in samples]).to(device)
        target_full = torch.stack(
            [sample["target"].long() for sample in samples]
        ).to(device)
        mu, _ = source_statistics(source_model, image)
        targets = prepare_state_targets(
            target_full, num_classes=config["dataset"]["num_classes"],
            state_size=mu.shape[-2:], ignore_index=config["dataset"]["void_class_index"],
            mask_pixel_losses=True,
        )
        x1_hard, target = targets.one_hot_state, targets.target_state
        x1 = smooth_categorical_target(x1_hard, args.target_smoothing_p)
        source_prediction = mu.argmax(dim=1)
        entropy = difficulty = entropy_valid_mask = None
        lambda_maps: list[torch.Tensor] | None = None
        if scheduler_settings is not None:
            entropy, difficulty, entropy_valid_mask = (
                source_entropy_difficulty_from_raw_logits(
                    mu, target=target,
                    void_index=config["dataset"]["void_class_index"],
                    settings=scheduler_settings,
                )
            )
            difficulty = emphasize_difficulty(
                difficulty, scheduler_settings["difficulty_gamma"]
            )
            lambda_maps = [
                entropy_scheduler_lambda(
                    mu.new_full((mu.shape[0],), time, dtype=torch.float32),
                    difficulty, beta=scheduler_settings["beta"],
                    scheduler=scheduler_settings["scheduler"],
                )
                for time in times
            ]

        def trajectory_states(
            initial: torch.Tensor, endpoint: torch.Tensor
        ) -> list[torch.Tensor]:
            if lambda_maps is None:
                return [
                    linear_interpolation(
                        initial, endpoint, time, args.path_exponent
                    )
                    for time in times
                ]
            return [
                coefficient[:, None].to(initial.dtype) * endpoint
                + (1.0 - coefficient[:, None].to(initial.dtype)) * initial
                for coefficient in lambda_maps
            ]

        sample_seeds = [int(args.seed + index * 2) for index in batch_indices]
        mode_states: dict[str, list[torch.Tensor]] = {}
        mode_x0: dict[str, torch.Tensor] = {}
        mode_mu_new: dict[str, torch.Tensor | None] = {}
        variance_entropy = variance_difficulty = variance_map = variance_std = None
        if "bounded_gaussian" in modes and args.variance_type == "entropy_adaptive":
            (
                variance_entropy, variance_difficulty, variance_map, variance_std
            ) = bounded_gaussian_variance_maps(
                mu, base_std=args.sigma, variance_type="entropy_adaptive",
                rho=args.variance_rho, normalization="rank", eps=1.0e-8,
            )
        if "simplex" in modes:
            components = [
                sample_image_simplex_components(
                    mu[offset:offset + 1], lambda_value=args.lambda_value,
                    temperature=args.temperature,
                    dirichlet_alpha=args.dirichlet_alpha, seed=sample_seed,
                )
                for offset, sample_seed in enumerate(sample_seeds)
            ]
            q = torch.cat([component[0] for component in components], dim=0)
            x0 = torch.cat([component[2] for component in components], dim=0)
            source_q = source_prediction != q.argmax(dim=1)
            if source_q.any():
                raise AssertionError("softmax temperature changed source argmax")
            simplex_flips["pixels"] += source_q.numel()
            simplex_flips["source_q"] += int(source_q.sum())
            simplex_flips["q_x0"] += int(
                (q.argmax(dim=1) != x0.argmax(dim=1)).sum()
            )
            mode_states["simplex"] = trajectory_states(x0, x1)
            mode_x0["simplex"] = x0
            mode_mu_new["simplex"] = None
            if args.compare_hard_target:
                mode_states["simplex_hard_target"] = trajectory_states(
                    x0, x1_hard
                )
                mode_x0["simplex_hard_target"] = x0
                mode_mu_new["simplex_hard_target"] = None
        if "bounded_gaussian" in modes:
            components = [
                bounded_gaussian_components(
                    mu[offset:offset + 1], amplitude=args.amplitude,
                    tanh_temperature=args.tanh_temperature, sigma=args.sigma,
                    seed=sample_seed + 1,
                    sigma_map=(
                        None if variance_std is None
                        else variance_std[offset:offset + 1]
                    ),
                )
                for offset, sample_seed in enumerate(sample_seeds)
            ]
            mu_new = torch.cat([component[0] for component in components], dim=0)
            noise = torch.cat([component[1] for component in components], dim=0)
            x0 = torch.cat([component[2] for component in components], dim=0)
            flips = source_prediction != mu_new.argmax(dim=1)
            values = mu.numel()
            gaussian_totals["pixels"] += values
            gaussian_totals["raw_abs"] += float(mu.float().abs().sum())
            gaussian_totals["new_abs"] += float(mu_new.abs().sum())
            gaussian_totals["noise_abs"] += float(noise.abs().sum())
            gaussian_totals["x0_abs"] += float(x0.abs().sum())
            gaussian_totals["raw_min"] = min(
                gaussian_totals["raw_min"], float(mu.min())
            )
            gaussian_totals["raw_max"] = max(
                gaussian_totals["raw_max"], float(mu.max())
            )
            gaussian_totals["new_min"] = min(
                gaussian_totals["new_min"], float(mu_new.min())
            )
            gaussian_totals["new_max"] = max(
                gaussian_totals["new_max"], float(mu_new.max())
            )
            gaussian_totals["flips"] += int(flips.sum())
            mode_states["bounded_gaussian"] = trajectory_states(x0, x1)
            mode_x0["bounded_gaussian"] = x0
            mode_mu_new["bounded_gaussian"] = mu_new
        if RAW_GAUSSIAN_MODE in modes:
            components = [
                raw_gaussian_components(
                    mu[offset:offset + 1], sigma=args.sigma, seed=sample_seed,
                )
                for offset, sample_seed in enumerate(sample_seeds)
            ]
            epsilon = torch.cat([component[0] for component in components], dim=0)
            x0 = torch.cat([component[1] for component in components], dim=0)
            elements = mu.numel()
            raw_gaussian_totals["elements"] += elements
            raw_gaussian_totals["semantic_pixels"] += source_prediction.numel()
            raw_gaussian_totals["mu_abs"] += float(mu.abs().sum())
            raw_gaussian_totals["mu_min"] = min(
                raw_gaussian_totals["mu_min"], float(mu.min())
            )
            raw_gaussian_totals["mu_max"] = max(
                raw_gaussian_totals["mu_max"], float(mu.max())
            )
            raw_gaussian_totals["noise_abs"] += float(epsilon.abs().sum())
            raw_gaussian_totals["x0_abs"] += float(x0.abs().sum())
            raw_gaussian_totals["x0_min"] = min(
                raw_gaussian_totals["x0_min"], float(x0.min())
            )
            raw_gaussian_totals["x0_max"] = max(
                raw_gaussian_totals["x0_max"], float(x0.max())
            )
            raw_gaussian_totals["flips"] += int(
                (source_prediction != x0.argmax(dim=1)).sum()
            )
            mode_states[RAW_GAUSSIAN_MODE] = trajectory_states(x0, x1_hard)
            mode_x0[RAW_GAUSSIAN_MODE] = x0
            mode_mu_new[RAW_GAUSSIAN_MODE] = None

        batch_trajectory_statistics = {
            mode: batched_interpolation_statistics(
                states, times, target, source_prediction,
                config["dataset"]["void_class_index"],
            )
            for mode, states in mode_states.items()
        }
        batch_scheduler_statistics = None
        if lambda_maps is not None:
            statistics_mask = (
                entropy_valid_mask
                if scheduler_settings["exclude_ignore"]
                else torch.ones_like(entropy_valid_mask)
            )
            batch_scheduler_statistics = [
                adaptive_scheduler_statistics(
                    entropy, difficulty, coefficient, statistics_mask
                )
                for coefficient in lambda_maps
            ]
        for offset, (dataset_index, sample) in enumerate(
            zip(batch_indices, samples, strict=True)
        ):
            ordinal = batch_start + offset
            sample_states = {
                mode: [state[offset:offset + 1] for state in states]
                for mode, states in mode_states.items()
            }
            sample_x0 = {
                mode: state[offset:offset + 1]
                for mode, state in mode_x0.items()
            }
            sample_mu_new = {
                mode: None if state is None else state[offset:offset + 1]
                for mode, state in mode_mu_new.items()
            }
            display_target = sample["target"].cpu()
            display_image = _inverse_normalized_image(sample["image"], config)
            display_image = _state_to_display(
                display_image[None], sample
            )[0].cpu()
            source_display = _semantic_display(
                mu[offset:offset + 1], sample
            )
            sample_id = str(sample.get("sample_id", dataset_index))
            trajectory_modes = tuple(sample_states)
            for mode in trajectory_modes:
                first_times.setdefault(mode, [])
                stats, first, sample_margins = (
                    batch_trajectory_statistics[mode][offset]
                )
                if batch_scheduler_statistics is not None:
                    for time_index, row in enumerate(stats):
                        row.update(
                            batch_scheduler_statistics[time_index][offset]
                        )
                rows.extend({
                    "mode": mode, "image_index": dataset_index,
                    "sample_id": sample_id, **row,
                } for row in stats)
                for key, values in sample_margins.items():
                    time_index, subset = key.split(":", 1)
                    margin_values.setdefault(
                        (mode, int(time_index), subset), []
                    ).extend(values)
                first_times[mode].extend([
                    times[index] if index >= 0 else None
                    for index in first.tolist()
                ])
                path_description = (
                    f"Entropy Adaptive | {scheduler_settings['scheduler']} | "
                    f"beta={scheduler_settings['beta']:g} | "
                    f"gamma={scheduler_settings['difficulty_gamma']:g} | "
                    f"normalization={scheduler_settings['normalization']}"
                    if scheduler_settings is not None
                    else f"Power Path | exponent={args.path_exponent:g}"
                )
                sample_lambda_maps = (
                    [coefficient[offset] for coefficient in lambda_maps]
                    if lambda_maps is not None else None
                )
                _save_mode_figure(
                    output / "figures" / mode / f"sample_{ordinal:04d}.png",
                    display_image, display_target, source_display,
                    sample_x0[mode], sample_states[mode], times, sample, mode,
                    sample_mu_new[mode], path_description,
                    None if sample_lambda_maps is None else [
                        float(coefficient.mean())
                        for coefficient in sample_lambda_maps
                    ],
                )
                if scheduler_settings is not None and mode in modes:
                    _save_scheduler_figure(
                        output / "scheduler" / mode
                        / f"sample_{ordinal:04d}.png",
                        display_image, source_display, entropy[offset].cpu(),
                        difficulty[offset].cpu(), [
                            coefficient.cpu()
                            for coefficient in sample_lambda_maps
                        ],
                        times, path_description,
                    )
            if variance_std is not None:
                _save_variance_figure(
                    output / "variance" / "bounded_gaussian"
                    / f"sample_{ordinal:04d}.png",
                    display_image, variance_entropy[offset].cpu(),
                    variance_difficulty[offset].cpu(), variance_map[offset].cpu(),
                    variance_std[offset].cpu(),
                    sample_x0["bounded_gaussian"], sample,
                )
            if args.mode == "both":
                _save_comparison(
                    output / "comparison" / f"sample_{ordinal:04d}.png",
                    display_image, display_target, sample_states, sample_x0,
                    times, sample,
                )
            if args.compare_hard_target:
                _save_hard_target_comparison(
                    output / "comparison"
                    / f"hard_vs_smoothed_sample_{ordinal:04d}.png",
                    display_image, display_target, sample_states,
                    sample_x0["simplex"], times, sample,
                )
        shapes = {
            "mu": [1, *mu.shape[1:]],
            "x1": [1, *x1.shape[1:]],
            "state_resolution": list(mu.shape[-2:]),
        }

    aggregate = _aggregate_rows(rows, times)
    for row in aggregate:
        time_index = times.index(row["t"])
        for subset, prefix in (
            ("all", ""), ("_source_correct", "source_correct_"),
            ("_source_incorrect", "source_incorrect_"),
        ):
            values = torch.tensor(margin_values[(row["mode"], time_index, subset)])
            row[prefix + "median_gt_margin"] = (
                float(values.median()) if values.numel() else float("nan")
            )
    with (output / "trajectory_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0])); writer.writeheader(); writer.writerows(aggregate)
    _save_summary_plots(aggregate, output)
    figure, axis = plt.subplots(figsize=(8, 5))
    bins = np.arange(len(times) + 2) - 0.5
    for mode, values in first_times.items():
        encoded = [times.index(value) if value is not None else len(times) for value in values]
        counts, _ = np.histogram(encoded, bins=bins)
        axis.step(range(len(counts)), counts / max(sum(counts), 1), where="mid", label=mode.replace("_", " "))
    axis.set_xticks(range(len(times) + 1), [f"{time:g}" for time in times] + ["never"])
    axis.set_xlabel("First specified t with GT argmax"); axis.set_ylabel("Semantic pixel ratio"); axis.legend(); axis.grid(alpha=0.3)
    figure.tight_layout(); figure.savefig(output / "first_gt_argmax_time.png", dpi=150); plt.close(figure)

    gaussian = None
    if "bounded_gaussian" in modes:
        n = gaussian_totals["pixels"]
        gaussian = {
            "mu_raw_min": gaussian_totals["raw_min"], "mu_raw_max": gaussian_totals["raw_max"],
            "mu_raw_abs_mean": gaussian_totals["raw_abs"] / n,
            "mu_new_min": gaussian_totals["new_min"], "mu_new_max": gaussian_totals["new_max"],
            "mu_new_abs_mean": gaussian_totals["new_abs"] / n,
            "noise_abs_mean": gaussian_totals["noise_abs"] / n,
            "x0_abs_mean": gaussian_totals["x0_abs"] / n,
            "mu_new_argmax_flip_ratio": gaussian_totals["flips"] / (n // mu.shape[1]),
        }
    simplex = None
    if "simplex" in modes:
        n = simplex_flips["pixels"]
        simplex = {"source_to_q_argmax_flip_ratio": simplex_flips["source_q"] / n,
                   "q_to_x0_argmax_flip_ratio": simplex_flips["q_x0"] / n}
    raw_gaussian = None
    if RAW_GAUSSIAN_MODE in modes:
        n = raw_gaussian_totals["elements"]
        semantic_pixels = raw_gaussian_totals["semantic_pixels"]
        raw_gaussian = {
            "mu_raw_abs": raw_gaussian_totals["mu_abs"] / n,
            "mu_raw_min": raw_gaussian_totals["mu_min"],
            "mu_raw_max": raw_gaussian_totals["mu_max"],
            "noise_abs_mean": raw_gaussian_totals["noise_abs"] / n,
            "sigma": args.sigma,
            "x0_abs": raw_gaussian_totals["x0_abs"] / n,
            "x0_min": raw_gaussian_totals["x0_min"],
            "x0_max": raw_gaussian_totals["x0_max"],
            "mu_to_x0_argmax_flip_ratio": (
                raw_gaussian_totals["flips"] / semantic_pixels
            ),
            "abs_reduction": "mean",
        }
    effective_smoothing_p = (
        0.0 if args.mode == RAW_GAUSSIAN_MODE else args.target_smoothing_p
    )
    summary_x1 = x1_hard if args.mode == RAW_GAUSSIAN_MODE else x1
    summary = {
        "checkpoint": str(checkpoint_path), "checkpoint_stage": checkpoint.get("stage"),
        "indices": indices, "batch_size": args.batch_size,
        "source_forward_batches": math.ceil(len(indices) / args.batch_size),
        "mode": args.mode, "times": times, "shapes": shapes,
        "path_type": args.path_type,
        "interpolation": (
            "x_t = lambda(t,d) * x1 + (1 - lambda(t,d)) * x0"
            if scheduler_settings is not None
            else "x_t = t^p * x1 + (1 - t^p) * x0"
        ),
        "path_exponent": args.path_exponent,
        "bounded_gaussian_variance": {
            "type": args.variance_type,
            "base_std": args.sigma,
            "rho": args.variance_rho,
            "normalization": "rank",
            "eps": 1.0e-8,
            "entropy_source": "softmax(raw_source_logits_mu)",
            "gt_independent": True,
            "rho_controls": "standard_deviation",
            "std_formula": "sigma_i = sigma0 * (1 + rho*d)",
            "variance_formula": "variance_i = sigma_i^2",
        },
        "entropy_scheduler": (
            {
                "entropy_source": "softmax(raw_source_logits_mu)",
                **scheduler_settings,
                "difficulty_formula": "d' = sign(d)*abs(d)^gamma",
                "formula": (
                    "lambda(t,d') = t - beta*t*(1-t)*d'"
                    if scheduler_settings["scheduler"] == "additive"
                    else "lambda(t,d') = t^exp(beta*d')"
                ),
            }
            if scheduler_settings is not None else None
        ),
        "void_excluded": config["dataset"]["void_class_index"],
        "parameters": {"lambda": args.lambda_value, "temperature": args.temperature,
                       "dirichlet_alpha": args.dirichlet_alpha, "amplitude": args.amplitude,
                       "tanh_temperature": args.tanh_temperature, "sigma": args.sigma,
                       "path_exponent": args.path_exponent, "seed": args.seed},
        "target_smoothing_enabled": effective_smoothing_p > 0.0,
        "target_smoothing_p": effective_smoothing_p,
        "x1_min": float(summary_x1.min()), "x1_max": float(summary_x1.max()),
        "x1_sum_error": float((summary_x1.sum(dim=1) - 1.0).abs().max()),
        "x1_gt_value": 1.0 - effective_smoothing_p + effective_smoothing_p / summary_x1.shape[1],
        "x1_non_gt_value": effective_smoothing_p / summary_x1.shape[1],
        "x1_gt_margin": 1.0 - effective_smoothing_p,
        "trajectory": aggregate, "simplex_diagnostics": simplex, "bounded_gaussian_diagnostics": gaussian,
        "raw_gaussian_diagnostics": raw_gaussian,
        "raw_gaussian_x0": "x0 = mu + sigma * epsilon; epsilon ~ N(0, I)",
        "raw_gaussian_x1": "hard one-hot e_y",
        "raw_gaussian_target_smoothing_enabled": False,
        "first_gt_argmax_time": {mode: {str(time): values.count(time) for time in (*times, None)} for mode, values in first_times.items()},
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    print(json.dumps(summary, indent=2, allow_nan=True))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
