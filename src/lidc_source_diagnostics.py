"""Numerical diagnostics for sampled LIDC source initial states."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _stats(values: torch.Tensor) -> dict[str, float | None]:
    values = values.detach().float().reshape(-1).cpu()
    if values.numel() == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _masked_flip_rates(
    flips: torch.Tensor, mask: torch.Tensor
) -> list[float | None]:
    mask = mask.bool()
    if not bool(mask.any()):
        return [None] * flips.shape[0]
    return [float(sample[mask].float().mean()) for sample in flips]


def _valid_mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def _binary_overlap(first: torch.Tensor, second: torch.Tensor) -> tuple[float, float]:
    first = first.bool()
    second = second.bool()
    intersection = int((first & second).sum())
    first_count = int(first.sum())
    second_count = int(second.sum())
    union = first_count + second_count - intersection
    iou = 1.0 if union == 0 else intersection / union
    denominator = first_count + second_count
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    return float(iou), float(dice)


def _gt_boundary(target: torch.Tensor) -> torch.Tensor:
    foreground = target.bool()[None, None].float()
    dilated = F.max_pool2d(foreground, 3, stride=1, padding=1).bool()
    eroded = ~F.max_pool2d(1.0 - foreground, 3, stride=1, padding=1).bool()
    return (dilated & ~eroded)[0, 0]


def compute_lidc_source_diagnostics(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    x0_samples: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_index: int,
    sample_id: str | None = None,
    eps: float = 1.0e-8,
) -> dict[str, Any]:
    """Describe source-state diversity; these are not final prediction metrics."""
    if mu.ndim != 3 or mu.shape[0] != 2:
        raise ValueError("LIDC mu must have shape [2,H,W]")
    if logvar.shape != mu.shape:
        raise ValueError("logvar must have the same shape as mu")
    if x0_samples.ndim != 4 or x0_samples.shape[1:] != mu.shape:
        raise ValueError("x0_samples must have shape [N,2,H,W]")
    if x0_samples.shape[0] == 0:
        raise ValueError("x0_samples must not be empty")
    if target.shape != mu.shape[-2:]:
        raise ValueError("target must have shape [H,W] matching mu")

    mu = mu.detach().float()
    logvar = logvar.detach().float()
    x0_samples = x0_samples.detach().float()
    target = target.detach().to(device=mu.device)
    mu_margin = mu[1] - mu[0]
    abs_margin = mu_margin.abs()
    std = torch.exp(0.5 * logvar)
    margin_noise_std = torch.sqrt(std[0].square() + std[1].square())
    mu_argmax = torch.argmax(mu, dim=0)
    x0_argmax = torch.argmax(x0_samples, dim=1)
    flips = x0_argmax != mu_argmax.unsqueeze(0)
    flip_per_sample = flips.float().mean(dim=(1, 2))

    foreground_flip = _masked_flip_rates(flips, target == 1)
    background_flip = _masked_flip_rates(flips, target == 0)
    boundary_flip = _masked_flip_rates(flips, _gt_boundary(target))

    pairwise_iou = []
    pairwise_dice = []
    for first in range(x0_argmax.shape[0]):
        for second in range(first + 1, x0_argmax.shape[0]):
            iou, dice = _binary_overlap(
                x0_argmax[first] == 1, x0_argmax[second] == 1
            )
            pairwise_iou.append(iou)
            pairwise_dice.append(dice)
    pairwise_iou_stats = _stats(torch.tensor(pairwise_iou))
    pairwise_dice_stats = _stats(torch.tensor(pairwise_dice))
    foreground_fraction = (x0_argmax == 1).float().mean(dim=(1, 2))
    mu_gt_iou, mu_gt_dice = _binary_overlap(mu_argmax == 1, target == 1)

    result: dict[str, Any] = {
        "sample_index": int(sample_index),
        "num_source_samples": int(x0_samples.shape[0]),
        "mu_margin": {
            **_stats(mu_margin),
            "abs_mean": float(abs_margin.mean()),
            "abs_median": float(abs_margin.median()),
            "abs_p90": float(torch.quantile(abs_margin, 0.9)),
        },
        "noise": {
            "source_std_mean": float(std.mean()),
            "margin_noise_std_mean": float(margin_noise_std.mean()),
            "margin_to_noise_ratio_mean": float(
                (abs_margin / margin_noise_std.clamp_min(eps)).mean()
            ),
        },
        "argmax_flip": {
            **_stats(flip_per_sample),
            "per_sample": [float(value) for value in flip_per_sample.cpu()],
        },
        "foreground_flip": {
            "mean": _valid_mean(foreground_flip),
            "per_sample": foreground_flip,
        },
        "background_flip": {
            "mean": _valid_mean(background_flip),
            "per_sample": background_flip,
        },
        "boundary_flip": {
            "mean": _valid_mean(boundary_flip),
            "per_sample": boundary_flip,
        },
        "source_diversity": {
            **{f"pairwise_iou_{key}": value for key, value in pairwise_iou_stats.items()},
            **{
                f"pairwise_dice_{key}": value
                for key, value in pairwise_dice_stats.items()
            },
        },
        "foreground_fraction": {
            **_stats(foreground_fraction),
            "per_sample": [float(value) for value in foreground_fraction.cpu()],
        },
        "mu_vs_gt_annotation0": {
            "iou": mu_gt_iou,
            "dice": mu_gt_dice,
            "note": "source mean diagnostic, not final DFM prediction performance",
        },
    }
    if sample_id is not None:
        result["sample_id"] = str(sample_id)
    return result


def summarize_lidc_source_diagnostics(
    samples: list[dict[str, Any]], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Aggregate selected sample diagnostics across visualized CT images."""
    paths = {
        "flip_rate": ("argmax_flip", "mean"),
        "foreground_flip_rate": ("foreground_flip", "mean"),
        "background_flip_rate": ("background_flip", "mean"),
        "boundary_flip_rate": ("boundary_flip", "mean"),
        "pairwise_iou": ("source_diversity", "pairwise_iou_mean"),
        "pairwise_dice": ("source_diversity", "pairwise_dice_mean"),
        "mu_margin_abs": ("mu_margin", "abs_mean"),
        "margin_to_noise_ratio": ("noise", "margin_to_noise_ratio_mean"),
        "foreground_fraction": ("foreground_fraction", "mean"),
        "mu_gt_iou": ("mu_vs_gt_annotation0", "iou"),
        "mu_gt_dice": ("mu_vs_gt_annotation0", "dice"),
    }
    summary: dict[str, Any] = {
        **metadata,
        "num_images": len(samples),
    }
    for output_key, (section, key) in paths.items():
        values = [sample[section][key] for sample in samples]
        valid = torch.tensor([value for value in values if value is not None])
        stats = _stats(valid)
        summary[f"{output_key}_mean"] = stats["mean"]
        summary[f"{output_key}_std"] = stats["std"]
    return summary


def save_diagnostics_json(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
