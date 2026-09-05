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

from config import load_config
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
    parser.add_argument("--lambda", dest="lambda_value", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=6.0)
    parser.add_argument("--dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--tanh-temperature", type=float, default=5.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--target-smoothing-p", type=float, default=0.0)
    parser.add_argument("--compare-hard-target", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    if args.indices is not None and any(index < 0 for index in args.indices):
        raise ValueError("--indices must be non-negative")
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
    if not 0.0 <= args.target_smoothing_p < 1.0:
        raise ValueError("--target-smoothing-p must satisfy 0 <= p < 1")
    if args.compare_hard_target and args.target_smoothing_p == 0.0:
        raise ValueError("--compare-hard-target requires --target-smoothing-p > 0")
    if args.compare_hard_target and args.mode not in ("simplex", "both"):
        raise ValueError("--compare-hard-target requires simplex or both mode")
    return args


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the diagnostic bounded Gaussian from raw source logits."""
    if amplitude <= 0 or tanh_temperature <= 0 or sigma < 0:
        raise ValueError("amplitude/tanh_temperature must be positive and sigma non-negative")
    mu_new = float(amplitude) * torch.tanh(mu.float() / float(tanh_temperature))
    devices = [] if mu.device.type != "cuda" else [mu.device.index or 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        noise = torch.randn(mu.shape, device=mu.device, dtype=torch.float32)
    x0 = mu_new + float(sigma) * noise
    if not torch.equal(mu.argmax(dim=1), mu_new.argmax(dim=1)):
        raise AssertionError("Positive tanh transform changed raw-logit argmax")
    return mu_new, noise, x0


def linear_interpolation(
    x0: torch.Tensor, x1: torch.Tensor, time: float
) -> torch.Tensor:
    return float(time) * x1 + (1.0 - float(time)) * x0


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
        (f"t={time:g}", _semantic_display(state, sample), False)
        for time, state in zip(times, states, strict=True)
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
    figure.suptitle(mode.replace("_", " ").title())
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=130, bbox_inches="tight"); plt.close(figure)


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

    for ordinal, dataset_index in enumerate(indices):
        sample = dataset[dataset_index]
        image = sample["image"].unsqueeze(0).to(device)
        target_full = sample["target"].long().unsqueeze(0).to(device)
        mu, _ = source_statistics(source_model, image)
        targets = prepare_state_targets(
            target_full, num_classes=config["dataset"]["num_classes"],
            state_size=mu.shape[-2:], ignore_index=config["dataset"]["void_class_index"],
            mask_pixel_losses=True,
        )
        x1_hard, target = targets.one_hot_state, targets.target_state
        x1 = smooth_categorical_target(x1_hard, args.target_smoothing_p)
        source_prediction = mu.argmax(dim=1)
        sample_seed = int(args.seed + dataset_index * 2)
        mode_states: dict[str, list[torch.Tensor]] = {}
        mode_x0: dict[str, torch.Tensor] = {}
        mode_mu_new: dict[str, torch.Tensor | None] = {}
        if "simplex" in modes:
            q, _, x0 = sample_image_simplex_components(
                mu, lambda_value=args.lambda_value, temperature=args.temperature,
                dirichlet_alpha=args.dirichlet_alpha, seed=sample_seed,
            )
            source_q = source_prediction != q.argmax(dim=1)
            if source_q.any(): raise AssertionError("softmax temperature changed source argmax")
            simplex_flips["pixels"] += source_q.numel()
            simplex_flips["source_q"] += int(source_q.sum())
            simplex_flips["q_x0"] += int((q.argmax(dim=1) != x0.argmax(dim=1)).sum())
            mode_states["simplex"] = [linear_interpolation(x0, x1, time) for time in times]
            mode_x0["simplex"] = x0
            mode_mu_new["simplex"] = None
            if args.compare_hard_target:
                mode_states["simplex_hard_target"] = [
                    linear_interpolation(x0, x1_hard, time) for time in times
                ]
                mode_x0["simplex_hard_target"] = x0
                mode_mu_new["simplex_hard_target"] = None
        if "bounded_gaussian" in modes:
            mu_new, noise, x0 = bounded_gaussian_components(
                mu, amplitude=args.amplitude, tanh_temperature=args.tanh_temperature,
                sigma=args.sigma, seed=sample_seed + 1,
            )
            flips = source_prediction != mu_new.argmax(dim=1)
            values = mu.numel(); gaussian_totals["pixels"] += values
            gaussian_totals["raw_abs"] += float(mu.float().abs().sum())
            gaussian_totals["new_abs"] += float(mu_new.abs().sum())
            gaussian_totals["noise_abs"] += float(noise.abs().sum())
            gaussian_totals["x0_abs"] += float(x0.abs().sum())
            gaussian_totals["raw_min"] = min(gaussian_totals["raw_min"], float(mu.min()))
            gaussian_totals["raw_max"] = max(gaussian_totals["raw_max"], float(mu.max()))
            gaussian_totals["new_min"] = min(gaussian_totals["new_min"], float(mu_new.min()))
            gaussian_totals["new_max"] = max(gaussian_totals["new_max"], float(mu_new.max()))
            gaussian_totals["flips"] += int(flips.sum())
            mode_states["bounded_gaussian"] = [linear_interpolation(x0, x1, time) for time in times]
            mode_x0["bounded_gaussian"] = x0
            mode_mu_new["bounded_gaussian"] = mu_new
        if RAW_GAUSSIAN_MODE in modes:
            epsilon, x0 = raw_gaussian_components(
                mu, sigma=args.sigma, seed=sample_seed,
            )
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
            mode_states[RAW_GAUSSIAN_MODE] = [
                linear_interpolation(x0, x1_hard, time) for time in times
            ]
            mode_x0[RAW_GAUSSIAN_MODE] = x0
            mode_mu_new[RAW_GAUSSIAN_MODE] = None

        display_target = sample["target"].cpu()
        display_image = _inverse_normalized_image(sample["image"], config)
        display_image = _state_to_display(display_image[None], sample)[0].cpu()
        source_display = _semantic_display(mu, sample)
        sample_id = str(sample.get("sample_id", dataset_index))
        trajectory_modes = tuple(mode_states)
        for mode in trajectory_modes:
            first_times.setdefault(mode, [])
            stats, first, sample_margins = interpolation_statistics(
                mode_states[mode], times, target, source_prediction,
                config["dataset"]["void_class_index"],
            )
            rows.extend({"mode": mode, "image_index": dataset_index, "sample_id": sample_id, **row} for row in stats)
            for key, values in sample_margins.items():
                time_index, subset = key.split(":", 1)
                margin_values.setdefault((mode, int(time_index), subset), []).extend(values)
            first_times[mode].extend([times[index] if index >= 0 else None for index in first.tolist()])
            _save_mode_figure(
                output / "figures" / mode / f"sample_{ordinal:04d}.png",
                display_image, display_target, source_display, mode_x0[mode], mode_states[mode],
                times, sample, mode, mode_mu_new[mode],
            )
        if args.mode == "both":
            _save_comparison(output / "comparison" / f"sample_{ordinal:04d}.png", display_image, display_target, mode_states, mode_x0, times, sample)
        if args.compare_hard_target:
            _save_hard_target_comparison(
                output / "comparison" / f"hard_vs_smoothed_sample_{ordinal:04d}.png",
                display_image, display_target, mode_states, mode_x0["simplex"],
                times, sample,
            )
        shapes = {"mu": list(mu.shape), "x1": list(x1.shape), "state_resolution": list(mu.shape[-2:])}

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
        "indices": indices, "mode": args.mode, "times": times, "shapes": shapes,
        "interpolation": "x_t = t * x1 + (1 - t) * x0", "void_excluded": config["dataset"]["void_class_index"],
        "parameters": {"lambda": args.lambda_value, "temperature": args.temperature,
                       "dirichlet_alpha": args.dirichlet_alpha, "amplitude": args.amplitude,
                       "tanh_temperature": args.tanh_temperature, "sigma": args.sigma, "seed": args.seed},
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
