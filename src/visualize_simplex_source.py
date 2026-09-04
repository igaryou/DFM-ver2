from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from checkpoint import _without_module_prefix
from config import load_config
from dataset import build_dataset
from discrete_flow_maps import (
    sample_image_simplex_components,
    sample_symmetric_dirichlet,
)
from source_diagnostics import _checkpoint_source_state_for_model
from source_model import build_source_model, source_statistics
from utils import resolve_device
from visualization import colorize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize an image-conditioned simplex source without a DFM endpoint"
    )
    parser.add_argument("--config", required=True)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint")
    checkpoint.add_argument("--checkpoint-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--lambda", dest="lambda_value", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--dirichlet-alpha", type=float)
    parser.add_argument("--lambda-values", type=float, nargs="+")
    parser.add_argument("--temperature-values", type=float, nargs="+")
    parser.add_argument("--alpha-values", type=float, nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    for scalar, values, label in (
        (args.lambda_value, args.lambda_values, "lambda"),
        (args.temperature, args.temperature_values, "temperature"),
        (args.dirichlet_alpha, args.alpha_values, "dirichlet alpha"),
    ):
        if scalar is not None and values is not None:
            raise ValueError(f"Specify either scalar {label} or its sweep values, not both")
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    if args.indices is not None and any(index < 0 for index in args.indices):
        raise ValueError("--indices must be non-negative")
    args.lambda_value = 0.8 if args.lambda_value is None else args.lambda_value
    args.temperature = 1.0 if args.temperature is None else args.temperature
    args.dirichlet_alpha = (
        1.0 if args.dirichlet_alpha is None else args.dirichlet_alpha
    )
    _validate_parameters(args.lambda_value, args.temperature, args.dirichlet_alpha)
    for value in args.lambda_values or []:
        _validate_parameters(value, args.temperature, args.dirichlet_alpha)
    for value in args.temperature_values or []:
        _validate_parameters(args.lambda_value, value, args.dirichlet_alpha)
    for value in args.alpha_values or []:
        _validate_parameters(args.lambda_value, args.temperature, value)
    return args


def _validate_parameters(lambda_value: float, temperature: float, alpha: float) -> None:
    if not 0 <= lambda_value <= 1:
        raise ValueError("lambda must be in [0,1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if alpha <= 0:
        raise ValueError("dirichlet alpha must be positive")


def resolve_checkpoint(
    checkpoint: str | None, checkpoint_dir: str | None
) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path.resolve()
    directory = Path(checkpoint_dir).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")
    for filename in ("best.pt", "latest.pt", "last.pt"):
        candidate = directory / filename
        if candidate.is_file():
            return candidate.resolve()
    candidates = sorted(directory.glob("*.pt"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(f"No checkpoint was found in: {directory}")
    raise ValueError(
        "Ambiguous checkpoint directory; specify --checkpoint explicitly: "
        + ", ".join(path.name for path in candidates)
    )


def load_source_checkpoint(
    config: dict, checkpoint_path: Path, device: torch.device
) -> tuple[dict, torch.nn.Module]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    state = checkpoint.get("source_model")
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint has no source_model state")
    saved_source = checkpoint.get("config", {}).get("source", {})
    saved_variant = saved_source.get("segformer_variant")
    current_variant = config["source"]["segformer_variant"]
    if saved_variant is not None and saved_variant != current_variant:
        raise RuntimeError(
            f"Source variant mismatch: checkpoint={saved_variant}, config={current_variant}"
        )
    build_config = copy.deepcopy(config)
    build_config["source"]["pretrained"] = False
    build_config["source"]["_load_pretrained"] = False
    build_config["source"]["checkpoint"] = None
    build_config["source"]["prior_type"] = "image_gaussian"
    source_model = build_source_model(build_config)
    if source_model is None:
        raise RuntimeError("Config does not construct an image-conditioned source model")
    source_state = _without_module_prefix(state)
    compatible = _checkpoint_source_state_for_model(source_state, source_model)
    source_model.load_state_dict(compatible, strict=True)
    source_model.to(device).eval().requires_grad_(False)
    return checkpoint, source_model


def _inverse_normalized_image(image: torch.Tensor, config: dict) -> torch.Tensor:
    image = image.detach().float().cpu()
    normalize = config["augmentation"]["normalize"]
    if normalize["enabled"]:
        mean = image.new_tensor(normalize["mean"])[:, None, None]
        std = image.new_tensor(normalize["std"])[:, None, None]
        image = image * std + mean
    elif config["augmentation"]["imagenet_normalize"]:
        mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        image = image * std + mean
    return image.clamp(0, 1)


def _state_to_display(
    state: torch.Tensor, sample: dict, *, mode: str = "bilinear"
) -> torch.Tensor:
    model_shape = tuple(sample.get("model_shape", sample["target"].shape))
    padded_shape = tuple(sample.get("padded_shape", model_shape))
    original_shape = tuple(sample.get("original_shape", sample["target"].shape))
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    full = F.interpolate(state.float(), padded_shape, mode=mode, **kwargs)
    full = full[..., : model_shape[0], : model_shape[1]]
    return F.interpolate(full, original_shape, mode=mode, **kwargs)


def _entropy(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.float()
    return -(probability * probability.clamp_min(1.0e-12).log()).sum(dim=1)


def _safe_mean(values: torch.Tensor) -> float:
    return float(values.mean()) if values.numel() else float("nan")


def condition_statistics(
    q: torch.Tensor, noise: torch.Tensor, z0: torch.Tensor,
    target: torch.Tensor, void_index: int,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    difference = z0.float() - q.float()
    l1 = difference.abs().sum(dim=1)
    l2 = difference.square().sum(dim=1).sqrt()
    q_entropy = _entropy(q)
    z_entropy = _entropy(z0)
    noise_entropy = _entropy(noise)
    q_confidence, q_prediction = q.float().max(dim=1)
    z_confidence, z_prediction = z0.float().max(dim=1)
    noise_confidence = noise.float().amax(dim=1)
    flip = q_prediction != z_prediction
    valid = target != void_index
    correct = valid & (q_prediction == target)
    incorrect = valid & ~correct
    stats = {
        "mean_l1": float(l1.mean()), "median_l1": float(l1.median()),
        "max_l1": float(l1.max()), "mean_l2": float(l2.mean()),
        "median_l2": float(l2.median()), "max_l2": float(l2.max()),
        "mean_source_entropy": float(q_entropy.mean()),
        "mean_z0_entropy": float(z_entropy.mean()),
        "mean_entropy_change": float((z_entropy - q_entropy).mean()),
        "argmax_flip_ratio": float(flip.float().mean()),
        "mean_source_confidence": float(q_confidence.mean()),
        "mean_z0_confidence": float(z_confidence.mean()),
        "mean_epsilon_max_probability": float(noise_confidence.mean()),
        "mean_epsilon_entropy": float(noise_entropy.mean()),
        "simplex_sum_error": float((z0.float().sum(dim=1) - 1.0).abs().amax()),
    }
    for name, mask in (("source_correct", correct), ("source_incorrect", incorrect)):
        stats.update({
            f"{name}_mean_l1": _safe_mean(l1[mask]),
            f"{name}_mean_l2": _safe_mean(l2[mask]),
            f"{name}_argmax_flip_ratio": _safe_mean(flip[mask].float()),
            f"{name}_mean_source_confidence": _safe_mean(q_confidence[mask]),
            f"{name}_mean_z0_confidence": _safe_mean(z_confidence[mask]),
        })
    maps = {
        "l1": l1, "l2": l2, "source_entropy": q_entropy,
        "z0_entropy": z_entropy, "entropy_change": z_entropy - q_entropy,
        "source_confidence": q_confidence, "z0_confidence": z_confidence,
        "source_prediction": q_prediction, "noise_prediction": noise.argmax(dim=1),
        "z0_prediction": z_prediction, "flip": flip,
        "noise_confidence": noise_confidence,
    }
    return stats, maps


def _heatmap(axis, values, title, *, cmap="viridis", vmin=None, vmax=None, center=False):
    array = values.detach().float().cpu().numpy()
    if center:
        limit = max(float(np.abs(array).max()), 1.0e-8)
        vmin, vmax = -limit, limit
    plotted = axis.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.axis("off")
    return plotted


def save_single_figure(
    image: torch.Tensor, target: torch.Tensor, maps: dict[str, torch.Tensor],
    path: Path, *, dataset_name: str,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.reshape(-1)
    axes[0].imshow(image.permute(1, 2, 0)); axes[0].set_title("Input image")
    axes[1].imshow(colorize(target, dataset_name)); axes[1].set_title("Ground truth")
    axes[2].imshow(colorize(maps["source_prediction"], dataset_name)); axes[2].set_title("Source prediction")
    handles = []
    handles.append((axes[3], _heatmap(axes[3], maps["source_confidence"], "Source confidence", vmin=0, vmax=1)))
    axes[4].imshow(colorize(maps["noise_prediction"], dataset_name)); axes[4].set_title("Dirichlet noise prediction")
    axes[5].imshow(colorize(maps["z0_prediction"], dataset_name)); axes[5].set_title("Final z0 prediction")
    handles.append((axes[6], _heatmap(axes[6], maps["z0_confidence"], "Final z0 confidence", vmin=0, vmax=1)))
    handles.append((axes[7], _heatmap(axes[7], maps["l1"], "||z0 - q||_1", cmap="magma", vmin=0)))
    for axis in axes:
        axis.axis("off")
    for axis, handle in handles:
        figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=140, bbox_inches="tight"); plt.close(figure)


def save_entropy_figure(maps: dict[str, torch.Tensor], path: Path, classes: int) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    upper = math.log(classes)
    for axis, key, title in zip(
        axes[:3], ("source_entropy", "z0_entropy", "entropy_change"),
        ("Source entropy", "z0 entropy", "Delta entropy"),
    ):
        handle = _heatmap(
            axis, maps[key], title,
            cmap="coolwarm" if key == "entropy_change" else "magma",
            vmin=None if key == "entropy_change" else 0,
            vmax=None if key == "entropy_change" else upper,
            center=key == "entropy_change",
        )
        figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)
    axes[3].imshow(maps["flip"].detach().cpu(), cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Argmax flip map"); axes[3].axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=140, bbox_inches="tight"); plt.close(figure)


def save_sweep_figure(
    image: torch.Tensor, target: torch.Tensor, source_prediction: torch.Tensor,
    conditions: list[tuple[str, dict[str, torch.Tensor]]], path: Path,
    *, dataset_name: str, sweep_type: str,
) -> None:
    columns = 5
    figure, axes = plt.subplots(len(conditions) + 1, columns, figsize=(4 * columns, 4 * (len(conditions) + 1)))
    axes = np.asarray(axes).reshape(len(conditions) + 1, columns)
    axes[0, 0].imshow(image.permute(1, 2, 0)); axes[0, 0].set_title("Input")
    axes[0, 1].imshow(colorize(target, dataset_name)); axes[0, 1].set_title("GT")
    axes[0, 2].imshow(colorize(source_prediction, dataset_name)); axes[0, 2].set_title("Source prediction")
    for axis in axes[0, 3:]: axis.set_visible(False)
    for row, (label, maps) in enumerate(conditions, start=1):
        if sweep_type == "temperature":
            panels = (
                ("source_confidence", "Source confidence", "heat"),
                ("source_entropy", "Source entropy", "entropy"),
                ("z0_prediction", "z0 prediction", "mask"),
                ("z0_confidence", "z0 confidence", "heat"),
                ("l1", "||z0-q||_1", "l1"),
            )
        elif sweep_type == "alpha":
            panels = (
                ("noise_prediction", "epsilon prediction", "mask"),
                ("noise_confidence", "epsilon max probability", "heat"),
                ("z0_prediction", "z0 prediction", "mask"),
                ("l1", "||z0-q||_1", "l1"),
                ("flip", "argmax flip", "binary"),
            )
        else:
            panels = (
                ("z0_prediction", "z0 prediction", "mask"),
                ("z0_confidence", "z0 confidence", "heat"),
                ("l1", "||z0-q||_1", "l1"),
                ("flip", "argmax flip", "binary"),
            )
        for column, (key, title, kind) in enumerate(panels):
            axis = axes[row, column]
            if kind == "mask": axis.imshow(colorize(maps[key], dataset_name))
            else:
                axis.imshow(
                    maps[key].detach().float().cpu(),
                    cmap="gray" if kind == "binary" else "magma" if kind in {"l1", "entropy"} else "viridis",
                    vmin=0, vmax=math.log(20) if kind == "entropy" else 1 if kind in {"heat", "binary"} else None,
                )
            axis.set_title(f"{label}\n{title}"); axis.axis("off")
        for axis in axes[row, len(panels):]: axis.set_visible(False)
    for axis in axes[0, :3]: axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=120, bbox_inches="tight"); plt.close(figure)


def _aggregate(rows: Iterable[dict], condition: str) -> dict:
    selected = [row for row in rows if row["condition"] == condition]
    result = {"num_images": len(selected)}
    if not selected:
        return result
    for key in (
        "mean_l1", "mean_l2", "mean_source_entropy", "mean_z0_entropy",
        "mean_entropy_change", "argmax_flip_ratio", "mean_source_confidence",
        "mean_z0_confidence", "mean_epsilon_max_probability", "mean_epsilon_entropy",
        "simplex_sum_error",
    ):
        values = [row[key] for row in selected if math.isfinite(row[key])]
        result[key] = sum(values) / len(values) if values else float("nan")
    return result


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    config = load_config(args.config, args.set)
    if config["dataset"]["name"] != "cityscapes":
        raise ValueError("This visualization currently targets Cityscapes")
    if config["source"]["type"] != "trainable_segformer":
        raise ValueError("Visualization requires source.type=trainable_segformer")
    if config["source"]["segformer_variant"] != "b1":
        raise ValueError("Visualization requires source.segformer_variant=b1")
    checkpoint_path = resolve_checkpoint(args.checkpoint, args.checkpoint_dir)
    device = resolve_device(args.device or config["runtime"]["device"])
    checkpoint, source_model = load_source_checkpoint(config, checkpoint_path, device)
    dataset = build_dataset(config, args.split, augment=False)
    indices = args.indices or list(range(min(args.num_images, len(dataset))))
    if any(index >= len(dataset) for index in indices):
        raise IndexError(f"Dataset index exceeds split size {len(dataset)}")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    shapes = {}

    for ordinal, dataset_index in enumerate(indices):
        sample = dataset[dataset_index]
        if not isinstance(sample, dict):
            image, target = sample
            sample = {"image": image, "target": target}
        image = sample["image"].unsqueeze(0).to(device)
        target = sample["target"].long()
        mu, _ = source_statistics(source_model, image)
        sample_seed = int(args.seed + dataset_index)
        base_noise = sample_symmetric_dirichlet(
            tuple(mu.shape), args.dirichlet_alpha, device=device, seed=sample_seed
        )
        q, _, z0 = sample_image_simplex_components(
            mu, lambda_value=args.lambda_value, temperature=args.temperature,
            dirichlet_alpha=args.dirichlet_alpha, dirichlet_noise=base_noise,
        )
        q_display = _state_to_display(q, sample)
        noise_display = _state_to_display(base_noise, sample)
        z_display = _state_to_display(z0, sample)
        display_target = target.unsqueeze(0).to(device)
        stats, maps = condition_statistics(
            q_display, noise_display, z_display, display_target,
            config["dataset"]["void_class_index"],
        )
        maps = {key: value[0].cpu() for key, value in maps.items()}
        row = {
            "image_index": dataset_index, "sample_id": sample.get("sample_id", str(dataset_index)),
            "condition": "single", "lambda": args.lambda_value,
            "temperature": args.temperature, "dirichlet_alpha": args.dirichlet_alpha,
            **stats,
        }
        rows.append(row)
        display_image = _inverse_normalized_image(sample["image"], config)
        model_shape = tuple(sample.get("model_shape", target.shape))
        display_image = display_image[..., : model_shape[0], : model_shape[1]]
        display_image = F.interpolate(
            display_image[None], target.shape, mode="bilinear", align_corners=False
        )[0]
        filename = f"image_{ordinal:04d}.png"
        save_single_figure(display_image, target, maps, output / "single" / filename, dataset_name="cityscapes")
        save_entropy_figure(maps, output / "entropy" / f"image_{ordinal:04d}_entropy.png", mu.shape[1])
        shapes = {
            "mu": list(mu.shape), "q": list(q.shape),
            "dirichlet_noise": list(base_noise.shape), "z0": list(z0.shape),
        }

        sweep_specs = (
            ("lambda", args.lambda_values),
            ("alpha", args.alpha_values),
            ("temperature", args.temperature_values),
        )
        for sweep_type, values in sweep_specs:
            if not values:
                continue
            conditions = []
            reference_prediction = None
            for value_index, value in enumerate(values):
                lam = value if sweep_type == "lambda" else args.lambda_value
                temperature = value if sweep_type == "temperature" else args.temperature
                alpha = value if sweep_type == "alpha" else args.dirichlet_alpha
                noise = (
                    sample_symmetric_dirichlet(
                        tuple(mu.shape), alpha, device=device,
                        seed=sample_seed + 100_000 * value_index,
                    )
                    if sweep_type == "alpha" else base_noise
                )
                sweep_q, _, sweep_z = sample_image_simplex_components(
                    mu, lambda_value=lam, temperature=temperature,
                    dirichlet_alpha=alpha, dirichlet_noise=noise,
                )
                sweep_q_display = _state_to_display(sweep_q, sample)
                sweep_noise_display = _state_to_display(noise, sample)
                sweep_z_display = _state_to_display(sweep_z, sample)
                sweep_stats, sweep_maps = condition_statistics(
                    sweep_q_display, sweep_noise_display, sweep_z_display,
                    display_target, config["dataset"]["void_class_index"],
                )
                source_prediction = sweep_maps["source_prediction"]
                if reference_prediction is None:
                    reference_prediction = source_prediction
                elif sweep_type == "temperature" and not torch.equal(
                    reference_prediction, source_prediction
                ):
                    raise AssertionError("Positive temperature changed source argmax")
                sweep_maps = {key: tensor[0].cpu() for key, tensor in sweep_maps.items()}
                conditions.append((f"{sweep_type}={value:g}", sweep_maps))
                rows.append({
                    "image_index": dataset_index, "sample_id": sample.get("sample_id", str(dataset_index)),
                    "condition": f"{sweep_type}:{value:g}", "lambda": lam,
                    "temperature": temperature, "dirichlet_alpha": alpha,
                    **sweep_stats,
                })
            save_sweep_figure(
                display_image, target, maps["source_prediction"], conditions,
                output / f"{sweep_type}_sweep" / filename,
                dataset_name="cityscapes", sweep_type=sweep_type,
            )

    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    primary = _aggregate(rows, "single")
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_stage": checkpoint.get("stage"),
        "source_variant": config["source"]["segformer_variant"],
        "shapes": shapes,
        "parameters": {
            "lambda": args.lambda_value, "temperature": args.temperature,
            "dirichlet_alpha": args.dirichlet_alpha, "seed": args.seed,
        },
        "lambda": args.lambda_value,
        "temperature": args.temperature,
        "dirichlet_alpha": args.dirichlet_alpha,
        **primary,
        "conditions": {condition: _aggregate(rows, condition) for condition in conditions},
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    invocation = vars(args).copy()
    invocation.update({"checkpoint_resolved": str(checkpoint_path), "indices_resolved": indices})
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(invocation, handle, indent=2)
    print(json.dumps(summary, indent=2, allow_nan=True))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
