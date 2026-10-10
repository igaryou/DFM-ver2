from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import load_config
from dataset import build_dataset
from source_model import source_statistics
from utils import resolve_device
from visualization import colorize
from visualize_simplex_source import (
    _inverse_normalized_image,
    _state_to_display,
    load_source_checkpoint,
    resolve_checkpoint,
)


DEFAULT_SIGMAS = (1.0, 0.8, 0.6, 0.5, 0.3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare argmax(mu) with argmax(mu + sigma * epsilon)."
    )
    parser.add_argument("--config", required=True)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint")
    checkpoint.add_argument("--checkpoint-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument(
        "--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS
    )
    parser.add_argument(
        "--mean-mode", choices=("raw", "effective"), default="raw",
        help=(
            "raw: use source_model mu as-is; effective: for an image_bounded_gaussian "
            "source, use amplitude*tanh(mu/temperature), matching its sampling mean"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def select_mean(mu: torch.Tensor, config: dict, mode: str) -> torch.Tensor:
    if mode not in {"raw", "effective"}:
        raise ValueError("mean mode must be raw or effective")
    if mode == "raw" or config["source"]["prior_type"] != "image_bounded_gaussian":
        return mu
    settings = config["source"]["bounded_gaussian"]
    return float(settings["amplitude"]) * torch.tanh(
        mu / float(settings["temperature"])
    )


def make_epsilon(mu: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=mu.device)
    generator.manual_seed(int(seed))
    return torch.randn(
        mu.shape, device=mu.device, dtype=mu.dtype, generator=generator
    )


def prediction_from_state(state: torch.Tensor, sample: dict) -> torch.Tensor:
    # Resize continuous channels before argmax, matching normal evaluation.
    return _state_to_display(state, sample).argmax(dim=1)[0].cpu()


def display_image(sample: dict, config: dict) -> torch.Tensor:
    target = sample["target"]
    image = _inverse_normalized_image(sample["image"], config)
    model_shape = tuple(sample.get("model_shape", target.shape))
    image = image[..., : model_shape[0], : model_shape[1]]
    if image.shape[-2:] != target.shape:
        image = F.interpolate(
            image[None], size=target.shape, mode="bilinear",
            align_corners=False,
        )[0]
    return image


def save_comparison(
    path: Path,
    image: torch.Tensor,
    target: torch.Tensor,
    source_prediction: torch.Tensor,
    predictions: list[torch.Tensor],
    sigmas: tuple[float, ...],
    mean_mode: str,
    dataset_name: str,
) -> None:
    panels = [
        ("Image", image.permute(1, 2, 0), True),
        ("GT", target, False),
        (f"argmax(mu{'_effective' if mean_mode == 'effective' else ''})",
         source_prediction, False),
    ]
    panels.extend(
        (f"argmax(mu + {sigma:g}*eps)", mask, False)
        for sigma, mask in zip(sigmas, predictions, strict=True)
    )
    columns = len(panels)
    figure, axes = plt.subplots(
        1, columns, figsize=(4.3 * columns, 4.2), squeeze=False
    )
    for axis, (title, values, is_image) in zip(
        axes.ravel(), panels, strict=True
    ):
        if is_image:
            axis.imshow(values)
        else:
            axis.imshow(colorize(values, dataset_name))
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.set)
    if config["dataset"]["name"] != "cityscapes":
        raise ValueError("This script currently supports Cityscapes.")
    prior_type = config["source"]["prior_type"]
    if prior_type not in {"image_gaussian", "image_bounded_gaussian"}:
        raise ValueError(
            "Expected image_gaussian or image_bounded_gaussian source, "
            f"got {prior_type}"
        )
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    sigmas = tuple(float(value) for value in args.sigmas)
    if not sigmas or any(not math.isfinite(s) or s < 0 for s in sigmas):
        raise ValueError("--sigmas must contain finite nonnegative values")

    device = resolve_device(args.device or config["runtime"]["device"])
    checkpoint_path = resolve_checkpoint(args.checkpoint, args.checkpoint_dir)
    _, source_model = load_source_checkpoint(config, checkpoint_path, device)
    dataset = build_dataset(config, args.split, augment=False)
    indices = (
        list(range(min(args.num_images, len(dataset))))
        if args.indices is None else args.indices
    )
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise IndexError("Dataset index is outside the selected split.")

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for ordinal, index in enumerate(indices):
        sample = dataset[index]
        if not isinstance(sample, dict):
            image, target = sample
            sample = {"image": image, "target": target}
        image = sample["image"].unsqueeze(0).to(device)
        mu_raw, _ = source_statistics(source_model, image)
        mu = select_mean(mu_raw.float(), config, args.mean_mode)
        # All sigma conditions for one image share exactly the same epsilon.
        epsilon = make_epsilon(mu, args.seed + int(index))
        source_prediction = prediction_from_state(mu, sample)
        predictions = [
            prediction_from_state(mu + sigma * epsilon, sample)
            for sigma in sigmas
        ]
        filename = output / f"image_{ordinal:04d}_idx{index:04d}.png"
        save_comparison(
            filename, display_image(sample, config), sample["target"],
            source_prediction, predictions, sigmas, args.mean_mode, "cityscapes"
        )
        print(f"[{ordinal + 1}/{len(indices)}] saved {filename}")

    metadata = {
        "config": str(args.config),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "indices": indices,
        "sigmas": list(sigmas),
        "mean_mode": args.mean_mode,
        "prior_type": prior_type,
        "seed": args.seed,
        "shared_epsilon_across_sigmas": True,
        "noise_distribution": "standard_normal",
        "note": (
            "For image_bounded_gaussian, effective mode applies the configured "
            "bounded mean. This visualizer uses scalar sigma and does not "
            "apply an entropy-adaptive per-pixel variance map."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
