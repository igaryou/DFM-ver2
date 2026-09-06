from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import load_config
from dataset import build_dataset
from state_space import prepare_state_targets
from utils import resolve_device
from visualization import colorize
from visualize_simplex_source import (
    _inverse_normalized_image,
    _state_to_display,
)


DEFAULT_TIMES = (0.0, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize linear interpolation from a standard-normal source "
            "x0 ~ N(0, I) to hard one-hot GT."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-images", type=int, default=32)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--times", type=float, nargs="+", default=DEFAULT_TIMES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser


def parse_args() -> argparse.Namespace:
    args = build_parser().parse_args()

    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")

    if args.indices is not None and any(index < 0 for index in args.indices):
        raise ValueError("--indices must be non-negative")

    if not args.times:
        raise ValueError("--times must not be empty")

    if any(not 0.0 <= t <= 1.0 for t in args.times):
        raise ValueError("--times must lie in [0, 1]")

    if any(b <= a for a, b in zip(args.times, args.times[1:])):
        raise ValueError("--times must be strictly increasing")

    return args


def sample_standard_normal(
    shape: torch.Size,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    devices = []
    if device.type == "cuda":
        devices = [
            torch.cuda.current_device()
            if device.index is None
            else device.index
        ]

    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        x0 = torch.randn(shape, device=device, dtype=dtype)

    return x0


def linear_interpolation(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: float,
) -> torch.Tensor:
    return float(t) * x1 + (1.0 - float(t)) * x0


def gt_margin(
    state: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    gt = state.gather(
        1, target[:, None]
    ).squeeze(1)

    competitors = state.clone()
    competitors.scatter_(
        1,
        target[:, None],
        -torch.inf,
    )

    return gt - competitors.amax(dim=1)


def state_to_display(
    state: torch.Tensor,
    sample: dict,
) -> torch.Tensor:
    return _state_to_display(
        state, sample
    ).argmax(dim=1)[0].cpu()


def save_figure(
    path: Path,
    *,
    image: torch.Tensor,
    target_full: torch.Tensor,
    x0: torch.Tensor,
    states: list[torch.Tensor],
    times: list[float],
    sample: dict,
) -> None:
    panels: list[tuple[str, object, bool]] = [
        ("Input", image.permute(1, 2, 0), True),
        ("GT", target_full, False),
        ("x0 ~ N(0, I)", state_to_display(x0, sample), False),
    ]

    panels.extend(
        (
            f"t={t:g}",
            state_to_display(state, sample),
            False,
        )
        for t, state in zip(times, states, strict=True)
        if t != 0.0
    )

    columns = 4
    rows = int(np.ceil(len(panels) / columns))

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4 * rows),
    )

    axes = np.asarray(axes).reshape(-1)

    for axis, (title, value, is_rgb) in zip(
        axes, panels, strict=False
    ):
        if is_rgb:
            axis.imshow(value)
        else:
            axis.imshow(
                colorize(value, "cityscapes")
            )

        axis.set_title(title)
        axis.axis("off")

    for axis in axes[len(panels):]:
        axis.set_visible(False)

    figure.suptitle(
        "Standard Normal → Hard GT / Linear Interpolation"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=130,
        bbox_inches="tight",
    )
    plt.close(figure)


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)

    if config["dataset"]["name"] != "cityscapes":
        raise ValueError(
            "This script currently supports Cityscapes."
        )

    device = resolve_device(
        args.device
        or config["runtime"]["device"]
    )

    dataset = build_dataset(
        config,
        args.split,
        augment=False,
    )

    if args.indices is None:
        indices = list(
            range(
                min(
                    args.num_images,
                    len(dataset),
                )
            )
        )
    else:
        indices = args.indices

    if any(index >= len(dataset) for index in indices):
        raise IndexError(
            f"Dataset index exceeds split size {len(dataset)}"
        )

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    times = [
        float(t)
        for t in args.times
    ]

    rows: list[dict] = []

    total_elements = 0
    total_abs = 0.0
    global_min = float("inf")
    global_max = float("-inf")

    for ordinal, dataset_index in enumerate(indices):
        sample = dataset[dataset_index]

        image = (
            sample["image"]
            .unsqueeze(0)
            .to(device)
        )

        target_full = (
            sample["target"]
            .long()
            .unsqueeze(0)
            .to(device)
        )

        # state resolution is determined by the same
        # state_downsample_factor as production.
        factor = config["model"].get(
            "state_downsample_factor",
            1,
        )

        state_height = image.shape[-2] // factor
        state_width = image.shape[-1] // factor

        targets = prepare_state_targets(
            target_full,
            num_classes=config["dataset"]["num_classes"],
            state_size=(state_height, state_width),
            ignore_index=config["dataset"]["void_class_index"],
            mask_pixel_losses=True,
        )

        x1 = targets.one_hot_state
        target = targets.target_state

        # IMPORTANT:
        # x0 is independent of image and source model.
        sample_seed = (
            int(args.seed)
            + int(dataset_index) * 1009
        )

        x0 = sample_standard_normal(
            x1.shape,
            device=device,
            dtype=x1.dtype,
            seed=sample_seed,
        )

        states = [
            linear_interpolation(
                x0,
                x1,
                t,
            )
            for t in times
        ]

        valid = (
            target
            != config["dataset"]["void_class_index"]
        )

        for t, state in zip(
            times,
            states,
            strict=True,
        ):
            prediction = state.argmax(dim=1)
            hits = prediction == target

            numerator = int(
                (hits & valid).sum()
            )
            denominator = int(
                valid.sum()
            )

            ratio = (
                numerator / denominator
                if denominator
                else float("nan")
            )

            margin = gt_margin(
                state.float(),
                target,
            )

            valid_margin = (
                margin[valid]
                .detach()
                .float()
                .cpu()
            )

            rows.append(
                {
                    "image_index": dataset_index,
                    "t": t,
                    "gt_argmax_ratio": ratio,
                    "gt_argmax_numerator": numerator,
                    "gt_argmax_denominator": denominator,
                    "mean_gt_margin": (
                        float(valid_margin.mean())
                        if valid_margin.numel()
                        else float("nan")
                    ),
                    "median_gt_margin": (
                        float(valid_margin.median())
                        if valid_margin.numel()
                        else float("nan")
                    ),
                }
            )

        x0_float = x0.detach().float()

        total_elements += x0.numel()
        total_abs += float(
            x0_float.abs().sum()
        )

        global_min = min(
            global_min,
            float(x0_float.min()),
        )

        global_max = max(
            global_max,
            float(x0_float.max()),
        )

        display_image = _inverse_normalized_image(
            sample["image"],
            config,
        )

        display_image = _state_to_display(
            display_image[None],
            sample,
        )[0].cpu()

        display_target = sample[
            "target"
        ].cpu()

        save_figure(
            output_dir
            / "figures"
            / f"{dataset_index:06d}.png",
            image=display_image,
            target_full=display_target,
            x0=x0,
            states=states,
            times=times,
            sample=sample,
        )

        print(
            f"[{ordinal + 1}/{len(indices)}] "
            f"index={dataset_index}"
        )

    # Aggregate over images for each t.
    aggregate: list[dict] = []

    for t in times:
        selected = [
            row
            for row in rows
            if row["t"] == t
        ]

        numerator = sum(
            row["gt_argmax_numerator"]
            for row in selected
        )

        denominator = sum(
            row["gt_argmax_denominator"]
            for row in selected
        )

        aggregate.append(
            {
                "t": t,
                "gt_argmax_ratio": (
                    numerator / denominator
                    if denominator
                    else float("nan")
                ),
                "gt_argmax_numerator": numerator,
                "gt_argmax_denominator": denominator,
                "mean_gt_margin": float(
                    np.mean(
                        [
                            row["mean_gt_margin"]
                            for row in selected
                        ]
                    )
                ),
                "median_gt_margin": float(
                    np.mean(
                        [
                            row["median_gt_margin"]
                            for row in selected
                        ]
                    )
                ),
            }
        )

    # CSV
    with (
        output_dir
        / "trajectory_stats.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=aggregate[0].keys(),
        )
        writer.writeheader()
        writer.writerows(aggregate)

    # GT argmax plot
    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.plot(
        [row["t"] for row in aggregate],
        [
            row["gt_argmax_ratio"]
            for row in aggregate
        ],
        marker="o",
    )

    axis.set_xlabel("t")
    axis.set_ylabel(
        "P[argmax(x_t) = y]"
    )
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_dir
        / "gt_argmax_ratio.png",
        dpi=150,
    )
    plt.close(figure)

    # Margin plot
    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.plot(
        [row["t"] for row in aggregate],
        [
            row["mean_gt_margin"]
            for row in aggregate
        ],
        marker="o",
    )

    axis.axhline(
        0.0,
        linewidth=1,
    )

    axis.set_xlabel("t")
    axis.set_ylabel(
        "Mean GT margin"
    )
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_dir
        / "gt_margin.png",
        dpi=150,
    )
    plt.close(figure)

    summary = {
        "mode": "standard_normal",
        "source": "x0 ~ N(0, I)",
        "target": "hard one-hot e_y",
        "interpolation": (
            "x_t = t * x1 + (1 - t) * x0"
        ),
        "seed": args.seed,
        "indices": indices,
        "times": times,
        "num_classes": config[
            "dataset"
        ]["num_classes"],
        "void_excluded": config[
            "dataset"
        ]["void_class_index"],
        "x0_statistics": {
            "abs_mean": (
                total_abs / total_elements
            ),
            "min": global_min,
            "max": global_max,
        },
        "trajectory": aggregate,
    }

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Done.")
    print(
        f"Output: {output_dir}"
    )


if __name__ == "__main__":
    run(parse_args())