from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import load_config
from dataset import build_dataset
from metrics import SegmentationMetrics
from source_model import source_statistics
from utils import resolve_device
from visualization import colorize
from visualize_simplex_source import (
    _inverse_normalized_image,
    _state_to_display,
    load_source_checkpoint,
    resolve_checkpoint,
)


DEFAULT_SIGMAS = (0.0, 1.0, 0.8, 0.6, 0.5, 0.3)
CITYSCAPES_CLASSES = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and visualize argmax(mu + sigma * epsilon) from a "
            "learned image-conditioned Gaussian source without running the DFM"
        )
    )
    parser.add_argument("--config", required=True)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint")
    checkpoint.add_argument("--checkpoint-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument(
        "--max-eval-images", type=int,
        help="Limit quantitative evaluation images; default evaluates the full split",
    )
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.num_images < 0:
        raise ValueError("--num-images must be non-negative")
    if args.max_eval_images is not None and args.max_eval_images <= 0:
        raise ValueError("--max-eval-images must be positive")
    if not args.sigmas:
        raise ValueError("--sigmas must contain at least one value")
    if any(not np.isfinite(sigma) or sigma < 0 for sigma in args.sigmas):
        raise ValueError("--sigmas must contain finite, non-negative values")
    if len(set(args.sigmas)) != len(args.sigmas):
        raise ValueError("--sigmas must not contain duplicates")
    if args.indices is not None:
        if not args.indices:
            raise ValueError("--indices must contain at least one index")
        if any(index < 0 for index in args.indices):
            raise ValueError("--indices must be non-negative")
        if len(set(args.indices)) != len(args.indices):
            raise ValueError("--indices must not contain duplicates")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    return args


def _as_sample(item: Any, dataset_index: int) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    image, target = item
    return {
        "image": image,
        "target": target,
        "sample_id": str(dataset_index),
    }


def _make_epsilon_like(mu: torch.Tensor, seed: int) -> torch.Tensor:
    """Draw exactly one reproducible epsilon tensor for one image."""
    generator = torch.Generator(device=mu.device)
    generator.manual_seed(seed)
    return torch.randn(
        mu.shape, dtype=mu.dtype, device=mu.device, generator=generator
    )


def noise_sweep_predictions(
    mu: torch.Tensor,
    epsilon: torch.Tensor,
    sigmas: list[float] | tuple[float, ...],
    sample: dict[str, Any],
    void_class_index: int = 19,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Resize states, then return full and prediction-void-excluded argmaxes."""
    if mu.ndim != 4 or mu.shape[0] != 1:
        raise ValueError(f"mu must have shape [1,C,H,W], got {tuple(mu.shape)}")
    if epsilon.shape != mu.shape:
        raise ValueError("epsilon must have the same shape as mu")

    mu_display = _state_to_display(mu, sample, mode="bilinear")
    mu_prediction = mu_display.argmax(dim=1)
    predictions = []
    void_excluded_predictions = []
    for sigma in sigmas:
        # epsilon is deliberately reused for every sigma in this image.
        state = mu + float(sigma) * epsilon
        state_display = _state_to_display(state, sample, mode="bilinear")
        predictions.append(state_display.argmax(dim=1))
        semantic = torch.cat(
            (
                state_display[:, :void_class_index],
                state_display[:, void_class_index + 1:],
            ),
            dim=1,
        )
        excluded = semantic.argmax(dim=1)
        excluded = excluded + (excluded >= void_class_index).to(excluded.dtype)
        void_excluded_predictions.append(excluded)
    return mu_prediction, predictions, void_excluded_predictions


def _ratio(numerator: int | torch.Tensor, denominator: int | torch.Tensor) -> float:
    numerator_value = int(numerator)
    denominator_value = int(denominator)
    return (
        numerator_value / denominator_value
        if denominator_value > 0 else float("nan")
    )


def _class_counts(
    target: torch.Tensor, mask: torch.Tensor, num_classes: int = 19
) -> torch.Tensor:
    return torch.bincount(target[mask], minlength=num_classes)[:num_classes]


class NoiseSweepMetrics:
    """Dataset-level confusion matrices and pixel-count ratio accumulators."""

    def __init__(self, sigmas: list[float] | tuple[float, ...]) -> None:
        self.sigmas = [float(sigma) for sigma in sigmas]
        self.basic_metrics = [
            SegmentationMetrics(
                20, 19, evaluated_class_indices=range(19), nanmean=True,
                prediction_void_retained=True,
            )
            for _ in self.sigmas
        ]
        self.void_excluded_metrics = [
            SegmentationMetrics(
                20, 19, evaluated_class_indices=range(19), nanmean=True,
                prediction_void_retained=False,
            )
            for _ in self.sigmas
        ]
        shape = (len(self.sigmas),)
        class_shape = (len(self.sigmas), 19)
        self.valid_pixels = 0
        self.baseline_correct = 0
        self.baseline_wrong = 0
        self.class_pixels = torch.zeros(19, dtype=torch.int64)
        self.class_baseline_correct = torch.zeros(19, dtype=torch.int64)
        self.class_baseline_wrong = torch.zeros(19, dtype=torch.int64)
        self.correct = torch.zeros(shape, dtype=torch.int64)
        self.void_excluded_correct = torch.zeros(shape, dtype=torch.int64)
        self.flips = torch.zeros(shape, dtype=torch.int64)
        self.correct_to_wrong = torch.zeros(shape, dtype=torch.int64)
        self.wrong_to_correct = torch.zeros(shape, dtype=torch.int64)
        self.void_predictions = torch.zeros(shape, dtype=torch.int64)
        self.class_flips = torch.zeros(class_shape, dtype=torch.int64)
        self.class_correct_to_wrong = torch.zeros(class_shape, dtype=torch.int64)
        self.class_wrong_to_correct = torch.zeros(class_shape, dtype=torch.int64)

    def update(
        self,
        *,
        dataset_index: int,
        sample_id: str,
        target: torch.Tensor,
        baseline: torch.Tensor,
        predictions: list[torch.Tensor],
        void_excluded_predictions: list[torch.Tensor],
    ) -> list[dict[str, Any]]:
        target = target.detach().long().cpu().reshape(-1)
        baseline = baseline.detach().long().cpu().reshape(-1)
        predictions = [
            prediction.detach().long().cpu().reshape(-1)
            for prediction in predictions
        ]
        void_excluded_predictions = [
            prediction.detach().long().cpu().reshape(-1)
            for prediction in void_excluded_predictions
        ]
        valid = target != 19
        valid_count = int(valid.sum())
        baseline_correct_mask = valid & (baseline == target)
        baseline_wrong_mask = valid & ~baseline_correct_mask
        baseline_correct = int(baseline_correct_mask.sum())
        baseline_wrong = int(baseline_wrong_mask.sum())

        self.valid_pixels += valid_count
        self.baseline_correct += baseline_correct
        self.baseline_wrong += baseline_wrong
        self.class_pixels += _class_counts(target, valid)
        self.class_baseline_correct += _class_counts(target, baseline_correct_mask)
        self.class_baseline_wrong += _class_counts(target, baseline_wrong_mask)

        rows = []
        for sigma_index, (sigma, prediction, excluded_prediction) in enumerate(zip(
            self.sigmas, predictions, void_excluded_predictions
        )):
            self.basic_metrics[sigma_index].update(prediction, target)
            self.void_excluded_metrics[sigma_index].update(
                excluded_prediction, target
            )
            correct_mask = valid & (prediction == target)
            flip_mask = valid & (prediction != baseline)
            correct_to_wrong_mask = baseline_correct_mask & ~correct_mask
            wrong_to_correct_mask = baseline_wrong_mask & correct_mask
            void_prediction_mask = valid & (prediction == 19)
            excluded_correct = valid & (excluded_prediction == target)

            correct = int(correct_mask.sum())
            flips = int(flip_mask.sum())
            correct_to_wrong = int(correct_to_wrong_mask.sum())
            wrong_to_correct = int(wrong_to_correct_mask.sum())
            void_predictions = int(void_prediction_mask.sum())
            self.correct[sigma_index] += correct
            self.void_excluded_correct[sigma_index] += int(excluded_correct.sum())
            self.flips[sigma_index] += flips
            self.correct_to_wrong[sigma_index] += correct_to_wrong
            self.wrong_to_correct[sigma_index] += wrong_to_correct
            self.void_predictions[sigma_index] += void_predictions
            self.class_flips[sigma_index] += _class_counts(target, flip_mask)
            self.class_correct_to_wrong[sigma_index] += _class_counts(
                target, correct_to_wrong_mask
            )
            self.class_wrong_to_correct[sigma_index] += _class_counts(
                target, wrong_to_correct_mask
            )
            rows.append({
                "dataset_index": dataset_index,
                "sample_id": sample_id,
                "sigma": sigma,
                "valid_pixels": valid_count,
                "pixel_accuracy": _ratio(correct, valid_count),
                "flip_ratio": _ratio(flips, valid_count),
                "correct_to_wrong_ratio": _ratio(
                    correct_to_wrong, baseline_correct
                ),
                "wrong_to_correct_ratio": _ratio(
                    wrong_to_correct, baseline_wrong
                ),
                "void_prediction_ratio": _ratio(void_predictions, valid_count),
                "pixel_accuracy_void_excluded_prediction": _ratio(
                    int(excluded_correct.sum()), valid_count
                ),
            })
        return rows

    def compute(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        summary_rows = []
        class_rows = []
        for sigma_index, sigma in enumerate(self.sigmas):
            basic = self.basic_metrics[sigma_index].compute()
            excluded = self.void_excluded_metrics[sigma_index].compute()
            summary_rows.append({
                "sigma": sigma,
                "miou": basic["mIoU"],
                "pixel_accuracy": _ratio(
                    self.correct[sigma_index], self.valid_pixels
                ),
                "flip_ratio": _ratio(
                    self.flips[sigma_index], self.valid_pixels
                ),
                "correct_to_wrong_ratio": _ratio(
                    self.correct_to_wrong[sigma_index], self.baseline_correct
                ),
                "wrong_to_correct_ratio": _ratio(
                    self.wrong_to_correct[sigma_index], self.baseline_wrong
                ),
                "void_prediction_ratio": _ratio(
                    self.void_predictions[sigma_index], self.valid_pixels
                ),
                "miou_void_excluded_prediction": excluded["mIoU"],
                "pixel_accuracy_void_excluded_prediction": _ratio(
                    self.void_excluded_correct[sigma_index], self.valid_pixels
                ),
            })
            for class_index, class_name in enumerate(CITYSCAPES_CLASSES):
                class_rows.append({
                    "sigma": sigma,
                    "class_index": class_index,
                    "class_name": class_name,
                    "iou": basic["class_iou"][class_index],
                    "iou_void_excluded_prediction": excluded["class_iou"][class_index],
                    "flip_ratio": _ratio(
                        self.class_flips[sigma_index, class_index],
                        self.class_pixels[class_index],
                    ),
                    "correct_to_wrong_ratio": _ratio(
                        self.class_correct_to_wrong[sigma_index, class_index],
                        self.class_baseline_correct[class_index],
                    ),
                    "wrong_to_correct_ratio": _ratio(
                        self.class_wrong_to_correct[sigma_index, class_index],
                        self.class_baseline_wrong[class_index],
                    ),
                    "gt_pixels": int(self.class_pixels[class_index]),
                    "baseline_correct_pixels": int(
                        self.class_baseline_correct[class_index]
                    ),
                    "baseline_wrong_pixels": int(
                        self.class_baseline_wrong[class_index]
                    ),
                })
        return summary_rows, class_rows


def _display_image(sample: dict[str, Any], config: dict) -> torch.Tensor:
    image = _inverse_normalized_image(sample["image"], config)
    target = sample["target"]
    model_shape = tuple(sample.get("model_shape", target.shape))
    image = image[..., : model_shape[0], : model_shape[1]]
    if image.shape[-2:] != target.shape:
        image = F.interpolate(
            image[None], target.shape, mode="bilinear", align_corners=False
        )[0]
    return image


def save_comparison(
    image: torch.Tensor,
    target: torch.Tensor,
    mu_prediction: torch.Tensor,
    sigma_predictions: list[torch.Tensor],
    sigmas: list[float] | tuple[float, ...],
    path: Path,
    dpi: int,
) -> None:
    panels: list[tuple[str, Any]] = [
        ("Input Image", image.permute(1, 2, 0).cpu().numpy()),
        ("Ground Truth", colorize(target, "cityscapes")),
        ("argmax(μ)", colorize(mu_prediction, "cityscapes")),
    ]
    panels.extend(
        (
            f"argmax(μ + {sigma:g}ε)",
            colorize(prediction, "cityscapes"),
        )
        for sigma, prediction in zip(sigmas, sigma_predictions)
        if sigma != 0.0
    )

    # Each 256x512 panel gets four inches of width; dpi controls export quality.
    figure, axes = plt.subplots(
        1, len(panels), figsize=(4.0 * len(panels), 2.8), squeeze=False
    )
    for axis, (title, values) in zip(axes[0], panels):
        axis.imshow(values)
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(pad=0.5)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty metrics CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_metric_plot(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    metric: str,
    ylabel: str,
    comparison_metric: str | None = None,
) -> None:
    ordered = sorted(rows, key=lambda row: row["sigma"])
    sigmas = [row["sigma"] for row in ordered]
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(
        sigmas, [row[metric] for row in ordered], marker="o", label="Basic"
    )
    if comparison_metric is not None:
        axis.plot(
            sigmas,
            [row[comparison_metric] for row in ordered],
            marker="s",
            label="Prediction void excluded",
        )
        axis.legend()
    axis.set_xlabel("Gaussian noise standard deviation σ")
    axis.set_ylabel(ylabel)
    axis.set_xticks(sigmas)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_metric_outputs(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    class_rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    accumulator: NoiseSweepMetrics,
) -> None:
    _write_csv(output_dir / "metrics_summary.csv", summary_rows)
    _write_csv(output_dir / "metrics_per_class.csv", class_rows)
    _write_csv(output_dir / "metrics_per_image.csv", image_rows)
    payload = {
        "evaluation_policies": {
            "basic": (
                "GT void (class 19) excluded; prediction void retained and "
                "counted as incorrect"
            ),
            "void_excluded_prediction": (
                "GT void excluded; class 19 removed before prediction argmax"
            ),
            "miou_aggregation": (
                "Computed after summing dataset-level confusion matrices"
            ),
            "pixel_ratio_aggregation": (
                "Computed from dataset-level numerator and denominator counts"
            ),
            "undefined_ratio": "NaN when denominator is zero",
        },
        "counts": {
            "valid_pixels": accumulator.valid_pixels,
            "baseline_correct_pixels": accumulator.baseline_correct,
            "baseline_wrong_pixels": accumulator.baseline_wrong,
        },
        "metrics": summary_rows,
    }
    with (output_dir / "metrics_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    _save_metric_plot(
        summary_rows,
        output_dir / "sigma_vs_miou.png",
        metric="miou",
        comparison_metric="miou_void_excluded_prediction",
        ylabel="mIoU",
    )
    _save_metric_plot(
        summary_rows,
        output_dir / "sigma_vs_flip_ratio.png",
        metric="flip_ratio",
        ylabel="Flip Ratio vs. argmax(μ)",
    )
    _save_metric_plot(
        summary_rows,
        output_dir / "sigma_vs_pixel_accuracy.png",
        metric="pixel_accuracy",
        comparison_metric="pixel_accuracy_void_excluded_prediction",
        ylabel="Pixel Accuracy",
    )


def _validate_config(config: dict) -> None:
    if config["dataset"]["name"] != "cityscapes":
        raise ValueError("This visualization requires dataset.name=cityscapes")
    if config["source"]["prior_type"] != "image_gaussian":
        raise ValueError("This visualization requires source.prior_type=image_gaussian")
    if config["source"].get("mu_tanh_scale", 0.0) != 0.0:
        raise ValueError(
            "Expected source.mu_tanh_scale=0.0 so that raw source mu is visualized"
        )
    if config["dataset"]["num_classes"] != 20:
        raise ValueError("Expected the 20-state Cityscapes protocol")
    if config["dataset"]["void_class_index"] != 19:
        raise ValueError("Expected Cityscapes void_class_index=19")


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config, args.set)
    _validate_config(config)
    if args.split != config["dataset"]["val_split"]:
        raise ValueError(
            "Quantitative source-noise evaluation must use the validation split "
            f"{config['dataset']['val_split']!r}"
        )
    checkpoint_path = resolve_checkpoint(args.checkpoint, args.checkpoint_dir)
    device = resolve_device(args.device or config["runtime"]["device"])
    checkpoint, source_model = load_source_checkpoint(
        config, checkpoint_path, device
    )
    dataset = build_dataset(config, args.split, augment=False)
    visualization_indices = (
        args.indices
        if args.indices is not None
        else list(range(min(args.num_images, len(dataset))))
    )
    if any(index >= len(dataset) for index in visualization_indices):
        raise IndexError(f"Dataset index exceeds split size {len(dataset)}")
    evaluation_count = min(args.max_eval_images or len(dataset), len(dataset))
    evaluation_indices = list(range(evaluation_count))
    evaluation_set = set(evaluation_indices)
    processing_indices = evaluation_indices + [
        index for index in visualization_indices if index not in evaluation_set
    ]
    visualization_ordinals = {
        index: ordinal for ordinal, index in enumerate(visualization_indices)
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir = output_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    image_records = []
    per_image_rows = []
    source_shape = None
    accumulator = NoiseSweepMetrics(args.sigmas)

    for dataset_index in processing_indices:
        sample = _as_sample(dataset[dataset_index], dataset_index)
        image = sample["image"].unsqueeze(0).to(device)
        target = sample["target"].long()

        # source_statistics calls forward_statistics, avoiding source sampling.
        mu, _ = source_statistics(source_model, image)
        if mu.shape[1] != config["dataset"]["num_classes"]:
            raise RuntimeError(
                f"Source returned {mu.shape[1]} channels; expected "
                f"{config['dataset']['num_classes']}"
            )
        sample_seed = int(args.seed + dataset_index)
        epsilon = _make_epsilon_like(mu, sample_seed)
        (
            mu_prediction,
            sigma_predictions,
            void_excluded_predictions,
        ) = noise_sweep_predictions(
            mu, epsilon, args.sigmas, sample,
            void_class_index=config["dataset"]["void_class_index"],
        )

        expected_shape = tuple(target.shape)
        all_predictions = [mu_prediction, *sigma_predictions]
        if any(
            tuple(prediction.shape[-2:]) != expected_shape
            for prediction in all_predictions
        ):
            raise AssertionError("GT and prediction spatial resolutions do not match")
        for sigma, prediction in zip(args.sigmas, sigma_predictions):
            if sigma == 0.0 and not torch.equal(prediction, mu_prediction):
                raise AssertionError("sigma=0 prediction differs from argmax(mu)")

        sample_id = str(sample.get("sample_id", dataset_index))
        if dataset_index in evaluation_set:
            per_image_rows.extend(accumulator.update(
                dataset_index=dataset_index,
                sample_id=sample_id,
                target=target,
                baseline=mu_prediction[0],
                predictions=[prediction[0] for prediction in sigma_predictions],
                void_excluded_predictions=[
                    prediction[0] for prediction in void_excluded_predictions
                ],
            ))

        if dataset_index in visualization_ordinals:
            ordinal = visualization_ordinals[dataset_index]
            filename = f"image_{ordinal:04d}.png"
            save_comparison(
                _display_image(sample, config),
                target,
                mu_prediction[0].cpu(),
                [prediction[0].cpu() for prediction in sigma_predictions],
                args.sigmas,
                visualizations_dir / filename,
                args.dpi,
            )
            image_records.append(
                {
                    "ordinal": ordinal,
                    "dataset_index": dataset_index,
                    "sample_id": sample_id,
                    "seed": sample_seed,
                    "filename": f"visualizations/{filename}",
                    "target_shape": list(target.shape),
                    "source_shape": list(mu.shape),
                }
            )
        source_shape = list(mu.shape)

    image_records.sort(key=lambda row: row["ordinal"])
    summary_rows, class_rows = accumulator.compute()
    zero_rows = [row for row in summary_rows if row["sigma"] == 0.0]
    if zero_rows and zero_rows[0]["flip_ratio"] != 0.0:
        raise AssertionError("sigma=0 aggregate flip ratio must be exactly zero")
    save_metric_outputs(
        output_dir, summary_rows, class_rows, per_image_rows, accumulator
    )

    metadata = {
        "config": str(Path(args.config).expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_stage": checkpoint.get("stage"),
        "split": args.split,
        "sigmas": [float(sigma) for sigma in args.sigmas],
        "base_seed": int(args.seed),
        "seed_policy": "base_seed + dataset_index",
        "same_epsilon_for_all_sigmas": True,
        "softmax_applied_to_mu": False,
        "argmax_dimension": 1,
        "num_classes": config["dataset"]["num_classes"],
        "void_class_index": config["dataset"]["void_class_index"],
        "source_shape": source_shape,
        "dataset_size": len(dataset),
        "num_evaluated_images": evaluation_count,
        "evaluation_indices": evaluation_indices,
        "num_visualized_images": len(image_records),
        "visualization_indices": visualization_indices,
        "images": image_records,
        "outputs": {
            "metrics_summary_csv": "metrics_summary.csv",
            "metrics_per_class_csv": "metrics_per_class.csv",
            "metrics_per_image_csv": "metrics_per_image.csv",
            "metrics_summary_json": "metrics_summary.json",
            "sigma_vs_miou": "sigma_vs_miou.png",
            "sigma_vs_flip_ratio": "sigma_vs_flip_ratio.png",
            "sigma_vs_pixel_accuracy": "sigma_vs_pixel_accuracy.png",
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    result = {"metadata": metadata, "metrics": summary_rows}
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True))
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
