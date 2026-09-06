from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from adaptive_path import normalize_entropy, shannon_entropy
from config import load_config
from dataset import build_dataset
from source_model import source_statistics
from utils import resolve_device
from visualize_simplex_source import load_source_checkpoint, resolve_checkpoint


DIFFICULTY_BINS = (
    ("[-1.0,-0.6)", -1.0, -0.6, False),
    ("[-0.6,-0.2)", -0.6, -0.2, False),
    ("[-0.2,0.2)", -0.2, 0.2, False),
    ("[0.2,0.6)", 0.2, 0.6, False),
    ("[0.6,1.0]", 0.6, 1.0, True),
)
BINARY_DIFFICULTY_BINS = (
    ("easy", -1.0, 0.0, False),
    ("hard", 0.0, 1.0, True),
)
CITYSCAPES_CLASS_NAMES = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic_light", "traffic_sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate source accuracy as a function of GT-independent "
            "entropy difficulty"
        )
    )
    parser.add_argument("--config", required=True)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint")
    checkpoint.add_argument("--checkpoint-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--entropy-histogram-bins", type=int, default=65536)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_images is not None and args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    if args.entropy_histogram_bins < 100:
        raise ValueError("--entropy-histogram-bins must be at least 100")
    return args


def entropy_difficulty_from_logits(
    mu_raw: torch.Tensor, *, eps: float = 1.0e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return GT-independent H(softmax(mu_raw)) and image-wise rank difficulty."""
    entropy = shannon_entropy(mu_raw, representation="logits", eps=eps)
    difficulty = normalize_entropy(
        entropy,
        "rank",
        valid_mask=None,
        eps=eps,
        num_classes=mu_raw.shape[1],
    )
    return entropy, difficulty


def interval_mask(
    values: torch.Tensor, lower: float, upper: float, include_upper: bool
) -> torch.Tensor:
    upper_mask = values <= upper if include_upper else values < upper
    return (values >= lower) & upper_mask


@dataclass
class BinAccumulator:
    name: str
    lower: float
    upper: float
    include_upper: bool
    num_classes: int
    evaluated_classes: tuple[int, ...]
    nanmean: bool
    kind: str
    confusion: torch.Tensor | None = None
    pixel_count: int = 0
    entropy_sum: float = 0.0
    difficulty_sum: float = 0.0

    def __post_init__(self) -> None:
        self.confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int64
        )

    @torch.no_grad()
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        entropy: torch.Tensor,
        difficulty: torch.Tensor,
        valid: torch.Tensor,
        *,
        bin_values: torch.Tensor,
    ) -> None:
        selected = valid & interval_mask(
            bin_values, self.lower, self.upper, self.include_upper
        )
        count = int(selected.sum())
        if not count:
            return
        selected_target = target[selected].long().cpu()
        selected_prediction = prediction[selected].long().cpu()
        indices = selected_target * self.num_classes + selected_prediction
        self.confusion += torch.bincount(
            indices, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        self.pixel_count += count
        self.entropy_sum += float(entropy[selected].double().sum())
        self.difficulty_sum += float(difficulty[selected].double().sum())

    def compute(self) -> dict[str, Any]:
        confusion = self.confusion.double()
        true_positive = confusion.diag()
        ground_truth = confusion.sum(dim=1)
        predicted = confusion.sum(dim=0)
        union = ground_truth + predicted - true_positive
        evaluated = torch.tensor(self.evaluated_classes, dtype=torch.long)
        present = union > 0
        if self.nanmean:
            iou = torch.where(present, true_positive / union, torch.nan)
            miou = torch.nanmean(iou[evaluated])
        else:
            iou = true_positive / union.clamp_min(1.0)
            miou = iou[evaluated].mean()
        evaluated_present = present[evaluated]
        present_class_miou = (
            (true_positive[evaluated][evaluated_present]
             / union[evaluated][evaluated_present]).mean()
            if bool(evaluated_present.any()) else confusion.new_tensor(float("nan"))
        )
        class_iou = {
            name: (
                float(true_positive[index] / union[index])
                if bool(present[index]) else float("nan")
            )
            for name, index in zip(CITYSCAPES_CLASS_NAMES, self.evaluated_classes)
        }
        correct = true_positive[evaluated].sum()
        return {
            "kind": self.kind,
            "bin": self.name,
            "lower": self.lower,
            "upper": self.upper,
            "upper_inclusive": self.include_upper,
            "pixel_accuracy": (
                float(correct / self.pixel_count) if self.pixel_count else float("nan")
            ),
            "miou": float(miou) if self.pixel_count else float("nan"),
            "present_class_miou": (
                float(present_class_miou) if self.pixel_count else float("nan")
            ),
            "pixel_count": self.pixel_count,
            "mean_entropy": (
                self.entropy_sum / self.pixel_count
                if self.pixel_count else float("nan")
            ),
            "mean_difficulty": (
                self.difficulty_sum / self.pixel_count
                if self.pixel_count else float("nan")
            ),
            "class_iou": class_iou,
            "confusion_matrix": self.confusion.tolist(),
        }


class EntropyHistogram:
    """Bounded-memory estimator for dataset-global entropy quantiles."""

    def __init__(self, num_classes: int, bins: int) -> None:
        self.minimum = 0.0
        self.maximum = math.log(float(num_classes))
        self.bins = bins
        self.counts = torch.zeros(bins, dtype=torch.int64)

    def update(self, entropy: torch.Tensor, valid: torch.Tensor) -> None:
        values = entropy[valid].float().cpu()
        if values.numel():
            self.counts += torch.histc(
                values,
                bins=self.bins,
                min=self.minimum,
                max=self.maximum,
            ).to(torch.int64)

    def quantiles(self, probabilities: Iterable[float]) -> list[float]:
        cumulative = self.counts.cumsum(0)
        total = int(cumulative[-1]) if cumulative.numel() else 0
        if not total:
            raise RuntimeError("No valid semantic pixels were found")
        boundaries = []
        width = (self.maximum - self.minimum) / self.bins
        for probability in probabilities:
            target = max(1, math.ceil(float(probability) * total))
            index = int(torch.searchsorted(cumulative, target))
            boundaries.append(self.minimum + (index + 1) * width)
        return boundaries


def _unpack_sample(sample: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(sample, dict):
        return sample["image"], sample["target"]
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return sample[0], sample[1]
    raise TypeError("Dataset sample must be a mapping or image/target tuple")


def _batches(dataset, indices: list[int], batch_size: int):
    for start in range(0, len(indices), batch_size):
        samples = [_unpack_sample(dataset[index]) for index in indices[start:start + batch_size]]
        image_shapes = {tuple(image.shape) for image, _ in samples}
        target_shapes = {tuple(target.shape) for _, target in samples}
        if len(image_shapes) != 1 or len(target_shapes) != 1:
            raise ValueError(
                "Samples in one batch must share shapes; reduce --batch-size to 1"
            )
        yield (
            torch.stack([image for image, _ in samples]),
            torch.stack([target.long() for _, target in samples]),
        )


def _state_target(target: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(target[:, None].float(), size=size, mode="nearest")[:, 0].long()


@torch.inference_mode()
def _source_batch(
    source_model, image: torch.Tensor, target: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image = image.to(device, non_blocking=True)
    mu_raw, _ = source_statistics(source_model, image)
    entropy, difficulty = entropy_difficulty_from_logits(mu_raw)
    target_state = _state_target(target.to(device), mu_raw.shape[-2:])
    prediction = mu_raw.argmax(dim=1)
    return prediction, target_state, entropy, difficulty


def _make_accumulators(
    definitions, *, kind: str, num_classes: int, void_index: int, nanmean: bool
) -> list[BinAccumulator]:
    evaluated = tuple(index for index in range(num_classes) if index != void_index)
    return [
        BinAccumulator(
            name, lower, upper, include_upper, num_classes, evaluated, nanmean, kind
        )
        for name, lower, upper, include_upper in definitions
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "kind", "bin", "lower", "upper", "upper_inclusive",
        "pixel_accuracy", "miou", "present_class_miou", "pixel_count", "mean_entropy",
        "mean_difficulty",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def _write_class_iou_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["kind", "bin", *CITYSCAPES_CLASS_NAMES]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "kind": row["kind"],
                "bin": row["bin"],
                **row["class_iou"],
            })


def _save_plot(
    path: Path, rows: list[dict[str, Any]], metric: str, ylabel: str
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        [row["bin"] for row in rows],
        [row[metric] for row in rows],
        marker="o",
    )
    axis.set_xlabel("Difficulty bin")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.3)
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _nonincreasing(rows: list[dict[str, Any]], metric: str) -> bool | None:
    values = [float(row[metric]) for row in rows if math.isfinite(float(row[metric]))]
    if len(values) < 2:
        return None
    return all(right <= left for left, right in zip(values, values[1:]))


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config, args.set)
    if config["dataset"]["name"] != "cityscapes":
        raise ValueError("This evaluator currently supports Cityscapes only")
    if config["dataset"]["num_classes"] != 20:
        raise ValueError("Cityscapes evaluation requires 20 source-state classes")
    void_index = int(config["dataset"]["void_class_index"])
    if void_index != 19:
        raise ValueError("Cityscapes void_class_index must be 19")
    device = resolve_device(args.device or config["runtime"]["device"])
    checkpoint_path = resolve_checkpoint(args.checkpoint, args.checkpoint_dir)
    checkpoint, source_model = load_source_checkpoint(config, checkpoint_path, device)
    dataset = build_dataset(config, args.split, augment=False)
    count = len(dataset) if args.num_images is None else min(args.num_images, len(dataset))
    indices = list(range(count))
    if not indices:
        raise RuntimeError("The selected split is empty")

    histogram = EntropyHistogram(20, args.entropy_histogram_bins)
    for batch_index, (image, target) in enumerate(
        _batches(dataset, indices, args.batch_size), start=1
    ):
        _, target_state, entropy, _ = _source_batch(
            source_model, image, target, device
        )
        histogram.update(entropy, target_state != void_index)
        if batch_index % 50 == 0:
            print(f"entropy quantile pass: {batch_index} batches")
    quantiles = histogram.quantiles((0.2, 0.4, 0.6, 0.8))
    entropy_edges = [0.0, *quantiles, math.log(20.0)]
    entropy_definitions = [
        (
            f"Q{index + 1}", entropy_edges[index], entropy_edges[index + 1],
            index == 4,
        )
        for index in range(5)
    ]

    kwargs = {
        "num_classes": 20,
        "void_index": void_index,
        "nanmean": bool(config["evaluation"].get("nanmean", False)),
    }
    binary = _make_accumulators(
        BINARY_DIFFICULTY_BINS, kind="binary", **kwargs
    )
    difficulty_bins = _make_accumulators(
        DIFFICULTY_BINS, kind="five_bin", **kwargs
    )
    entropy_bins = _make_accumulators(
        entropy_definitions, kind="entropy_quantile", **kwargs
    )

    for batch_index, (image, target) in enumerate(
        _batches(dataset, indices, args.batch_size), start=1
    ):
        prediction, target_state, entropy, difficulty = _source_batch(
            source_model, image, target, device
        )
        valid = target_state != void_index
        for accumulator in (*binary, *difficulty_bins):
            accumulator.update(
                prediction, target_state, entropy, difficulty, valid,
                bin_values=difficulty,
            )
        for accumulator in entropy_bins:
            accumulator.update(
                prediction, target_state, entropy, difficulty, valid,
                bin_values=entropy,
            )
        if batch_index % 50 == 0:
            print(f"metric pass: {batch_index} batches")

    binary_rows = [accumulator.compute() for accumulator in binary]
    difficulty_rows = [accumulator.compute() for accumulator in difficulty_bins]
    entropy_rows = [accumulator.compute() for accumulator in entropy_bins]
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "difficulty_bins.csv", [*binary_rows, *difficulty_rows])
    _write_csv(output / "entropy_bins.csv", entropy_rows)
    _write_class_iou_csv(
        output / "difficulty_class_iou.csv", [*binary_rows, *difficulty_rows]
    )
    _save_plot(
        output / "difficulty_vs_pixel_accuracy.png",
        difficulty_rows,
        "pixel_accuracy",
        "Pixel Accuracy",
    )
    _save_plot(
        output / "difficulty_vs_miou.png", difficulty_rows, "miou", "mIoU"
    )
    summary = {
        "config": str(Path(args.config).expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_stage": checkpoint.get("stage"),
        "split": args.split,
        "num_images": count,
        "batch_size": args.batch_size,
        "source_prediction": "argmax(raw_source_logits_mu)",
        "entropy_source": "softmax(raw_source_logits_mu)",
        "entropy_formula": "H = -sum_k p_k*log(p_k)",
        "difficulty": {
            "normalization": "image_wise_rank",
            "range": [-1.0, 1.0],
            "gt_used": False,
        },
        "evaluation": {
            "void_gt_excluded": void_index,
            "evaluated_class_indices": list(range(19)),
            "prediction_void_retained": True,
            "nanmean": kwargs["nanmean"],
        },
        "binary_difficulty_regions": binary_rows,
        "difficulty_bins": difficulty_rows,
        "entropy_quantile_boundaries": entropy_edges,
        "entropy_quantile_estimator": {
            "type": "dataset_global_streaming_histogram",
            "histogram_bins": args.entropy_histogram_bins,
            "range": [0.0, math.log(20.0)],
            "void_excluded_for_evaluation_quantiles": True,
        },
        "entropy_bins": entropy_rows,
        "difficulty_accuracy_trend": {
            "pixel_accuracy_nonincreasing": _nonincreasing(
                difficulty_rows, "pixel_accuracy"
            ),
            "miou_nonincreasing": _nonincreasing(difficulty_rows, "miou"),
            "hard_minus_easy_pixel_accuracy": (
                binary_rows[1]["pixel_accuracy"]
                - binary_rows[0]["pixel_accuracy"]
            ),
            "hard_minus_easy_miou": (
                binary_rows[1]["miou"] - binary_rows[0]["miou"]
            ),
        },
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    print(json.dumps(summary, indent=2, allow_nan=True))
    return summary


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
