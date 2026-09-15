"""Best-of-N/oracle coverage diagnostics for final LIDC DFM predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from metrics import binary_pairwise_dice, binary_pairwise_iou
from visualization import colorize


ORACLE_NOTE = (
    "Best-IoU is a best-of-N / oracle coverage diagnostic, not a "
    "distribution-wide quality metric or a replacement for GED/HM-IoU/MDM."
)


def compute_lidc_gt_best_diagnostics(
    predictions: torch.Tensor,
    ground_truths: torch.Tensor,
    *,
    sample_index: int,
    num_steps: int,
    sample_id: str | None = None,
) -> dict[str, Any]:
    """Compare N final prediction masks against all four LIDC annotations."""
    if predictions.ndim != 3 or predictions.shape[0] == 0:
        raise ValueError("predictions must have shape [N,H,W] with N > 0")
    if ground_truths.ndim != 3 or ground_truths.shape[0] != 4:
        raise ValueError("ground_truths must have shape [4,H,W]")
    if predictions.shape[-2:] != ground_truths.shape[-2:]:
        raise ValueError("predictions and ground_truths must share spatial shape")
    predictions = predictions.detach().cpu()
    ground_truths = ground_truths.detach().cpu()
    # Existing helpers produce [B,N,G]; JSON uses the requested [G,N].
    iou_matrix = binary_pairwise_iou(
        predictions[None], ground_truths[None]
    )[0].transpose(0, 1)
    dice_matrix = binary_pairwise_dice(
        predictions[None], ground_truths[None]
    )[0].transpose(0, 1)
    best_iou, best_iou_indices = iou_matrix.max(dim=1)
    best_dice, best_dice_indices = dice_matrix.max(dim=1)
    gt_gt_iou = binary_pairwise_iou(
        ground_truths[None], ground_truths[None]
    )[0]
    gt_pairs = [
        {
            "gt_a": first,
            "gt_b": second,
            "iou": float(gt_gt_iou[first, second]),
        }
        for first in range(4)
        for second in range(first + 1, 4)
    ]
    gt_pair_values = torch.tensor([pair["iou"] for pair in gt_pairs])
    best_iou_mean = float(best_iou.mean())
    result: dict[str, Any] = {
        "sample_index": int(sample_index),
        "num_samples": int(predictions.shape[0]),
        "num_steps": int(num_steps),
        "diagnostic_note": ORACLE_NOTE,
        "gt_best_iou": [
            {
                "gt_index": gt_index,
                "best_iou": float(best_iou[gt_index]),
                "best_prediction_index": int(best_iou_indices[gt_index]),
            }
            for gt_index in range(4)
        ],
        "best_iou_mean": best_iou_mean,
        "best_iou_std": float(best_iou.std(unbiased=False)),
        "best_iou_min": float(best_iou.min()),
        "best_iou_max": float(best_iou.max()),
        "best_iou_range": float(best_iou.max() - best_iou.min()),
        "num_unique_best_predictions": len(set(best_iou_indices.tolist())),
        "gt_best_dice": [
            {
                "gt_index": gt_index,
                "best_dice": float(best_dice[gt_index]),
                "best_prediction_index": int(best_dice_indices[gt_index]),
            }
            for gt_index in range(4)
        ],
        # This is the same per-GT maximum then GT mean used by image-level MDM.
        "best_dice_mean": float(best_dice.mean()),
        "gt_pairwise_iou": {
            "mean": float(gt_pair_values.mean()),
            "min": float(gt_pair_values.min()),
            "max": float(gt_pair_values.max()),
            "pairs": gt_pairs,
        },
        "gt_prediction_iou_matrix": iou_matrix.tolist(),
        "gt_prediction_dice_matrix": dice_matrix.tolist(),
    }
    if sample_id is not None:
        result["sample_id"] = str(sample_id)
    return result


def summarize_lidc_gt_best_diagnostics(
    samples: list[dict[str, Any]], metadata: dict[str, Any]
) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        if not samples:
            return None
        return sum(float(sample[key]) for sample in samples) / len(samples)

    summary: dict[str, Any] = {
        **metadata,
        "num_images": len(samples),
        "diagnostic_note": ORACLE_NOTE,
        "best_iou_mean_across_images": mean("best_iou_mean"),
        "best_iou_min_mean_across_images": mean("best_iou_min"),
        "best_iou_range_mean_across_images": mean("best_iou_range"),
        "best_dice_mean_across_images": mean("best_dice_mean"),
        "num_unique_best_predictions_mean": mean(
            "num_unique_best_predictions"
        ),
        "gt_pairwise_iou_mean_across_images": (
            None
            if not samples
            else sum(
                sample["gt_pairwise_iou"]["mean"] for sample in samples
            ) / len(samples)
        ),
    }
    for gt_index in range(4):
        summary[f"gt{gt_index}_best_iou_mean"] = (
            None
            if not samples
            else sum(
                sample["gt_best_iou"][gt_index]["best_iou"]
                for sample in samples
            ) / len(samples)
        )
    return summary


def save_lidc_gt_best_visualization(
    image: torch.Tensor,
    ground_truths: torch.Tensor,
    predictions: torch.Tensor,
    diagnostics: dict[str, Any],
    path: str | Path,
) -> None:
    """Place each GT beside its best-IoU final prediction and FP/FN map."""
    image = image.detach().float().cpu()
    ground_truths = ground_truths.detach().cpu()
    predictions = predictions.detach().cpu()
    if image.shape[0] != 1 or ground_truths.shape[0] != 4:
        raise ValueError("expected image [1,H,W] and four ground truths")
    figure = plt.figure(figsize=(12, 17))
    grid = figure.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, 1])
    image_axis = figure.add_subplot(grid[0, :])
    image_axis.imshow(((image[0] + 1.0) / 2.0).clamp(0, 1), cmap="gray")
    image_axis.set_title("Input CT")
    image_axis.axis("off")
    error_colors = np.asarray([
        [0, 0, 0],       # TN
        [0, 220, 0],     # TP
        [255, 210, 0],   # FP
        [0, 140, 255],   # FN
    ], dtype=np.uint8)
    for gt_index, best in enumerate(diagnostics["gt_best_iou"]):
        gt = ground_truths[gt_index].bool()
        prediction_index = best["best_prediction_index"]
        prediction = predictions[prediction_index].bool()
        axes = [
            figure.add_subplot(grid[gt_index + 1, column])
            for column in range(3)
        ]
        axes[0].imshow(colorize(gt, "lidc"))
        axes[0].set_title(f"GT annotation {gt_index}")
        axes[1].imshow(colorize(prediction, "lidc"))
        axes[1].set_title(
            f"Best-IoU final sample #{prediction_index + 1:02d}\n"
            f"IoU={best['best_iou']:.4f}"
        )
        error = torch.zeros_like(gt, dtype=torch.long)
        error[gt & prediction] = 1
        error[~gt & prediction] = 2
        error[gt & ~prediction] = 3
        axes[2].imshow(error_colors[error.numpy()])
        axes[2].set_title("Error: TP=green, FP=yellow, FN=blue")
        for axis in axes:
            axis.axis("off")
    figure.suptitle(ORACLE_NOTE, fontsize=10)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
