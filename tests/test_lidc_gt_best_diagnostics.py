from __future__ import annotations

import json

import pytest
import torch
from PIL import Image

from lidc_gt_best_diagnostics import (
    compute_lidc_gt_best_diagnostics,
    save_json,
    save_lidc_gt_best_visualization,
    summarize_lidc_gt_best_diagnostics,
)


def _masks():
    empty = torch.zeros(2, 2, dtype=torch.long)
    top = torch.tensor([[1, 1], [0, 0]])
    bottom = torch.tensor([[0, 0], [1, 1]])
    full = torch.ones(2, 2, dtype=torch.long)
    predictions = torch.stack((empty, top, bottom))
    ground_truths = torch.stack((empty, top, bottom, full))
    return predictions, ground_truths


def test_iou_best_indices_statistics_dice_and_gt_pairs_are_correct():
    predictions, ground_truths = _masks()
    result = compute_lidc_gt_best_diagnostics(
        predictions, ground_truths, sample_index=0, num_steps=1
    )

    assert len(result["gt_prediction_iou_matrix"]) == 4
    assert all(len(row) == 3 for row in result["gt_prediction_iou_matrix"])
    assert [entry["best_iou"] for entry in result["gt_best_iou"]] == [
        1.0, 1.0, 1.0, 0.5
    ]
    assert [
        entry["best_prediction_index"] for entry in result["gt_best_iou"]
    ] == [0, 1, 2, 1]
    expected = torch.tensor([1.0, 1.0, 1.0, 0.5])
    assert result["best_iou_mean"] == pytest.approx(float(expected.mean()))
    assert result["best_iou_std"] == pytest.approx(
        float(expected.std(unbiased=False))
    )
    assert result["best_iou_min"] == 0.5
    assert result["best_iou_max"] == 1.0
    assert result["best_iou_range"] == 0.5
    assert result["num_unique_best_predictions"] == 3
    assert result["best_dice_mean"] == pytest.approx(11.0 / 12.0)
    assert len(result["gt_pairwise_iou"]["pairs"]) == 6
    assert result["gt_pairwise_iou"]["mean"] == pytest.approx(1.0 / 6.0)
    # The empty GT and empty prediction use the existing empty-empty IoU=1 rule.
    assert result["gt_prediction_iou_matrix"][0][0] == 1.0


def test_png_sample_json_and_summary_json_are_saved(tmp_path):
    predictions, ground_truths = _masks()
    result = compute_lidc_gt_best_diagnostics(
        predictions, ground_truths,
        sample_index=0, num_steps=1, sample_id="synthetic",
    )
    png = tmp_path / "sample_0000.png"
    save_lidc_gt_best_visualization(
        torch.zeros(1, 2, 2), ground_truths, predictions, result, png
    )
    sample_json = tmp_path / "metrics/sample_0000.json"
    save_json(result, sample_json)
    summary = summarize_lidc_gt_best_diagnostics(
        [result], {"num_samples": 3, "num_steps": 1}
    )
    summary_json = tmp_path / "metrics/summary.json"
    save_json(summary, summary_json)

    with Image.open(png) as saved:
        assert saved.format == "PNG"
    assert json.loads(sample_json.read_text())["sample_id"] == "synthetic"
    loaded_summary = json.loads(summary_json.read_text())
    assert loaded_summary["num_images"] == 1
    assert loaded_summary["gt3_best_iou_mean"] == 0.5
