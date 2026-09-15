from __future__ import annotations

import json
import math

import pytest
import torch

from lidc_source_diagnostics import (
    compute_lidc_source_diagnostics,
    save_diagnostics_json,
    summarize_lidc_source_diagnostics,
)


def _known_diagnostics():
    mu = torch.tensor([
        [[2.0, 0.0], [2.0, 0.0]],
        [[0.0, 2.0], [0.0, 2.0]],
    ])
    # mu argmax is [[0,1],[0,1]]. Samples flip respectively 0/4 and 2/4 pixels.
    x0_samples = torch.stack((
        mu,
        torch.tensor([
            [[0.0, 2.0], [2.0, 0.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ]),
    ))
    target = torch.tensor([[0, 1], [0, 1]])
    result = compute_lidc_source_diagnostics(
        mu, torch.zeros_like(mu), x0_samples, target,
        sample_index=0, sample_id="known",
    )
    return result


def test_known_margin_noise_flip_and_foreground_fraction():
    result = _known_diagnostics()
    assert result["mu_margin"]["mean"] == 0.0
    assert result["mu_margin"]["abs_mean"] == 2.0
    assert result["noise"]["source_std_mean"] == 1.0
    assert result["noise"]["margin_noise_std_mean"] == pytest.approx(math.sqrt(2))
    assert result["argmax_flip"]["per_sample"] == [0.0, 0.5]
    assert result["argmax_flip"]["mean"] == 0.25
    assert result["foreground_flip"]["per_sample"] == [0.0, 0.5]
    assert result["background_flip"]["per_sample"] == [0.0, 0.5]
    assert result["foreground_fraction"]["per_sample"] == [0.5, 0.5]


def test_pairwise_metrics_exclude_self_and_treat_empty_pairs_as_identical():
    mu = torch.stack((torch.ones(2, 2), torch.zeros(2, 2)))
    empty_samples = mu.unsqueeze(0).repeat(3, 1, 1, 1)
    result = compute_lidc_source_diagnostics(
        mu, torch.zeros_like(mu), empty_samples,
        torch.zeros(2, 2, dtype=torch.long), sample_index=0,
    )
    diversity = result["source_diversity"]
    assert diversity["pairwise_iou_mean"] == 1.0
    assert diversity["pairwise_iou_std"] == 0.0
    assert diversity["pairwise_dice_mean"] == 1.0
    assert diversity["pairwise_dice_std"] == 0.0

    # With two masks there is exactly one unique i<j pair. Including self-pairs
    # would make this mean non-zero.
    opposite = empty_samples[:2].clone()
    opposite[1] = opposite[1].flip(0)
    result = compute_lidc_source_diagnostics(
        mu, torch.zeros_like(mu), opposite,
        torch.zeros(2, 2, dtype=torch.long), sample_index=0,
    )
    assert result["source_diversity"]["pairwise_iou_mean"] == 0.0
    assert result["source_diversity"]["pairwise_dice_mean"] == 0.0


def test_empty_gt_region_is_json_null_and_sample_and_summary_save(tmp_path):
    result = _known_diagnostics()
    mu = torch.stack((torch.ones(2, 2), torch.zeros(2, 2)))
    no_foreground = compute_lidc_source_diagnostics(
        mu, torch.zeros_like(mu), mu.unsqueeze(0),
        torch.zeros(2, 2, dtype=torch.long), sample_index=1,
    )
    assert no_foreground["foreground_flip"]["mean"] is None
    assert no_foreground["foreground_flip"]["per_sample"] == [None]

    sample_path = tmp_path / "metrics/sample_0000.json"
    save_diagnostics_json(result, sample_path)
    summary = summarize_lidc_source_diagnostics(
        [result, no_foreground],
        {"checkpoint_path": "epoch.pt", "num_source_samples": 2},
    )
    summary_path = tmp_path / "metrics/summary.json"
    save_diagnostics_json(summary, summary_path)

    assert json.loads(sample_path.read_text())["sample_id"] == "known"
    loaded_summary = json.loads(summary_path.read_text())
    assert loaded_summary["num_images"] == 2
    assert loaded_summary["num_source_samples"] == 2
    assert "flip_rate_mean" in loaded_summary
    assert "flip_rate_std" in loaded_summary
