from __future__ import annotations

import pytest
import torch

import inference
from inference import sample_lidc_distribution
from metrics import (
    LIDCStochasticMetrics,
    binary_pairwise_dice,
    binary_pairwise_iou,
)


@pytest.mark.parametrize("samples", [16, 32])
def test_perfect_lidc_distribution_metrics(samples):
    ground_truths = torch.zeros(2, 4, 8, 8, dtype=torch.long)
    ground_truths[:, :, 2:6, 3:7] = 1
    predictions = ground_truths[:, :1].expand(-1, samples, -1, -1).clone()
    metrics = LIDCStochasticMetrics([samples])
    metrics.update(predictions, ground_truths, predictions[:, 0])
    result = metrics.compute()
    assert result[f"ged{samples}"] == pytest.approx(0.0, abs=1.0e-12)
    assert result[f"hm_iou{samples}"] == pytest.approx(1.0)
    assert result[f"mdm{samples}"] == pytest.approx(1.0)
    for key in ("dice", "foreground_iou", "precision", "sensitivity", "specificity"):
        assert result[key] == pytest.approx(1.0)


def test_empty_masks_are_perfect_and_never_nan():
    prediction = torch.zeros(2, 32, 8, 8, dtype=torch.long)
    target = torch.zeros(2, 4, 8, 8, dtype=torch.long)
    assert torch.all(binary_pairwise_iou(prediction, target) == 1)
    assert torch.all(binary_pairwise_dice(prediction, target) == 1)
    metrics = LIDCStochasticMetrics([16, 32])
    metrics.update(prediction, target, prediction[:, 0])
    result = metrics.compute()
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())
    assert result["ged16"] == pytest.approx(0.0)
    assert result["ged32"] == pytest.approx(0.0)


def test_completely_disjoint_distributions():
    target = torch.zeros(2, 4, 8, 8, dtype=torch.long)
    target[..., :4, :] = 1
    prediction = torch.zeros(2, 32, 8, 8, dtype=torch.long)
    prediction[..., 4:, :] = 1
    metrics = LIDCStochasticMetrics([16, 32])
    metrics.update(prediction, target, prediction[:, 0])
    result = metrics.compute()
    assert result["ged16"] == pytest.approx(2.0)
    assert result["ged32"] == pytest.approx(2.0)
    assert result["hm_iou16"] == pytest.approx(0.0)
    assert result["hm_iou32"] == pytest.approx(0.0)
    assert result["mdm16"] == pytest.approx(0.0)
    assert result["mdm32"] == pytest.approx(0.0)
    assert result["dice"] == pytest.approx(0.0)
    assert result["foreground_iou"] == pytest.approx(0.0)


def test_hungarian_matching_uses_distinct_predictions_for_four_gt_masks():
    target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    for index in range(4):
        target[0, index, index, :] = 1
    prediction = torch.zeros(1, 16, 4, 4, dtype=torch.long)
    prediction[:, :4] = target
    metrics = LIDCStochasticMetrics([16])
    metrics.update(prediction, target, prediction[:, :4].amax(dim=1))
    result = metrics.compute()
    assert result["hm_iou16"] == pytest.approx(1.0)
    assert result["mdm16"] == pytest.approx(1.0)


def test_lidc_distribution_generates_max_count_and_probability_mean(monkeypatch):
    calls = 0

    def fake_probability(_model, _source, image, _config):
        nonlocal calls
        calls += 1
        foreground = torch.full(
            (image.shape[0], *image.shape[-2:]),
            0.75 if calls <= 16 else 0.25,
        )
        return torch.stack((1.0 - foreground, foreground), dim=1)

    monkeypatch.setattr(inference, "sample_segmentation_probabilities", fake_probability)
    predictions, deterministic = sample_lidc_distribution(
        object(), object(), torch.zeros(2, 1, 8, 8),
        {"dataset": {"protocol": "lidc"}}, [16, 32],
    )
    assert calls == 32
    assert predictions.shape == (2, 32, 8, 8)
    assert torch.all(predictions[:, :16] == 1)
    assert torch.all(predictions[:, 16:] == 0)
    # Exactly 0.5 is foreground by the documented >= threshold.
    assert torch.all(deterministic == 1)
