from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from visualize_source_noise_sweep import (
    NoiseSweepMetrics,
    _make_epsilon_like,
    noise_sweep_predictions,
)


def test_epsilon_is_reproducible_per_image_seed():
    reference = torch.empty(1, 20, 4, 6)

    first = _make_epsilon_like(reference, 42)
    repeated = _make_epsilon_like(reference, 42)
    different = _make_epsilon_like(reference, 43)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)


def test_noise_sweep_reuses_epsilon_and_resizes_before_argmax():
    generator = torch.Generator().manual_seed(7)
    mu = torch.randn(1, 20, 4, 6, generator=generator)
    epsilon = torch.randn(1, 20, 4, 6, generator=generator)
    sample = {"target": torch.zeros(8, 12, dtype=torch.long)}
    sigmas = [1.0, 0.5, 0.0]

    mu_prediction, predictions, excluded_predictions = noise_sweep_predictions(
        mu, epsilon, sigmas, sample
    )

    expected_mu = F.interpolate(
        mu, sample["target"].shape, mode="bilinear", align_corners=False
    ).argmax(dim=1)
    assert torch.equal(mu_prediction, expected_mu)
    for sigma, prediction in zip(sigmas, predictions):
        expected = F.interpolate(
            mu + sigma * epsilon,
            sample["target"].shape,
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        assert torch.equal(prediction, expected)
        expected_excluded = F.interpolate(
            (mu + sigma * epsilon)[:, :19],
            sample["target"].shape,
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        assert torch.equal(excluded_predictions[sigmas.index(sigma)], expected_excluded)
    assert torch.equal(predictions[-1], mu_prediction)


def test_void_channel_remains_an_argmax_candidate():
    mu = torch.zeros(1, 20, 2, 3)
    mu[:, 19] = 1.0
    epsilon = torch.zeros_like(mu)
    sample = {"target": torch.zeros(2, 3, dtype=torch.long)}

    mu_prediction, predictions, excluded_predictions = noise_sweep_predictions(
        mu, epsilon, [1.0], sample
    )

    assert torch.equal(mu_prediction, torch.full((1, 2, 3), 19))
    assert torch.equal(predictions[0], mu_prediction)
    assert torch.equal(
        excluded_predictions[0], torch.zeros((1, 2, 3), dtype=torch.long)
    )


def test_dataset_metrics_use_global_pixel_counts_and_keep_void_errors():
    accumulator = NoiseSweepMetrics([0.0, 1.0])
    target = torch.tensor([0, 0, 1, 1, 19])
    baseline = torch.tensor([0, 1, 1, 0, 0])
    noisy = torch.tensor([19, 0, 1, 1, 19])
    excluded_noisy = torch.tensor([0, 0, 1, 1, 0])

    image_rows = accumulator.update(
        dataset_index=3,
        sample_id="sample",
        target=target,
        baseline=baseline,
        predictions=[baseline, noisy],
        void_excluded_predictions=[baseline, excluded_noisy],
    )
    summary, per_class = accumulator.compute()

    assert summary[0]["flip_ratio"] == 0.0
    assert summary[1]["pixel_accuracy"] == pytest.approx(0.75)
    assert summary[1]["flip_ratio"] == pytest.approx(0.75)
    assert summary[1]["correct_to_wrong_ratio"] == pytest.approx(0.5)
    assert summary[1]["wrong_to_correct_ratio"] == pytest.approx(1.0)
    assert summary[1]["void_prediction_ratio"] == pytest.approx(0.25)
    assert summary[1]["pixel_accuracy_void_excluded_prediction"] == 1.0
    assert image_rows[1]["flip_ratio"] == pytest.approx(0.75)
    class_zero = next(
        row for row in per_class
        if row["sigma"] == 1.0 and row["class_index"] == 0
    )
    assert class_zero["flip_ratio"] == 1.0
    assert class_zero["correct_to_wrong_ratio"] == 1.0
    assert class_zero["wrong_to_correct_ratio"] == 1.0
