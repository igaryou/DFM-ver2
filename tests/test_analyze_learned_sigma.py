from __future__ import annotations

import pytest
import torch

from analyze_learned_sigma import save_sigma_histogram, save_sigma_panel
from sigma_analysis_utils import (
    StreamingDistribution,
    per_image_sigma_statistics,
    sigma_variation_summary,
)


def test_sigma_distribution_statistics_percentiles_and_target_distances(tmp_path):
    accumulator = StreamingDistribution(seed=7, max_samples=100)
    values = torch.tensor([0.5, 1.0, 1.0, 1.5])
    accumulator.update(values)
    result = accumulator.compute()

    assert result["sigma_mean"] == pytest.approx(1.0)
    assert result["sigma_std"] == pytest.approx((0.125) ** 0.5)
    assert result["sigma_pixel_variance"] == pytest.approx(0.125)
    assert result["sigma_min"] == 0.5
    assert result["sigma_max"] == 1.5
    assert result["sigma_p50"] == pytest.approx(1.0)
    assert result["sigma_abs_deviation_mean"] == pytest.approx(0.25)
    assert result["sigma_abs_deviation_median"] == pytest.approx(0.25)
    assert result["sigma_mse_from_target"] == pytest.approx(0.125)
    assert result["fraction_abs_sigma_minus_1_lt_0.01"] == pytest.approx(0.5)
    assert result["fraction_sigma_lt_0.8"] == pytest.approx(0.25)
    assert result["fraction_sigma_gt_1.2"] == pytest.approx(0.25)

    destination = tmp_path / "sigma_histogram.png"
    save_sigma_histogram(values, values, result, destination)
    assert destination.is_file() and destination.stat().st_size > 0


def test_valid_mask_excludes_void_values():
    accumulator = StreamingDistribution(seed=3, max_samples=100)
    values = torch.tensor([[0.5, 1.0, 3.0]])
    valid = torch.tensor([[True, True, False]])
    accumulator.update(values, valid)
    result = accumulator.compute()
    assert result["sigma_count"] == 2
    assert result["sigma_mean"] == pytest.approx(0.75)
    assert result["sigma_max"] == 1.0


def test_disk_backed_quantiles_and_histogram_use_every_pixel(tmp_path):
    raw_path = tmp_path / "sigma.float32"
    accumulator = StreamingDistribution(
        seed=3, max_samples=2, exact_values_path=raw_path
    )
    values = torch.linspace(0.5, 1.5, 101)
    accumulator.update(values)
    result = accumulator.compute()
    assert result["sigma_quantile_method"] == "exact_all_pixels_disk_backed"
    assert result["sigma_quantile_sample_count"] == values.numel()
    assert result["sigma_p50"] == pytest.approx(1.0)
    assert int(accumulator.histogram_counts.sum()) == values.numel()
    assert not raw_path.exists()


def test_between_and_within_image_variation_are_distinct():
    first = torch.tensor([0.0, 2.0])
    second = torch.tensor([2.0, 4.0])
    rows = [
        {"image_index": 0, "scope": "all", **per_image_sigma_statistics(first)},
        {"image_index": 1, "scope": "all", **per_image_sigma_statistics(second)},
    ]
    result = sigma_variation_summary(rows, "all")
    # Image means are 1 and 3; population std is 1. Each image has std 1.
    assert result["sigma_between_image_std"] == pytest.approx(1.0)
    assert result["sigma_within_image_std_mean"] == pytest.approx(1.0)


def test_per_image_sigma_statistics_include_requested_quantiles():
    result = per_image_sigma_statistics(torch.arange(1.0, 6.0))
    assert result["sigma_mean"] == pytest.approx(3.0)
    assert result["sigma_p50"] == pytest.approx(3.0)
    assert set(("sigma_p05", "sigma_p50", "sigma_p95")) <= result.keys()


def test_sigma_four_panel_absolute_and_contrast_visualizations(tmp_path):
    config = {
        "dataset": {"name": "cityscapes"},
        "augmentation": {
            "imagenet_normalize": False,
            "normalize": {"enabled": False, "mean": [0, 0, 0], "std": [1, 1, 1]},
        },
    }
    for scale in ("absolute", "contrast"):
        destination = tmp_path / f"sigma_{scale}.png"
        save_sigma_panel(
            torch.rand(3, 8, 12), torch.randint(0, 20, (8, 12)),
            torch.linspace(0.9, 1.1, 96).reshape(8, 12), destination,
            config, scale=scale, contrast_limits=(0.92, 1.08),
        )
        assert destination.is_file() and destination.stat().st_size > 0
