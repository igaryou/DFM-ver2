from pathlib import Path

import pytest
import torch
import torch.nn as nn

import evaluate_entropy_difficulty as evaluation


def test_entropy_difficulty_is_softmax_entropy_and_gt_independent():
    mu_raw = torch.tensor([
        [[[5.0, 0.0, -1.0]], [[0.0, 0.0, 1.0]], [[-2.0, 0.0, 0.0]]],
        [[[1.0, 3.0, 0.5]], [[0.0, -2.0, 0.5]], [[-1.0, 0.0, 0.5]]],
    ])
    entropy, difficulty = evaluation.entropy_difficulty_from_logits(mu_raw)
    probability = torch.softmax(mu_raw.float(), dim=1)
    expected = -(probability * probability.clamp_min(1.0e-8).log()).sum(dim=1)
    torch.testing.assert_close(entropy, expected)
    assert difficulty.shape == (2, 1, 3)
    torch.testing.assert_close(
        difficulty.mean(dim=(1, 2)), torch.zeros(2), atol=1.0e-7, rtol=0
    )
    assert difficulty.min() >= -1 and difficulty.max() <= 1


def test_bin_accumulator_uses_only_selected_nonvoid_pixels_and_own_confusion():
    accumulator = evaluation.BinAccumulator(
        "easy", -1.0, 0.0, False, 4, (0, 1, 2), True, "binary"
    )
    prediction = torch.tensor([[[0, 1, 2, 3]]])
    target = torch.tensor([[[0, 0, 2, 3]]])
    entropy = torch.tensor([[[0.1, 0.2, 0.8, 0.9]]])
    difficulty = torch.tensor([[[-0.9, -0.2, 0.5, -0.8]]])
    valid = target != 3
    accumulator.update(
        prediction, target, entropy, difficulty, valid, bin_values=difficulty
    )
    result = accumulator.compute()
    assert result["pixel_count"] == 2
    assert result["pixel_accuracy"] == pytest.approx(0.5)
    assert result["mean_entropy"] == pytest.approx(0.15)
    assert sum(map(sum, result["confusion_matrix"])) == 2
    assert result["confusion_matrix"][3][3] == 0


def test_histogram_quantiles_are_ordered_and_near_fifths():
    histogram = evaluation.EntropyHistogram(num_classes=20, bins=10000)
    values = torch.linspace(0.0, torch.log(torch.tensor(20.0)), 10001)[None]
    histogram.update(values, torch.ones_like(values, dtype=torch.bool))
    boundaries = histogram.quantiles((0.2, 0.4, 0.6, 0.8))
    expected = [float(torch.log(torch.tensor(20.0))) * q for q in (0.2, 0.4, 0.6, 0.8)]
    assert boundaries == pytest.approx(expected, abs=1.0e-3)
    assert boundaries == sorted(boundaries)


class _Dataset:
    def __len__(self):
        return 4

    def __getitem__(self, index):
        target = torch.zeros(4, 6, dtype=torch.long)
        target[:, 3:] = 1
        target[0, 0] = 19
        return {
            "image": torch.full((3, 4, 6), index / 10.0),
            "target": target,
        }


class _Source(nn.Module):
    def forward_statistics(self, image):
        batch, _, height, width = image.shape
        logits = torch.zeros(batch, 20, height, width)
        logits[:, 0, :, : width // 2] = 6.0
        logits[:, 2, :, width // 2 :] = 0.2
        logits[:, 1, :, width // 2 :] = 0.1
        return logits, torch.zeros_like(logits)


def test_evaluate_writes_all_outputs_and_excludes_void_only_in_metrics(
    monkeypatch, tmp_path,
):
    config = {
        "dataset": {"name": "cityscapes", "num_classes": 20, "void_class_index": 19},
        "runtime": {"device": "cpu"},
        "evaluation": {"nanmean": True},
    }
    source = _Source()
    monkeypatch.setattr(evaluation, "load_config", lambda *args: config)
    monkeypatch.setattr(evaluation, "build_dataset", lambda *args, **kwargs: _Dataset())
    monkeypatch.setattr(evaluation, "resolve_device", lambda value: torch.device("cpu"))
    monkeypatch.setattr(evaluation, "resolve_checkpoint", lambda *args: Path("checkpoint.pt"))
    monkeypatch.setattr(
        evaluation, "load_source_checkpoint",
        lambda *args: ({"stage": "joint_training"}, source),
    )
    args = evaluation.parse_args([
        "--config", "config.yaml", "--checkpoint", "checkpoint.pt",
        "--output-dir", str(tmp_path), "--split", "val",
        "--batch-size", "2", "--entropy-histogram-bins", "1000",
    ])
    summary = evaluation.evaluate(args)
    for filename in (
        "summary.json", "difficulty_bins.csv", "entropy_bins.csv",
        "difficulty_vs_pixel_accuracy.png", "difficulty_vs_miou.png",
    ):
        assert (tmp_path / filename).is_file()
    assert summary["difficulty"]["gt_used"] is False
    assert summary["evaluation"]["void_gt_excluded"] == 19
    assert summary["evaluation"]["evaluated_class_indices"] == list(range(19))
    assert sum(row["pixel_count"] for row in summary["difficulty_bins"]) == 4 * 23
    assert len(summary["entropy_bins"]) == 5
    assert "pixel_accuracy_nonincreasing" in summary["difficulty_accuracy_trend"]
    assert summary["entropy_quantile_estimator"]["histogram_bins"] == 1000
    assert summary["binary_difficulty_regions"][0]["pixel_count"] + summary[
        "binary_difficulty_regions"
    ][1]["pixel_count"] == 4 * 23
