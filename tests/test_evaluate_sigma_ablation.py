from __future__ import annotations

import pytest
import torch

import sigma_analysis_utils as analysis_utils
from evaluate_sigma_ablation import comparison_result, save_ablation_panel
from inference import sample_segmentation_from_x0
from sigma_analysis_utils import (
    PredictionComparisonAccumulator,
    build_sigma_loader,
    deterministic_epsilon_for_indices,
    new_segmentation_metrics,
    paired_initial_states,
    parameter_versions,
)


def test_epsilon_is_shared_and_batch_size_independent():
    reference = torch.zeros(3, 4, 2, 2)
    together = deterministic_epsilon_for_indices(reference, [5, 8, 11], seed=123)
    separate = torch.cat([
        deterministic_epsilon_for_indices(reference[index:index + 1], [image_index], seed=123)
        for index, image_index in enumerate((5, 8, 11))
    ])
    assert torch.equal(together, separate)
    # Reusing this tensor for every NFE is independent of the NFE value.
    assert torch.equal(together, deterministic_epsilon_for_indices(
        reference, [5, 8, 11], seed=123
    ))


def test_sigma_one_makes_learned_and_fixed_initial_states_identical():
    mu = torch.randn(2, 3, 2, 4)
    epsilon = torch.randn_like(mu)
    logvar = torch.zeros(2, 1, 2, 4)
    learned, fixed, sigma = paired_initial_states(
        mu, logvar, epsilon, fixed_std=1.0
    )
    assert torch.equal(sigma, torch.ones_like(sigma))
    torch.testing.assert_close(learned, fixed)


class _TinyFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward_logits(self, state, image, s, t):
        del image, s, t
        return self.scale * state


def _config() -> dict:
    return {
        "dataset": {
            "num_classes": 3,
            "void_class_index": 2,
            "protocol": "fixed",
        },
        "model": {"state_downsample_factor": 1},
        "evaluation": {
            "num_steps": 1,
            "eval_class_indices": [0, 1],
            "nanmean": False,
            "exclude_void_from_prediction": True,
        },
        "flow": {"time_eps": 1.0e-5, "path": {"type": "linear"}},
    }


def test_same_x0_gives_identical_prediction_and_metrics():
    config = _config()
    model = _TinyFlow()
    image = torch.randn(1, 3, 2, 3)
    x0 = torch.randn(1, 3, 2, 3)
    first = sample_segmentation_from_x0(model, image, x0, config, num_steps=3)
    second = sample_segmentation_from_x0(model, image, x0, config, num_steps=3)
    assert torch.equal(first, second)
    target = torch.tensor([[[0, 1, 2], [1, 0, 1]]])
    first_metrics = new_segmentation_metrics(config, torch.device("cpu"))
    second_metrics = new_segmentation_metrics(config, torch.device("cpu"))
    first_metrics.update(first, target)
    second_metrics.update(second, target)
    assert first_metrics.compute()["mIoU"] == second_metrics.compute()["mIoU"]


def test_disagreement_correctness_and_void_exclusion():
    target = torch.tensor([[[0, 1, 2, 1]]])
    learned = torch.tensor([[[0, 0, 0, 1]]])
    fixed = torch.tensor([[[1, 1, 1, 1]]])
    sigma = torch.tensor([[[0.5, 1.0, 3.0, 1.2]]])
    accumulator = PredictionComparisonAccumulator()
    accumulator.update(
        learned, fixed, target, sigma, void_index=2, num_classes=3
    )
    result = accumulator.compute()
    assert result["valid_pixel_count"] == 3
    assert result["prediction_disagreement_rate"] == pytest.approx(2 / 3)
    assert result["both_correct_fraction"] == pytest.approx(1 / 3)
    assert result["learned_only_correct_fraction"] == pytest.approx(1 / 3)
    assert result["fixed_only_correct_fraction"] == pytest.approx(1 / 3)
    assert result["both_wrong_fraction"] == 0.0
    assert sum(row["pixel_count"] for row in accumulator.bin_rows()) == 3


def test_comparison_deltas_are_learned_minus_fixed():
    learned = {
        "mIoU": 0.6, "pixel_acc": 0.8, "mAcc": 0.7,
        "class_iou": [0.5, 0.7], "class_accuracy": [0.6, 0.8],
    }
    fixed = {
        "mIoU": 0.4, "pixel_acc": 0.75, "mAcc": 0.5,
        "class_iou": [0.3, 0.5], "class_accuracy": [0.4, 0.6],
    }
    result = comparison_result(
        learned, fixed, PredictionComparisonAccumulator()
    )
    assert result["delta_mIoU"] == pytest.approx(0.2)
    assert result["delta_pixel_acc"] == pytest.approx(0.05)
    assert result["delta_mAcc"] == pytest.approx(0.2)
    assert result["delta_class_iou"] == pytest.approx([0.2, 0.2])


def test_parameter_versions_remain_unchanged_during_evaluation():
    model = _TinyFlow()
    before = parameter_versions(model)
    with torch.no_grad():
        sample_segmentation_from_x0(
            model, torch.randn(1, 3, 2, 2), torch.randn(1, 3, 2, 2),
            _config(), num_steps=1,
        )
    assert parameter_versions(model) == before


def test_max_images_limits_loader(monkeypatch):
    dataset = torch.utils.data.TensorDataset(
        torch.randn(5, 3, 2, 2), torch.zeros(5, 2, 2, dtype=torch.long)
    )
    monkeypatch.setattr(analysis_utils, "build_dataset", lambda *args, **kwargs: dataset)
    config = {
        "evaluation": {"split": "val", "batch_size": 2, "original_resolution": False},
        "dataset": {
            "name": "cityscapes", "num_workers": 0, "pin_memory": False,
            "persistent_workers": False,
        },
    }
    loader, count = build_sigma_loader(config, max_images=3)
    assert count == 3
    assert sum(batch[0].shape[0] for batch in loader) == 3


def test_ablation_twelve_panel_visualization(tmp_path):
    config = {
        "dataset": {"name": "cityscapes", "void_class_index": 19},
        "augmentation": {
            "imagenet_normalize": False,
            "normalize": {"enabled": False, "mean": [0, 0, 0], "std": [1, 1, 1]},
        },
    }
    masks = [torch.randint(0, 19, (8, 12)) for _ in range(6)]
    destination = tmp_path / "ablation.png"
    save_ablation_panel(
        torch.rand(3, 8, 12), masks[0], masks[1],
        torch.linspace(0.9, 1.1, 96).reshape(8, 12),
        masks[2], masks[3], masks[4], masks[5], destination, config,
    )
    assert destination.is_file() and destination.stat().st_size > 0
