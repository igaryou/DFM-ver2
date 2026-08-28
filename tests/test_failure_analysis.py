import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from failure_analysis import (
    DistributionAccumulator,
    RetentionAccumulator,
    class_score_maps,
    endpoint_probability,
    original_continuous,
    run_flow_with_image_feat,
    state_resolution_oracle,
    valid_class_scores,
)
from inference import sample_segmentation_from_x0, state_to_prediction
from metrics import SegmentationMetrics


def test_margin_calculation_uses_correct_and_maximum_wrong_class():
    scores = torch.tensor([[[[0.0]], [[1.0]], [[3.0]], [[2.0]]]])
    target = torch.tensor([[[2]]])
    correct, wrong, margin = valid_class_scores(scores, target)
    torch.testing.assert_close(correct, torch.tensor([3.0]))
    torch.testing.assert_close(wrong, torch.tensor([2.0]))
    torch.testing.assert_close(margin, torch.tensor([1.0]))


def test_void_pixels_are_excluded_from_margin_values():
    scores = torch.tensor([[[[5.0, 0.0]], [[1.0, 2.0]], [[0.0, 3.0]]]])
    target = torch.tensor([[[0, 2]]])
    correct, wrong, margin = valid_class_scores(scores, target, ignore_index=0)
    assert correct.numel() == wrong.numel() == margin.numel() == 1
    torch.testing.assert_close(margin, torch.tensor([1.0]))


def test_top1_retention_counts_only_mu_correct_conditioning_set():
    accumulator = RetentionAccumulator()
    accumulator.update(
        torch.tensor([1.0, 0.5, -0.2, -1.0]),
        torch.tensor([0.2, -0.1, 0.5, -0.5]),
    )
    result = accumulator.compute()
    assert result["top1_accuracy"] == pytest.approx(0.5)
    assert result["top1_retention_given_mu_correct"] == pytest.approx(0.5)


def test_distribution_statistics_cast_bfloat16_inputs_to_float32():
    accumulator = DistributionAccumulator()
    accumulator.update(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    assert accumulator.quantiles.chunks[0].dtype == torch.float32
    result = accumulator.compute()
    assert result["aggregation_dtype"] == "float32"
    assert result["mean"] == pytest.approx(1.5)


class _CachedEndpoint(nn.Module):
    def encode_image(self, image):
        return F.adaptive_avg_pool2d(image[:, :1], (2, 3)).expand(-1, 3, -1, -1)

    def forward_logits_with_image_feat(self, state, image_feat, s, t):
        return state + image_feat + (s + t)[:, None, None, None]

    def forward_logits(self, state, image, s, t):
        return self.forward_logits_with_image_feat(
            state, self.encode_image(image), s, t
        )


def _config():
    return {
        "dataset": {"void_class_index": 0},
        "model": {"state_downsample_factor": 4},
        "flow": {"time_eps": 1.0e-5},
        "evaluation": {
            "num_steps": 1,
            "align_corners": False,
            "exclude_void_from_prediction": True,
        },
    }


@pytest.mark.parametrize("steps", [1, 3])
def test_cached_image_feature_flow_matches_production_inference(steps):
    model = _CachedEndpoint()
    image = torch.randn(1, 3, 8, 12)
    x0 = torch.randn(1, 3, 2, 3)
    expected = sample_segmentation_from_x0(
        model, image, x0, _config(), num_steps=steps, return_terminal_state=True
    )
    actual = run_flow_with_image_feat(
        model, model.encode_image(image), x0, _config(), steps
    )
    torch.testing.assert_close(actual, expected)


def test_pi01_equals_one_step_flow_map():
    model = _CachedEndpoint()
    image = torch.randn(1, 3, 8, 12)
    x0 = torch.randn(1, 3, 2, 3)
    image_feat = model.encode_image(image)
    pi01 = endpoint_probability(
        model, image_feat, x0, s_value=0.0, t_value=1.0
    )
    terminal = run_flow_with_image_feat(model, image_feat, x0, _config(), 1)
    torch.testing.assert_close(pi01, terminal)


def test_semantic_only_argmax_never_returns_ade_void():
    scores = torch.zeros(1, 151, 2, 2)
    scores[:, 0] = 100
    prediction = state_to_prediction(
        scores, void_class_index=0, exclude_void=True
    )
    assert prediction.min() >= 1
    assert prediction.max() <= 150


def test_h4_oracle_shape_and_perfect_block_metric():
    target = torch.tensor([[[1, 1, 2, 2], [1, 1, 2, 2],
                            [3, 3, 1, 1], [3, 3, 1, 1]]])
    sample = {
        "model_shape": (4, 4),
        "padded_shape": (4, 4),
        "original_shape": (4, 4),
    }
    state = state_resolution_oracle(
        target, sample, state_size=(2, 2), num_classes=4
    )
    assert state.shape == (1, 4, 2, 2)
    continuous = original_continuous(state, sample, _config())
    prediction = state_to_prediction(
        continuous, void_class_index=0, exclude_void=True
    )
    metrics = SegmentationMetrics(
        num_classes=4,
        void_class_index=0,
        evaluated_class_indices=range(1, 4),
        nanmean=True,
        prediction_void_retained=False,
    )
    metrics.update(prediction, target)
    assert metrics.compute()["mIoU"] == pytest.approx(1.0)


def test_class_score_maps_reject_shape_mismatch():
    with pytest.raises(ValueError, match="scores/target shapes"):
        class_score_maps(torch.randn(1, 3, 2, 2), torch.zeros(1, 3, 2).long())
