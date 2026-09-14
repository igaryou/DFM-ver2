from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from diagnose_cityscapes_endpoint import (
    deterministic_epsilon_like,
    fixed_t_state,
    run_diagnostics,
    validate_checkpoint_architecture,
    zero_image_feature,
)
from metrics import SegmentationMetrics


def _config() -> dict:
    return {
        "runtime": {"device": "cpu", "amp": False, "amp_dtype": "bf16"},
        "dataset": {"name": "cityscapes", "num_classes": 20,
                    "void_class_index": 19, "num_workers": 0, "pin_memory": False},
        "evaluation": {"split": "val", "batch_size": 1, "original_resolution": True,
                       "eval_class_indices": [0, 18], "nanmean": False,
                       "exclude_void_from_prediction": False, "align_corners": False},
        "augmentation": {"normalize": {"enabled": False}, "imagenet_normalize": False},
        "source": {"segformer_variant": "b0", "segformer_decoder": "standard",
                   "fixed_std": 1.0, "type": "trainable_segformer"},
        "model": {"state_downsample_factor": 1,
                  "endpoint": {"type": "segformer", "segformer_variant": "b3"}},
        "flow": {"path": {"type": "power", "exponent": 1.0},
                 "target_smoothing": {"enabled": False}},
    }


class TinySource(nn.Module):
    def forward_statistics(self, image):
        mu = torch.zeros(image.shape[0], 20, *image.shape[-2:], device=image.device)
        mu[:, 0] = 2.0
        return mu, torch.zeros_like(mu)


class TinyEndpoint(nn.Module):
    def encode_image(self, image):
        return image[:, :1]

    def forward_logits_with_image_feat(self, state, image_feat, s, t):
        assert image_feat.shape[-2:] == state.shape[-2:]
        return state + image_feat[:, :1] * 0.01


class TinyDataset:
    def __len__(self): return 2

    def __getitem__(self, index):
        return {"image": torch.zeros(3, 4, 5), "target": torch.tensor([
            [0, 0, 1, 1, 19], [0, 0, 1, 1, 19],
            [0, 0, 1, 1, 19], [0, 0, 1, 1, 19]]),
            "model_shape": (4, 5), "original_shape": (4, 5),
            "padded_shape": (4, 5), "sample_id": f"sample-{index}"}


def test_fixed_path_endpoints():
    config = _config(); x0 = torch.randn(2, 20, 3, 4); x1 = torch.randn_like(x0)
    assert torch.equal(fixed_t_state(x0, x1, 0, config), x0)
    assert torch.equal(fixed_t_state(x0, x1, 1, config), x1)


def test_zero_image_feature_is_exactly_zero():
    feature = torch.randn(2, 7, 3, 4)
    zero = zero_image_feature(feature)
    assert zero.shape == feature.shape and torch.count_nonzero(zero) == 0


def test_deterministic_epsilon_is_sample_stable():
    reference = torch.empty(1, 20, 3, 4)
    first = deterministic_epsilon_like(reference, 42, "frankfurt_1")
    second = deterministic_epsilon_like(reference, 42, "frankfurt_1")
    assert torch.equal(first, second)
    assert not torch.equal(first, deterministic_epsilon_like(reference, 42, "frankfurt_2"))


@pytest.mark.parametrize("path,value", [
    (("source", "segformer_variant"), "b1"),
    (("model", "endpoint", "segformer_variant"), "b0"),
])
def test_architecture_mismatch_is_detected(path, value):
    config = _config(); saved = copy.deepcopy(config); node = saved
    for key in path[:-1]: node = node[key]
    node[path[-1]] = value
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        validate_checkpoint_architecture(config, {"config": saved})


def test_void_gt_is_excluded_from_metrics():
    metrics = SegmentationMetrics(20, 19, evaluated_class_indices=range(19))
    metrics.update(torch.tensor([[0, 3]]), torch.tensor([[0, 19]]))
    result = metrics.compute()
    assert result["pixel_acc"] == 1.0
    assert sum(sum(row) for row in result["confusion_matrix"]) == 1


def test_fixed_t_endpoint_shape_matches_state():
    model = TinyEndpoint(); state = torch.randn(1, 20, 4, 5)
    feature = model.encode_image(torch.randn(1, 3, 4, 5)); t = torch.tensor([.5])
    assert model.forward_logits_with_image_feat(state, feature, t, t).shape == state.shape


def test_synthetic_end_to_end(tmp_path):
    config = _config(); checkpoint = {"config": copy.deepcopy(config), "global_step": 16_000}
    result = run_diagnostics(config, checkpoint_path="synthetic.pt", output_dir=tmp_path,
        t_values=[0, .5, 1], num_visualizations=0, seed=42, max_batches=1,
        device="cpu", dataset=TinyDataset(), models=(checkpoint, TinyEndpoint(), TinySource()))
    assert result["samples_evaluated"] == 1
    assert set(result["fixed_t"]) == {"0.0", "0.5", "1.0"}
    for name in ("diagnostics.json", "diagnostics.csv", "per_class_iou.csv", "summary.txt"):
        assert (tmp_path / name).is_file()
