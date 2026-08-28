from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import model as model_module
from failure_analysis import BoundedQuantiles
from image_fusion_analysis import (
    ActivationCapture,
    ChannelRMS,
    TensorStatistics,
    deterministic_wrong_image_feature,
    diagnostic_fusion,
    extract_image_features,
    feature_geometry,
    fusion_logits,
)
from model import DiscreteFlowMapModel, TransformerImageEncoder
from metrics import SegmentationMetrics
from source_diagnostics import diagnostic_initial_state


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (4, 8, 16, 32)
        self.config = SimpleNamespace(hidden_sizes=list(channels))
        self.stages = nn.ModuleList(nn.Conv2d(3, channel, 1) for channel in channels)

    def forward(self, pixel_values, output_hidden_states=True, return_dict=True):
        del output_hidden_states, return_dict
        height, width = pixel_values.shape[-2:]
        features = []
        for index, stage in enumerate(self.stages):
            divisor = 4 * 2**index
            resized = F.interpolate(
                pixel_values,
                size=((height + divisor - 1) // divisor,
                      (width + divisor - 1) // divisor),
                mode="bilinear",
                align_corners=False,
            )
            features.append(stage(resized))
        return SimpleNamespace(hidden_states=tuple(features), feature_maps=tuple(features))


def tiny_model() -> DiscreteFlowMapModel:
    model = DiscreteFlowMapModel({
        "num_classes": 3,
        "state_downsample_factor": 4,
        "fusion_channels": 4,
        "rrdb_blocks": 0,
        "rrdb_growth_channels": 2,
        "unet": {
            "base_channels": 4,
            "channel_mults": [1, 1, 1, 1, 1],
            "num_res_blocks": 1,
            "attention_levels": [],
            "num_heads": 1,
            "dropout": 0.0,
            "time_embedding_dim": 8,
        },
    })
    return model.eval()


def test_diagnostic_normal_is_exact_production_fusion():
    endpoint = tiny_model()
    state = torch.randn(1, 3, 20, 24)
    image_feat = torch.randn(1, 4, 20, 24)
    mask_feat = endpoint.mask_encoder(state)
    s = torch.tensor([0.0])
    production = endpoint.forward_logits_with_image_feat(state, image_feat, s, s)
    diagnostic = fusion_logits(
        endpoint, mask_feat, image_feat, s_value=0.0, t_value=0.0
    )
    assert torch.equal(production, diagnostic)


def test_branch_ablation_and_scale_idententities():
    mask = torch.randn(2, 5, 7, 9)
    image = torch.randn_like(mask)
    assert torch.equal(diagnostic_fusion(mask, image), mask + image)
    assert torch.equal(diagnostic_fusion(mask, image, image_scale=0), mask)
    assert torch.equal(diagnostic_fusion(mask, image, mask_scale=0), image)
    assert torch.count_nonzero(
        diagnostic_fusion(mask, image, mask_scale=0, image_scale=0)
    ) == 0
    assert torch.equal(
        diagnostic_fusion(mask, image, image_scale=2), mask + 2 * image
    )
    assert torch.equal(
        diagnostic_fusion(mask, image, mask_scale=2), 2 * mask + image
    )


def test_activation_hooks_do_not_change_output_and_are_removed():
    endpoint = tiny_model()
    mask = torch.randn(1, 4, 20, 24)
    image = torch.randn_like(mask)
    reference = fusion_logits(endpoint, mask, image, s_value=0.0, t_value=0.0)
    with ActivationCapture(endpoint) as capture:
        observed = fusion_logits(endpoint, mask, image, s_value=0.0, t_value=0.0)
    assert torch.equal(reference, observed)
    assert set(capture.activations) == {
        "input_conv", "down1", "down2", "down3", "down4", "down5",
        "bottleneck", "up1", "up2", "up3", "up4", "up5", "output",
    }
    assert all(not module._forward_hooks for module in capture.modules.values())


def test_exposed_swin_fpn_path_is_exact(monkeypatch):
    monkeypatch.setattr(
        model_module,
        "load_transformer_image_backbone",
        lambda kind, variant, pretrained: (FakeBackbone(), [4, 8, 16, 32]),
    )
    encoder = TransformerImageEncoder(
        "swin", "tiny", False, False, "ddp_fpn_merge", 6, 4, 4, True
    ).eval()
    wrapper = SimpleNamespace(image_encoder=encoder, state_downsample_factor=4)
    image = torch.randn(1, 3, 33, 49)
    exposed = extract_image_features(wrapper, image)
    production = encoder(image)
    assert torch.equal(exposed["image_feat"], production)
    assert [feature.shape[1] for feature in exposed["stages"]] == [4, 8, 16, 32]
    assert exposed["concat"].shape[1] == 24
    assert exposed["merge_gn"].shape[1] == 6


def test_feature_geometry_detects_alignment_and_cancellation():
    image = torch.ones(1, 2, 2, 2)
    aligned = feature_geometry(image, image)
    opposed = feature_geometry(image, -image)
    assert torch.allclose(aligned["per_pixel_ratio"], torch.ones(1, 2, 2))
    assert torch.allclose(aligned["cosine"], torch.ones(1, 2, 2))
    assert torch.allclose(opposed["cosine"], -torch.ones(1, 2, 2))
    assert torch.count_nonzero(opposed["cancellation_ratio"]) == 0


def test_statistics_accumulate_in_float32_and_channel_ratio_is_exact():
    values = torch.tensor([[[[1.0]], [[2.0]]]], dtype=torch.bfloat16)
    statistics = TensorStatistics(collect_pixel_l2=False)
    statistics.update(values)
    result = statistics.compute()
    assert result["aggregation_dtype"] == "float32"
    assert result["rms"] == pytest.approx((2.5) ** 0.5)
    channels = ChannelRMS()
    channels.update(values)
    assert torch.equal(channels.rms(), torch.tensor([1.0, 2.0]))
    mask_channels = ChannelRMS()
    mask_channels.update(torch.ones_like(values))
    assert torch.equal(
        channels.rms() / mask_channels.rms(), torch.tensor([1.0, 2.0])
    )


def test_wrong_image_feature_mapping_is_deterministic_and_resizable():
    previous = torch.arange(12.0).reshape(1, 1, 3, 4)
    first = deterministic_wrong_image_feature(previous, (6, 8))
    second = deterministic_wrong_image_feature(previous, (6, 8))
    assert torch.equal(first, second)
    assert first.shape == (1, 1, 6, 8)
    assert deterministic_wrong_image_feature(previous, (3, 4)) is previous


def test_bounded_quantile_subsampling_uses_safe_integer_endpoints():
    accumulator = BoundedQuantiles(max_per_update=4, max_total=5)
    accumulator.update(torch.arange(10.0))
    assert torch.equal(accumulator.chunks[0], torch.tensor([0.0, 3.0, 6.0, 9.0]))
    accumulator.update(torch.arange(10.0, 20.0))
    result = accumulator.compute()
    assert result["quantile_sample_count"] == 5


def test_source_off_initial_state_is_exact_epsilon():
    mu = torch.randn(1, 3, 4, 5)
    epsilon = torch.randn_like(mu)
    assert torch.equal(
        diagnostic_initial_state(mu, epsilon, 1.0, mu_zero=True), epsilon
    )


def test_void_zero_is_excluded_from_metric_evaluation():
    metric = SegmentationMetrics(
        num_classes=3,
        void_class_index=0,
        evaluated_class_indices=[1, 2],
        prediction_void_retained=True,
    )
    metric.update(
        torch.tensor([[2, 1, 0]]),
        torch.tensor([[0, 1, 2]]),
    )
    result = metric.compute()
    assert result["void_gt_excluded"] == 0
    assert result["evaluated_class_indices"] == [1, 2]
    assert sum(map(sum, result["confusion_matrix"])) == 2
