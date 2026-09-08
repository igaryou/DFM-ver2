from __future__ import annotations

from pathlib import Path

import pytest
import torch

from config import load_config
from model import DiscreteFlowMapModel
from segformer_architecture import SEGFORMER_DEPTHS, SEGFORMER_HIDDEN_SIZES
from source_model import SegFormerSourceGenerator, UNetSourceGenerator
from training_objectives import (
    DDPCompatibleTrainingModel,
    compute_model_training_objectives,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/cityscapes/mmseg/psd/segformer/joint_ce_160k.yaml"


@pytest.mark.parametrize("source_variant", ["b0", "b1", "b2", "b3"])
@pytest.mark.parametrize("endpoint_variant", ["b0", "b1", "b2", "b3"])
def test_source_and_endpoint_variants_are_independently_configurable(
    source_variant, endpoint_variant,
):
    config = load_config(CONFIG, [
        f"source.segformer_variant={source_variant}",
        f"model.endpoint.segformer_variant={endpoint_variant}",
    ])
    assert config["source"]["segformer_variant"] == source_variant
    assert config["model"]["endpoint"]["segformer_variant"] == endpoint_variant
    assert SEGFORMER_DEPTHS[source_variant]
    assert SEGFORMER_HIDDEN_SIZES[endpoint_variant]
    assert SegFormerSourceGenerator.DEPTHS is SEGFORMER_DEPTHS


def test_segformer_endpoint_uses_fused_channels_and_returns_quarter_state_logits():
    config = load_config(CONFIG, [
        "source.segformer_variant=b1",
        "model.endpoint.segformer_variant=b0",
        "model.endpoint.image_encoder.channels=4",
        "model.endpoint.image_encoder.blocks=0",
        "model.endpoint.image_encoder.growth_channels=2",
        "model.endpoint.state_encoder.channels=5",
        "model.endpoint.state_encoder.blocks=1",
        "model.endpoint.fusion.channels=7",
        "model.endpoint.decoder_channels=8",
        "model.endpoint.time_embedding_dim=16",
    ])
    endpoint = DiscreteFlowMapModel(config["model"])

    assert config["source"]["segformer_variant"] == "b1"
    assert endpoint.segformer_variant == "b0"
    assert endpoint.image_encoder.first.in_channels == 3
    assert endpoint.image_encoder.downsample_factor == 1
    assert endpoint.fusion_projection.in_channels == 9
    assert endpoint.segformer_encoder.encoder.config.num_channels == 7
    assert endpoint.segformer_encoder.encoder.config.depths == SEGFORMER_DEPTHS["b0"]
    assert len(endpoint.segformer_encoder.stage_time) == 4

    image = torch.randn(1, 3, 32, 64)
    state = torch.randn(1, 20, 8, 16)
    time = torch.tensor([0.25])
    endpoint.eval()
    with torch.no_grad():
        image_feature = endpoint.encode_image(image)
        logits = endpoint.forward_logits_with_image_feat(
            state, image_feature, time, time + 0.5
        )
    assert image_feature.shape == (1, 4, 32, 64)
    assert logits.shape == state.shape


def test_endpoint_segformer_pretrained_is_false_by_default_and_rgb_pretraining_rejected():
    config = load_config(CONFIG)
    assert config["model"]["endpoint"]["pretrained"] is False
    with pytest.raises(ValueError, match="requires pretrained=false"):
        load_config(CONFIG, ["model.endpoint.pretrained=true"])


def test_segformer_endpoint_runs_full_resolution_feature_training_path():
    config = load_config(CONFIG, [
        "model.endpoint.segformer_variant=b0",
        "model.endpoint.image_encoder.channels=4",
        "model.endpoint.image_encoder.blocks=0",
        "model.endpoint.image_encoder.growth_channels=2",
        "model.endpoint.state_encoder.channels=4",
        "model.endpoint.state_encoder.blocks=0",
        "model.endpoint.fusion.channels=8",
        "model.endpoint.decoder_channels=8",
        "model.endpoint.time_embedding_dim=16",
    ])
    endpoint = DiscreteFlowMapModel(config["model"])
    source = UNetSourceGenerator(20, 4, False, 1.0, 4)
    adapter = DDPCompatibleTrainingModel(endpoint, source, config)
    image = torch.randn(1, 3, 32, 64)
    target = torch.randint(0, 20, (1, 32, 64))

    result = compute_model_training_objectives(
        adapter,
        operation="stage1_objectives",
        image=image,
        target=target,
        epoch_index=0,
        progress_in_epoch=0.0,
    )
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert endpoint.fusion_projection.weight.grad is not None


def test_all_original_and_segformer_psd_configs_resolve():
    paths = sorted((ROOT / "configs/cityscapes/mmseg/psd").rglob("*.yaml"))
    assert paths
    for path in paths:
        load_config(path)


def test_ablation_configs_keep_source_path_and_variance_axes_independent():
    directory = CONFIG.parent
    ce = load_config(directory / "joint_ce_160k.yaml")
    align = load_config(directory / "joint_align_160k.yaml")
    full = load_config(
        directory / "joint_bounded_gaussian_ce_exponential_path_adaptive_std_160k.yaml"
    )

    assert ce["source"]["prior_type"] == "image_gaussian"
    assert ce["source"]["mu_tanh_scale"] == 0.0
    assert ce["source"]["supervision"]["type"] == "cross_entropy"
    assert ce["flow"]["path"]["type"] == "power"
    assert align["source"]["supervision"]["type"] == "align"
    assert full["source"]["prior_type"] == "image_bounded_gaussian"
    assert full["source"]["bounded_gaussian"]["amplitude"] == 0.5
    assert full["source"]["bounded_gaussian"]["temperature"] == 4.0
    assert full["source"]["bounded_gaussian"]["variance"]["type"] == "entropy_adaptive"
    assert full["source"]["bounded_gaussian"]["variance"]["rho"] == 0.95
    assert full["flow"]["path"]["type"] == "entropy_adaptive"
    assert full["flow"]["path"]["scheduler"]["beta"] == 2.0
