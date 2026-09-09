from __future__ import annotations

from pathlib import Path

import pytest
import torch

from adaptive_path import bounded_gaussian_variance_maps, source_entropy_difficulty
from config import load_config
from model import DiscreteFlowMapModel
from segformer_architecture import SEGFORMER_DEPTHS, SEGFORMER_HIDDEN_SIZES
from discrete_flow_maps import linear_path, sample_prior
from inference import sample_segmentation_from_x0
from source_model import SegFormerSourceGenerator, UNetSourceGenerator
from state_space import prepare_state_targets, state_spatial_size
from training_objectives import (
    DDPCompatibleTrainingModel,
    compute_model_training_objectives,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/cityscapes/mmseg/psd/segformer/joint_ce_160k.yaml"
SEGFORMER_CONFIG_NAMES = (
    "joint_ce_160k.yaml",
    "joint_align_160k.yaml",
    "joint_bounded_gaussian_ce_160k.yaml",
    "joint_bounded_gaussian_align_160k.yaml",
    "joint_bounded_gaussian_ce_exponential_path_160k.yaml",
    "joint_bounded_gaussian_ce_entropy_adaptive_variance_160k.yaml",
    "joint_bounded_gaussian_ce_exponential_path_adaptive_std_160k.yaml",
)


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


def test_segformer_endpoint_uses_fused_channels_and_returns_full_state_logits():
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
    state = torch.randn(1, 20, 32, 64)
    time = torch.tensor([0.25])
    endpoint.eval()
    stage_shapes = []
    hooks = [
        stage.register_forward_hook(
            lambda _module, _inputs, output: stage_shapes.append(output.shape[-2:])
        )
        for stage in endpoint.segformer_encoder.encoder.stages
    ]
    with torch.no_grad():
        image_feature = endpoint.encode_image(image)
        logits = endpoint.forward_logits_with_image_feat(
            state, image_feature, time, time + 0.5
        )
    for hook in hooks:
        hook.remove()
    assert image_feature.shape == (1, 4, 32, 64)
    assert stage_shapes == [(8, 16), (4, 8), (2, 4), (1, 2)]
    assert logits.shape == state.shape


def test_segformer_endpoint_rejects_quarter_resolution_state():
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
    image = torch.randn(1, 3, 32, 64)
    quarter_state = torch.randn(1, 20, 8, 16)
    time = torch.tensor([0.25])
    with pytest.raises(AssertionError, match="full-resolution state"):
        endpoint.forward_logits_with_image_feat(
            quarter_state, endpoint.encode_image(image), time, time + 0.5
        )


def test_endpoint_segformer_pretrained_is_false_by_default_and_rgb_pretraining_rejected():
    config = load_config(CONFIG)
    assert config["model"]["endpoint"]["pretrained"] is False
    with pytest.raises(ValueError, match="requires pretrained=false"):
        load_config(CONFIG, ["model.endpoint.pretrained=true"])


@pytest.mark.parametrize(
    ("config_name", "supervision_type"),
    [("joint_ce_160k.yaml", "cross_entropy"), ("joint_align_160k.yaml", "align")],
)
def test_segformer_endpoint_runs_full_resolution_feature_training_path(
    config_name, supervision_type,
):
    config = load_config(CONFIG.parent / config_name, [
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
    source = UNetSourceGenerator(20, 4, False, 1.0, 1)
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
    assert config["source"]["supervision"]["type"] == supervision_type
    assert torch.isfinite(result["stats"]["loss_source_supervision"])


def test_segformer_and_original_configs_keep_independent_state_resolutions():
    segformer = load_config(CONFIG)
    original = load_config(
        ROOT / "configs/cityscapes/mmseg/psd/original/"
        "joint_swin_t_segformer_b1_standard_ce_160k.yaml"
    )
    assert segformer["model"]["state_downsample_factor"] == 1
    assert original["model"]["state_downsample_factor"] == 4


@pytest.mark.parametrize("config_name", SEGFORMER_CONFIG_NAMES)
def test_all_segformer_endpoint_configs_use_full_resolution_state(config_name):
    config = load_config(CONFIG.parent / config_name)
    assert config["model"]["state_downsample_factor"] == 1


def test_full_resolution_source_target_path_and_one_step_inference():
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
    image = torch.randn(1, 3, 32, 64)
    target = torch.randint(0, 20, (1, 32, 64))
    state_size = state_spatial_size(
        image, config["model"]["state_downsample_factor"]
    )
    targets = prepare_state_targets(
        target,
        num_classes=20,
        state_size=state_size,
        ignore_index=config["loss"]["ignore_index"],
        mask_pixel_losses=config["loss"]["mask_pixel_losses"],
    )
    source = UNetSourceGenerator(20, 4, False, 1.0, 1)
    x0, _ = sample_prior(
        config, image, targets.one_hot_state, source,
        target_full=target, sampling_mode="training",
    )
    x1 = targets.one_hot_state
    s = torch.tensor([0.25])
    t = torch.tensor([0.75])
    xs = linear_path(x0, x1, s, config)
    xt = linear_path(x0, x1, t, config)
    expected = (1, 20, 32, 64)
    assert x0.shape == x1.shape == xs.shape == xt.shape == expected

    endpoint = DiscreteFlowMapModel(config["model"])
    final_state = sample_segmentation_from_x0(
        endpoint, image, x0, config, num_steps=1, return_terminal_state=True
    )
    assert final_state.shape == expected


def test_source_segformer_statistics_are_resized_to_full_resolution():
    source = SegFormerSourceGenerator(
        num_classes=20,
        variant="b0",
        pretrained=False,
        decoder_channels=8,
        freeze_encoder=False,
        learned_logvar=False,
        fixed_std=1.0,
        mu_tanh_scale=0.0,
        state_downsample_factor=1,
        decoder_type="standard",
    )
    image = torch.randn(1, 3, 32, 64)
    with torch.no_grad():
        mu_raw, logvar = source.forward_statistics(image)
    assert mu_raw.shape == logvar.shape == (1, 20, 32, 64)


def test_psd_teacher_student_backward_stays_at_full_state_resolution():
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
    source = UNetSourceGenerator(20, 4, False, 1.0, 1)
    adapter = DDPCompatibleTrainingModel(endpoint, source, config)
    image = torch.randn(1, 3, 32, 64)
    target = torch.randint(0, 20, (1, 32, 64))
    result = compute_model_training_objectives(
        adapter,
        operation="joint_objectives",
        image=image,
        target=target,
        epoch_index=0,
        progress_in_epoch=0.5,
    )
    assert result["stats"]["state_height"] == 32
    assert result["stats"]["state_width"] == 64
    assert result["stats"]["psd_loss_height"] == 32
    assert result["stats"]["psd_loss_width"] == 64
    result["loss"].backward()
    assert endpoint.fusion_projection.weight.grad is not None


def test_adaptive_path_and_variance_maps_are_full_resolution():
    config = load_config(
        CONFIG.parent
        / "joint_bounded_gaussian_ce_exponential_path_adaptive_std_160k.yaml"
    )
    source_state = torch.randn(2, 20, 32, 64)
    entropy, path_difficulty = source_entropy_difficulty(
        source_state, config, spatial_size=(32, 64)
    )
    variance_entropy, variance_difficulty, variance, std = (
        bounded_gaussian_variance_maps(
            source_state,
            base_std=config["source"]["fixed_std"],
            variance_type="entropy_adaptive",
            rho=config["source"]["bounded_gaussian"]["variance"]["rho"],
            normalization="rank",
            eps=1.0e-8,
        )
    )
    assert entropy.shape == path_difficulty.shape == (2, 32, 64)
    assert variance_entropy.shape == variance_difficulty.shape == (2, 32, 64)
    assert variance.shape == std.shape == (2, 32, 64)


def test_segformer_endpoint_state_dict_round_trip():
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
    restored = DiscreteFlowMapModel(config["model"])
    restored.load_state_dict(endpoint.state_dict(), strict=True)
    for expected, observed in zip(endpoint.parameters(), restored.parameters()):
        torch.testing.assert_close(observed, expected)


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
