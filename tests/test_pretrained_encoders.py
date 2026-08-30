from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import discrete_flow_maps
import model as model_module
from checkpoint import model_signature
from config import load_config
from discrete_flow_maps import sample_prior
from model import (
    DDPFPNMultiStageMerging,
    DiscreteFlowMapModel,
    ImageEncoder,
    TransformerImageEncoder,
)
from model_inspection import inspect_model_parameters
from source_model import (
    TaskFinetunedSegFormerSourceGenerator,
    build_source_model,
)
from trainer import build_optimizer
from training_objectives import DDPCompatibleTrainingModel
from training_objectives import compute_model_training_objectives


ROOT = Path(__file__).parents[1]
ADE_CONFIG = ROOT / "configs" / "joint_psd_ade20k.yaml"
CITY_CONFIG = ROOT / "configs" / "debug_diagonal_cityscapes.yaml"


class FakeBackbone(nn.Module):
    def __init__(self, channels=(4, 8, 16, 32)):
        super().__init__()
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
                size=((height + divisor - 1) // divisor, (width + divisor - 1) // divisor),
                mode="bilinear",
                align_corners=False,
            )
            features.append(stage(resized))
        return SimpleNamespace(
            hidden_states=tuple(features), feature_maps=tuple(features)
        )


class FakeSegmentationModel(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.config = SimpleNamespace(num_labels=classes)
        self.decode = nn.Conv2d(3, classes, 1)

    def forward(self, pixel_values, return_dict=True):
        del return_dict
        return SimpleNamespace(logits=F.avg_pool2d(self.decode(pixel_values), 4))


def _tiny_model_config(config: dict) -> dict:
    result = deepcopy(config["model"])
    result.update({
        "fusion_channels": 8,
        "rrdb_blocks": 1,
        "rrdb_growth_channels": 2,
        "unet": {
            "base_channels": 8,
            "channel_mults": [1, 2],
            "num_res_blocks": 1,
            "attention_levels": [],
            "num_heads": 1,
            "dropout": 0.0,
            "time_embedding_dim": 16,
        },
    })
    return result


def _task_source(classes: int, void_index: int, *, freeze: bool = False):
    return TaskFinetunedSegFormerSourceGenerator(
        num_classes=classes,
        void_class_index=void_index,
        model_id="unit-test/no-download",
        representation="probability",
        freeze=freeze,
        fixed_std=1.0,
        input_already_normalized=True,
        state_downsample_factor=4,
        segmentation_model=FakeSegmentationModel(classes - 1),
    )


def test_legacy_ade_yaml_resolves_to_rrdb_and_legacy_source():
    config = load_config(ADE_CONFIG)
    endpoint = DiscreteFlowMapModel(_tiny_model_config(config))
    assert config["model"]["image_encoder"]["type"] == "rrdb"
    assert isinstance(endpoint.image_encoder, ImageEncoder)
    assert config["source"]["type"] == "trainable_segformer"
    assert config["source"]["backbone"] == "segformer"


def test_legacy_rrdb_state_keys_and_forward_shape_are_unchanged():
    config = load_config(ADE_CONFIG)
    legacy = _tiny_model_config(config)
    legacy.pop("image_encoder")
    explicit = _tiny_model_config(config)
    first = DiscreteFlowMapModel(legacy)
    second = DiscreteFlowMapModel(explicit)
    assert set(first.state_dict()) == set(second.state_dict())
    assert "image_encoder.first.weight" in first.state_dict()
    assert not any(key.startswith("image_encoder.encoder.") for key in first.state_dict())
    output = first.encode_image(torch.randn(2, 3, 31, 47))
    assert output.shape == (2, 8, 8, 12)


@pytest.mark.parametrize("encoder_type", ["swin", "convnext"])
@pytest.mark.parametrize("variant", ["tiny", "small", "base", "large"])
def test_transformer_encoder_variants_validate_without_download(encoder_type, variant):
    config = load_config(ADE_CONFIG, [
        f"model.image_encoder.type={encoder_type}",
        f"model.image_encoder.variant={variant}",
        "model.image_encoder.pretrained=false",
    ])
    assert config["model"]["image_encoder"]["type"] == encoder_type
    assert config["model"]["image_encoder"]["variant"] == variant
    assert config["model"]["image_encoder"]["input_already_normalized"] is True


@pytest.mark.parametrize("encoder_type", ["swin", "convnext"])
def test_normalized_ade_input_requires_transformer_normalized_flag(encoder_type):
    valid = load_config(ADE_CONFIG, [
        f"model.image_encoder.type={encoder_type}",
        "model.image_encoder.pretrained=false",
        "model.image_encoder.input_already_normalized=true",
    ])
    assert valid["augmentation"]["normalize"]["enabled"] is True
    assert valid["model"]["image_encoder"]["input_already_normalized"] is True

    with pytest.raises(
        ValueError, match="Normalized dataset input requires"
    ):
        load_config(ADE_CONFIG, [
            f"model.image_encoder.type={encoder_type}",
            "model.image_encoder.pretrained=false",
            "model.image_encoder.input_already_normalized=false",
        ])


def test_transformer_cli_type_override_auto_matches_dataset_normalization():
    config = load_config(ADE_CONFIG, [
        "model.image_encoder.type=swin",
        "model.image_encoder.pretrained=false",
    ])
    assert config["model"]["image_encoder"]["input_already_normalized"] is True


def test_legacy_imagenet_normalize_is_treated_as_dataset_normalization():
    config = load_config(CITY_CONFIG, [
        "model.image_encoder.type=convnext",
        "model.image_encoder.pretrained=false",
        "augmentation.imagenet_normalize=true",
    ])
    assert config["augmentation"]["normalize"]["enabled"] is False
    assert config["augmentation"]["imagenet_normalize"] is True
    assert config["model"]["image_encoder"]["input_already_normalized"] is True


def test_unnormalized_dataset_rejects_transformer_normalized_flag():
    with pytest.raises(ValueError, match="Unnormalized dataset input requires"):
        load_config(CITY_CONFIG, [
            "model.image_encoder.type=swin",
            "model.image_encoder.pretrained=false",
            "model.image_encoder.input_already_normalized=true",
        ])


def test_rrdb_is_exempt_from_transformer_normalization_validation():
    config = load_config(ADE_CONFIG, [
        "model.image_encoder.input_already_normalized=false",
    ])
    assert config["model"]["image_encoder"]["type"] == "rrdb"
    assert config["augmentation"]["normalize"]["enabled"] is True
    assert config["model"]["image_encoder"]["input_already_normalized"] is False


@pytest.mark.parametrize("encoder_type", ["swin", "convnext"])
def test_scratch_transformer_factory_forward_and_frozen_neck(
    monkeypatch, encoder_type
):
    calls = []

    def fake_loader(kind, variant, pretrained):
        calls.append((kind, variant, pretrained))
        return FakeBackbone(), [4, 8, 16, 32]

    monkeypatch.setattr(model_module, "load_transformer_image_backbone", fake_loader)
    encoder = TransformerImageEncoder(
        encoder_type, "tiny", False, True, "ddp_fpn_merge", 6, 8, 4, True
    )
    encoder.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = encoder(torch.randn(2, 3, 33, 49))
    assert calls == [(encoder_type, "tiny", False)]
    assert output.shape == (2, 8, 9, 13)
    assert torch.isfinite(output).all()
    assert not encoder.backbone.training
    assert not any(parameter.requires_grad for parameter in encoder.backbone.parameters())
    assert all(parameter.requires_grad for parameter in encoder.neck.parameters())
    assert all(parameter.requires_grad for parameter in encoder.projection.parameters())


def test_real_scratch_swin_uses_four_stages_and_all_trainable_params_get_grad():
    encoder = TransformerImageEncoder(
        "swin", "tiny", False, False, "ddp_fpn_merge", 8, 6, 4, True
    )
    image = torch.randn(1, 3, 224, 224)
    features = encoder._extract_backbone_features(image)
    assert [feature.shape[1] for feature in features] == [96, 192, 384, 768]
    assert [feature.shape[-2:] for feature in features] == [
        (56, 56), (28, 28), (14, 14), (7, 7),
    ]
    output = encoder.projection(encoder.neck(features))
    assert output.shape == (1, 6, 56, 56)
    output.float().square().mean().backward()

    for stage_index in range(4):
        stage_parameters = [
            parameter
            for name, parameter in encoder.backbone.named_parameters()
            if (
                f"swin.encoder.layers.{stage_index}." in name
                or f"encoder.layers.{stage_index}." in name
            )
            and parameter.requires_grad
        ]
        assert stage_parameters
        assert all(parameter.grad is not None for parameter in stage_parameters)
    missing = [
        name for name, parameter in encoder.backbone.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []
    if hasattr(encoder.backbone, "swin"):
        assert not encoder.backbone.swin.layernorm.weight.requires_grad
        assert not encoder.backbone.swin.layernorm.bias.requires_grad
    else:
        assert not hasattr(encoder.backbone, "layernorm")


def test_real_scratch_convnext_has_no_trainable_unused_parameters():
    encoder = TransformerImageEncoder(
        "convnext", "tiny", False, False, "ddp_fpn_merge", 8, 6, 4, True
    )
    output = encoder(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 6, 16, 16)
    output.float().square().mean().backward()
    missing = [
        name for name, parameter in encoder.backbone.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []
    assert not encoder.backbone.layernorm.weight.requires_grad
    assert not encoder.backbone.layernorm.bias.requires_grad


def test_ddp_fpn_merge_uses_highest_resolution():
    neck = DDPFPNMultiStageMerging([4, 8, 16, 32], channels=5)
    result = neck([
        torch.randn(2, 4, 9, 13),
        torch.randn(2, 8, 5, 7),
        torch.randn(2, 16, 3, 4),
        torch.randn(2, 32, 2, 2),
    ])
    assert result.shape == (2, 5, 9, 13)


def test_ddp_fpn_merge_uses_adaptive_group_norm_without_activation():
    neck = DDPFPNMultiStageMerging([8, 16, 32, 64], channels=256)
    normalized_blocks = [*neck.lateral, *neck.fpn_output, neck.merge]
    for block in normalized_blocks:
        assert isinstance(block, nn.Sequential)
        assert len(block) == 2
        assert isinstance(block[0], nn.Conv2d)
        assert isinstance(block[1], nn.GroupNorm)
        assert block[1].num_groups == 32

    small = DDPFPNMultiStageMerging([4, 8, 16, 32], channels=6)
    small_norms = [
        module for module in small.modules() if isinstance(module, nn.GroupNorm)
    ]
    assert len(small_norms) == 9
    assert all(norm.num_groups == 6 for norm in small_norms)
    result = small([
        torch.randn(1, 4, 9, 13),
        torch.randn(1, 8, 5, 7),
        torch.randn(1, 16, 3, 4),
        torch.randn(1, 32, 2, 2),
    ])
    assert result.shape == (1, 6, 9, 13)


@pytest.mark.parametrize(
    "classes,void_index,semantic_slice",
    [(20, 19, slice(0, 19)), (151, 0, slice(1, 151))],
)
def test_task_source_maps_semantics_and_outputs_state_resolution(
    classes, void_index, semantic_slice
):
    source = _task_source(classes, void_index)
    x0, mu, logvar = source(torch.randn(1, 3, 32, 48))
    assert x0.shape == mu.shape == logvar.shape == (1, classes, 8, 12)
    assert torch.count_nonzero(mu[:, void_index]) == 0
    probabilities = mu[:, semantic_slice]
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(
        probabilities.sum(dim=1), torch.ones_like(probabilities[:, 0]),
        rtol=1e-5, atol=1e-6,
    )


def test_frozen_task_source_stays_eval_and_skips_alignment(monkeypatch):
    source = _task_source(20, 19, freeze=True)
    source.train(True)
    assert not source.training
    assert not any(parameter.requires_grad for parameter in source.parameters())

    def alignment_must_not_run(*args, **kwargs):
        raise AssertionError("alignment loss was evaluated")

    monkeypatch.setattr(
        discrete_flow_maps, "source_alignment_map_from_indices", alignment_must_not_run
    )
    config = {
        "model": {"state_downsample_factor": 4},
        "dataset": {"num_classes": 20},
        "loss": {"ignore_index": None},
        "source": {
            "type": "task_finetuned_segformer",
            "prior_type": "image_gaussian",
            "freeze": True,
            "fixed_std": 1.0,
            "var_weight": 0.0,
            "align_weight": 0.15,
            "align_eps": 1e-8,
            "supervision": {"type": "none", "weight": 0.0},
        },
    }
    image = torch.randn(1, 3, 32, 48)
    target = torch.randint(0, 20, (1, 32, 48))
    one_hot = F.one_hot(target[:, ::4, ::4], 20).permute(0, 3, 1, 2).float()
    _, stats = sample_prior(
        config, image, one_hot, source, target_full=target
    )
    assert stats["loss_source_align"].item() == 0.0
    assert stats["weighted_align"].item() == 0.0
    assert not stats["loss_source_align"].requires_grad


def test_frozen_task_source_adds_no_supervision_to_training_objective():
    config = load_config(ADE_CONFIG)
    config = deepcopy(config)
    config["model"] = _tiny_model_config(config)
    config["source"].update({
        "type": "task_finetuned_segformer",
        "model_id": "unit-test/no-download",
        "freeze": True,
        "supervision": {"type": "none", "weight": 0.0},
    })
    endpoint = DiscreteFlowMapModel(config["model"])
    source = _task_source(151, 0, freeze=True)
    adapter = DDPCompatibleTrainingModel(endpoint, source, config)
    result = compute_model_training_objectives(
        adapter,
        operation="stage1_objectives",
        image=torch.randn(1, 3, 32, 32),
        target=torch.randint(0, 151, (1, 32, 32)),
        epoch_index=0,
        progress_in_epoch=0.0,
    )
    stats = result["stats"]
    assert stats["loss_source_align"].item() == 0.0
    assert stats["weighted_source_supervision"].item() == 0.0
    torch.testing.assert_close(
        result["loss"].detach(),
        config["loss"]["primary"]["weight"] * stats["loss_diagonal"],
    )


def test_source_factory_keeps_legacy_default_and_selects_task(monkeypatch):
    config = load_config(ADE_CONFIG)
    sentinel = nn.Conv2d(1, 1, 1)
    import source_model as source_module
    monkeypatch.setattr(source_module, "SegFormerSourceGenerator", lambda *args: sentinel)
    assert build_source_model(config) is sentinel

    task_config = deepcopy(config)
    task_config["source"].update({
        "type": "task_finetuned_segformer",
        "model_id": "unit-test/no-download",
        "freeze": True,
        "supervision": {"type": "none", "weight": 0.0},
    })
    task = _task_source(151, 0, freeze=True)
    monkeypatch.setattr(
        source_module, "TaskFinetunedSegFormerSourceGenerator", lambda *args, **kwargs: task
    )
    assert build_source_model(task_config) is task


def test_optimizer_excludes_frozen_source_and_backbone(monkeypatch):
    monkeypatch.setattr(
        model_module,
        "load_transformer_image_backbone",
        lambda *args: (FakeBackbone(), [4, 8, 16, 32]),
    )
    config = load_config(ADE_CONFIG, [
        "model.image_encoder.type=swin",
        "model.image_encoder.pretrained=false",
        "model.image_encoder.freeze=true",
        "model.image_encoder.neck.type=ddp_fpn_merge",
        "model.image_encoder.neck.channels=6",
    ])
    endpoint = DiscreteFlowMapModel(_tiny_model_config(config))
    source = _task_source(151, 0, freeze=True)
    config["source"]["freeze"] = True
    adapter = DDPCompatibleTrainingModel(endpoint, source, config)
    optimizer = build_optimizer(config, adapter)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert all(
        id(parameter) not in optimized for parameter in endpoint.image_encoder.backbone.parameters()
    )
    assert all(id(parameter) in optimized for parameter in endpoint.image_encoder.neck.parameters())
    assert all(id(parameter) not in optimized for parameter in source.parameters())


@pytest.mark.parametrize("encoder_type", ["swin", "convnext"])
def test_new_encoder_checkpoint_and_parameter_report_round_trip(
    monkeypatch, tmp_path, encoder_type
):
    monkeypatch.setattr(
        model_module,
        "load_transformer_image_backbone",
        lambda *args: (FakeBackbone(), [4, 8, 16, 32]),
    )
    config = load_config(ADE_CONFIG, [
        f"model.image_encoder.type={encoder_type}",
        "model.image_encoder.pretrained=false",
        "model.image_encoder.neck.type=ddp_fpn_merge",
        "model.image_encoder.neck.channels=6",
    ])
    first = DiscreteFlowMapModel(_tiny_model_config(config))
    second = DiscreteFlowMapModel(_tiny_model_config(config))
    checkpoint_path = tmp_path / f"{encoder_type}.pt"
    torch.save({"model": first.state_dict()}, checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    second.load_state_dict(saved["model"], strict=True)
    report = inspect_model_parameters(first, None)
    assert report["total_parameters"] == report["endpoint"]["total"]
    assert report["backbone"]["total"] > 0
    assert report["neck"]["total"] > 0
    assert model_signature(config)["model"]["image_encoder"]["type"] == encoder_type


def test_legacy_model_signature_omits_new_rrdb_default_block():
    config = load_config(ADE_CONFIG)
    current_signature = model_signature(config)
    assert "image_encoder" not in current_signature["model"]
    old_resolved_config = deepcopy(config)
    old_resolved_config["model"].pop("image_encoder")
    for key in ("type", "model_id", "representation", "void_channel_value"):
        old_resolved_config["source"].pop(key)
    assert model_signature(old_resolved_config) == current_signature


@pytest.mark.parametrize(
    "name",
    [
        "joint_psd_ade20k_swin_t.yaml",
        "joint_psd_ade20k_frozen_source.yaml",
        "joint_psd_ade20k_swin_t_frozen_source.yaml",
    ],
)
def test_ade_examples_extend_base_and_preserve_training_protocol(name):
    raw = yaml.safe_load((ROOT / "configs" / name).read_text())
    assert raw["extends"] == "joint_psd_ade20k.yaml"
    assert set(raw) <= {"extends", "model", "source"}
    base = load_config(ADE_CONFIG)
    example = load_config(ROOT / "configs" / name)
    assert example["training"] == base["training"]
    assert example["loss"] == base["loss"]
    assert example["evaluation"] == base["evaluation"]
    assert example["augmentation"] == base["augmentation"]
    assert example["model"]["state_downsample_factor"] == 4
