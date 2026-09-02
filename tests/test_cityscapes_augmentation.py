from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from config import load_config, validate_config
from dataset import (
    Cityscapes20ClassDataset,
    _crop_has_acceptable_class_ratio,
    _random_resize_pair,
)
from source_model import SegFormerSourceGenerator
from state_space import prepare_state_targets, state_spatial_size


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "_base_" / "cityscapes" / "joint_psd_160k.yaml"


def _config() -> dict:
    return load_config(CONFIG_PATH)


def _synthetic_dataset(config: dict) -> Cityscapes20ClassDataset:
    dataset = object.__new__(Cityscapes20ClassDataset)
    dataset.config = config
    dataset.split = "train"
    dataset.augment = True
    dataset.photo_distortion = lambda image: image
    dataset.jitter = lambda image: image
    dataset.dataset = SimpleNamespace(images=["sample_leftImg8bit.png"])
    return dataset


def test_cityscapes_main_augmentation_protocol_config():
    config = _config()
    assert config["dataset"]["image_size"] == [512, 1024]
    assert config["dataset"]["num_classes"] == 20
    assert config["dataset"]["eval_num_classes"] == 19
    assert config["dataset"]["void_class_index"] == 19
    assert config["training"]["batch_size"] == 16
    assert config["augmentation"]["random_crop"] == {
        "enabled": True,
        "size": [512, 1024],
        "cat_max_ratio": 0.75,
        "ignore_index": 19,
        "max_attempts": 10,
    }
    assert config["augmentation"]["pad"]["mask_value"] == 19
    assert config["augmentation"]["normalize"]["enabled"] is True
    assert config["augmentation"]["imagenet_normalize"] is False
    assert config["source"]["input_already_normalized"] is True
    assert config["evaluation"]["original_resolution"] is True


def test_cityscapes_mapping_remains_20_state_protocol():
    raw = np.array([[7, 26, 33, 0, 255]], dtype=np.uint8)
    mapped = Cityscapes20ClassDataset._map_target(raw)
    assert mapped.tolist() == [[0, 13, 18, 19, 19]]
    assert mapped.dtype == torch.long


def test_random_resize_uses_nearest_for_mask():
    image = torch.rand(3, 4, 8)
    mask = torch.tensor([
        [0, 0, 0, 0, 13, 13, 13, 13],
        [0, 0, 0, 0, 13, 13, 13, 13],
        [18, 18, 18, 18, 19, 19, 19, 19],
        [18, 18, 18, 18, 19, 19, 19, 19],
    ])
    _, resized = _random_resize_pair(image, mask, {
        "enabled": True,
        "base_scale": {"width": 16, "height": 8},
        "ratio_range": [1.0, 1.0],
        "keep_ratio": True,
    })
    assert resized.shape == (8, 16)
    assert set(resized.unique().tolist()) == {0, 13, 18, 19}


def test_cat_max_ratio_excludes_void_19():
    candidate = torch.full((10, 10), 19, dtype=torch.long)
    candidate[0, :5] = 0
    candidate[0, 5:] = 1
    assert _crop_has_acceptable_class_ratio(
        candidate, ignore_index=19, cat_max_ratio=0.75
    )


def test_train_pipeline_pads_before_crop_and_returns_exact_shape_and_dtype():
    config = _config()
    config["dataset"]["image_size"] = [32, 64]
    config["augmentation"]["random_resize"]["enabled"] = False
    config["augmentation"]["random_crop"].update({"size": [32, 64]})
    config["augmentation"]["horizontal_flip"]["enabled"] = False
    config["augmentation"]["photometric_distortion"]["enabled"] = False
    config["augmentation"]["pad"].update({"size": [32, 64]})
    dataset = _synthetic_dataset(config)
    image = torch.rand(3, 16, 32)
    mask = torch.randint(0, 20, (16, 32), dtype=torch.long)

    transformed_image, transformed_mask = dataset._train_item(image, mask)

    assert transformed_image.shape == (3, 32, 64)
    assert transformed_mask.shape == (32, 64)
    assert transformed_image.dtype == torch.float32
    assert transformed_mask.dtype == torch.long
    assert bool((transformed_mask == 19).any())


def test_train_and_original_resolution_validation_are_separate():
    config = _config()
    config["dataset"]["image_size"] = [32, 64]
    config["augmentation"]["random_resize"].update({
        "base_scale": {"width": 64, "height": 32},
        "ratio_range": [1.0, 1.0],
    })
    config["augmentation"]["random_crop"]["size"] = [32, 64]
    config["augmentation"]["horizontal_flip"]["enabled"] = False
    config["augmentation"]["photometric_distortion"]["enabled"] = False
    config["augmentation"]["pad"]["size"] = [32, 64]
    config["evaluation"]["resize"] = {
        "width": 128, "height": 64, "keep_ratio": True,
    }
    dataset = _synthetic_dataset(config)
    image = torch.rand(3, 64, 128)
    mask = torch.randint(0, 20, (64, 128), dtype=torch.long)

    train_image, train_mask = dataset._train_item(image, mask)
    validation = dataset._validation_item(image, mask, 0)

    assert train_image.shape == (3, 32, 64)
    assert train_mask.shape == (32, 64)
    assert validation["image"].shape == (3, 64, 128)
    assert validation["target"].shape == (64, 128)
    assert validation["model_shape"] == (64, 128)


def test_512x1024_target_creates_only_quarter_resolution_one_hot():
    image = torch.zeros(1, 3, 512, 1024)
    target = torch.randint(0, 20, (1, 512, 1024))
    state_size = state_spatial_size(image, 4)
    targets = prepare_state_targets(
        target, num_classes=20, state_size=state_size,
        ignore_index=None, mask_pixel_losses=False,
    )
    assert state_size == (128, 256)
    assert targets.target_full.shape == (1, 512, 1024)
    assert targets.target_state.shape == (1, 128, 256)
    assert targets.one_hot_state.shape == (1, 20, 128, 256)
    assert set(targets.target_state.unique().tolist()) <= set(range(20))


class _RecordingEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen = None

    def forward(self, pixel_values, **kwargs):
        del kwargs
        self.seen = pixel_values.detach().clone()
        return SimpleNamespace(hidden_states=(
            pixel_values, pixel_values, pixel_values, pixel_values,
        ))


def test_pre_normalized_dataset_input_is_not_normalized_again_by_segformer():
    source = object.__new__(SegFormerSourceGenerator)
    nn.Module.__init__(source)
    source.encoder = _RecordingEncoder()
    source.projections = nn.ModuleList([nn.Identity() for _ in range(4)])
    source.decoder = nn.Conv2d(12, 20, 1)
    source.fixed_std = 1.0
    source.mu_tanh_scale = 0.0
    source.input_already_normalized = True
    source.state_downsample_factor = 4
    source.register_buffer(
        "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    )
    source.register_buffer(
        "std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    )
    normalized = torch.randn(1, 3, 8, 12)

    _, mu, _ = source(normalized)

    assert torch.equal(source.encoder.seen, normalized)
    assert mu.shape == (1, 20, 2, 3)


def test_cityscapes_normalization_validation_rejects_double_or_missing_contract():
    config = _config()
    invalid = copy.deepcopy(config)
    invalid["source"]["input_already_normalized"] = False
    with pytest.raises(ValueError, match="input_already_normalized"):
        validate_config(invalid)
    invalid = copy.deepcopy(config)
    invalid["augmentation"]["imagenet_normalize"] = True
    with pytest.raises(ValueError, match="must not both"):
        validate_config(invalid)
