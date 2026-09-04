from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from checkpoint import _validate_stage2_init_checkpoint, model_signature
from config import load_config, validate_config
from dataset import Cityscapes20ClassDataset, _random_crop_triplet
from losses import masked_mean
from state_space import prepare_state_targets
from trainer import _unpack_training_batch
from training_objectives import build_psd_valid_masks


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "cityscapes" / "psd" / "simplex_source_rank_128k.yaml"


def _targets():
    # Pixel 0 is semantic, pixel 1 is real void, and pixel 2 is padding.
    target = torch.tensor([[[3, 19, 19]]])
    spatial = torch.tensor([[[True, True, False]]])
    return prepare_state_targets(
        target,
        num_classes=20,
        state_size=(1, 3),
        ignore_index=19,
        mask_pixel_losses=True,
        spatial_valid_mask_full=spatial,
    )


def _masked_cross_entropy(student_logits, teacher, mask):
    loss_map = -(teacher * F.log_softmax(student_logits, dim=1)).sum(dim=1)
    return masked_mean(loss_map, mask)


def test_psd_masks_distinguish_semantic_real_void_and_padding():
    targets = _targets()
    ignored = build_psd_valid_masks(
        targets, void_class_index=19, ignore_void=True
    )
    included = build_psd_valid_masks(
        targets, void_class_index=19, ignore_void=False
    )

    assert ignored.psd_full.tolist() == [[[True, False, False]]]
    assert included.psd_full.tolist() == [[[True, True, False]]]
    assert ignored.spatial_full.tolist() == [[[True, True, False]]]
    assert ignored.semantic_full.tolist() == [[[True, False, False]]]


def test_psd_numerical_reduction_excludes_padding_in_both_modes():
    targets = _targets()
    loss_map = torch.tensor([[[-1.0, -3.0, -100.0]]]).neg()
    ignored = build_psd_valid_masks(
        targets, void_class_index=19, ignore_void=True
    )
    included = build_psd_valid_masks(
        targets, void_class_index=19, ignore_void=False
    )

    assert masked_mean(loss_map, ignored.psd_full) == 1.0
    assert masked_mean(loss_map, included.psd_full) == 2.0


@pytest.mark.parametrize("ignore_void", [True, False])
def test_padding_logits_do_not_change_psd_or_receive_gradient(ignore_void):
    masks = build_psd_valid_masks(
        _targets(), void_class_index=19, ignore_void=ignore_void
    )
    teacher = torch.tensor([[[[1.0, 0.0, 0.0]], [[0.0, 1.0, 1.0]]]])
    logits = torch.zeros(1, 2, 1, 3, requires_grad=True)
    reference = _masked_cross_entropy(logits, teacher, masks.psd_full)
    changed = logits.detach().clone()
    changed[0, :, 0, 2] = torch.tensor([100.0, -100.0])
    changed.requires_grad_(True)
    perturbed = _masked_cross_entropy(changed, teacher, masks.psd_full)

    torch.testing.assert_close(perturbed, reference)
    perturbed.backward()
    assert not bool((changed.grad[..., 2] != 0).any())


def test_real_void_logits_only_affect_include_void_psd():
    targets = _targets()
    teacher = torch.tensor([[[[1.0, 0.0, 0.0]], [[0.0, 1.0, 1.0]]]])
    baseline = torch.zeros(1, 2, 1, 3)
    changed = baseline.clone()
    changed[0, :, 0, 1] = torch.tensor([100.0, -100.0])
    ignored = build_psd_valid_masks(
        targets, void_class_index=19, ignore_void=True
    )
    included = build_psd_valid_masks(
        targets, void_class_index=19, ignore_void=False
    )

    torch.testing.assert_close(
        _masked_cross_entropy(baseline, teacher, ignored.psd_full),
        _masked_cross_entropy(changed, teacher, ignored.psd_full),
    )
    assert not torch.isclose(
        _masked_cross_entropy(baseline, teacher, included.psd_full),
        _masked_cross_entropy(changed, teacher, included.psd_full),
    )


def test_safety_padding_marks_only_artificial_pixels_invalid():
    image = torch.zeros(3, 2, 3)
    target = torch.tensor([[19, 1, 1], [2, 2, 2]])
    spatial = torch.ones_like(target, dtype=torch.bool)
    _, padded_target, padded_spatial = _random_crop_triplet(
        image,
        target,
        spatial,
        {
            "enabled": True,
            "size": [4, 5],
            "cat_max_ratio": 1.01,
            "ignore_index": 19,
            "max_attempts": 1,
        },
        ensure_crop_size=True,
        mask_pad_value=19,
    )

    assert padded_target.shape == padded_spatial.shape == (4, 5)
    assert padded_spatial[:2, :3].all()
    assert padded_spatial[0, 0]  # Real void stays spatially valid.
    assert not padded_spatial[2:, :].any()
    assert not padded_spatial[:, 3:].any()
    assert (padded_target[~padded_spatial] == 19).all()


def test_full_train_geometry_and_rng_match_legacy_pair():
    config = load_config(CONFIG)
    config["dataset"]["image_size"] = [4, 6]
    config["augmentation"]["random_resize"].update({
        "enabled": True,
        "base_scale": {"width": 3, "height": 2},
        "ratio_range": [1.0, 1.0],
    })
    config["augmentation"]["random_crop"]["enabled"] = False
    config["augmentation"]["horizontal_flip"].update({
        "enabled": True, "probability": 1.0,
    })
    config["augmentation"]["photometric_distortion"]["enabled"] = False
    config["augmentation"]["normalize"]["enabled"] = False
    config["augmentation"]["pad"].update({
        "enabled": True, "size": [4, 6], "image_value": 0.0, "mask_value": 19,
    })
    dataset = object.__new__(Cityscapes20ClassDataset)
    dataset.config = config
    dataset.photo_distortion = lambda image: image
    dataset.jitter = lambda image: image
    image = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
    target = torch.tensor([[19, 1, 2], [3, 4, 5]])

    torch.manual_seed(123)
    pair_image, pair_target = dataset._train_item(image, target)
    torch.manual_seed(123)
    triplet_image, triplet_target, spatial = (
        dataset._train_item_with_spatial_mask(image, target)
    )

    torch.testing.assert_close(triplet_image, pair_image)
    torch.testing.assert_close(triplet_target, pair_target)
    assert spatial.shape == triplet_target.shape == (4, 6)
    assert spatial[:2, :3].all()
    assert spatial[0, 2]  # Flipped real void remains valid.
    assert not spatial[2:, :].any()
    assert not spatial[:, 3:].any()


def test_spatial_mask_nearest_resize_to_state_resolution():
    target = torch.zeros(1, 4, 6, dtype=torch.long)
    spatial = torch.tensor([[
        [True, True, True, False, False, False],
        [True, True, True, False, False, False],
        [False, False, False, False, False, False],
        [False, False, False, False, False, False],
    ]])
    targets = prepare_state_targets(
        target,
        num_classes=20,
        state_size=(2, 3),
        ignore_index=19,
        mask_pixel_losses=True,
        spatial_valid_mask_full=spatial,
    )
    expected = F.interpolate(
        spatial[:, None].float(), size=(2, 3), mode="nearest"
    )[:, 0].bool()
    torch.testing.assert_close(targets.spatial_valid_mask_state, expected)


def test_ignore_void_true_matches_legacy_mask_without_padding():
    target = torch.tensor([[[1, 19], [2, 3]]])
    spatial = torch.ones_like(target, dtype=torch.bool)
    targets = prepare_state_targets(
        target,
        num_classes=20,
        state_size=(2, 2),
        ignore_index=19,
        mask_pixel_losses=True,
        spatial_valid_mask_full=spatial,
    )
    masks = build_psd_valid_masks(
        targets, void_class_index=19, ignore_void=True
    )
    torch.testing.assert_close(masks.psd_full, targets.valid_mask_full)
    torch.testing.assert_close(masks.psd_state, targets.valid_mask_state)


def test_config_default_cli_overrides_and_boolean_validation():
    default = load_config(CONFIG)
    ignored = load_config(CONFIG, ["loss.consistency.psd.ignore_void=true"])
    included = load_config(CONFIG, ["loss.consistency.psd.ignore_void=false"])
    assert default["loss"]["consistency"]["psd"]["ignore_void"] is True
    assert ignored["loss"]["consistency"]["psd"]["ignore_void"] is True
    assert included["loss"]["consistency"]["psd"]["ignore_void"] is False

    for invalid in (0, "false"):
        config = deepcopy(default)
        config["loss"]["consistency"]["psd"]["ignore_void"] = invalid
        with pytest.raises(ValueError, match="ignore_void must be a boolean"):
            validate_config(config)


def test_include_void_requires_explicit_spatial_mask():
    target = torch.tensor([[[1, 19]]])
    targets = prepare_state_targets(
        target,
        num_classes=20,
        state_size=(1, 2),
        ignore_index=19,
        mask_pixel_losses=True,
    )
    with pytest.raises(ValueError, match="explicit spatial valid mask"):
        build_psd_valid_masks(
            targets, void_class_index=19, ignore_void=False
        )


def test_legacy_pair_falls_back_to_all_valid_when_void_is_ignored():
    config = load_config(CONFIG, ["loss.consistency.psd.ignore_void=true"])
    image = torch.zeros(2, 3, 4, 5)
    target = torch.full((2, 4, 5), 19)
    unpacked_image, unpacked_target, spatial = _unpack_training_batch(
        (image, target), config
    )
    assert unpacked_image is image
    assert unpacked_target is target
    assert spatial.dtype == torch.bool
    assert spatial.all()


def test_legacy_pair_is_rejected_when_real_void_is_included():
    config = load_config(CONFIG, ["loss.consistency.psd.ignore_void=false"])
    batch = (torch.zeros(1, 3, 2, 3), torch.full((1, 2, 3), 19))
    with pytest.raises(
        ValueError,
        match="ignore_void=false requires an explicit spatial_valid_mask",
    ):
        _unpack_training_batch(batch, config)


def test_explicit_spatial_mask_is_accepted_when_real_void_is_included():
    config = load_config(CONFIG, ["loss.consistency.psd.ignore_void=false"])
    image = torch.zeros(1, 3, 2, 3)
    target = torch.tensor([[[1, 19, 19], [2, 3, 19]]])
    spatial = torch.tensor([[[True, True, False], [True, True, False]]])
    unpacked = _unpack_training_batch((image, target, spatial), config)
    assert unpacked[0] is image
    assert unpacked[1] is target
    assert unpacked[2] is spatial

    targets = prepare_state_targets(
        unpacked[1], num_classes=20, state_size=(2, 3),
        ignore_index=19, mask_pixel_losses=True,
        spatial_valid_mask_full=unpacked[2],
    )
    masks = build_psd_valid_masks(
        targets, void_class_index=19, ignore_void=False
    )
    torch.testing.assert_close(masks.psd_full, spatial)


def test_production_cityscapes_train_item_returns_explicit_spatial_mask():
    config = load_config(
        ROOT / "configs" / "cityscapes" / "psd"
        / "joint_simplex_b1_ce_include_void_160k.yaml"
    )
    assert config["loss"]["consistency"]["psd"]["ignore_void"] is False
    config["dataset"]["image_size"] = [4, 8]
    config["augmentation"]["random_resize"]["enabled"] = False
    config["augmentation"]["random_crop"]["enabled"] = False
    config["augmentation"]["horizontal_flip"]["enabled"] = False
    config["augmentation"]["photometric_distortion"]["enabled"] = False
    config["augmentation"]["normalize"]["enabled"] = False
    config["augmentation"]["pad"].update({
        "enabled": True, "size": [4, 8], "image_value": 0.0, "mask_value": 19,
    })

    class _RawDataset:
        images = ["sample_leftImg8bit.png"]

        def __getitem__(self, index):
            del index
            image = Image.fromarray(np.zeros((2, 4, 3), dtype=np.uint8))
            raw_target = np.array([[7, 7, 0, 0], [7, 7, 0, 0]], dtype=np.uint8)
            return image, Image.fromarray(raw_target)

    dataset = object.__new__(Cityscapes20ClassDataset)
    dataset.config = config
    dataset.split = "train"
    dataset.augment = True
    dataset.return_spatial_valid_mask = True
    dataset.photo_distortion = lambda image: image
    dataset.jitter = lambda image: image
    dataset.dataset = _RawDataset()

    image, target, spatial = dataset[0]
    assert image.shape == (3, 4, 8)
    assert target.shape == spatial.shape == (4, 8)
    assert spatial[:2, :4].all()
    assert not spatial[2:, :].any()
    assert not spatial[:, 4:].any()
    assert target[0, 2] == 19 and spatial[0, 2]  # Real void.
    assert target[3, 7] == 19 and not spatial[3, 7]  # Padding.


def test_simplex_stage2_resolves_to_b1_best_checkpoint_and_is_compatible():
    stage1 = load_config(
        ROOT / "configs" / "cityscapes" / "diagonal"
        / "source_segformer_b1_32k.yaml"
    )
    stage2 = load_config(CONFIG)
    expected = (
        "/home/igarashi_25/DFM/results/cityscapes/"
        "source_segformer_b1_32k/best.pt"
    )
    assert stage2["source"]["type"] == "trainable_segformer"
    assert stage2["source"]["segformer_variant"] == "b1"
    assert stage2["source"]["prior_type"] == "image_simplex_mixture"
    assert stage2["source"]["freeze"] is True
    assert stage2["checkpoint"]["init_from"] == expected
    assert stage2["source"]["supervision"] == {
        "type": "cross_entropy", "weight": 1.0, "include_void": True,
    }
    assert "segformer-b1" in stage2["wandb"]["tags"]
    assert "segformer-b1" in stage1["wandb"]["tags"]

    checkpoint = {
        "stage": "diagonal_pretrain",
        "model": {},
        "source_model": {},
        "model_signature": model_signature(stage1),
    }
    _validate_stage2_init_checkpoint(checkpoint, stage2, expected)
