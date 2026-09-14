from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import inference
from config import load_config
from dataset import LIDCDataset, build_dataset
from inference import sample_segmentation
from model_factory import build_models
from state_space import prepare_state_targets, state_spatial_size
from training_objectives import (
    DDPCompatibleTrainingModel,
    compute_model_training_objectives,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs/lidc/diagonal/standard.yaml"
REAL_CACHE = Path("/home/igarashi_25/datasets/LIDC/LIDC_data/lidc_memmap_v1")


def _synthetic_lidc(tmp_path: Path, monkeypatch) -> tuple[dict, Path, Path]:
    monkeypatch.setattr(
        LIDCDataset, "EXPECTED_SAMPLE_COUNTS", {"train": 1, "val": 1, "test": 1}
    )
    monkeypatch.setattr(
        LIDCDataset, "EXPECTED_SERIES_COUNTS", {"train": 1, "val": 1, "test": 1}
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    images = np.zeros((3, 128, 128), dtype=np.float32)
    images[:, 10:20, 30:40] = 1.0
    masks = np.zeros((3, 4, 128, 128), dtype=np.uint8)
    masks[:, :, 10:20, 30:40] = 1
    np.save(cache / "images.npy", images)
    np.save(cache / "masks.npy", masks)
    (cache / "metadata.json").write_text(json.dumps({
        "sample_ids": ["a", "b", "c"],
        "series_uids": ["series-a", "series-b", "series-c"],
    }))
    (cache / "manifest.json").write_text(json.dumps({
        "image_shape": [3, 128, 128], "mask_shape": [3, 4, 128, 128],
    }))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({
        "seed": 42,
        "samples": {"train": ["a"], "val": ["b"], "test": ["c"]},
        "series": {
            "train": ["series-a"], "val": ["series-b"], "test": ["series-c"],
        },
    }))
    config = load_config(CONFIG_PATH)
    config["dataset"]["cache_dir"] = str(cache)
    config["dataset"]["split_path"] = str(split)
    return config, cache, split


def test_lidc_train_and_eval_contracts(tmp_path, monkeypatch):
    config, cache, split = _synthetic_lidc(tmp_path, monkeypatch)
    train = LIDCDataset(str(cache), str(split), "train", config, True)
    image, target = train[0]
    assert image.shape == (1, 128, 128)
    assert target.shape == (128, 128)
    assert set(target.unique().tolist()) <= {0, 1}
    assert image.min() >= -1 and image.max() <= 1
    assert torch.equal(image[0] > 0, target.bool())

    for split_name in ("val", "test"):
        sample = LIDCDataset(str(cache), str(split), split_name, config, False)[0]
        assert sample["image"].shape == (1, 128, 128)
        assert sample["masks"].shape == (4, 128, 128)
        assert set(sample["masks"].unique().tolist()) <= {0, 1}
        assert torch.equal(sample["target"], sample["masks"][0])


def test_lidc_factory_and_one_channel_models(tmp_path, monkeypatch):
    config, _, _ = _synthetic_lidc(tmp_path, monkeypatch)
    dataset = build_dataset(config, "train", augment=True)
    assert isinstance(dataset, LIDCDataset)
    endpoint, source = build_models(config, torch.device("cpu"))
    assert endpoint.image_encoder.first.in_channels == 1
    assert source.network[0].in_channels == 1


def test_lidc_resolved_config_is_independent_and_full_resolution():
    config = load_config(CONFIG_PATH)
    assert config["dataset"]["name"] == "lidc"
    assert config["dataset"]["protocol"] == "lidc"
    assert config["dataset"]["num_classes"] == 2
    assert config["dataset"]["in_channels"] == 1
    assert config["dataset"]["image_size"] == [128, 128]
    assert config["model"]["num_classes"] == 2
    assert config["model"]["state_downsample_factor"] == 1
    assert config["training"]["batch_size"] == 16


def test_lidc_full_resolution_training_and_sampling(tmp_path, monkeypatch):
    config, cache, split = _synthetic_lidc(tmp_path, monkeypatch)
    config["model"]["fusion_channels"] = 8
    config["model"]["rrdb_blocks"] = 0
    config["model"]["rrdb_growth_channels"] = 4
    config["model"]["unet"].update({
        "base_channels": 8,
        "channel_mults": [1, 2],
        "num_res_blocks": 1,
        "attention_levels": [],
        "num_heads": 1,
        "time_embedding_dim": 16,
    })
    config["source"]["decoder_channels"] = 8
    image, target, spatial_valid = LIDCDataset(
        str(cache), str(split), "train", config, True,
        return_spatial_valid_mask=True,
    )[0]
    image, target, spatial_valid = (
        image[None], target[None], spatial_valid[None]
    )
    state_size = state_spatial_size(
        image, config["model"]["state_downsample_factor"]
    )
    targets = prepare_state_targets(
        target,
        num_classes=2,
        state_size=state_size,
        ignore_index=None,
        mask_pixel_losses=False,
        spatial_valid_mask_full=spatial_valid,
    )
    assert state_size == (128, 128)
    assert targets.target_state.shape == (1, 128, 128)
    assert targets.one_hot_state.shape == (1, 2, 128, 128)
    assert targets.target_state.data_ptr() == target.data_ptr()

    endpoint, source = build_models(config, torch.device("cpu"))
    assert endpoint.state_downsample_factor == 1
    assert source.state_downsample_factor == 1
    with torch.no_grad():
        x0, mu, logvar = source(image)
        logits, probability = endpoint(
            x0, image, torch.tensor([0.25]), torch.tensor([0.75])
        )
    expected_state = (1, 2, 128, 128)
    assert x0.shape == mu.shape == logvar.shape == expected_state
    assert logits.shape == probability.shape == expected_state

    adapter = DDPCompatibleTrainingModel(endpoint, source, config)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1.0e-4)
    result = compute_model_training_objectives(
        adapter,
        operation="stage1_objectives",
        image=image,
        target=target,
        spatial_valid_mask=spatial_valid,
        epoch_index=0,
        progress_in_epoch=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    result["loss"].backward()
    optimizer.step()
    assert torch.isfinite(result["loss"])
    assert int(result["stats"]["state_height"]) == 128
    assert int(result["stats"]["state_width"]) == 128

    terminal = sample_segmentation(
        endpoint, source, image, config, num_steps=1,
        return_terminal_state=True,
    )
    monkeypatch.setattr(
        inference,
        "resize_continuous",
        lambda *_args, **_kwargs: pytest.fail(
            "LIDC prediction must not use state-to-image upsampling"
        ),
    )
    prediction = sample_segmentation(
        endpoint, source, image, config, num_steps=1,
    )
    assert terminal.shape == expected_state
    assert prediction.shape == (1, 128, 128)


def test_lidc_fixed_split_counts_and_disjoint_series():
    split = LIDCDataset._load_and_validate_split(Path(
        "/home/igarashi_25/datasets/LIDC/LIDC_data/lidc_split_seed42.json"
    ))
    assert {name: len(values) for name, values in split["samples"].items()} == {
        "train": 9348, "val": 2748, "test": 3000,
    }
    assert {name: len(values) for name, values in split["series"].items()} == {
        "train": 560, "val": 140, "test": 175,
    }
    series = {name: set(values) for name, values in split["series"].items()}
    assert series["train"].isdisjoint(series["val"])
    assert series["train"].isdisjoint(series["test"])
    assert series["val"].isdisjoint(series["test"])


@pytest.mark.skipif(
    not (REAL_CACHE / "manifest.json").is_file(), reason="LIDC cache not built"
)
def test_real_lidc_cache_contracts():
    config = load_config(CONFIG_PATH)
    for split, count in (("train", 9348), ("val", 2748), ("test", 3000)):
        dataset = build_dataset(config, split, augment=split == "train")
        assert len(dataset) == count
        sample = dataset[0]
        image = sample[0] if split == "train" else sample["image"]
        assert image.shape == (1, 128, 128)
