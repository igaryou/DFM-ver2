from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from config import load_config
from dataset import LIDCDataset, build_dataset
from model_factory import build_models


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
