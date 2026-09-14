#!/usr/bin/env python3
"""Convert the monolithic LIDC pickle into worker-safe NumPy memmaps."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def convert(pickle_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Completed cache already exists: {output_dir}. "
            "Remove it explicitly before rebuilding."
        )
    with pickle_path.open("rb") as handle:
        samples = pickle.load(handle)
    if not isinstance(samples, dict) or len(samples) != 15096:
        raise RuntimeError("Expected a dict containing exactly 15096 LIDC samples")

    sample_ids = list(samples)
    images = np.lib.format.open_memmap(
        output_dir / "images.npy", mode="w+", dtype=np.float32,
        shape=(len(samples), 128, 128),
    )
    masks = np.lib.format.open_memmap(
        output_dir / "masks.npy", mode="w+", dtype=np.uint8,
        shape=(len(samples), 4, 128, 128),
    )
    series_uids: list[str] = []
    for index, sample_id in enumerate(sample_ids):
        sample = samples[sample_id]
        image = np.asarray(sample["image"], dtype=np.float32)
        annotation = np.asarray(sample["masks"])
        if image.shape != (128, 128):
            raise RuntimeError(f"{sample_id}: unexpected image shape {image.shape}")
        if annotation.shape != (4, 128, 128):
            raise RuntimeError(f"{sample_id}: unexpected mask shape {annotation.shape}")
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise RuntimeError(f"{sample_id}: image must be finite and within [0, 1]")
        images[index] = image
        masks[index] = annotation > 0
        series_uids.append(str(sample["series_uid"]))
    images.flush()
    masks.flush()
    del images, masks, samples

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"sample_ids": sample_ids, "series_uids": series_uids},
            handle, ensure_ascii=False,
        )
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "format_version": 1,
                "source_pickle": str(pickle_path.resolve()),
                "image_shape": [15096, 128, 128],
                "image_dtype": "float32",
                "mask_shape": [15096, 4, 128, 128],
                "mask_dtype": "uint8",
            },
            handle, indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    convert(args.pickle.expanduser(), args.output_dir.expanduser())


if __name__ == "__main__":
    main()
