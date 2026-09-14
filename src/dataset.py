from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import Cityscapes
from torchvision.transforms import functional as TF


ID_TO_20CLASS = np.full(256, 19, dtype=np.uint8)
for cityscapes_id, train_id in {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
    22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
    32: 17, 33: 18,
}.items():
    ID_TO_20CLASS[cityscapes_id] = train_id


def _normalize(image: torch.Tensor, config: dict) -> torch.Tensor:
    if not config["enabled"]:
        return image
    mean = image.new_tensor(config["mean"])[:, None, None]
    std = image.new_tensor(config["std"])[:, None, None]
    return (image - mean) / std


def _resize_keep_ratio_size(
    height: int, width: int, target_width: float, target_height: float
) -> tuple[int, int]:
    max_long_edge = max(target_width, target_height)
    max_short_edge = min(target_width, target_height)
    scale = min(
        max_long_edge / max(height, width),
        max_short_edge / min(height, width),
    )
    new_height = max(1, int(height * scale + 0.5))
    new_width = max(1, int(width * scale + 0.5))
    return new_height, new_width


def _pad_to(
    image: torch.Tensor,
    mask: torch.Tensor | None,
    target_height: int,
    target_width: int,
    image_value: float,
    mask_value: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    pad_height = max(target_height - image.shape[-2], 0)
    pad_width = max(target_width - image.shape[-1], 0)
    image = F.pad(image, (0, pad_width, 0, pad_height), value=float(image_value))
    if mask is not None:
        mask = F.pad(mask, (0, pad_width, 0, pad_height), value=int(mask_value))
    return image, mask


def _random_resize_pair(
    image: torch.Tensor, mask: torch.Tensor, config: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    image, mask, _ = _random_resize_triplet(
        image, mask, torch.ones_like(mask, dtype=torch.bool), config
    )
    return image, mask


def _random_resize_triplet(
    image: torch.Tensor,
    mask: torch.Tensor,
    spatial_valid_mask: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not config["enabled"]:
        return image, mask, spatial_valid_mask
    ratio = float(torch.empty(()).uniform_(*config["ratio_range"]))
    target_width = config["base_scale"]["width"] * ratio
    target_height = config["base_scale"]["height"] * ratio
    if config["keep_ratio"]:
        size = _resize_keep_ratio_size(
            *mask.shape, target_width, target_height
        )
    else:
        size = (round(target_height), round(target_width))
    image = TF.resize(
        image, size, TF.InterpolationMode.BILINEAR, antialias=True
    )
    mask = TF.resize(
        mask[None], size, TF.InterpolationMode.NEAREST
    )[0].long()
    spatial_valid_mask = TF.resize(
        spatial_valid_mask[None], size, TF.InterpolationMode.NEAREST
    )[0].bool()
    return image, mask, spatial_valid_mask


def _crop_has_acceptable_class_ratio(
    candidate: torch.Tensor, *, ignore_index: int, cat_max_ratio: float
) -> bool:
    valid = candidate[candidate != ignore_index]
    if valid.numel() == 0:
        return False
    counts = torch.bincount(valid)
    return float(counts.max()) / valid.numel() < cat_max_ratio


def _random_crop_pair(
    image: torch.Tensor,
    mask: torch.Tensor,
    config: dict,
    *,
    ensure_crop_size: bool = False,
    image_pad_value: float = 0.0,
    mask_pad_value: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    image, mask, _ = _random_crop_triplet(
        image,
        mask,
        torch.ones_like(mask, dtype=torch.bool),
        config,
        ensure_crop_size=ensure_crop_size,
        image_pad_value=image_pad_value,
        mask_pad_value=mask_pad_value,
    )
    return image, mask


def _random_crop_triplet(
    image: torch.Tensor,
    mask: torch.Tensor,
    spatial_valid_mask: torch.Tensor,
    config: dict,
    *,
    ensure_crop_size: bool = False,
    image_pad_value: float = 0.0,
    mask_pad_value: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not config["enabled"]:
        return image, mask, spatial_valid_mask
    crop_h, crop_w = config["size"]
    if ensure_crop_size:
        # Safety padding precedes photometric distortion and normalization.
        # The main Cityscapes scale range normally makes this a no-op.
        image, padded_mask = _pad_to(
            image, mask, crop_h, crop_w, image_pad_value, mask_pad_value
        )
        assert padded_mask is not None
        mask = padded_mask
        spatial_valid_mask = F.pad(
            spatial_valid_mask,
            (0, image.shape[-1] - spatial_valid_mask.shape[-1],
             0, image.shape[-2] - spatial_valid_mask.shape[-2]),
            value=False,
        )
    height, width = mask.shape
    out_h, out_w = min(crop_h, height), min(crop_w, width)
    selected = (0, 0)
    for _ in range(config["max_attempts"]):
        top = int(torch.randint(0, height - out_h + 1, ()))
        left = int(torch.randint(0, width - out_w + 1, ()))
        selected = top, left
        candidate = mask[top:top + out_h, left:left + out_w]
        if _crop_has_acceptable_class_ratio(
            candidate,
            ignore_index=config["ignore_index"],
            cat_max_ratio=config["cat_max_ratio"],
        ):
            break
    top, left = selected
    return (
        image[:, top:top + out_h, left:left + out_w],
        mask[top:top + out_h, left:left + out_w],
        spatial_valid_mask[top:top + out_h, left:left + out_w],
    )


class PhotoMetricDistortion:
    """MMSeg-style random ordering, operating on an RGB tensor in [0, 1]."""

    def __init__(self, config: dict) -> None:
        self.brightness_delta = float(config["brightness_delta"]) / 255.0
        self.contrast_range = tuple(float(value) for value in config["contrast_range"])
        self.saturation_range = tuple(float(value) for value in config["saturation_range"])
        self.hue_delta = float(config["hue_delta"]) / 360.0

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return float(torch.empty(()).uniform_(low, high))

    @staticmethod
    def _rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
        red, green, blue = image.unbind(dim=0)
        maximum, maximum_index = image.max(dim=0)
        minimum = image.min(dim=0).values
        delta = maximum - minimum
        safe_maximum = torch.where(
            maximum.abs() > 1.0e-12, maximum, torch.ones_like(maximum)
        )
        saturation = torch.where(
            maximum.abs() > 1.0e-12, delta / safe_maximum, torch.zeros_like(delta)
        )
        safe_delta = delta.clamp_min(1.0e-12)
        hue = torch.zeros_like(maximum)
        hue = torch.where(maximum_index == 0, (green - blue) / safe_delta, hue)
        hue = torch.where(maximum_index == 1, 2.0 + (blue - red) / safe_delta, hue)
        hue = torch.where(maximum_index == 2, 4.0 + (red - green) / safe_delta, hue)
        hue = torch.where(delta == 0, torch.zeros_like(hue), (hue / 6.0).remainder(1.0))
        return torch.stack((hue, saturation, maximum))

    @staticmethod
    def _hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
        hue, saturation, value = image.unbind(dim=0)
        sector = torch.floor(hue.remainder(1.0) * 6.0).to(torch.int64)
        fraction = hue.remainder(1.0) * 6.0 - sector
        p = value * (1.0 - saturation)
        q = value * (1.0 - fraction * saturation)
        t = value * (1.0 - (1.0 - fraction) * saturation)
        choices = torch.stack((
            torch.stack((value, t, p)), torch.stack((q, value, p)),
            torch.stack((p, value, t)), torch.stack((p, q, value)),
            torch.stack((t, p, value)), torch.stack((value, p, q)),
        ))
        gather_index = sector.remainder(6)[None, None].expand(1, 3, *sector.shape)
        return choices.gather(0, gather_index).squeeze(0)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if bool(torch.randint(0, 2, ())):
            image = image + self._uniform(-self.brightness_delta, self.brightness_delta)
        contrast_first = bool(torch.randint(0, 2, ()))
        if contrast_first and bool(torch.randint(0, 2, ())):
            image = image * self._uniform(*self.contrast_range)
        if bool(torch.randint(0, 2, ())):
            hsv = self._rgb_to_hsv(image)
            hsv[1] *= self._uniform(*self.saturation_range)
            image = self._hsv_to_rgb(hsv)
        if bool(torch.randint(0, 2, ())):
            hsv = self._rgb_to_hsv(image)
            hsv[0] = (hsv[0] + self._uniform(-self.hue_delta, self.hue_delta)).remainder(1.0)
            image = self._hsv_to_rgb(hsv)
        if not contrast_first and bool(torch.randint(0, 2, ())):
            image = image * self._uniform(*self.contrast_range)
        return image


class Cityscapes20ClassDataset(Dataset):
    """Cityscapes with 19 semantic classes plus void at class index 19."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        config: dict | None = None,
        augment: bool = False,
        return_spatial_valid_mask: bool = False,
    ) -> None:
        if config is None:
            raise ValueError("Cityscapes20ClassDataset requires config")
        self.config = config
        self.split = split
        self.augment = augment and split == config["dataset"]["train_split"]
        self.return_spatial_valid_mask = return_spatial_valid_mask
        photo_config = config["augmentation"]["photometric_distortion"]
        self.photo_distortion = PhotoMetricDistortion(photo_config)
        jitter = config["augmentation"]["color_jitter"]
        self.jitter = transforms.ColorJitter(
            jitter["brightness"], jitter["contrast"],
            jitter["saturation"], jitter["hue"],
        )
        self.dataset = Cityscapes(
            root=root, split=split, mode="fine", target_type="semantic"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _map_target(target) -> torch.Tensor:
        return torch.from_numpy(
            ID_TO_20CLASS[np.asarray(target, dtype=np.uint8)]
        ).long()

    def _legacy_train_item(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask, _ = self._legacy_train_item_with_spatial_mask(image, mask)
        return image, mask

    def _legacy_train_item_with_spatial_mask(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        augmentation = self.config["augmentation"]
        spatial_valid_mask = torch.ones_like(mask, dtype=torch.bool)
        flip = augmentation["horizontal_flip"]
        if flip["enabled"] and torch.rand(()) < flip["probability"]:
            image = torch.flip(image, (2,))
            mask = torch.flip(mask, (1,))
            spatial_valid_mask = torch.flip(spatial_valid_mask, (1,))
        image_size = self.config["dataset"]["image_size"]
        if image_size is not None:
            image = TF.resize(
                image, image_size, interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            mask = TF.resize(
                mask[None], image_size, interpolation=TF.InterpolationMode.NEAREST
            )[0].long()
            spatial_valid_mask = TF.resize(
                spatial_valid_mask[None], image_size,
                interpolation=TF.InterpolationMode.NEAREST,
            )[0].bool()
        crop_size = self.config["dataset"]["crop_size"]
        if crop_size is not None:
            if crop_size[0] > mask.shape[0] or crop_size[1] > mask.shape[1]:
                raise ValueError("dataset.crop_size exceeds the resized image")
            image, mask, spatial_valid_mask = _random_crop_triplet(
                image, mask, spatial_valid_mask, {
                    "enabled": True,
                    "size": crop_size,
                    "cat_max_ratio": 1.01,
                    "ignore_index": self.config["dataset"]["void_class_index"],
                    "max_attempts": 1,
                }
            )
        if augmentation["color_jitter"]["enabled"]:
            image = self.jitter(image).clamp(0.0, 1.0)
        if augmentation["imagenet_normalize"]:
            image = _normalize(image, {
                "enabled": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            })
        return image, mask, spatial_valid_mask

    def _original_train_item_with_spatial_mask(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CFM/CCDM fixed-resolution Cityscapes training pipeline."""
        augmentation = self.config["augmentation"]
        spatial_valid_mask = torch.ones_like(mask, dtype=torch.bool)
        flip = augmentation["horizontal_flip"]
        if flip["enabled"] and torch.rand(()) < flip["probability"]:
            image = torch.flip(image, (2,))
            mask = torch.flip(mask, (1,))
            spatial_valid_mask = torch.flip(spatial_valid_mask, (1,))
        size = tuple(self.config["dataset"]["image_size"])
        resize = self.config["dataset"]["fixed_resize"]
        image = TF.resize(
            image, size, interpolation=TF.InterpolationMode.BILINEAR,
            antialias=resize["antialias"],
        )
        mask = TF.resize(
            mask[None], size, interpolation=TF.InterpolationMode.NEAREST
        )[0].long()
        spatial_valid_mask = TF.resize(
            spatial_valid_mask[None], size,
            interpolation=TF.InterpolationMode.NEAREST,
        )[0].bool()
        if augmentation["color_jitter"]["enabled"]:
            image = self.jitter(image).clamp(0.0, 1.0)
        if augmentation["imagenet_normalize"]:
            image = _normalize(image, {
                "enabled": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            })
        else:
            image = _normalize(image, augmentation["normalize"])
        return image, mask, spatial_valid_mask

    def _train_item(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask, _ = self._train_item_with_spatial_mask(image, mask)
        return image, mask

    def _train_item_with_spatial_mask(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config["dataset"]["protocol"] == "original":
            return self._original_train_item_with_spatial_mask(image, mask)
        augmentation = self.config["augmentation"]
        spatial_valid_mask = torch.ones_like(mask, dtype=torch.bool)
        modern_pipeline = any(
            augmentation[name]["enabled"]
            for name in (
                "random_resize", "random_crop", "photometric_distortion",
                "normalize", "pad",
            )
        )
        if not modern_pipeline:
            return self._legacy_train_item_with_spatial_mask(image, mask)
        image, mask, spatial_valid_mask = _random_resize_triplet(
            image, mask, spatial_valid_mask, augmentation["random_resize"]
        )
        image, mask, spatial_valid_mask = _random_crop_triplet(
            image,
            mask,
            spatial_valid_mask,
            augmentation["random_crop"],
            ensure_crop_size=True,
            image_pad_value=0.0,
            mask_pad_value=self.config["dataset"]["void_class_index"],
        )
        flip = augmentation["horizontal_flip"]
        if flip["enabled"] and torch.rand(()) < flip["probability"]:
            image = torch.flip(image, (2,))
            mask = torch.flip(mask, (1,))
            spatial_valid_mask = torch.flip(spatial_valid_mask, (1,))
        photo = augmentation["photometric_distortion"]
        if photo["enabled"]:
            image = self.photo_distortion(image)
        image = _normalize(image, augmentation["normalize"])
        pad = augmentation["pad"]
        if pad["enabled"]:
            # Padding is intentionally after normalization, so image_value=0
            # denotes zero in normalized space. The main crop normally means
            # that no final padding is required.
            image, padded_mask = _pad_to(
                image, mask, *pad["size"], pad["image_value"], pad["mask_value"]
            )
            assert padded_mask is not None
            mask = padded_mask
            spatial_valid_mask = F.pad(
                spatial_valid_mask,
                (0, image.shape[-1] - spatial_valid_mask.shape[-1],
                 0, image.shape[-2] - spatial_valid_mask.shape[-2]),
                value=False,
            )
        expected = tuple(self.config["dataset"]["image_size"])
        enforce_expected = (
            augmentation["random_crop"]["enabled"]
            and tuple(augmentation["random_crop"]["size"]) == expected
        )
        if enforce_expected and (
            image.shape[-2:] != expected or mask.shape != expected
        ):
            raise RuntimeError(
                "Cityscapes train augmentation must produce dataset.image_size: "
                f"image={tuple(image.shape[-2:])}, mask={tuple(mask.shape)}, "
                f"expected={expected}"
            )
        return image, mask, spatial_valid_mask

    def _validation_item(
        self, image: torch.Tensor, mask: torch.Tensor, index: int
    ):
        evaluation = self.config["evaluation"]
        if (
            self.config["dataset"]["protocol"] == "original"
            or not evaluation["original_resolution"]
        ):
            size = self.config["dataset"]["image_size"]
            antialias = self.config["dataset"]["fixed_resize"]["antialias"]
            image = TF.resize(
                image, size, TF.InterpolationMode.BILINEAR, antialias=antialias
            )
            mask = TF.resize(
                mask[None], size, TF.InterpolationMode.NEAREST
            )[0].long()
            if self.config["augmentation"]["imagenet_normalize"]:
                image = _normalize(image, {
                    "enabled": True,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                })
            else:
                image = _normalize(
                    image, self.config["augmentation"]["normalize"]
                )
            return image, mask
        original_shape = tuple(mask.shape)
        resize = evaluation["resize"]
        model_shape = (
            _resize_keep_ratio_size(
                *original_shape, resize["width"], resize["height"]
            )
            if resize["keep_ratio"]
            else (resize["height"], resize["width"])
        )
        image = TF.resize(
            image, model_shape, TF.InterpolationMode.BILINEAR, antialias=True
        )
        image = _normalize(image, self.config["augmentation"]["normalize"])
        padded_shape = model_shape
        divisor = evaluation["size_divisor"]
        if divisor is not None:
            padded_shape = (
                math.ceil(model_shape[0] / divisor) * divisor,
                math.ceil(model_shape[1] / divisor) * divisor,
            )
            image, _ = _pad_to(image, None, *padded_shape, 0.0, 19)
        return {
            "image": image,
            "target": mask,
            "original_shape": original_shape,
            "model_shape": model_shape,
            "padded_shape": padded_shape,
            "sample_id": Path(self.dataset.images[index]).stem,
        }

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        image = TF.pil_to_tensor(image).float() / 255.0
        mask = self._map_target(target)

        if not self.augment:
            return self._validation_item(image, mask, index)
        if self.return_spatial_valid_mask:
            return self._train_item_with_spatial_mask(image, mask)
        return self._train_item(image, mask)


class ADE20KDataset(Dataset):
    """ADE20K loader preserving annotation values 0..150 as 151 flow states."""

    SPLITS = {"train": "training", "training": "training", "val": "validation", "validation": "validation"}
    EXPECTED_COUNTS = {"training": 20210, "validation": 2000}

    def __init__(
        self, root: str, split: str, config: dict, augment: bool,
        return_spatial_valid_mask: bool = False,
    ) -> None:
        self.root = Path(root)
        if split not in self.SPLITS:
            raise ValueError(f"Unknown ADE20K split: {split}")
        self.split = self.SPLITS[split]
        self.config = config
        self.augment = augment and self.split == "training"
        self.return_spatial_valid_mask = return_spatial_valid_mask
        image_dir = self.root / "images" / self.split
        annotation_dir = self.root / "annotations" / self.split
        if not image_dir.is_dir() or not annotation_dir.is_dir():
            raise FileNotFoundError(
                f"ADE20K requires images/{self.split} and annotations/{self.split} under {self.root}"
            )
        self.images = sorted(image_dir.glob("*.jpg"))
        self.annotations = [annotation_dir / f"{path.stem}.png" for path in self.images]
        missing = [path for path in self.annotations if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing ADE20K annotation: {missing[0]}")
        expected = self.EXPECTED_COUNTS[self.split]
        if len(self.images) != expected:
            raise RuntimeError(
                f"ADE20K {self.split} expected {expected} images, found {len(self.images)}"
            )
        photo_config = config["augmentation"]["photometric_distortion"]
        self.photo_distortion = PhotoMetricDistortion(photo_config)

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def _load(path: Path, annotation: Path) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(path) as handle:
            image = TF.pil_to_tensor(handle.convert("RGB")).float() / 255.0
        with Image.open(annotation) as handle:
            mask_array = np.array(handle, dtype=np.uint8, copy=True)
        mask = torch.from_numpy(mask_array).long()
        minimum, maximum = int(mask.min()), int(mask.max())
        if minimum < 0 or maximum > 150:
            raise ValueError(f"ADE20K labels must be in [0, 150], got [{minimum}, {maximum}]")
        return image, mask

    def _random_resize(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _random_resize_pair(
            image, mask, self.config["augmentation"]["random_resize"]
        )

    def _random_crop(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _random_crop_pair(
            image, mask, self.config["augmentation"]["random_crop"]
        )

    def _train_item(self, image: torch.Tensor, mask: torch.Tensor):
        image, mask, _ = self._train_item_with_spatial_mask(image, mask)
        return image, mask

    def _train_item_with_spatial_mask(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial_valid_mask = torch.ones_like(mask, dtype=torch.bool)
        image, mask, spatial_valid_mask = _random_resize_triplet(
            image, mask, spatial_valid_mask,
            self.config["augmentation"]["random_resize"],
        )
        image, mask, spatial_valid_mask = _random_crop_triplet(
            image, mask, spatial_valid_mask,
            self.config["augmentation"]["random_crop"],
        )
        flip = self.config["augmentation"]["horizontal_flip"]
        if flip["enabled"] and torch.rand(()) < flip["probability"]:
            image = torch.flip(image, (2,))
            mask = torch.flip(mask, (1,))
            spatial_valid_mask = torch.flip(spatial_valid_mask, (1,))
        photo = self.config["augmentation"]["photometric_distortion"]
        if photo["enabled"]:
            image = self.photo_distortion(image)
        image = _normalize(image, self.config["augmentation"]["normalize"])
        pad = self.config["augmentation"]["pad"]
        if pad["enabled"]:
            image, mask = _pad_to(
                image, mask, *pad["size"], pad["image_value"], pad["mask_value"]
            )
            spatial_valid_mask = F.pad(
                spatial_valid_mask,
                (0, image.shape[-1] - spatial_valid_mask.shape[-1],
                 0, image.shape[-2] - spatial_valid_mask.shape[-2]),
                value=False,
            )
        return image, mask, spatial_valid_mask

    def _validation_item(self, image: torch.Tensor, mask: torch.Tensor, index: int) -> dict:
        evaluation = self.config["evaluation"]
        original_shape = tuple(mask.shape)
        resize = evaluation["resize"]
        if resize["keep_ratio"]:
            model_shape = _resize_keep_ratio_size(
                *original_shape, resize["width"], resize["height"]
            )
        else:
            model_shape = (resize["height"], resize["width"])
        image = TF.resize(
            image, model_shape, TF.InterpolationMode.BILINEAR, antialias=True
        )
        image = _normalize(image, self.config["augmentation"]["normalize"])
        divisor = evaluation["size_divisor"]
        padded_shape = model_shape
        if divisor is not None:
            padded_shape = (
                math.ceil(model_shape[0] / divisor) * divisor,
                math.ceil(model_shape[1] / divisor) * divisor,
            )
            image, _ = _pad_to(image, None, *padded_shape, 0.0, 0)
        return {
            "image": image,
            "target": mask,
            "original_shape": original_shape,
            "model_shape": model_shape,
            "padded_shape": padded_shape,
            "sample_id": self.images[index].stem,
        }

    def __getitem__(self, index: int):
        image, mask = self._load(self.images[index], self.annotations[index])
        if self.augment:
            if self.return_spatial_valid_mask:
                return self._train_item_with_spatial_mask(image, mask)
            return self._train_item(image, mask)
        return self._validation_item(image, mask, index)


class LIDCDataset(Dataset):
    """LIDC-IDRI samples backed by worker-safe read-only NumPy memmaps."""

    EXPECTED_SAMPLE_COUNTS = {"train": 9348, "val": 2748, "test": 3000}
    EXPECTED_SERIES_COUNTS = {"train": 560, "val": 140, "test": 175}
    IMAGE_SHAPE = (128, 128)
    ANNOTATIONS = 4

    def __init__(
        self, cache_dir: str, split_path: str, split: str, config: dict,
        augment: bool, return_spatial_valid_mask: bool = False,
    ) -> None:
        if split not in self.EXPECTED_SAMPLE_COUNTS:
            raise ValueError(f"Unknown LIDC split: {split}")
        self.cache_dir = Path(cache_dir)
        self.split_path = Path(split_path)
        self.split = split
        self.config = config
        self.augment = augment and split == "train"
        self.return_spatial_valid_mask = return_spatial_valid_mask
        self._images = None
        self._masks = None

        split_data = self._load_and_validate_split(self.split_path)
        self.sample_ids = list(split_data["samples"][split])
        self.series_uids = list(split_data["series"][split])
        metadata_path = self.cache_dir / "metadata.json"
        manifest_path = self.cache_dir / "manifest.json"
        if not metadata_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"LIDC cache is incomplete under {self.cache_dir}. Run "
                "scripts/convert_lidc_pickle.py first."
            )
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        cached_ids = metadata.get("sample_ids", [])
        cached_series = metadata.get("series_uids", [])
        expected_total = sum(self.EXPECTED_SAMPLE_COUNTS.values())
        if (
            len(cached_ids) != expected_total
            or len(set(cached_ids)) != expected_total
            or len(cached_series) != len(cached_ids)
        ):
            raise RuntimeError(
                f"LIDC cache metadata must contain {expected_total} unique samples"
            )
        source_pickle = manifest.get("source_pickle")
        configured_pickle = self.config["dataset"].get("pickle_path")
        if source_pickle and configured_pickle and (
            Path(source_pickle).resolve() != Path(configured_pickle).resolve()
        ):
            raise RuntimeError("LIDC cache was built from a different pickle")
        if manifest.get("image_shape") != [expected_total, 128, 128]:
            raise RuntimeError("LIDC cache has an unexpected image shape")
        if manifest.get("mask_shape") != [expected_total, 4, 128, 128]:
            raise RuntimeError("LIDC cache has an unexpected mask shape")
        id_to_index = {sample_id: index for index, sample_id in enumerate(cached_ids)}
        missing = [
            sample_id for sample_id in self.sample_ids
            if sample_id not in id_to_index
        ]
        if missing:
            raise RuntimeError(f"LIDC split sample is missing from cache: {missing[0]}")
        self.indices = [id_to_index[sample_id] for sample_id in self.sample_ids]
        self.sample_series_uids = [cached_series[index] for index in self.indices]
        allowed_series = set(self.series_uids)
        mismatched = [
            sample_id for sample_id, index in zip(self.sample_ids, self.indices)
            if cached_series[index] not in allowed_series
        ]
        if mismatched:
            raise RuntimeError(
                f"LIDC sample/series mismatch in fixed split: {mismatched[0]}"
            )

    @classmethod
    def _load_and_validate_split(cls, path: Path) -> dict:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("seed") != 42:
            raise RuntimeError("LIDC fixed split must have seed=42")
        if set(data.get("samples", {})) != set(cls.EXPECTED_SAMPLE_COUNTS):
            raise RuntimeError("LIDC split JSON must contain train/val/test samples")
        if set(data.get("series", {})) != set(cls.EXPECTED_SERIES_COUNTS):
            raise RuntimeError("LIDC split JSON must contain train/val/test series")
        for split, expected in cls.EXPECTED_SAMPLE_COUNTS.items():
            if len(data["samples"][split]) != expected:
                raise RuntimeError(
                    f"LIDC {split} expected {expected} samples, "
                    f"found {len(data['samples'][split])}"
                )
        series_sets = {}
        for split, expected in cls.EXPECTED_SERIES_COUNTS.items():
            values = data["series"][split]
            if len(values) != expected or len(set(values)) != expected:
                raise RuntimeError(
                    f"LIDC {split} expected {expected} unique series, "
                    f"found {len(set(values))}"
                )
            series_sets[split] = set(values)
        assert series_sets["train"].isdisjoint(series_sets["val"])
        assert series_sets["train"].isdisjoint(series_sets["test"])
        assert series_sets["val"].isdisjoint(series_sets["test"])
        sample_sets = {name: set(values) for name, values in data["samples"].items()}
        if any(
            len(sample_sets[name]) != cls.EXPECTED_SAMPLE_COUNTS[name]
            for name in cls.EXPECTED_SAMPLE_COUNTS
        ):
            raise RuntimeError("LIDC sample IDs must be unique within each split")
        assert sample_sets["train"].isdisjoint(sample_sets["val"])
        assert sample_sets["train"].isdisjoint(sample_sets["test"])
        assert sample_sets["val"].isdisjoint(sample_sets["test"])
        expected_total = sum(cls.EXPECTED_SAMPLE_COUNTS.values())
        if len(set().union(*sample_sets.values())) != expected_total:
            raise RuntimeError(
                f"LIDC fixed split must cover all {expected_total} samples"
            )
        return data

    def _open_arrays(self) -> None:
        if self._images is None:
            self._images = np.load(
                self.cache_dir / "images.npy", mmap_mode="r", allow_pickle=False
            )
            self._masks = np.load(
                self.cache_dir / "masks.npy", mmap_mode="r", allow_pickle=False
            )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_images"] = None
        state["_masks"] = None
        return state

    def __len__(self) -> int:
        return len(self.indices)

    def _augment_pair(
        self, image: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        augmentation = self.config["augmentation"]
        horizontal = augmentation["horizontal_flip"]
        if horizontal["enabled"] and torch.rand(()) < horizontal["probability"]:
            image = torch.flip(image, (-1,))
            masks = torch.flip(masks, (-1,))
        vertical = augmentation["vertical_flip"]
        if vertical["enabled"] and torch.rand(()) < vertical["probability"]:
            image = torch.flip(image, (-2,))
            masks = torch.flip(masks, (-2,))
        if augmentation["random_rotation_90"]["enabled"]:
            turns = int(torch.randint(0, 4, ()))
            image = torch.rot90(image, turns, (-2, -1))
            masks = torch.rot90(masks, turns, (-2, -1))
        return image, masks

    def __getitem__(self, index: int):
        self._open_arrays()
        cache_index = self.indices[index]
        image = torch.from_numpy(np.array(
            self._images[cache_index], dtype=np.float32, copy=True
        ))[None]
        masks = torch.from_numpy(
            (np.array(self._masks[cache_index], copy=True) > 0).astype(np.int64)
        )
        if image.shape != (1, *self.IMAGE_SHAPE):
            raise RuntimeError(f"Unexpected LIDC image shape: {tuple(image.shape)}")
        if masks.shape != (self.ANNOTATIONS, *self.IMAGE_SHAPE):
            raise RuntimeError(f"Unexpected LIDC mask shape: {tuple(masks.shape)}")
        if self.augment:
            image, masks = self._augment_pair(image, masks)
        image = image * 2.0 - 1.0
        if self.augment:
            target = masks[int(torch.randint(0, self.ANNOTATIONS, ()))]
            if self.return_spatial_valid_mask:
                return image, target, torch.ones_like(target, dtype=torch.bool)
            return image, target
        return {
            "image": image,
            # Keep a deterministic target for the existing single-GT metrics.
            "target": masks[0],
            "masks": masks,
            "sample_id": self.sample_ids[index],
            "series_uid": self.sample_series_uids[index],
        }


def ade20k_eval_collate(batch: list[dict]) -> list[dict]:
    """Keep original-resolution evaluation samples separate until inference."""
    return batch


def build_dataset(
    config: dict,
    split: str,
    augment: bool | None = None,
    return_spatial_valid_mask: bool = False,
):
    if config["dataset"]["name"] == "ade20k":
        enabled = config["augmentation"]["enabled"] if augment is None else augment
        return ADE20KDataset(
            config["dataset"]["root"], split, config, enabled,
            return_spatial_valid_mask=return_spatial_valid_mask,
        )
    if config["dataset"]["name"] == "lidc":
        enabled = config["augmentation"]["enabled"] if augment is None else augment
        return LIDCDataset(
            config["dataset"]["cache_dir"], config["dataset"]["split_path"],
            split, config, enabled,
            return_spatial_valid_mask=return_spatial_valid_mask,
        )

    enabled = config["augmentation"]["enabled"] if augment is None else augment
    return Cityscapes20ClassDataset(
        root=config["dataset"]["root"],
        split=split,
        config=config,
        augment=enabled and split == config["dataset"]["train_split"],
        return_spatial_valid_mask=return_spatial_valid_mask,
    )
