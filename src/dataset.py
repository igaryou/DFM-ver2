from __future__ import annotations

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
        image_size: list[int] | tuple[int, int] | None = None,
        crop_size: list[int] | tuple[int, int] | None = None,
        augment: bool = False,
        hflip_prob: float = 0.5,
        color_jitter: bool = False,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        imagenet_normalize: bool = False,
    ) -> None:
        self.image_size = tuple(image_size) if image_size is not None else None
        self.crop_size = tuple(crop_size) if crop_size is not None else None
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.color_jitter_enabled = color_jitter
        self.imagenet_normalize = imagenet_normalize
        self.jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.dataset = Cityscapes(
            root=root, split=split, mode="fine", target_type="semantic"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        image = TF.pil_to_tensor(image).float() / 255.0
        mask = torch.from_numpy(ID_TO_20CLASS[np.asarray(target, dtype=np.uint8)]).long()

        if self.augment and torch.rand(()) < self.hflip_prob:
            image = torch.flip(image, (2,))
            mask = torch.flip(mask, (1,))
        if self.image_size is not None:
            image = TF.resize(
                image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            mask = TF.resize(
                mask[None], self.image_size, interpolation=TF.InterpolationMode.NEAREST
            )[0].long()
        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            _, height, width = image.shape
            if crop_h > height or crop_w > width:
                raise ValueError("dataset.crop_size exceeds the resized image")
            top = torch.randint(0, height - crop_h + 1, ()).item()
            left = torch.randint(0, width - crop_w + 1, ()).item()
            image = TF.crop(image, top, left, crop_h, crop_w)
            mask = TF.crop(mask, top, left, crop_h, crop_w)
        if self.augment and self.color_jitter_enabled:
            image = self.jitter(image).clamp(0.0, 1.0)
        if self.imagenet_normalize:
            image = _normalize(image, {
                "enabled": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            })

        one_hot = F.one_hot(mask, num_classes=20).permute(2, 0, 1).float()
        return image, one_hot, mask


class ADE20KDataset(Dataset):
    """ADE20K loader preserving annotation values 0..150 as 151 flow states."""

    SPLITS = {"train": "training", "training": "training", "val": "validation", "validation": "validation"}
    EXPECTED_COUNTS = {"training": 20210, "validation": 2000}

    def __init__(self, root: str, split: str, config: dict, augment: bool) -> None:
        self.root = Path(root)
        if split not in self.SPLITS:
            raise ValueError(f"Unknown ADE20K split: {split}")
        self.split = self.SPLITS[split]
        self.config = config
        self.augment = augment and self.split == "training"
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
        resize = self.config["augmentation"]["random_resize"]
        if not resize["enabled"]:
            return image, mask
        ratio = float(torch.empty(()).uniform_(*resize["ratio_range"]))
        target_width = resize["base_scale"]["width"] * ratio
        target_height = resize["base_scale"]["height"] * ratio
        if resize["keep_ratio"]:
            size = _resize_keep_ratio_size(*mask.shape, target_width, target_height)
        else:
            size = (round(target_height), round(target_width))
        image = TF.resize(image, size, TF.InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask[None], size, TF.InterpolationMode.NEAREST)[0].long()
        return image, mask

    def _random_crop(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        crop = self.config["augmentation"]["random_crop"]
        if not crop["enabled"]:
            return image, mask
        crop_h, crop_w = crop["size"]
        height, width = mask.shape
        out_h, out_w = min(crop_h, height), min(crop_w, width)
        selected = (0, 0)
        for _ in range(crop["max_attempts"]):
            top = int(torch.randint(0, height - out_h + 1, ()))
            left = int(torch.randint(0, width - out_w + 1, ()))
            selected = top, left
            candidate = mask[top:top + out_h, left:left + out_w]
            valid = candidate[candidate != crop["ignore_index"]]
            if valid.numel() == 0:
                continue
            counts = torch.bincount(valid, minlength=151)
            if float(counts.max()) / valid.numel() < crop["cat_max_ratio"]:
                break
        top, left = selected
        return (
            image[:, top:top + out_h, left:left + out_w],
            mask[top:top + out_h, left:left + out_w],
        )

    def _train_item(self, image: torch.Tensor, mask: torch.Tensor):
        image, mask = self._random_resize(image, mask)
        image, mask = self._random_crop(image, mask)
        flip = self.config["augmentation"]["horizontal_flip"]
        if flip["enabled"] and torch.rand(()) < flip["probability"]:
            image, mask = torch.flip(image, (2,)), torch.flip(mask, (1,))
        photo = self.config["augmentation"]["photometric_distortion"]
        if photo["enabled"]:
            image = self.photo_distortion(image)
        image = _normalize(image, self.config["augmentation"]["normalize"])
        pad = self.config["augmentation"]["pad"]
        if pad["enabled"]:
            image, mask = _pad_to(
                image, mask, *pad["size"], pad["image_value"], pad["mask_value"]
            )
        one_hot = F.one_hot(mask, num_classes=151).permute(2, 0, 1).float()
        return image, one_hot, mask

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
            return self._train_item(image, mask)
        return self._validation_item(image, mask, index)


def ade20k_eval_collate(batch: list[dict]) -> list[dict]:
    """Keep variable-resolution ADE20K samples separate until evaluation."""
    return batch


def build_dataset(config: dict, split: str, augment: bool | None = None):
    if config["dataset"]["name"] == "ade20k":
        enabled = config["augmentation"]["enabled"] if augment is None else augment
        return ADE20KDataset(config["dataset"]["root"], split, config, enabled)

    aug = config["augmentation"]
    flip = aug["horizontal_flip"]
    jitter = aug["color_jitter"]
    enabled = aug["enabled"] if augment is None else augment
    return Cityscapes20ClassDataset(
        root=config["dataset"]["root"],
        split=split,
        image_size=config["dataset"]["image_size"],
        crop_size=config["dataset"]["crop_size"] if split == "train" else None,
        augment=enabled and split == "train",
        hflip_prob=flip["probability"] if flip["enabled"] else 0.0,
        color_jitter=enabled and jitter["enabled"],
        brightness=jitter["brightness"],
        contrast=jitter["contrast"],
        saturation=jitter["saturation"],
        hue=jitter["hue"],
        imagenet_normalize=aug["imagenet_normalize"],
    )
