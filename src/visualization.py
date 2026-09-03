from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


CITYSCAPES_PALETTE = np.asarray([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
    [0, 0, 230], [119, 11, 32], [0, 0, 0],
], dtype=np.uint8)


_ADE20K_SEMANTIC_PALETTE = np.asarray([
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
    [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
    [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
    [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
    [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
    [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
    [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
    [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
    [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
    [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
    [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
    [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
    [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
    [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
    [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
    [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
    [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
    [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
    [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
    [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
    [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
    [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
    [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
    [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
    [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
    [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
    [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
    [102, 255, 0], [92, 0, 255],
], dtype=np.uint8)

# MMSegmentation's official ADE20K palette is indexed 0..149 after
# reduce_zero_label. DFM preserves annotation state 0 as void, so semantic
# state n uses the official palette entry n - 1.
ADE20K_PALETTE = np.concatenate(
    (np.zeros((1, 3), dtype=np.uint8), _ADE20K_SEMANTIC_PALETTE), axis=0
)

_PALETTES = {
    "cityscapes": (CITYSCAPES_PALETTE, 19),
    "ade20k": (ADE20K_PALETTE, 0),
}


def colorize(
    mask: torch.Tensor, dataset_name: str = "cityscapes"
) -> np.ndarray:
    if dataset_name not in _PALETTES:
        raise ValueError(f"Unsupported visualization dataset: {dataset_name}")
    palette, void_class_index = _PALETTES[dataset_name]
    indices = mask.detach().cpu().numpy().astype(np.int64)
    indices = np.where(
        (indices >= 0) & (indices < len(palette)), indices, void_class_index
    )
    return palette[indices]


def save_prediction(
    image: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    path: str | Path,
    imagenet_normalize: bool = False,
    dataset_name: str = "cityscapes",
) -> None:
    image = image.detach().cpu()
    if imagenet_normalize:
        mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        image = image * std + mean
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image.clamp(0, 1).permute(1, 2, 0))
    axes[1].imshow(colorize(target, dataset_name))
    axes[2].imshow(colorize(prediction, dataset_name))
    for axis, title in zip(axes, ("image", "ground truth", "DFM prediction")):
        axis.set_title(title)
        axis.axis("off")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def save_adaptive_path_debug(
    image: torch.Tensor,
    target: torch.Tensor,
    debug: dict[str, torch.Tensor | tuple[float, ...]],
    path: str | Path,
    *,
    imagenet_normalize: bool = False,
    dataset_name: str = "cityscapes",
    num_classes: int = 20,
) -> None:
    """Save source prediction, entropy, difficulty, and configured lambda maps."""
    image = image.detach().cpu()
    if imagenet_normalize:
        mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        image = image * std + mean
    target = target.detach().cpu()
    display_size = tuple(target.shape[-2:])
    source_mean = debug["source_mean"][0:1].detach().float()
    prediction = F.interpolate(
        source_mean, display_size, mode="bilinear", align_corners=False
    )[0].argmax(dim=0).cpu()
    entropy = F.interpolate(
        debug["entropy"][0:1, None].detach().float(),
        display_size, mode="bilinear", align_corners=False,
    )[0, 0].cpu()
    difficulty = F.interpolate(
        debug["difficulty"][0:1, None].detach().float(),
        display_size, mode="bilinear", align_corners=False,
    )[0, 0].cpu()
    lambdas = F.interpolate(
        debug["lambdas"][0].detach().float()[:, None],
        display_size, mode="bilinear", align_corners=False,
    )[:, 0].cpu()
    times = debug["times"]
    semantic_mask = F.interpolate(
        debug["source_semantic_mask"][0:1, None].detach().float(),
        display_size, mode="nearest",
    )[0, 0].bool().cpu()
    panels = 6 + len(times)
    columns = 4
    rows = (panels + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.asarray(axes).reshape(-1)
    axes[0].imshow(image.clamp(0, 1).permute(1, 2, 0))
    axes[0].set_title("image")
    axes[1].imshow(colorize(target, dataset_name))
    axes[1].set_title("ground truth")
    axes[2].imshow(colorize(prediction, dataset_name))
    axes[2].set_title("source prediction")
    entropy_image = axes[3].imshow(
        entropy, cmap="magma", vmin=0.0, vmax=math.log(num_classes)
    )
    axes[3].set_title(
        f"entropy\nmin={entropy.min():.3f} mean={entropy.mean():.3f} "
        f"max={entropy.max():.3f}"
    )
    figure.colorbar(entropy_image, ax=axes[3], fraction=0.046, pad=0.04)
    axes[4].imshow(~semantic_mask, cmap="gray", vmin=0, vmax=1)
    axes[4].set_title("source-predicted void")
    difficulty_image = axes[5].imshow(
        difficulty, cmap="coolwarm", vmin=-1.0, vmax=1.0
    )
    axes[5].set_title("difficulty (easy / neutral / hard)")
    figure.colorbar(difficulty_image, ax=axes[5], fraction=0.046, pad=0.04)
    for index, value in enumerate(times):
        axes[6 + index].imshow(lambdas[index], cmap="viridis", vmin=0.0, vmax=1.0)
        axes[6 + index].set_title(f"lambda(t={value:g})")
    for axis in axes:
        axis.axis("off")
    for axis in axes[panels:]:
        axis.set_visible(False)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def save_source_diagnostics(
    image: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    entropy: torch.Tensor,
    path: str | Path,
    *,
    num_classes: int,
    imagenet_normalize: bool = False,
    dataset_name: str = "cityscapes",
) -> None:
    """Save the source-only validation view with a fixed entropy scale."""
    image = image.detach().float().cpu()
    if imagenet_normalize:
        mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        image = image * std + mean
    target = target.detach().cpu()
    prediction = prediction.detach().cpu()
    entropy = entropy.detach().float().cpu()
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.reshape(-1)
    axes[0].imshow(image.clamp(0, 1).permute(1, 2, 0))
    axes[0].set_title("input image")
    axes[1].imshow(colorize(target, dataset_name))
    axes[1].set_title("ground truth")
    axes[2].imshow(colorize(prediction, dataset_name))
    axes[2].set_title("source mean prediction")
    entropy_image = axes[3].imshow(
        entropy, cmap="magma", vmin=0.0, vmax=math.log(num_classes)
    )
    axes[3].set_title(
        f"entropy\nmin={entropy.min():.3f} mean={entropy.mean():.3f} "
        f"max={entropy.max():.3f}"
    )
    figure.colorbar(entropy_image, ax=axes[3], fraction=0.046, pad=0.04)
    for axis in axes:
        axis.axis("off")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
