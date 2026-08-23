from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


CITYSCAPES_PALETTE = np.asarray([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
    [0, 0, 230], [119, 11, 32], [0, 0, 0],
], dtype=np.uint8)


def colorize(mask: torch.Tensor) -> np.ndarray:
    indices = mask.detach().cpu().numpy().astype(np.int64)
    indices = np.where((indices >= 0) & (indices < 20), indices, 19)
    return CITYSCAPES_PALETTE[indices]


def save_prediction(
    image: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    path: str | Path,
    imagenet_normalize: bool = False,
) -> None:
    image = image.detach().cpu()
    if imagenet_normalize:
        mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        image = image * std + mean
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image.clamp(0, 1).permute(1, 2, 0))
    axes[1].imshow(colorize(target))
    axes[2].imshow(colorize(prediction))
    for axis, title in zip(axes, ("image", "ground truth", "DFM prediction")):
        axis.set_title(title)
        axis.axis("off")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)

