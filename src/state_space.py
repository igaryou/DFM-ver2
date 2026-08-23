from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def state_spatial_size(
    image_or_size: torch.Tensor | tuple[int, int] | list[int],
    downsample_factor: int,
) -> tuple[int, int]:
    """Return the stride-convolution state size derived from the input image."""
    if downsample_factor < 1:
        raise ValueError("downsample_factor must be positive")
    if torch.is_tensor(image_or_size):
        height, width = image_or_size.shape[-2:]
    else:
        height, width = (int(value) for value in image_or_size)
    return (
        math.ceil(height / downsample_factor),
        math.ceil(width / downsample_factor),
    )


def resize_continuous(
    tensor: torch.Tensor, size: tuple[int, int] | list[int]
) -> torch.Tensor:
    """Bilinear resize for logits, probabilities, Gaussian fields, and features."""
    if tensor.ndim != 4:
        raise ValueError(f"continuous tensor must have shape [B,C,H,W], got {tensor.shape}")
    size = tuple(int(value) for value in size)
    if tensor.shape[-2:] == size:
        return tensor
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)


@dataclass(frozen=True)
class StateTargets:
    target_full: torch.Tensor
    target_state: torch.Tensor
    one_hot_state: torch.Tensor
    valid_mask_full: torch.Tensor | None
    valid_mask_state: torch.Tensor | None


def prepare_state_targets(
    target_full: torch.Tensor,
    *,
    num_classes: int,
    state_size: tuple[int, int] | list[int],
    ignore_index: int | None,
    mask_pixel_losses: bool,
) -> StateTargets:
    """Separate discrete full-resolution supervision from the DFM state target."""
    if target_full.ndim != 3:
        raise ValueError(
            f"target_full must have shape [B,H,W], got {tuple(target_full.shape)}"
        )
    state_size = tuple(int(value) for value in state_size)
    target_state = F.interpolate(
        target_full[:, None].float(), size=state_size, mode="nearest"
    )[:, 0].long()
    if target_state.numel() and (
        int(target_state.min()) < 0 or int(target_state.max()) >= num_classes
    ):
        raise ValueError(
            f"target labels must be in [0, {num_classes - 1}] before one-hot encoding"
        )
    one_hot_state = F.one_hot(target_state, num_classes=num_classes).permute(
        0, 3, 1, 2
    ).float()
    use_mask = mask_pixel_losses and ignore_index is not None
    valid_mask_full = target_full != ignore_index if use_mask else None
    valid_mask_state = target_state != ignore_index if use_mask else None

    assert target_state.shape[-2:] == state_size
    assert one_hot_state.shape[-2:] == state_size
    if valid_mask_full is not None:
        assert valid_mask_full.shape == target_full.shape
        assert valid_mask_state is not None
        assert valid_mask_state.shape == target_state.shape
    return StateTargets(
        target_full=target_full,
        target_state=target_state,
        one_hot_state=one_hot_state,
        valid_mask_full=valid_mask_full,
        valid_mask_state=valid_mask_state,
    )
