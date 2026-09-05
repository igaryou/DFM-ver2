from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def smooth_categorical_target(
    one_hot: torch.Tensor,
    smoothing_p: float,
) -> torch.Tensor:
    """Move categorical vertices into the open simplex without changing argmax."""
    if one_hot.ndim < 2:
        raise ValueError("categorical target must have a channel dimension")
    if isinstance(smoothing_p, bool) or not isinstance(smoothing_p, (int, float)):
        raise TypeError("smoothing_p must be a real number")
    smoothing_p = float(smoothing_p)
    if not 0.0 <= smoothing_p < 1.0:
        raise ValueError("smoothing_p must satisfy 0 <= p < 1")
    if smoothing_p == 0.0:
        return one_hot
    classes = one_hot.shape[1]
    if classes < 2:
        raise ValueError("categorical target must contain at least two classes")
    smoothed = (1.0 - smoothing_p) * one_hot + smoothing_p / classes
    assert smoothed.shape == one_hot.shape
    assert bool((smoothed >= 0).all())
    assert torch.allclose(
        smoothed.sum(dim=1),
        torch.ones_like(smoothed[:, 0]),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    return smoothed


def target_state_from_config(one_hot: torch.Tensor, config: dict) -> torch.Tensor:
    """Resolve the configured state-space endpoint, with legacy-safe defaults."""
    settings = config.get("flow", {}).get("target_smoothing", {})
    enabled = settings.get("enabled", False)
    smoothing_p = settings.get("p", 0.0) if enabled else 0.0
    return smooth_categorical_target(one_hot, smoothing_p)


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
    spatial_valid_mask_full: torch.Tensor | None
    spatial_valid_mask_state: torch.Tensor | None


def prepare_state_targets(
    target_full: torch.Tensor,
    *,
    num_classes: int,
    state_size: tuple[int, int] | list[int],
    ignore_index: int | None,
    mask_pixel_losses: bool,
    spatial_valid_mask_full: torch.Tensor | None = None,
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
    spatial_valid_mask_state = None
    if spatial_valid_mask_full is not None:
        if spatial_valid_mask_full.shape != target_full.shape:
            raise ValueError(
                "spatial_valid_mask_full must match target_full, got "
                f"{tuple(spatial_valid_mask_full.shape)} and {tuple(target_full.shape)}"
            )
        if spatial_valid_mask_full.dtype != torch.bool:
            raise ValueError("spatial_valid_mask_full must have dtype bool")
        spatial_valid_mask_state = F.interpolate(
            spatial_valid_mask_full[:, None].float(),
            size=state_size,
            mode="nearest",
        )[:, 0].bool()

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
        spatial_valid_mask_full=spatial_valid_mask_full,
        spatial_valid_mask_state=spatial_valid_mask_state,
    )
