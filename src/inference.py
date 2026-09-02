from __future__ import annotations

import torch
import torch.nn.functional as F

from discrete_flow_maps import flow_map, make_time_grid, sample_prior
from state_space import resize_continuous, state_spatial_size


def state_to_prediction(
    continuous: torch.Tensor,
    *,
    void_class_index: int | None = None,
    exclude_void: bool = False,
) -> torch.Tensor:
    """Convert continuous class scores to labels, optionally excluding void."""
    if continuous.ndim != 4:
        raise ValueError(
            "continuous state must have shape [batch, classes, height, width]"
        )
    if not exclude_void:
        return continuous.argmax(dim=1)
    classes = continuous.shape[1]
    if (
        isinstance(void_class_index, bool)
        or not isinstance(void_class_index, int)
        or not 0 <= void_class_index < classes
    ):
        raise ValueError(
            "void_class_index must be a valid class index when exclude_void=true"
        )
    if classes < 2:
        raise ValueError("exclude_void=true requires at least one non-void class")
    scores = continuous.clone()
    scores[:, void_class_index] = -torch.inf
    return scores.argmax(dim=1)


@torch.no_grad()
def run_flow_from_state(
    model,
    image: torch.Tensor,
    initial_state: torch.Tensor,
    config: dict,
    num_steps: int,
    return_trajectory: bool = False,
):
    """Run the production Flow Map update from a caller-provided state."""
    model.eval()
    x = initial_state
    expected_state_size = state_spatial_size(
        image, config.get("model", {}).get("state_downsample_factor", 1)
    )
    assert x.shape[-2:] == expected_state_size
    trajectory = [x.argmax(dim=1)]
    for scalar_s, scalar_t in make_time_grid(num_steps, image.device):
        batch = image.shape[0]
        s = scalar_s.expand(batch)
        t = scalar_t.expand(batch)
        logits = model.forward_logits(x, image, s, t)
        assert logits.shape == x.shape
        probability = torch.softmax(logits.float(), dim=1).to(x.dtype)
        x = flow_map(
            x, probability, s, t, config["flow"]["time_eps"], config["flow"]
        )
        trajectory.append(x.argmax(dim=1))
    if return_trajectory:
        return x, torch.stack(trajectory, dim=1)
    return x


@torch.no_grad()
def sample_segmentation_from_x0(
    model,
    image: torch.Tensor,
    x0: torch.Tensor,
    config: dict,
    num_steps: int | None = None,
    return_trajectory: bool = False,
    return_terminal_state: bool = False,
):
    """Production segmentation inference starting from a deterministic x0."""
    steps = num_steps or config["evaluation"]["num_steps"]
    flow_result = run_flow_from_state(
        model, image, x0, config, steps, return_trajectory=return_trajectory
    )
    if return_trajectory:
        x, trajectory = flow_result
    else:
        x = flow_result
    if return_terminal_state:
        return x
    full_resolution = resize_continuous(x, image.shape[-2:])
    evaluation = config.get("evaluation", {})
    prediction = state_to_prediction(
        full_resolution,
        void_class_index=config.get("dataset", {}).get("void_class_index"),
        exclude_void=evaluation.get("exclude_void_from_prediction", False),
    )
    if return_trajectory:
        return prediction, trajectory
    return prediction


@torch.no_grad()
def sample_segmentation(
    model,
    source_model,
    image: torch.Tensor,
    config: dict,
    num_steps: int | None = None,
    return_trajectory: bool = False,
    return_terminal_state: bool = False,
):
    model.eval()
    if source_model is not None:
        source_model.eval()
    x0, _ = sample_prior(config, image, None, source_model)
    return sample_segmentation_from_x0(
        model,
        image,
        x0,
        config,
        num_steps=num_steps,
        return_trajectory=return_trajectory,
        return_terminal_state=return_terminal_state,
    )


def state_to_original_continuous(
    state: torch.Tensor,
    model_shape: tuple[int, int] | list[int],
    original_shape: tuple[int, int] | list[int],
    *,
    padded_shape: tuple[int, int] | list[int] | None = None,
    align_corners: bool = False,
) -> torch.Tensor:
    """Remove input padding and bilinear-resize continuous state channels."""
    model_height, model_width = (int(value) for value in model_shape)
    original_height, original_width = (int(value) for value in original_shape)
    if padded_shape is None:
        padded_shape = state.shape[-2:]
    padded_height, padded_width = (int(value) for value in padded_shape)
    continuous = resize_continuous(
        state.float(), (padded_height, padded_width)
    )
    continuous = continuous[..., :model_height, :model_width]
    return F.interpolate(
        continuous,
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=align_corners,
    )


def terminal_state_to_original_prediction(
    terminal_state: torch.Tensor,
    model_shape: tuple[int, int] | list[int],
    original_shape: tuple[int, int] | list[int],
    *,
    padded_shape: tuple[int, int] | list[int] | None = None,
    align_corners: bool = False,
    void_class_index: int | None = None,
    exclude_void: bool = False,
) -> torch.Tensor:
    """Remove padding, bilinear-resize all channels, then take argmax."""
    continuous = state_to_original_continuous(
        terminal_state,
        model_shape,
        original_shape,
        padded_shape=padded_shape,
        align_corners=align_corners,
    )
    return state_to_prediction(
        continuous,
        void_class_index=void_class_index,
        exclude_void=exclude_void,
    )
