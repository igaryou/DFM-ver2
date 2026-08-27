import pytest
import torch

from inference import (
    sample_segmentation_from_x0,
    state_to_prediction,
    terminal_state_to_original_prediction,
)


class _VoidEndpoint(torch.nn.Module):
    def forward_logits(self, state, image, s, t):
        logits = torch.zeros_like(state)
        logits[:, 0] = 20
        return logits


def _inference_config(exclude_void=True):
    return {
        "dataset": {"void_class_index": 0},
        "model": {"state_downsample_factor": 4},
        "flow": {"time_eps": 1.0e-5},
        "evaluation": {
            "num_steps": 1,
            "exclude_void_from_prediction": exclude_void,
        },
    }


def test_ade_void_exclusion_matches_non_void_slice_argmax():
    state = torch.randn(2, 151, 3, 5)
    state[:, 0] = 100
    expected = state[:, 1:].argmax(dim=1) + 1

    retained = state_to_prediction(state, void_class_index=0, exclude_void=False)
    excluded = state_to_prediction(state, void_class_index=0, exclude_void=True)

    assert torch.all(retained == 0)
    torch.testing.assert_close(excluded, expected)
    assert excluded.min() >= 1
    assert excluded.max() <= 150


def test_cityscapes_void_exclusion_never_returns_class_19():
    state = torch.randn(2, 20, 3, 5)
    state[:, 19] = 100
    prediction = state_to_prediction(
        state, void_class_index=19, exclude_void=True
    )
    assert prediction.min() >= 0
    assert prediction.max() <= 18


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_void_exclusion_supports_low_precision_without_mutation(dtype):
    state = torch.randn(1, 4, 2, 3).to(dtype)
    state[:, 2] = 100
    original = state.clone()
    prediction = state_to_prediction(
        state, void_class_index=2, exclude_void=True
    )
    torch.testing.assert_close(state, original)
    assert torch.all(prediction != 2)


@pytest.mark.parametrize("void_class_index", [None, -1, 3, True])
def test_void_exclusion_rejects_invalid_void_index(void_class_index):
    with pytest.raises(ValueError, match="void_class_index"):
        state_to_prediction(
            torch.randn(1, 3, 2, 2),
            void_class_index=void_class_index,
            exclude_void=True,
        )


def test_original_resolution_prediction_excludes_void_after_resize():
    terminal = torch.zeros(1, 3, 2, 3)
    terminal[:, 0] = 100
    terminal[:, 2] = 2
    prediction = terminal_state_to_original_prediction(
        terminal,
        model_shape=(7, 10),
        original_shape=(5, 9),
        padded_shape=(8, 12),
        void_class_index=0,
        exclude_void=True,
    )
    assert prediction.shape == (1, 5, 9)
    assert torch.all(prediction == 2)


def test_normal_inference_excludes_void_but_terminal_and_trajectory_keep_it():
    model = _VoidEndpoint()
    image = torch.randn(1, 3, 8, 12)
    x0 = torch.zeros(1, 3, 2, 3)
    x0[:, 0] = 1
    config = _inference_config()

    terminal = sample_segmentation_from_x0(
        model, image, x0, config, return_terminal_state=True
    )
    prediction, trajectory = sample_segmentation_from_x0(
        model, image, x0, config, return_trajectory=True
    )

    assert terminal.shape == x0.shape
    assert torch.all(terminal.argmax(dim=1) == 0)
    assert torch.all(prediction != 0)
    assert torch.all(trajectory[:, -1] == 0)


def test_legacy_prediction_policy_remains_available():
    state = torch.zeros(1, 3, 2, 2)
    state[:, 0] = 10
    prediction = state_to_prediction(
        state, void_class_index=0, exclude_void=False
    )
    assert torch.all(prediction == 0)
