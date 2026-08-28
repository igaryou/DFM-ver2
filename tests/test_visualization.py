import numpy as np
from PIL import Image
import pytest
import torch

from visualization import (
    ADE20K_PALETTE,
    CITYSCAPES_PALETTE,
    colorize,
    save_prediction,
)


def test_palette_shapes_and_dtypes():
    assert len(CITYSCAPES_PALETTE) == 20
    assert len(ADE20K_PALETTE) == 151
    assert CITYSCAPES_PALETTE.shape == (20, 3)
    assert ADE20K_PALETTE.shape == (151, 3)
    assert CITYSCAPES_PALETTE.dtype == np.uint8
    assert ADE20K_PALETTE.dtype == np.uint8


def test_ade20k_state_colors_match_mmsegmentation_palette():
    expected = {
        0: [0, 0, 0],
        1: [120, 120, 120],
        2: [180, 120, 120],
        10: [4, 250, 7],
        50: [250, 10, 15],
        100: [255, 112, 0],
        150: [92, 0, 255],
    }
    for state, color in expected.items():
        np.testing.assert_array_equal(ADE20K_PALETTE[state], color)


def test_ade20k_colorize_keeps_high_semantic_classes_colored():
    states = torch.tensor([[0, 1, 19, 20, 50, 100, 150]])
    colored = colorize(states, dataset_name="ade20k")

    np.testing.assert_array_equal(colored[0], ADE20K_PALETTE[states[0]])
    np.testing.assert_array_equal(colored[0, 0], [0, 0, 0])
    assert np.any(colored[0, [3, 4, 5, 6]] != 0, axis=1).all()


def test_ade20k_full_state_grid_has_black_only_for_void():
    states = torch.arange(151).reshape(151, 1)
    colored = colorize(states, dataset_name="ade20k")
    is_black = np.all(colored == 0, axis=-1).reshape(-1)
    np.testing.assert_array_equal(np.flatnonzero(is_black), [0])


def test_invalid_labels_map_to_dataset_void_color():
    invalid = torch.tensor([[-1, 151, 999]])
    ade_colored = colorize(invalid, dataset_name="ade20k")
    city_colored = colorize(invalid, dataset_name="cityscapes")
    np.testing.assert_array_equal(
        ade_colored, np.broadcast_to(ADE20K_PALETTE[0], ade_colored.shape)
    )
    np.testing.assert_array_equal(
        city_colored, np.broadcast_to(CITYSCAPES_PALETTE[19], city_colored.shape)
    )


def test_cityscapes_palette_regression_and_default_interface():
    states = torch.tensor([[0, 13, 18, 19]])
    expected = np.asarray(
        [[[128, 64, 128], [0, 0, 142], [119, 11, 32], [0, 0, 0]]],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(colorize(states), expected)
    np.testing.assert_array_equal(
        colorize(states, dataset_name="cityscapes"), expected
    )


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="Unsupported visualization dataset"):
        colorize(torch.zeros(1, 1), dataset_name="unknown")


def test_save_prediction_supports_ade20k_classes_above_19(tmp_path):
    output = tmp_path / "ade20k_prediction.png"
    target = torch.tensor([[0, 1, 20], [50, 100, 150]])
    prediction = torch.tensor([[1, 20, 50], [100, 150, 2]])
    save_prediction(
        torch.rand(3, 2, 3),
        target,
        prediction,
        output,
        dataset_name="ade20k",
    )
    assert output.is_file()
    with Image.open(output) as saved:
        assert saved.format == "PNG"
        assert saved.width > 0
        assert saved.height > 0
