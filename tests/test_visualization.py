import numpy as np
from PIL import Image
import pytest
import torch

from visualization import (
    ADE20K_PALETTE,
    CITYSCAPES_PALETTE,
    colorize,
    save_adaptive_path_debug,
    save_prediction,
    save_source_diagnostics,
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


def test_source_and_adaptive_diagnostic_visualizations_are_png(tmp_path):
    image = torch.rand(3, 8, 12)
    target = torch.randint(0, 20, (8, 12))
    prediction = torch.randint(0, 20, (8, 12))
    entropy = torch.rand(8, 12) * np.log(20)
    source_path = tmp_path / "source.png"
    save_source_diagnostics(
        image, target, prediction, entropy, source_path, num_classes=20
    )
    adaptive_path = tmp_path / "adaptive.png"
    save_adaptive_path_debug(
        image, target,
        {
            "source_mean": torch.nn.functional.one_hot(
                prediction, 20
            ).permute(2, 0, 1)[None].float(),
            "entropy": entropy[None],
            "difficulty": torch.linspace(-1, 1, 96).reshape(1, 8, 12),
            "source_semantic_mask": (prediction != 19)[None],
            "lambdas": torch.rand(1, 4, 8, 12),
            "times": (0.2, 0.4, 0.6, 0.8),
        },
        adaptive_path,
        num_classes=20,
    )
    for path in (source_path, adaptive_path):
        assert path.is_file()
        with Image.open(path) as saved:
            assert saved.format == "PNG"
