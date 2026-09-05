from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import visualize_source_interpolation as diagnostic
import visualize_simplex_source
from discrete_flow_maps import sample_image_simplex_components


def _inputs():
    mu = torch.tensor([[[[2.0, -1.0]], [[0.0, 3.0]], [[-2.0, 0.0]]]])
    target = torch.tensor([[[0, 1]]])
    x1 = torch.nn.functional.one_hot(target, 3).permute(0, 3, 1, 2).float()
    return mu, target, x1


def test_simplex_x0_is_exactly_the_production_helper_and_sums_to_one():
    mu, _, _ = _inputs()
    expected = sample_image_simplex_components(
        mu, lambda_value=0.1, temperature=6.0,
        dirichlet_alpha=1.0, seed=42,
    )[2]
    actual = sample_image_simplex_components(
        mu, lambda_value=0.1, temperature=6.0,
        dirichlet_alpha=1.0, seed=42,
    )[2]
    assert torch.equal(actual, expected)
    torch.testing.assert_close(actual.sum(dim=1), torch.ones_like(actual[:, 0]))


def test_bounded_gaussian_formula_and_fixed_seed():
    mu, _, _ = _inputs()
    mu_new, epsilon, x0 = diagnostic.bounded_gaussian_components(
        mu, amplitude=1.5, tanh_temperature=5.0, sigma=0.7, seed=9,
    )
    torch.testing.assert_close(mu_new, 1.5 * torch.tanh(mu / 5.0))
    torch.testing.assert_close(x0, mu_new + 0.7 * epsilon)
    repeated = diagnostic.bounded_gaussian_components(
        mu, amplitude=1.5, tanh_temperature=5.0, sigma=0.7, seed=9,
    )
    for left, right in zip((mu_new, epsilon, x0), repeated, strict=True):
        assert torch.equal(left, right)


def test_bounded_gaussian_sigma_zero_equals_transformed_mean():
    mu, _, _ = _inputs()
    mu_new, _, x0 = diagnostic.bounded_gaussian_components(
        mu, amplitude=1.0, tanh_temperature=5.0, sigma=0.0, seed=42,
    )
    assert torch.equal(x0, mu_new)


def test_raw_gaussian_formula_fixed_seed_and_sigma_zero(monkeypatch):
    mu, _, _ = _inputs()
    epsilon, x0 = diagnostic.raw_gaussian_components(mu, sigma=0.7, seed=9)
    torch.testing.assert_close(x0, mu + 0.7 * epsilon)

    repeated = diagnostic.raw_gaussian_components(mu, sigma=0.7, seed=9)
    assert torch.equal(epsilon, repeated[0])
    assert torch.equal(x0, repeated[1])

    _, zero_sigma = diagnostic.raw_gaussian_components(mu, sigma=0.0, seed=10)
    assert torch.equal(zero_sigma, mu)

    def forbidden(*args, **kwargs):
        raise AssertionError("raw Gaussian must not use softmax or tanh")

    monkeypatch.setattr(torch, "softmax", forbidden)
    monkeypatch.setattr(torch, "tanh", forbidden)
    diagnostic.raw_gaussian_components(mu, sigma=1.0, seed=11)


@pytest.mark.parametrize("amplitude,tau", [(0.1, 0.2), (1.0, 5.0), (10.0, 20.0)])
def test_positive_tanh_transform_preserves_raw_logit_argmax(amplitude, tau):
    # Keep values away from floating-point tanh saturation; the mathematical
    # transform is strictly monotone for every positive amplitude/tau.
    mu = torch.empty(2, 20, 3, 4).uniform_(-0.2, 0.2)
    mu_new, _, _ = diagnostic.bounded_gaussian_components(
        mu, amplitude=amplitude, tanh_temperature=tau, sigma=1.0, seed=1,
    )
    assert torch.equal(mu.argmax(1), mu_new.argmax(1))


def test_linear_interpolation_endpoints_and_formula():
    x0 = torch.randn(1, 3, 1, 2)
    _, _, x1 = _inputs()
    assert torch.equal(diagnostic.linear_interpolation(x0, x1, 0.0), x0)
    assert torch.equal(diagnostic.linear_interpolation(x0, x1, 1.0), x1)
    torch.testing.assert_close(
        diagnostic.linear_interpolation(x0, x1, 0.5), 0.5 * x1 + 0.5 * x0
    )


def test_gt_margin_uses_largest_non_gt_channel():
    state = torch.tensor([[[[0.8, 0.1]], [[0.2, 0.7]], [[0.1, 0.4]]]])
    target = torch.tensor([[[0, 2]]])
    expected = torch.tensor([[[0.6, -0.3]]])
    torch.testing.assert_close(diagnostic.gt_margin(state, target), expected)


def test_statistics_exclude_void_and_split_source_correctness():
    target = torch.tensor([[[0, 1, 2]]])
    source = torch.tensor([[[0, 0, 2]]])
    state = torch.tensor([[[[2.0, 0.0, 9.0]], [[0.0, 2.0, 0.0]], [[1.0, 0.0, 0.0]]]])
    rows, first, _ = diagnostic.interpolation_statistics(
        [state], [0.0], target, source, void_index=2,
    )
    row = rows[0]
    assert row["gt_argmax_ratio_denominator"] == 2
    assert row["gt_argmax_ratio"] == 1.0
    assert row["gt_argmax_ratio_source_correct_denominator"] == 1
    assert row["gt_argmax_ratio_source_incorrect_denominator"] == 1
    assert row["gt_argmax_ratio_source_correct"] == 1.0
    assert row["gt_argmax_ratio_source_incorrect"] == 1.0
    assert first.numel() == 2


def test_source_correctness_is_based_on_mu_argmax_not_noisy_x0():
    mu = torch.tensor([[[[3.0, 0.0]], [[0.0, 3.0]], [[0.0, 0.0]]]])
    target = torch.tensor([[[0, 1]]])
    noisy_x0 = torch.tensor([[[[0.0, 3.0]], [[3.0, 0.0]], [[0.0, 0.0]]]])
    rows, _, _ = diagnostic.interpolation_statistics(
        [noisy_x0], [0.0], target, mu.argmax(dim=1), void_index=2,
    )
    row = rows[0]
    assert row["gt_argmax_ratio_source_correct_denominator"] == 2
    assert row["gt_argmax_ratio_source_incorrect_denominator"] == 0
    assert row["gt_argmax_ratio_source_correct"] == 0.0


def test_first_gt_argmax_grid_index_and_never():
    _, target, x1 = _inputs()
    x0 = torch.tensor([[[[3.0, 3.0]], [[0.0, 0.0]], [[0.0, 0.0]]]])
    times = [0.0, 0.5]
    states = [diagnostic.linear_interpolation(x0, x1, time) for time in times]
    _, first, _ = diagnostic.interpolation_statistics(
        states, times, target, x0.argmax(1), void_index=2,
    )
    assert first.tolist() == [0, -1]


def test_argument_defaults_and_validation():
    args = diagnostic.parse_args([
        "--config", "x", "--checkpoint", "y", "--output-dir", "z"
    ])
    assert tuple(args.times) == diagnostic.DEFAULT_TIMES
    assert (args.lambda_value, args.temperature, args.dirichlet_alpha) == (0.1, 6.0, 1.0)
    assert (args.amplitude, args.tanh_temperature, args.sigma) == (1.0, 5.0, 1.0)
    assert args.target_smoothing_p == 0.0
    raw = diagnostic.parse_args([
        "--config", "x", "--checkpoint", "y", "--output-dir", "z",
        "--mode", "raw_gaussian",
    ])
    assert raw.mode == diagnostic.RAW_GAUSSIAN_MODE
    assert diagnostic.MODES == ("simplex", "bounded_gaussian")
    with pytest.raises(ValueError, match="strictly increasing"):
        diagnostic.parse_args([
            "--config", "x", "--checkpoint", "y", "--output-dir", "z",
            "--times", "0.5", "0.25",
        ])


def test_visualizer_target_smoothing_matches_shared_helper():
    _, _, hard = _inputs()
    assert diagnostic.smooth_categorical_target(hard, 0.0) is hard
    actual = diagnostic.smooth_categorical_target(hard, 0.8)
    expected = 0.2 * hard + 0.8 / hard.shape[1]
    torch.testing.assert_close(actual, expected)


def test_cpu_synthetic_smoke_for_both_modes():
    mu, target, x1 = _inputs()
    q, _, simplex = sample_image_simplex_components(
        mu, lambda_value=0.1, temperature=6.0, dirichlet_alpha=1.0, seed=42,
    )
    mu_new, _, bounded = diagnostic.bounded_gaussian_components(
        mu, amplitude=1.0, tanh_temperature=5.0, sigma=1.0, seed=43,
    )
    assert torch.equal(mu.argmax(1), q.argmax(1))
    assert torch.equal(mu.argmax(1), mu_new.argmax(1))
    for x0 in (simplex, bounded):
        states = [diagnostic.linear_interpolation(x0, x1, t) for t in (0.0, 0.5, 1.0)]
        rows, _, _ = diagnostic.interpolation_statistics(
            states, [0.0, 0.5, 1.0], target, mu.argmax(1), void_index=2,
        )
        assert rows[-1]["gt_argmax_ratio"] == 1.0


class TinySource(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3))


def test_joint_checkpoint_loads_only_source_model(tmp_path, monkeypatch):
    expected = TinySource()
    expected.weight.data.copy_(torch.tensor([1.0, 2.0, 3.0]))
    checkpoint = tmp_path / "joint.pt"
    torch.save(
        {
            "source_model": expected.state_dict(),
            "model": {"endpoint.weight": torch.tensor([99.0])},
            "config": {"source": {"segformer_variant": "b1"}},
        },
        checkpoint,
    )
    built = TinySource()
    monkeypatch.setattr(visualize_simplex_source, "build_source_model", lambda config: built)
    config = {
        "source": {
            "segformer_variant": "b1", "pretrained": True,
            "checkpoint": None, "prior_type": "image_gaussian",
        }
    }
    loaded_checkpoint, loaded = diagnostic.load_source_checkpoint(
        config, Path(checkpoint), torch.device("cpu")
    )
    assert "model" in loaded_checkpoint
    torch.testing.assert_close(loaded.weight, expected.weight)
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
