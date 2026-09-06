from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import visualize_source_interpolation as diagnostic
import visualize_simplex_source
from adaptive_path import adaptive_lambda, normalize_entropy, shannon_entropy
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


def test_power_interpolation_endpoints_linear_compatibility_and_square_path():
    x0 = torch.randn(1, 3, 1, 2)
    _, _, x1 = _inputs()
    assert torch.equal(diagnostic.linear_interpolation(x0, x1, 0.0), x0)
    assert torch.equal(diagnostic.linear_interpolation(x0, x1, 1.0), x1)
    legacy = diagnostic.linear_interpolation(x0, x1, 0.5)
    explicit_linear = diagnostic.linear_interpolation(
        x0, x1, 0.5, path_exponent=1.0
    )
    torch.testing.assert_close(legacy, 0.5 * x1 + 0.5 * x0)
    torch.testing.assert_close(explicit_linear, legacy)
    torch.testing.assert_close(
        diagnostic.linear_interpolation(x0, x1, 0.5, path_exponent=2.0),
        0.25 * x1 + 0.75 * x0,
    )
    with pytest.raises(ValueError, match="path_exponent must be positive"):
        diagnostic.linear_interpolation(x0, x1, 0.5, path_exponent=0.0)


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


def test_batched_statistics_batch_one_matches_legacy_helper_exactly():
    mu, target, x1 = _inputs()
    source_prediction = torch.tensor([[[0, 0]]])
    states = [mu, diagnostic.linear_interpolation(mu, x1, 0.5), x1]
    times = [0.0, 0.5, 1.0]
    expected = diagnostic.interpolation_statistics(
        states, times, target, source_prediction, void_index=2,
    )
    actual = diagnostic.batched_interpolation_statistics(
        states, times, target, source_prediction, void_index=2,
    )[0]
    assert actual[0] == expected[0]
    assert torch.equal(actual[1], expected[1])
    assert actual[2] == expected[2]


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
    assert args.batch_size == 16
    assert args.path_type == "power"
    assert args.path_exponent == 1.0
    assert args.entropy_beta is None
    assert args.entropy_scheduler == "additive"
    assert args.difficulty_gamma == 1.0
    assert args.entropy_normalization is None
    assert args.variance_type == "fixed"
    assert args.variance_rho == 0.8
    raw = diagnostic.parse_args([
        "--config", "x", "--checkpoint", "y", "--output-dir", "z",
        "--mode", "raw_gaussian",
    ])
    assert raw.mode == diagnostic.RAW_GAUSSIAN_MODE
    assert diagnostic.MODES == ("simplex", "bounded_gaussian")
    with pytest.raises(ValueError, match="batch-size must be positive"):
        diagnostic.parse_args([
            "--config", "x", "--checkpoint", "y", "--output-dir", "z",
            "--batch-size", "0",
        ])
    with pytest.raises(ValueError, match="path-exponent must be positive"):
        diagnostic.parse_args([
            "--config", "x", "--checkpoint", "y", "--output-dir", "z",
            "--path-exponent", "0",
        ])
    with pytest.raises(ValueError, match=r"in \[0,1\].*additive"):
        diagnostic.parse_args([
            "--config", "x", "--checkpoint", "y", "--output-dir", "z",
            "--entropy-scheduler", "additive", "--entropy-beta", "1.1",
        ])
    exponential = diagnostic.parse_args([
        "--config", "x", "--checkpoint", "y", "--output-dir", "z",
        "--entropy-scheduler", "exponential", "--entropy-beta", "1.5",
    ])
    assert exponential.entropy_beta == 1.5
    with pytest.raises(ValueError, match="non-negative.*exponential"):
        diagnostic.parse_args([
            "--config", "x", "--checkpoint", "y", "--output-dir", "z",
            "--entropy-scheduler", "exponential", "--entropy-beta", "-0.1",
        ])
    with pytest.raises(ValueError, match="difficulty-gamma must be positive"):
        diagnostic.parse_args([
            "--config", "x", "--checkpoint", "y", "--output-dir", "z",
            "--difficulty-gamma", "0",
        ])
    with pytest.raises(ValueError, match="entropy-eps must be positive"):
        diagnostic.parse_args([
            "--config", "x", "--checkpoint", "y", "--output-dir", "z",
            "--entropy-eps", "0",
        ])
    with pytest.raises(ValueError, match="variance-rho must satisfy"):
        diagnostic.parse_args([
            "--config", "x", "--checkpoint", "y", "--output-dir", "z",
            "--variance-rho", "1",
        ])
    with pytest.raises(ValueError, match="requires bounded_gaussian"):
        diagnostic.parse_args([
            "--config", "x", "--checkpoint", "y", "--output-dir", "z",
            "--mode", "raw_gaussian",
            "--variance-type", "entropy_adaptive",
        ])
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


class _SyntheticVisualizationDataset:
    def __init__(self, length: int = 32):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        target = (
            torch.arange(24, dtype=torch.long).reshape(4, 6) + index
        ) % 19
        return {
            "image": torch.full((3, 4, 6), index / self.length),
            "target": target,
            "sample_id": f"synthetic-{index:03d}",
        }


class _CountingVisualizationSource(nn.Module):
    fixed_std = 1.0

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.batch_sizes: list[int] = []

    def forward_statistics(self, image):
        self.batch_sizes.append(image.shape[0])
        class_offsets = torch.linspace(-0.2, 0.2, 20, device=image.device)[
            None, :, None, None
        ]
        spatial_scale = 0.5 + image[:, :1] + torch.linspace(
            0.0, 2.0, image.shape[-1], device=image.device
        )[None, None, None, :]
        mu = class_offsets * spatial_scale + self.anchor * 0.0
        return mu.expand(-1, -1, image.shape[-2], -1), torch.zeros_like(
            mu.expand(-1, -1, image.shape[-2], -1)
        )


def _run_synthetic_visualizer(
    monkeypatch, tmp_path, *, batch_size: int, mode: str, num_images: int,
    path_exponent: float = 1.0, path_type: str = "power",
    entropy_beta: float | None = None, variance_type: str = "fixed",
    variance_rho: float = 0.8,
):
    dataset = _SyntheticVisualizationDataset()
    source = _CountingVisualizationSource()
    augment_values = []
    figure_sample_ids = []
    figure_x0 = {}
    scheduler_paths = []

    def fake_build_dataset(config, split, augment):
        del config, split
        augment_values.append(augment)
        return dataset

    def record_mode_figure(*args, **kwargs):
        del kwargs
        sample_id = args[7]["sample_id"]
        figure_sample_ids.append(sample_id)
        figure_x0[sample_id] = args[4].detach().cpu().clone()

    monkeypatch.setattr(diagnostic, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(diagnostic, "resolve_device", lambda value: torch.device("cpu"))
    monkeypatch.setattr(diagnostic, "resolve_checkpoint", lambda *args: Path("unused.pt"))
    monkeypatch.setattr(
        diagnostic, "load_source_checkpoint",
        lambda config, checkpoint, device: ({"stage": "joint_training"}, source),
    )
    monkeypatch.setattr(
        diagnostic, "_inverse_normalized_image", lambda image, config: image
    )
    monkeypatch.setattr(diagnostic, "_state_to_display", lambda state, sample: state)
    monkeypatch.setattr(diagnostic, "_save_mode_figure", record_mode_figure)
    monkeypatch.setattr(
        diagnostic, "_save_scheduler_figure",
        lambda path, *args, **kwargs: scheduler_paths.append(path),
    )
    monkeypatch.setattr(diagnostic, "_save_comparison", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        diagnostic, "_save_variance_figure", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        diagnostic, "_save_hard_target_comparison", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(diagnostic, "_save_summary_plots", lambda *args: None)
    args = diagnostic.parse_args([
        "--config", "configs/cityscapes/psd/joint_bounded_gaussian_b1_ce_160k.yaml",
        "--checkpoint", "unused.pt", "--output-dir", str(tmp_path),
        "--device", "cpu", "--mode", mode, "--num-images", str(num_images),
        "--batch-size", str(batch_size), "--times", "0", "0.5", "1",
        "--path-exponent", str(path_exponent), "--path-type", path_type,
        "--variance-type", variance_type, "--variance-rho", str(variance_rho),
        *([] if entropy_beta is None else [
            "--entropy-beta", str(entropy_beta)
        ]),
        "--seed", "17",
    ])
    summary = diagnostic.run(args)
    assert augment_values == [False]
    return (
        summary, source.batch_sizes, figure_sample_ids, figure_x0,
        scheduler_paths,
    )


def test_entropy_scheduler_config_beta_validation_depends_on_scheduler():
    config = {
        "flow": {"path": {
            "scheduler": {"beta": 1.5},
            "entropy": {
                "normalization": "rank", "eps": 1.0e-8,
                "zscore_clip": 3.0, "exclude_ignore": True,
            },
        }},
    }
    exponential = diagnostic.resolve_entropy_scheduler_settings(
        _adaptive_args("--entropy-scheduler", "exponential"), config
    )
    assert exponential["beta"] == 1.5
    with pytest.raises(ValueError, match=r"in \[0,1\].*additive"):
        diagnostic.resolve_entropy_scheduler_settings(
            _adaptive_args("--entropy-scheduler", "additive"), config
        )


@pytest.mark.parametrize(
    "mode", ["raw_gaussian", "bounded_gaussian", "simplex"]
)
def test_batch_size_one_and_four_have_identical_trajectory_statistics(
    monkeypatch, tmp_path, mode,
):
    single, single_batches, single_ids, single_x0, _ = _run_synthetic_visualizer(
        monkeypatch, tmp_path / f"{mode}-b1", batch_size=1,
        mode=mode, num_images=8, path_exponent=2.0,
    )
    batched, batched_batches, batched_ids, batched_x0, _ = _run_synthetic_visualizer(
        monkeypatch, tmp_path / f"{mode}-b4", batch_size=4,
        mode=mode, num_images=8, path_exponent=2.0,
    )
    assert json.dumps(single["trajectory"], sort_keys=True) == json.dumps(
        batched["trajectory"], sort_keys=True
    )
    assert single["indices"] == batched["indices"] == list(range(8))
    assert single["path_exponent"] == batched["path_exponent"] == 2.0
    assert single_batches == [1] * 8
    assert batched_batches == [4, 4]
    assert single_ids == batched_ids == [
        f"synthetic-{index:03d}" for index in range(8)
    ]
    assert single_x0.keys() == batched_x0.keys()
    for sample_id in single_x0:
        assert torch.equal(single_x0[sample_id], batched_x0[sample_id])


def test_32_images_with_batch_size_four_use_eight_source_forwards(
    monkeypatch, tmp_path,
):
    summary, batch_sizes, figure_sample_ids, _, _ = _run_synthetic_visualizer(
        monkeypatch, tmp_path / "thirty-two", batch_size=4,
        mode="raw_gaussian", num_images=32,
    )
    assert batch_sizes == [4] * 8
    assert summary["batch_size"] == 4
    assert summary["source_forward_batches"] == 8
    assert len(summary["indices"]) == 32
    assert figure_sample_ids == [
        f"synthetic-{index:03d}" for index in range(32)
    ]



def _adaptive_args(*extra: str):
    return diagnostic.parse_args([
        "--config", "x", "--checkpoint", "y", "--output-dir", "z",
        "--path-type", "entropy_adaptive", *extra,
    ])


def test_adaptive_entropy_is_always_computed_from_raw_logits():
    mu_raw = torch.tensor([
        [[[4.0, 0.0]], [[0.0, 1.0]], [[-3.0, 2.0]]]
    ])
    target = torch.tensor([[[0, 1]]])
    settings = {
        "beta": 0.5, "normalization": "rank", "eps": 1.0e-8,
        "zscore_clip": 3.0, "exclude_ignore": True,
    }
    entropy, difficulty, _ = (
        diagnostic.source_entropy_difficulty_from_raw_logits(
            mu_raw, target=target, void_index=2, settings=settings,
        )
    )
    probability = torch.softmax(mu_raw.float(), dim=1)
    expected_entropy = -(
        probability * probability.clamp_min(1.0e-8).log()
    ).sum(dim=1)
    torch.testing.assert_close(entropy, expected_entropy)

    bounded_state = 1.0 * torch.tanh(mu_raw / 5.0)
    bounded_entropy = shannon_entropy(bounded_state, representation="logits")
    simplex_probability = torch.softmax(mu_raw / 4.75, dim=1)
    simplex_entropy = shannon_entropy(
        simplex_probability, representation="probability"
    )
    assert not torch.allclose(entropy, bounded_entropy)
    assert not torch.allclose(entropy, simplex_entropy)
    torch.testing.assert_close(
        difficulty,
        normalize_entropy(
            expected_entropy, "rank", valid_mask=target != 2,
            eps=1.0e-8, zscore_clip=3.0, num_classes=3,
        ),
    )


@pytest.mark.parametrize("normalization", ["rank", "mean", "zscore", "minmax"])
def test_visualizer_entropy_normalizations_match_production(normalization):
    mu_raw = torch.tensor([
        [[[3.0, 0.0, 1.0]], [[0.0, 2.0, -1.0]], [[-2.0, 0.0, 2.0]]]
    ])
    target = torch.tensor([[[0, 1, 2]]])
    settings = {
        "beta": 0.5, "normalization": normalization, "eps": 1.0e-8,
        "zscore_clip": 2.0, "exclude_ignore": False,
    }
    entropy, difficulty, _ = (
        diagnostic.source_entropy_difficulty_from_raw_logits(
            mu_raw, target=target, void_index=2, settings=settings,
        )
    )
    expected = normalize_entropy(
        shannon_entropy(mu_raw, representation="logits"), normalization,
        eps=1.0e-8, zscore_clip=2.0, num_classes=3,
    )
    torch.testing.assert_close(difficulty, expected)
    assert difficulty.min() >= -1 and difficulty.max() <= 1
    assert torch.isfinite(entropy).all()


def test_scheduler_is_independent_of_simplex_bounded_and_raw_x0():
    mu_raw = torch.randn(2, 4, 3, 5)
    target = torch.randint(0, 3, (2, 3, 5))
    settings = {
        "beta": 0.6, "normalization": "rank", "eps": 1.0e-8,
        "zscore_clip": 3.0, "exclude_ignore": True,
    }
    reference = diagnostic.source_entropy_difficulty_from_raw_logits(
        mu_raw, target=target, void_index=3, settings=settings,
    )
    # Construct every x0 mode, but deliberately keep the scheduler input raw.
    sample_image_simplex_components(
        mu_raw, lambda_value=0.2, temperature=4.75,
        dirichlet_alpha=1.0, seed=1,
    )
    diagnostic.bounded_gaussian_components(
        mu_raw, amplitude=1.0, tanh_temperature=5.0, sigma=1.0, seed=2,
    )
    diagnostic.raw_gaussian_components(mu_raw, sigma=1.0, seed=1)
    for _mode in ("simplex", "bounded_gaussian", "raw_gaussian"):
        current = diagnostic.source_entropy_difficulty_from_raw_logits(
            mu_raw, target=target, void_index=3, settings=settings,
        )
        torch.testing.assert_close(current[0], reference[0])
        torch.testing.assert_close(current[1], reference[1])
        torch.testing.assert_close(
            adaptive_lambda(torch.tensor([0.4, 0.4]), current[1], beta=0.6),
            adaptive_lambda(torch.tensor([0.4, 0.4]), reference[1], beta=0.6),
        )


def test_difficulty_emphasis_matches_signed_power_and_gamma_one_regression():
    difficulty = torch.tensor([[[-1.0, -0.25, 0.0, 0.25, 1.0]]])
    torch.testing.assert_close(
        diagnostic.emphasize_difficulty(difficulty, 1.0), difficulty,
        rtol=0, atol=0,
    )
    expected = difficulty.sign() * difficulty.abs().sqrt()
    actual = diagnostic.emphasize_difficulty(difficulty, 0.5)
    torch.testing.assert_close(actual, expected)
    assert actual.min() >= -1 and actual.max() <= 1


@pytest.mark.parametrize("scheduler", ["additive", "exponential"])
def test_entropy_scheduler_formulas_endpoints_and_direction(scheduler):
    time = torch.tensor([0.4])
    difficulty = torch.tensor([[[-0.8, 0.0, 0.7]]])
    actual = diagnostic.entropy_scheduler_lambda(
        time, difficulty, beta=0.5, scheduler=scheduler,
    )
    t = time[:, None, None]
    expected = (
        t - 0.5 * t * (1.0 - t) * difficulty
        if scheduler == "additive"
        else t.pow(torch.exp(0.5 * difficulty))
    )
    torch.testing.assert_close(actual, expected)
    assert actual[0, 0, 0] > time[0]
    assert actual[0, 0, 2] < time[0]
    for endpoint in (0.0, 1.0):
        coefficient = diagnostic.entropy_scheduler_lambda(
            torch.tensor([endpoint]), difficulty, beta=0.5,
            scheduler=scheduler,
        )
        torch.testing.assert_close(
            coefficient, torch.full_like(difficulty, endpoint)
        )


def test_adaptive_path_endpoints_beta_zero_and_entropy_direction():
    x0 = torch.tensor([[[[2.0, -1.0]], [[0.0, 3.0]]]])
    x1 = torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]])
    difficulty = torch.tensor([[[-0.8, 0.7]]])
    at_zero, lambda_zero = diagnostic.interpolation_path(
        x0, x1, 0.0, path_type="entropy_adaptive", path_exponent=9.0,
        difficulty=difficulty, entropy_beta=0.5,
    )
    at_one, lambda_one = diagnostic.interpolation_path(
        x0, x1, 1.0, path_type="entropy_adaptive", path_exponent=9.0,
        difficulty=difficulty, entropy_beta=0.5,
    )
    torch.testing.assert_close(at_zero, x0)
    torch.testing.assert_close(at_one, x1)
    torch.testing.assert_close(lambda_zero, torch.zeros_like(difficulty))
    torch.testing.assert_close(lambda_one, torch.ones_like(difficulty))
    adaptive, coefficient = diagnostic.interpolation_path(
        x0, x1, 0.4, path_type="entropy_adaptive", path_exponent=2.0,
        difficulty=difficulty, entropy_beta=0.5,
    )
    assert coefficient[0, 0, 0] > 0.4
    assert coefficient[0, 0, 1] < 0.4
    beta_zero, coefficient_zero = diagnostic.interpolation_path(
        x0, x1, 0.4, path_type="entropy_adaptive", path_exponent=2.0,
        difficulty=difficulty, entropy_beta=0.0,
    )
    expected_linear = diagnostic.linear_interpolation(x0, x1, 0.4)
    torch.testing.assert_close(beta_zero, expected_linear, rtol=0, atol=0)
    torch.testing.assert_close(
        coefficient_zero, torch.full_like(difficulty, 0.4), rtol=0, atol=0
    )
    expected_adaptive = (
        coefficient[:, None] * x1 + (1 - coefficient[:, None]) * x0
    )
    torch.testing.assert_close(adaptive, expected_adaptive)


def test_entropy_scheduler_cli_overrides_and_config_fallbacks():
    config = {
        "flow": {"path": {
            "scheduler": {"beta": 0.35},
            "entropy": {
                "normalization": "minmax", "eps": 2.0e-7,
                "zscore_clip": 1.7, "exclude_ignore": False,
            },
        }},
    }
    fallback = diagnostic.resolve_entropy_scheduler_settings(
        _adaptive_args(), config
    )
    assert fallback == {
        "beta": 0.35, "normalization": "minmax", "eps": 2.0e-7,
        "zscore_clip": 1.7, "exclude_ignore": False,
        "scheduler": "additive", "difficulty_gamma": 1.0,
    }
    overridden = diagnostic.resolve_entropy_scheduler_settings(
        _adaptive_args(
            "--entropy-beta", "0.7", "--entropy-normalization", "zscore",
            "--entropy-eps", "1e-6", "--entropy-zscore-clip", "2.5",
            "--entropy-exclude-ignore", "--entropy-scheduler", "exponential",
            "--difficulty-gamma", "0.5",
        ),
        config,
    )
    assert overridden == {
        "beta": 0.7, "normalization": "zscore", "eps": 1.0e-6,
        "zscore_clip": 2.5, "exclude_ignore": True,
        "scheduler": "exponential", "difficulty_gamma": 0.5,
    }


@pytest.mark.parametrize(
    "mode", ["raw_gaussian", "bounded_gaussian", "simplex"]
)
def test_adaptive_batching_is_deterministic_and_exports_scheduler_stats(
    monkeypatch, tmp_path, mode,
):
    single, _, _, single_x0, single_scheduler = _run_synthetic_visualizer(
        monkeypatch, tmp_path / f"adaptive-{mode}-b1", batch_size=1,
        mode=mode, num_images=4, path_type="entropy_adaptive",
        entropy_beta=0.5,
    )
    batched, batch_sizes, _, batched_x0, batched_scheduler = (
        _run_synthetic_visualizer(
            monkeypatch, tmp_path / f"adaptive-{mode}-b4", batch_size=4,
            mode=mode, num_images=4, path_type="entropy_adaptive",
            entropy_beta=0.5,
        )
    )
    assert json.dumps(single["trajectory"], sort_keys=True) == json.dumps(
        batched["trajectory"], sort_keys=True
    )
    for sample_id in single_x0:
        assert torch.equal(single_x0[sample_id], batched_x0[sample_id])
    assert batch_sizes == [4]
    assert len(single_scheduler) == len(batched_scheduler) == 4
    assert all(f"scheduler/{mode}" in str(path) for path in batched_scheduler)
    assert batched["path_type"] == "entropy_adaptive"
    assert batched["entropy_scheduler"]["entropy_source"] == (
        "softmax(raw_source_logits_mu)"
    )
    assert batched["entropy_scheduler"]["beta"] == 0.5
    for row in batched["trajectory"]:
        for key in (
            "lambda_mean", "lambda_std", "lambda_min", "lambda_max",
            "lambda_easy_mean", "lambda_hard_mean", "difficulty_mean",
            "difficulty_std", "entropy_mean", "entropy_std",
        ):
            assert key in row



def test_visualizer_adaptive_variance_is_batch_deterministic_and_path_independent(
    monkeypatch, tmp_path,
):
    variance_only, _, _, x0_single, _ = _run_synthetic_visualizer(
        monkeypatch, tmp_path / "variance-only-b1", batch_size=1,
        mode="bounded_gaussian", num_images=4,
        variance_type="entropy_adaptive", variance_rho=0.8,
    )
    combined, batches, _, x0_batched, _ = _run_synthetic_visualizer(
        monkeypatch, tmp_path / "variance-path-b4", batch_size=4,
        mode="bounded_gaussian", num_images=4, path_type="entropy_adaptive",
        entropy_beta=0.5, variance_type="entropy_adaptive", variance_rho=0.8,
    )
    assert batches == [4]
    for sample_id in x0_single:
        assert torch.equal(x0_single[sample_id], x0_batched[sample_id])
    assert variance_only["bounded_gaussian_variance"]["type"] == (
        "entropy_adaptive"
    )
    assert combined["path_type"] == "entropy_adaptive"
    assert combined["bounded_gaussian_variance"]["gt_independent"] is True


def test_adaptive_variance_figure_smoke(tmp_path):
    shape = (3, 4)
    diagnostic._save_variance_figure(
        tmp_path / "variance.png", torch.rand(3, *shape),
        torch.rand(*shape), torch.linspace(-1, 1, 12).reshape(shape),
        torch.rand(*shape) + 0.1, torch.rand(*shape) + 0.1,
        torch.rand(1, 20, *shape), {"target": torch.zeros(shape)},
    )
    assert (tmp_path / "variance.png").is_file()


def test_adaptive_scheduler_statistics_direction_and_figure_smoke(tmp_path):
    entropy = torch.tensor([[[0.1, 0.4], [0.8, 1.2]]])
    difficulty = torch.tensor([[[-1.0, -0.3], [0.4, 0.9]]])
    coefficient = adaptive_lambda(
        torch.tensor([0.5]), difficulty, beta=0.5
    )
    stats = diagnostic.adaptive_scheduler_statistics(
        entropy, difficulty, coefficient, torch.ones_like(entropy).bool()
    )[0]
    assert stats["lambda_easy_mean"] > 0.5
    assert stats["lambda_hard_mean"] < 0.5
    assert stats["lambda_min"] <= stats["lambda_mean"] <= stats["lambda_max"]
    output = tmp_path / "scheduler" / "sample_0000.png"
    diagnostic._save_scheduler_figure(
        output,
        torch.rand(3, 2, 2),
        torch.tensor([[0, 1], [1, 0]]),
        entropy[0], difficulty[0], [coefficient[0]], [0.5],
        "Entropy Adaptive | beta=0.5 | normalization=rank",
    )
    assert output.is_file() and output.stat().st_size > 0
