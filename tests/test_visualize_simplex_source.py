from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

import visualize_simplex_source as diagnostic
from discrete_flow_maps import (
    sample_image_simplex_components,
    sample_symmetric_dirichlet,
)


def test_checkpoint_resolution_prefers_best_and_rejects_ambiguity(tmp_path):
    latest = tmp_path / "latest.pt"
    best = tmp_path / "best.pt"
    latest.touch(); best.touch()
    assert diagnostic.resolve_checkpoint(None, str(tmp_path)) == best.resolve()
    best.unlink(); latest.unlink()
    (tmp_path / "a.pt").touch(); (tmp_path / "b.pt").touch()
    with pytest.raises(ValueError, match="Ambiguous"):
        diagnostic.resolve_checkpoint(None, str(tmp_path))


class TinySource(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3))


def test_source_only_checkpoint_loader_strict_loads_and_freezes(tmp_path, monkeypatch):
    expected = TinySource()
    expected.weight.data.copy_(torch.tensor([1.0, 2.0, 3.0]))
    checkpoint = tmp_path / "source.pt"
    torch.save({
        "source_model": expected.state_dict(),
        "config": {"source": {"segformer_variant": "b1"}},
    }, checkpoint)
    built = TinySource()
    monkeypatch.setattr(diagnostic, "build_source_model", lambda config: built)
    config = {
        "source": {
            "segformer_variant": "b1", "pretrained": True,
            "checkpoint": None, "prior_type": "image_gaussian",
        }
    }
    _, loaded = diagnostic.load_source_checkpoint(config, checkpoint, torch.device("cpu"))
    torch.testing.assert_close(loaded.weight, expected.weight)
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())


def test_seeded_dirichlet_is_repeatable_and_does_not_change_global_rng():
    shape = (1, 5, 3, 4)
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    first = sample_symmetric_dirichlet(shape, 0.5, device=torch.device("cpu"), seed=9)
    after = torch.random.get_rng_state()
    second = sample_symmetric_dirichlet(shape, 0.5, device=torch.device("cpu"), seed=9)
    torch.testing.assert_close(first, second)
    assert torch.equal(before, after)
    torch.testing.assert_close(first.sum(dim=1), torch.ones_like(first[:, 0]))


def test_lambda_sweep_reuses_noise_and_matches_formula():
    mu = torch.randn(1, 5, 2, 3)
    noise = sample_symmetric_dirichlet(
        tuple(mu.shape), 1.0, device=mu.device, seed=4
    )
    q, returned, z0 = sample_image_simplex_components(
        mu, lambda_value=0.25, temperature=1.3, dirichlet_alpha=1.0,
        dirichlet_noise=noise,
    )
    _, returned_second, z1 = sample_image_simplex_components(
        mu, lambda_value=0.75, temperature=1.3, dirichlet_alpha=1.0,
        dirichlet_noise=noise,
    )
    torch.testing.assert_close(returned, noise)
    torch.testing.assert_close(returned_second, noise)
    torch.testing.assert_close(z0, 0.25 * q + 0.75 * noise)
    torch.testing.assert_close(z1, 0.75 * q + 0.25 * noise)


def _maps(height=4, width=6):
    prediction = torch.zeros(height, width, dtype=torch.long)
    scalar = torch.full((height, width), 0.5)
    return {
        "source_prediction": prediction,
        "noise_prediction": prediction,
        "z0_prediction": prediction,
        "source_confidence": scalar,
        "z0_confidence": scalar,
        "noise_confidence": scalar,
        "l1": scalar,
        "l2": scalar,
        "source_entropy": scalar,
        "z0_entropy": scalar,
        "entropy_change": torch.zeros_like(scalar),
        "flip": torch.zeros(height, width, dtype=torch.bool),
    }


def test_metrics_are_finite_bounded_and_entropy_is_valid():
    q = torch.softmax(torch.randn(1, 5, 4, 6), dim=1)
    noise = sample_symmetric_dirichlet(tuple(q.shape), 1.0, device=q.device, seed=1)
    _, _, z0 = sample_image_simplex_components(
        q.log(), lambda_value=0.6, temperature=1.0,
        dirichlet_alpha=1.0, dirichlet_noise=noise,
    )
    target = torch.randint(0, 5, (1, 4, 6))
    stats, _ = diagnostic.condition_statistics(q, noise, z0, target, void_index=4)
    assert all(torch.isfinite(torch.tensor(value)) for key, value in stats.items() if "incorrect" not in key and "correct" not in key)
    assert 0 <= stats["argmax_flip_ratio"] <= 1
    assert 0 <= stats["mean_source_entropy"] <= torch.log(torch.tensor(5.0))
    assert 0 <= stats["mean_z0_entropy"] <= torch.log(torch.tensor(5.0))


def test_visualization_smoke_writes_pngs(tmp_path):
    image = torch.rand(3, 4, 6)
    target = torch.zeros(4, 6, dtype=torch.long)
    maps = _maps()
    single = tmp_path / "single.png"
    entropy = tmp_path / "entropy.png"
    sweep = tmp_path / "sweep.png"
    diagnostic.save_single_figure(image, target, maps, single, dataset_name="cityscapes")
    diagnostic.save_entropy_figure(maps, entropy, classes=20)
    diagnostic.save_sweep_figure(
        image, target, maps["source_prediction"], [("lambda=0.5", maps)],
        sweep, dataset_name="cityscapes", sweep_type="lambda",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in (single, entropy, sweep))


def test_cli_argument_parsing_and_conflict_validation(tmp_path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.touch()
    args = diagnostic.parse_args([
        "--config", "config.yaml", "--checkpoint", str(checkpoint),
        "--output-dir", str(tmp_path / "out"), "--indices", "0", "10",
        "--lambda-values", "0", "0.5", "1", "--seed", "7",
    ])
    assert args.indices == [0, 10]
    assert args.lambda_values == [0.0, 0.5, 1.0]
    assert args.seed == 7
    with pytest.raises(ValueError, match="either scalar lambda"):
        diagnostic.parse_args([
            "--config", "config.yaml", "--checkpoint", str(checkpoint),
            "--output-dir", str(tmp_path / "out"),
            "--lambda", "0.8", "--lambda-values", "0", "1",
        ])
