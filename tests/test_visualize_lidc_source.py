from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts/visualize_lidc_source.py"
SPEC = importlib.util.spec_from_file_location("visualize_lidc_source", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _config() -> dict:
    return {
        "runtime": {"amp": False, "amp_dtype": "bf16"},
    }


def test_save_source_visualizations_uses_same_forward_mu_x0_and_limit(
    tmp_path, monkeypatch
):
    class CountingSource(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward_statistics(self, image):
            self.calls += 1
            mu = torch.cat((image, image + 2.0), dim=1)
            return mu, torch.zeros_like(mu)

    captured = []
    visualized_x0_ids = []
    diagnosed_x0_ids = []
    original_diagnostics = MODULE.compute_lidc_source_diagnostics

    def capture(image, target, mu, x0_samples, path):
        visualized_x0_ids.append(id(x0_samples))
        captured.append(
            (mu.clone(), x0_samples.clone(), Path(path), image, target)
        )

    def capture_diagnostics(mu, logvar, x0_samples, target, **kwargs):
        diagnosed_x0_ids.append(id(x0_samples))
        return original_diagnostics(mu, logvar, x0_samples, target, **kwargs)

    monkeypatch.setattr(MODULE, "save_lidc_source_multisample", capture)
    monkeypatch.setattr(
        MODULE, "compute_lidc_source_diagnostics", capture_diagnostics
    )
    source = CountingSource()
    loader = [
        {
            "image": torch.zeros(2, 1, 8, 8),
            "target": torch.zeros(2, 8, 8, dtype=torch.long),
        },
        {
            "image": torch.ones(2, 1, 8, 8),
            "target": torch.ones(2, 8, 8, dtype=torch.long),
        },
    ]
    diagnostics = MODULE.save_source_visualizations(
        source, loader, tmp_path, 3, _config(), torch.device("cpu")
    )

    assert len(diagnostics) == 3
    assert source.calls == 2
    assert visualized_x0_ids == diagnosed_x0_ids
    assert [item[2].name for item in captured] == [
        "sample_0000.png", "sample_0001.png", "sample_0002.png"
    ]
    for mu, x0_samples, _, _, target in captured:
        assert x0_samples.shape == (16, 2, 8, 8)
        assert target.shape == (8, 8)
        assert not torch.equal(x0_samples[0], x0_samples[1])
    assert (tmp_path / "metrics/sample_0000.json").is_file()
    assert (tmp_path / "metrics/summary.json").is_file()


def test_parser_uses_requested_hyphenated_arguments():
    arguments = MODULE.build_parser().parse_args([
        "--config", "config.yaml",
        "--checkpoint", "epoch_0100.pt",
        "--output-dir", "outputs",
        "--num-visualizations", "12",
        "--num-source-samples", "5",
        "--split", "val",
        "--seed", "123",
    ])
    assert arguments.output_dir == "outputs"
    assert arguments.num_visualizations == 12
    assert arguments.num_source_samples == 5
    assert arguments.split == "val"
    assert arguments.seed == 123


def test_source_sampling_is_reproducible_per_sample_and_checkpoint_independent():
    mu_a = torch.zeros(2, 8, 8)
    mu_b = torch.full((2, 8, 8), 10.0)
    logvar = torch.zeros_like(mu_a)
    x0_a, epsilon_a = MODULE.sample_source_distribution(
        mu_a, logvar, 5, seed=42, sample_index=3
    )
    x0_a_repeat, epsilon_repeat = MODULE.sample_source_distribution(
        mu_a, logvar, 5, seed=42, sample_index=3
    )
    x0_b, epsilon_b = MODULE.sample_source_distribution(
        mu_b, logvar, 5, seed=42, sample_index=3
    )
    _, epsilon_other_sample = MODULE.sample_source_distribution(
        mu_a, logvar, 5, seed=42, sample_index=4
    )

    torch.testing.assert_close(x0_a, x0_a_repeat)
    torch.testing.assert_close(epsilon_a, epsilon_repeat)
    torch.testing.assert_close(epsilon_a, epsilon_b)
    torch.testing.assert_close(x0_b - mu_b.unsqueeze(0), epsilon_b)
    assert not torch.equal(epsilon_a, epsilon_other_sample)
    assert not torch.equal(x0_a[0], x0_a[1])


def test_load_checkpoint_source_loads_only_source_model(tmp_path, monkeypatch):
    expected = torch.full((2, 1, 1, 1), 3.0)
    checkpoint_path = tmp_path / "epoch_0100.pt"
    torch.save({
        "model": {"unused": torch.tensor(99.0)},
        "source_model": {"weight": expected},
    }, checkpoint_path)
    source = torch.nn.Conv2d(1, 2, 1, bias=False)
    endpoint = object()
    monkeypatch.setattr(
        MODULE, "build_models", lambda config, device: (endpoint, source)
    )
    config = {
        "source": {},
        "checkpoint": {"strict_model": True},
    }

    loaded = MODULE.load_checkpoint_source(
        config, checkpoint_path, torch.device("cpu")
    )

    assert loaded is source
    assert not loaded.training
    torch.testing.assert_close(loaded.weight, expected)
