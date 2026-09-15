from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1] / "scripts/visualize_lidc_gt_best_predictions.py"
)
SPEC = importlib.util.spec_from_file_location("lidc_gt_best_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_script_reuses_final_stochastic_inference_helper(tmp_path, monkeypatch):
    calls = []

    def fake_sample(model, source, image, config, sample_counts):
        calls.append((model, source, sample_counts, config["evaluation"]["num_steps"]))
        predictions = torch.zeros(1, sample_counts[0], 4, 4, dtype=torch.long)
        predictions[:, 1, :2] = 1
        return predictions, predictions[:, 0]

    monkeypatch.setattr(MODULE, "sample_lidc_distribution", fake_sample)
    model = torch.nn.Identity()
    source = torch.nn.Identity()
    loader = [{
        "image": torch.zeros(1, 1, 4, 4),
        "masks": torch.zeros(1, 4, 4, 4, dtype=torch.long),
        "sample_id": ["sample-a"],
    }]
    config = {
        "evaluation": {"num_steps": 9},
        "runtime": {"amp": False, "amp_dtype": "bf16"},
    }
    results = MODULE.run_best_prediction_diagnostics(
        model, source, loader, tmp_path,
        config=config, device=torch.device("cpu"),
        num_visualizations=1, num_samples=3, num_steps=1, seed=42,
    )

    assert len(results) == 1
    assert calls == [(model, source, [3], 1)]
    assert results[0]["num_samples"] == 3
    assert (tmp_path / "sample_0000.png").is_file()
    assert (tmp_path / "metrics/sample_0000.json").is_file()
    assert (tmp_path / "metrics/summary.json").is_file()


def test_cli_contract_is_preserved():
    arguments = MODULE.build_parser().parse_args([
        "--config", "config.yaml",
        "--checkpoint", "checkpoint.pt",
        "--output-dir", "output",
        "--num-visualizations", "8",
        "--num-samples", "16",
        "--split", "val",
        "--num-steps", "2",
        "--seed", "7",
    ])
    assert arguments.num_visualizations == 8
    assert arguments.num_samples == 16
    assert arguments.split == "val"
    assert arguments.num_steps == 2
    assert arguments.seed == 7


def test_checkpoint_loader_restores_endpoint_and_source_only(tmp_path, monkeypatch):
    endpoint_weight = torch.full((2, 1, 1, 1), 2.0)
    source_weight = torch.full((2, 1, 1, 1), 3.0)
    checkpoint = tmp_path / "epoch_0010.pt"
    torch.save({
        "model": {"weight": endpoint_weight},
        "source_model": {"weight": source_weight},
        "epoch": 10,
        "optimizer": {"unused": True},
    }, checkpoint)
    endpoint = torch.nn.Conv2d(1, 2, 1, bias=False)
    source = torch.nn.Conv2d(1, 2, 1, bias=False)
    monkeypatch.setattr(
        MODULE, "build_models", lambda config, device: (endpoint, source)
    )
    config = {"source": {}, "checkpoint": {"strict_model": True}}

    loaded_endpoint, loaded_source, epoch = MODULE.load_models(
        config, checkpoint, torch.device("cpu")
    )

    torch.testing.assert_close(loaded_endpoint.weight, endpoint_weight)
    torch.testing.assert_close(loaded_source.weight, source_weight)
    assert not loaded_endpoint.training
    assert not loaded_source.training
    assert epoch == 10
