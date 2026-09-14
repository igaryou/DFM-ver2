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

        def forward(self, image):
            self.calls += 1
            mu = torch.cat((image, image + 2.0), dim=1)
            x0 = mu + 7.0
            return x0, mu, torch.zeros_like(mu)

    captured = []

    def capture(mu, x0, path, foreground_channel, image):
        captured.append(
            (mu.clone(), x0.clone(), Path(path), foreground_channel, image)
        )

    monkeypatch.setattr(MODULE, "save_lidc_source_mu_x0", capture)
    source = CountingSource()
    loader = [
        {"image": torch.zeros(2, 1, 8, 8)},
        {"image": torch.ones(2, 1, 8, 8)},
    ]
    saved = MODULE.save_source_visualizations(
        source, loader, tmp_path, 3, _config(), torch.device("cpu")
    )

    assert saved == 3
    assert source.calls == 2
    assert [item[2].name for item in captured] == [
        "sample_0000.png", "sample_0001.png", "sample_0002.png"
    ]
    assert all(item[3] == 1 for item in captured)
    for mu, x0, _, _, _ in captured:
        torch.testing.assert_close(x0, mu + 7.0)


def test_parser_uses_requested_hyphenated_arguments():
    arguments = MODULE.build_parser().parse_args([
        "--config", "config.yaml",
        "--checkpoint", "epoch_0100.pt",
        "--output-dir", "outputs",
        "--num-visualizations", "12",
        "--split", "val",
    ])
    assert arguments.output_dir == "outputs"
    assert arguments.num_visualizations == 12
    assert arguments.split == "val"


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
