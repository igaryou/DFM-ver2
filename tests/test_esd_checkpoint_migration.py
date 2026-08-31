from pathlib import Path

import torch

from checkpoint import checkpoint_payload, initialize_or_resume, save_checkpoint
from config import load_config
from dfm_stabilization import ESDTimeWeightNetwork, PSDTimeWeightNetwork
from trainer import build_scheduler


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "joint_esd_cityscapes_swin_t_dfm_recipe_resume.yaml"


def _config(tmp_path):
    config = load_config(CONFIG)
    config["checkpoint"]["resume"] = str(tmp_path / "resume.pt")
    return config


def _optimizer(endpoint, source, weight=None):
    groups = [
        {"params": endpoint.parameters(), "lr": 1.0e-4, "name": "model"},
        {"params": source.parameters(), "lr": 5.0e-5, "name": "source"},
    ]
    if weight is not None:
        groups.append({
            "params": weight.parameters(), "lr": 1.0e-4,
            "weight_decay": 0.0, "name": "esd_weight",
        })
    return torch.optim.AdamW(groups, lr=1.0e-4, weight_decay=1.0e-3)


def _step(endpoint, source, weight, optimizer):
    loss = endpoint(torch.ones(2, 3)).sum() + source(torch.ones(2, 3)).sum()
    if weight is not None:
        loss = loss + weight(
            torch.tensor([0.2, 0.5]), torch.tensor([0.6, 0.9])
        ).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_96k_checkpoint_adds_fresh_esd_group_at_current_scheduler_factor(tmp_path):
    config = _config(tmp_path)
    old_endpoint = torch.nn.Linear(3, 2)
    old_source = torch.nn.Linear(3, 2)
    old_optimizer = _optimizer(old_endpoint, old_source)
    old_scheduler = build_scheduler(config, old_optimizer)
    _step(old_endpoint, old_source, None, old_optimizer)
    old_scheduler.step(96000)
    expected_model_lr = old_optimizer.param_groups[0]["lr"]
    payload = checkpoint_payload(
        config=config, epoch=17, global_step=96000,
        model=old_endpoint, source_model=old_source,
        optimizer=old_optimizer, scheduler=old_scheduler, scaler=None, metrics={},
    )
    save_checkpoint(payload, tmp_path, "resume.pt")

    endpoint = torch.nn.Linear(3, 2)
    source = torch.nn.Linear(3, 2)
    weight = ESDTimeWeightNetwork()
    optimizer = _optimizer(endpoint, source, weight)
    scheduler = build_scheduler(config, optimizer)
    state = initialize_or_resume(
        config, endpoint, source, optimizer, scheduler, None,
        consistency_weight_model=weight,
    )
    assert state.global_step == 96000
    torch.testing.assert_close(endpoint.weight, old_endpoint.weight)
    torch.testing.assert_close(source.weight, old_source.weight)
    assert optimizer.state[endpoint.weight]
    assert optimizer.state[source.weight]
    assert all(parameter not in optimizer.state for parameter in weight.parameters())
    assert optimizer.param_groups[0]["lr"] == expected_model_lr
    assert optimizer.param_groups[2]["lr"] == expected_model_lr
    assert scheduler.last_epoch == 96000


def test_esd_weight_checkpoint_strict_round_trip(tmp_path):
    config = _config(tmp_path)
    old_endpoint = torch.nn.Linear(3, 2)
    old_source = torch.nn.Linear(3, 2)
    old_weight = ESDTimeWeightNetwork()
    old_optimizer = _optimizer(old_endpoint, old_source, old_weight)
    old_scheduler = build_scheduler(config, old_optimizer)
    _step(old_endpoint, old_source, old_weight, old_optimizer)
    old_scheduler.step()
    payload = checkpoint_payload(
        config=config, epoch=18, global_step=96001,
        model=old_endpoint, source_model=old_source,
        consistency_weight_model=old_weight,
        optimizer=old_optimizer, scheduler=old_scheduler, scaler=None, metrics={},
    )
    assert "consistency_weight_model" in payload
    assert "psd_weight_model" not in payload
    save_checkpoint(payload, tmp_path, "resume.pt")

    endpoint = torch.nn.Linear(3, 2)
    source = torch.nn.Linear(3, 2)
    weight = ESDTimeWeightNetwork()
    optimizer = _optimizer(endpoint, source, weight)
    scheduler = build_scheduler(config, optimizer)
    state = initialize_or_resume(
        config, endpoint, source, optimizer, scheduler, None,
        consistency_weight_model=weight,
    )
    assert state.global_step == 96001
    for actual, expected in zip(weight.parameters(), old_weight.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert optimizer.state[actual]


def test_new_psd_payload_also_emits_legacy_weight_key():
    config = load_config(
        ROOT / "configs" / "joint_psd_cityscapes_swin_t_dfm_recipe.yaml"
    )
    endpoint = torch.nn.Linear(3, 2)
    source = torch.nn.Linear(3, 2)
    weight = PSDTimeWeightNetwork()
    payload = checkpoint_payload(
        config=config, epoch=1, global_step=1,
        model=endpoint, source_model=source,
        consistency_weight_model=weight,
        optimizer=None, scheduler=None, scaler=None, metrics={},
    )
    assert payload["consistency_weight_model"].keys() == payload["psd_weight_model"].keys()
    for key in payload["consistency_weight_model"]:
        torch.testing.assert_close(
            payload["consistency_weight_model"][key],
            payload["psd_weight_model"][key], rtol=0, atol=0,
        )
