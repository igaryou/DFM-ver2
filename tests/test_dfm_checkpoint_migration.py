import copy

import torch

from checkpoint import checkpoint_payload, initialize_or_resume, save_checkpoint
from config import load_config
from dfm_stabilization import PSDTimeWeightNetwork
from trainer import build_scheduler


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message % args if args else message)


def _optimizer(endpoint, source, weight=None):
    groups = [
        {"params": endpoint.parameters(), "lr": 1e-4, "name": "model"},
        {"params": source.parameters(), "lr": 5e-5, "name": "source"},
    ]
    if weight is not None:
        groups.append({
            "params": weight.parameters(), "lr": 1e-4,
            "weight_decay": 0.0, "name": "psd_weight",
        })
    return torch.optim.AdamW(groups, lr=1e-4, weight_decay=1e-3)


def _step(endpoint, source, weight, optimizer, scheduler):
    loss = endpoint(torch.ones(2, 3)).sum() + source(torch.ones(2, 3)).sum()
    if weight is not None:
        loss = loss + weight(torch.tensor([0.2, 0.8])).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def _config(tmp_path):
    config = load_config("configs/cityscapes/psd/swin_t_dfm_recipe.yaml")
    config["checkpoint"]["resume"] = str(tmp_path / "resume.pt")
    return config


def test_old_checkpoint_restores_existing_optimizer_and_initializes_new_group(tmp_path):
    config = _config(tmp_path)
    old_endpoint = torch.nn.Linear(3, 2)
    old_source = torch.nn.Linear(3, 2)
    old_optimizer = _optimizer(old_endpoint, old_source)
    old_scheduler = build_scheduler(config, old_optimizer)
    _step(old_endpoint, old_source, None, old_optimizer, old_scheduler)
    payload = checkpoint_payload(
        config=config, epoch=17, global_step=96000,
        model=old_endpoint, source_model=old_source,
        optimizer=old_optimizer, scheduler=old_scheduler, scaler=None, metrics={},
    )
    save_checkpoint(payload, tmp_path, "resume.pt")

    endpoint = torch.nn.Linear(3, 2)
    source = torch.nn.Linear(3, 2)
    weight = PSDTimeWeightNetwork()
    optimizer = _optimizer(endpoint, source, weight)
    scheduler = build_scheduler(config, optimizer)
    logger = _Logger()
    state = initialize_or_resume(
        config, endpoint, source, optimizer, scheduler, None, logger,
        psd_weight_model=weight,
    )
    assert state.global_step == 96000 and state.start_epoch == 17
    torch.testing.assert_close(endpoint.weight, old_endpoint.weight)
    torch.testing.assert_close(source.weight, old_source.weight)
    assert optimizer.state[endpoint.weight]
    assert optimizer.state[source.weight]
    assert all(parameter not in optimizer.state for parameter in weight.parameters())
    torch.testing.assert_close(
        weight(torch.tensor([0.1, 0.9])), torch.full((2,), torch.log(torch.tensor(2.0)))
    )
    assert any("predates learnable PSD weighting" in message for message in logger.messages)


def test_new_checkpoint_strictly_restores_weight_and_optimizer_state(tmp_path):
    config = _config(tmp_path)
    old_endpoint = torch.nn.Linear(3, 2)
    old_source = torch.nn.Linear(3, 2)
    old_weight = PSDTimeWeightNetwork()
    old_optimizer = _optimizer(old_endpoint, old_source, old_weight)
    old_scheduler = build_scheduler(config, old_optimizer)
    _step(old_endpoint, old_source, old_weight, old_optimizer, old_scheduler)
    payload = checkpoint_payload(
        config=config, epoch=18, global_step=96001,
        model=old_endpoint, source_model=old_source, psd_weight_model=old_weight,
        optimizer=old_optimizer, scheduler=old_scheduler, scaler=None, metrics={},
    )
    save_checkpoint(payload, tmp_path, "resume.pt")

    endpoint = torch.nn.Linear(3, 2)
    source = torch.nn.Linear(3, 2)
    weight = PSDTimeWeightNetwork()
    optimizer = _optimizer(endpoint, source, weight)
    scheduler = build_scheduler(config, optimizer)
    state = initialize_or_resume(
        config, endpoint, source, optimizer, scheduler, None,
        psd_weight_model=weight,
    )
    assert state.global_step == 96001
    for actual, expected in zip(weight.parameters(), old_weight.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
        assert optimizer.state[actual]
