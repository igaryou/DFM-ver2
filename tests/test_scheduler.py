import copy
from pathlib import Path

import pytest
import torch

from config import load_config
from trainer import build_scheduler


CONFIG = Path(__file__).parents[1] / "configs" / "cityscapes" / "ecld" / "joint.yaml"


def _config():
    return load_config(
        CONFIG,
        [
            "training.epochs=800",
            "training.optimizer.lr=1.0e-4",
            "training.optimizer.parameter_groups.model.lr=1.0e-4",
            "training.optimizer.parameter_groups.source.lr=5.0e-5",
            "training.scheduler.warmup_epochs=10",
            "training.scheduler.eta_min=5.0e-7",
        ],
    )


def _optimizer():
    model = torch.nn.Parameter(torch.tensor(1.0))
    source = torch.nn.Parameter(torch.tensor(2.0))
    return torch.optim.AdamW([
        {"params": [model], "lr": 1.0e-4, "name": "model"},
        {"params": [source], "lr": 5.0e-5, "name": "source"},
    ])


def _lr_sequence(config, epochs=15):
    optimizer = _optimizer()
    scheduler = build_scheduler(config, optimizer)
    sequence = []
    for _ in range(epochs):
        sequence.append(tuple(group["lr"] for group in optimizer.param_groups))
        optimizer.step()
        scheduler.step()
    return sequence


def _cfm_model_lr_sequence(epochs=15):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-4)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=10,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=790,
                eta_min=5.0e-7,
            ),
        ],
        milestones=[10],
    )
    sequence = []
    for _ in range(epochs):
        sequence.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return sequence


def test_epoch_one_lr_and_source_ratio_match_cfm():
    sequence = _lr_sequence(_config())
    assert sequence[0] == pytest.approx((1.0e-5, 5.0e-6))
    assert [model_lr for model_lr, _ in sequence] == pytest.approx(
        _cfm_model_lr_sequence()
    )
    assert all(source_lr / model_lr == pytest.approx(0.5)
               for model_lr, source_lr in sequence)


def test_743_optimizer_steps_do_not_advance_scheduler_inside_epoch():
    optimizer = _optimizer()
    scheduler = build_scheduler(_config(), optimizer)
    initial_lrs = [group["lr"] for group in optimizer.param_groups]
    initial_scheduler_epoch = scheduler.last_epoch
    for _ in range(743):
        optimizer.step()
    assert [group["lr"] for group in optimizer.param_groups] == initial_lrs
    assert scheduler.last_epoch == initial_scheduler_epoch

    scheduler.step()
    assert scheduler.last_epoch == initial_scheduler_epoch + 1
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1.9e-5, 9.5e-6]
    )


@pytest.mark.parametrize(
    "override",
    [
        ("training.grad_accum_steps", 2),
        ("training.batch_size", 8),
        ("distributed.enabled", False),
    ],
)
def test_epoch_lr_sequence_is_independent_of_training_loop_shape(override):
    baseline = _config()
    changed = copy.deepcopy(baseline)
    section, key = override[0].split(".", 1)
    if "." in key:
        first, second = key.split(".", 1)
        changed[section][first][second] = override[1]
    else:
        changed[section][key] = override[1]
    assert _lr_sequence(changed) == pytest.approx(_lr_sequence(baseline))


def test_resume_lr_sequence_matches_continuous_sequence():
    config = _config()
    continuous = _lr_sequence(config, epochs=15)

    optimizer = _optimizer()
    scheduler = build_scheduler(config, optimizer)
    first = []
    for _ in range(7):
        first.append(tuple(group["lr"] for group in optimizer.param_groups))
        optimizer.step()
        scheduler.step()
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()

    resumed_optimizer = _optimizer()
    resumed_scheduler = build_scheduler(config, resumed_optimizer)
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)
    resumed = []
    for _ in range(8):
        resumed.append(
            tuple(group["lr"] for group in resumed_optimizer.param_groups)
        )
        resumed_optimizer.step()
        resumed_scheduler.step()

    assert first + resumed == pytest.approx(continuous)


def test_poly_optimizer_step_uses_160k_budget_and_1500_step_warmup():
    config = _config()
    config["training"]["max_optimizer_steps"] = 160000
    config["training"]["scheduler"].update({
        "name": "poly", "step_unit": "optimizer_step",
        "warmup_steps": 1500, "warmup_start_factor": 1.0e-6,
        "power": 1.0, "min_lr": 0.0,
    })
    optimizer = _optimizer()
    scheduler = build_scheduler(config, optimizer)

    assert scheduler.lr_lambdas[0](0) == pytest.approx(1.0e-6)
    assert scheduler.lr_lambdas[0](1500) == pytest.approx(1.0)
    assert scheduler.lr_lambdas[0](160000) == pytest.approx(0.0)


def test_cosine_optimizer_step_uses_remaining_step_budget_for_t_max():
    config = _config()
    config["training"]["max_optimizer_steps"] = 160000
    config["training"]["scheduler"].update({
        "name": "cosine", "step_unit": "optimizer_step", "warmup_steps": 1500,
    })
    scheduler = build_scheduler(config, _optimizer())
    assert scheduler._milestones == [1500]
    assert scheduler._schedulers[1].T_max == 158500


def test_cosine_epoch_keeps_epoch_budget_for_t_max():
    scheduler = build_scheduler(_config(), _optimizer())
    assert scheduler._milestones == [10]
    assert scheduler._schedulers[1].T_max == 790
