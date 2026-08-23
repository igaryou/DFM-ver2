import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import trainer
from config import load_config
from losses import esd_schedule_weight
from trainer import (
    _build_epoch_report,
    _create_epoch_progress,
    _epoch_total_iterations,
    _format_epoch_summary,
    _numbered_checkpoint_epochs,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "joint_ecld_cityscapes.yaml"


class _FakeProgress:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.updates = 0
        self.closed = False
        self.postfix_called = False

    def set_postfix(self, *args, **kwargs):
        del args, kwargs
        self.postfix_called = True
        raise AssertionError("The CFM-style progress bar must not use a postfix")

    def update(self, amount):
        self.updates += amount

    def close(self):
        self.closed = True


class _FakeTqdm:
    def __init__(self):
        self.calls = []
        self.writes = []

    def __call__(self, **kwargs):
        progress = _FakeProgress(**kwargs)
        self.calls.append(progress)
        return progress

    def write(self, value):
        self.writes.append(value)


def test_progress_is_created_only_for_rank0_and_has_no_postfix(monkeypatch):
    fake_tqdm = _FakeTqdm()
    monkeypatch.setattr(trainer, "tqdm", fake_tqdm)

    progress = _create_epoch_progress(
        epoch_index=7,
        total=200,
        is_main_process=True,
    )
    assert progress is fake_tqdm.calls[0]
    assert progress.kwargs == {
        "total": 200,
        "desc": "epoch 8",
        "dynamic_ncols": True,
        "leave": True,
        "unit": "batch",
        "mininterval": 0.5,
    }
    assert not progress.postfix_called

    assert _create_epoch_progress(
        epoch_index=7,
        total=200,
        is_main_process=False,
    ) is None
    assert len(fake_tqdm.calls) == 1


def test_epoch_total_supports_run_and_per_epoch_debug_limits():
    assert _epoch_total_iterations(743, 200, 0) == 200
    assert _epoch_total_iterations(743, 200, 150) == 50
    assert _epoch_total_iterations(743, None, 150) == 743
    assert _epoch_total_iterations(743, None, 0, 3) == 3
    assert _epoch_total_iterations(743, 2, 2, 3) == 0


def test_joint_numbered_checkpoints_include_last_pure_stage1_epoch():
    # start_epoch is compared with the internal 0-indexed epoch. Consequently,
    # start_epoch=500 first contributes at displayed epoch 501.
    assert esd_schedule_weight(499, 0.5, 500, 0) == 0.0
    assert esd_schedule_weight(500, 0.0, 500, 0) == 1.0
    assert _numbered_checkpoint_epochs(
        total_epochs=800,
        checkpoint_interval=200,
        joint_entrypoint=True,
        consistency_enabled=True,
        consistency_start_epoch=500,
    ) == {200, 400, 500, 600, 800}


@pytest.mark.parametrize(
    ("interval", "start_epoch", "joint", "enabled", "expected"),
    [
        (100, 501, True, True, {100, 200, 300, 400, 500, 501, 600, 700, 800}),
        (200, 0, True, True, {200, 400, 600, 800}),
        (200, 900, True, True, {200, 400, 600, 800}),
        (200, 500, False, True, {200, 400, 600, 800}),
        (200, 500, True, False, {200, 400, 600, 800}),
    ],
)
def test_numbered_checkpoint_epoch_edge_cases(
    interval, start_epoch, joint, enabled, expected
):
    assert _numbered_checkpoint_epochs(
        total_epochs=800,
        checkpoint_interval=interval,
        joint_entrypoint=joint,
        consistency_enabled=enabled,
        consistency_start_epoch=start_epoch,
    ) == expected


def test_numbered_checkpoint_epochs_deduplicate_and_preserve_final_handling():
    assert _numbered_checkpoint_epochs(
        total_epochs=750,
        checkpoint_interval=100,
        joint_entrypoint=True,
        consistency_enabled=True,
        consistency_start_epoch=500,
    ) == {100, 200, 300, 400, 500, 600, 700}


def test_epoch_report_has_cfm_names_and_does_not_invent_other_losses():
    report = _build_epoch_report(
        epoch=3,
        reduced_epoch={
            "loss_total": torch.tensor(1.760800),
            "loss_diagonal": torch.tensor(1.294519),
            "loss_consistency": torch.tensor(2.207328),
            "consistency_effective_weight": torch.tensor(0.5),
            "loss_ecld": torch.tensor(2.207328),
            "loss_ecld_ec": torch.tensor(2.207307),
            "loss_ecld_td": torch.tensor(0.000021),
            "ecld_dt_prob_norm": torch.tensor(0.004290),
            "loss_source_var": torch.tensor(0.0),
            "loss_source_align": torch.tensor(0.065842),
            "weighted_var": torch.tensor(0.0),
            "weighted_align": torch.tensor(0.009876),
            "source_mu_abs": torch.tensor(0.290600),
            "source_mu_min": torch.tensor(-2.475448),
            "source_mu_max": torch.tensor(2.937300),
            "source_logvar_mean": torch.tensor(0.0),
            "source_sigma_mean": torch.tensor(1.0),
            "source_x0_abs": torch.tensor(0.856801),
            "target_x1_abs": torch.tensor(0.05),
            "grad_norm": torch.tensor(2.710008),
        },
        consistency_type="ecld",
        primary_weight=0.5,
        local_batch_size=2,
        global_batch_size=4,
        grad_accum_steps=1,
        optimizer_step=743,
        processed_batches=743,
        optimizer_updates=743,
        elapsed_seconds=582.0,
        rank0_peak_allocated_mb=30473.0,
        max_peak_allocated_mb=30473.0,
        rank0_peak_reserved_mb=30598.0,
        max_peak_reserved_mb=30598.0,
        epoch_lr=1.0e-5,
        epoch_source_lr=5.0e-6,
    )

    assert list(report)[:13] == [
        "epoch",
        "loss_avg",
        "loss_base",
        "inf",
        "distill",
        "loss_total",
        "loss_primary",
        "loss_consistency",
        "distill_type",
        "consistency_loss_type",
        "consistency_weight",
        "ce_ec",
        "td",
    ]
    assert report["loss_base"] == pytest.approx(
        0.5 * report["loss_primary"]
        + report["consistency_weight"] * report["loss_consistency"]
    )
    for key in (
        "loss_var",
        "loss_align",
        "weighted_var",
        "weighted_align",
        "mu_abs",
        "mu_min",
        "mu_max",
        "logvar_mean",
        "sigma_mean",
        "x0_abs",
        "x1_abs",
    ):
        assert key in report
    assert "loss_psd" not in report
    assert "loss_csd" not in report
    assert "loss_esd" not in report

    summary = _format_epoch_summary(report)
    assert "lr:1.00000000e-05" in summary
    assert "source_lr:5.00000000e-06" in summary
    assert "ce_ec:2.207307 td:0.000021" in summary


def test_training_logs_only_one_epoch_record_and_one_epoch_wandb_call(
    tmp_path, monkeypatch, capsys
):
    output_dir = tmp_path / "run"
    config = load_config(
        CONFIG,
        [
            f"experiment.output_dir={output_dir}",
            "runtime.device=cpu",
            "training.epochs=1",
            "training.max_iterations=null",
            "training.max_batches_per_epoch=3",
            "training.grad_accum_steps=2",
            "training.log_interval=1",
            "training.validation_epochs=[]",
            "wandb.enabled=false",
        ],
    )
    batches = [
        (
            torch.full((2, 1), float(index)),
            torch.zeros(2, 1),
            torch.zeros(2, dtype=torch.long),
        )
        for index in range(1, 5)
    ]
    endpoint = nn.Linear(1, 1, bias=False)
    source = nn.Linear(1, 1, bias=False)
    fake_tqdm = _FakeTqdm()
    saved_checkpoints = []
    grad_norms = iter((torch.tensor(2.0), torch.tensor(4.0)))

    class _FakeWandb:
        def __init__(self):
            self.logs = []
            self.finished = False

        def log(self, payload, *, step):
            self.logs.append((payload, step))

        def finish(self):
            self.finished = True

    fake_wandb = _FakeWandb()

    def fake_objectives(model, *, operation, **kwargs):
        del operation, kwargs
        parameters = list(model.parameters())
        loss = sum(parameter.square().mean() for parameter in parameters)
        diagonal = loss.detach()
        consistency = diagonal.new_tensor(2.0)
        return {
            "loss": loss,
            "stats": {
                "loss_total": diagonal,
                "loss_diagonal": diagonal,
                "loss_consistency": consistency,
                "consistency_effective_weight": diagonal.new_tensor(0.5),
                "loss_source_var": diagonal.new_tensor(0.1),
                "loss_source_align": diagonal.new_tensor(0.2),
                "weighted_var": diagonal.new_tensor(0.01),
                "weighted_align": diagonal.new_tensor(0.02),
                "source_mu_abs": diagonal.new_tensor(0.3),
                "source_mu_min": diagonal.new_tensor(-1.0),
                "source_mu_max": diagonal.new_tensor(1.0),
                "source_logvar_mean": diagonal.new_tensor(0.0),
                "source_sigma_mean": diagonal.new_tensor(1.0),
                "source_x0_abs": diagonal.new_tensor(0.8),
                "target_x1_abs": diagonal.new_tensor(0.05),
                "loss_ecld": consistency,
                "loss_ecld_ec": diagonal.new_tensor(1.9),
                "loss_ecld_td": diagonal.new_tensor(0.1),
                "ecld_dt_prob_norm": diagonal.new_tensor(0.01),
            },
            "consistency_type": "ecld",
        }

    monkeypatch.setattr(
        trainer,
        "_build_loaders",
        lambda config, context, local_batch_size: (batches, [], None),
    )
    monkeypatch.setattr(
        trainer,
        "build_models",
        lambda config, device: (endpoint.to(device), source.to(device)),
    )
    monkeypatch.setattr(
        trainer,
        "initialize_or_resume",
        lambda *args, **kwargs: SimpleNamespace(
            start_epoch=0,
            global_step=0,
            best_miou=float("-inf"),
        ),
    )
    monkeypatch.setattr(
        trainer,
        "run_model_training_objectives",
        fake_objectives,
    )
    monkeypatch.setattr(
        trainer,
        "_save_training_checkpoint",
        lambda **kwargs: saved_checkpoints.append(kwargs),
    )
    monkeypatch.setattr(trainer, "init_wandb", lambda config: fake_wandb)
    monkeypatch.setattr(trainer, "tqdm", fake_tqdm)
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: next(grad_norms),
    )

    trainer.run_training(config, joint_entrypoint=True)

    assert len(fake_tqdm.calls) == 1
    progress = fake_tqdm.calls[0]
    assert progress.kwargs["total"] == 3
    assert progress.updates == 3
    assert progress.closed
    assert not progress.postfix_called
    assert len(fake_tqdm.writes) == 1

    records = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert [record["scope"] for record in records] == ["epoch", "runtime"]
    epoch_record = records[0]
    assert epoch_record["grad_norm"] == pytest.approx(3.0)
    assert epoch_record["loss_align"] == pytest.approx(0.2)
    assert epoch_record["weighted_align"] == pytest.approx(0.02)
    assert epoch_record["sigma_mean"] == pytest.approx(1.0)

    assert len(fake_wandb.logs) == 1
    epoch_payload, epoch_step = fake_wandb.logs[0]
    assert epoch_step == 2
    assert all(key.startswith("epoch/") for key in epoch_payload)
    assert epoch_payload["epoch/grad_norm"] == pytest.approx(3.0)
    assert fake_wandb.finished

    base_model_lr = config["training"]["optimizer"]["parameter_groups"]["model"]["lr"]
    base_model_lr = base_model_lr or config["training"]["optimizer"]["lr"]
    base_source_lr = (
        config["training"]["optimizer"]["parameter_groups"]["source"]["lr"]
        or config["training"]["optimizer"]["lr"]
    )
    expected_factor = config["training"]["scheduler"]["warmup_start_factor"]
    assert epoch_record["lr"] == pytest.approx(base_model_lr * expected_factor)
    assert epoch_record["source_lr"] == pytest.approx(
        base_source_lr * expected_factor
    )

    assert len(saved_checkpoints) == 1
    assert saved_checkpoints[0]["filenames"] == ["latest.pt"]
    saved_scheduler = saved_checkpoints[0]["scheduler"]
    assert saved_scheduler.last_epoch == 1

    console = capsys.readouterr().err
    assert "epoch=1 iter=" not in console
    file_log = (output_dir / "train_log.txt").read_text()
    assert "epoch=1 iter=" not in file_log
    assert file_log.count("| epoch:1 ") == 1
