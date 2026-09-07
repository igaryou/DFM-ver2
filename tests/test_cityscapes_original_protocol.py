from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

import inference
from checkpoint import checkpoint_payload
from config import load_config
from dataset import Cityscapes20ClassDataset
from discrete_flow_maps import source_supervision_schedule
from inference import sample_segmentation_ensemble
from metrics import SegmentationMetrics
from model import DiscreteFlowMapModel, ImageEncoder
from training_objectives import (
    DDPCompatibleTrainingModel, compute_model_training_objectives,
)
from trainer import _wandb_epoch_payload
from training_stages import (
    TrainingStage,
    schedule_value,
    set_stage_trainability,
    stage_operation,
    training_stage_for_epoch,
)


ROOT = Path(__file__).parents[1]
REPRESENTATIVE = ROOT / (
    "configs/cityscapes/original/psd/"
    "joint_bounded_gaussian_b1_exponential_path_adaptive_std_trainable.yaml"
)
MMSEG = ROOT / "configs/_base_/cityscapes/swin_t_160k.yaml"
MMSEG_WRAPPER = ROOT / "configs/cityscapes/mmseg/psd/swin_t_linear_160k.yaml"


def _synthetic_dataset(config: dict, *, augment: bool):
    dataset = object.__new__(Cityscapes20ClassDataset)
    dataset.config = config
    dataset.split = "train" if augment else "val"
    dataset.augment = augment
    dataset.photo_distortion = lambda image: image
    dataset.jitter = lambda image: image
    dataset.dataset = SimpleNamespace(images=["sample_leftImg8bit.png"])
    return dataset


def test_original_config_resolves_protocol_and_reference_recipe():
    config = load_config(REPRESENTATIVE)
    assert config["dataset"]["protocol"] == "original"
    assert config["dataset"]["image_size"] == [256, 512]
    assert config["dataset"]["fixed_resize"] == {
        "enabled": True,
        "image_interpolation": "bilinear",
        "mask_interpolation": "nearest",
        "antialias": True,
    }
    assert config["training"]["schedule_unit"] == "epoch"
    assert config["training"]["epochs"] == 800
    assert config["training"]["max_optimizer_steps"] is None
    assert config["source"]["bounded_gaussian"]["amplitude"] == 0.5
    assert config["source"]["bounded_gaussian"]["temperature"] == 4.0
    assert config["source"]["bounded_gaussian"]["variance"]["rho"] == 0.95
    assert config["flow"]["path"]["scheduler"] == {
        "type": "exponential", "beta": 2.0, "difficulty_gamma": 1.0
    }


def test_original_representative_resolves_and_builds_rrdb_flow_encoder():
    config = load_config(REPRESENTATIVE)
    model_config = config["model"]
    assert model_config["image_encoder"]["type"] == "rrdb"
    assert model_config["fusion_channels"] == 128
    assert model_config["rrdb_blocks"] == 3
    assert model_config["rrdb_growth_channels"] == 32
    assert model_config["state_downsample_factor"] == 4
    assert config["source"]["segformer_variant"] == "b1"
    assert config["source"]["segformer_decoder"] == "standard"

    endpoint = DiscreteFlowMapModel(model_config)
    assert isinstance(endpoint.image_encoder, ImageEncoder)
    assert endpoint.image_encoder.downsample_factor == 4
    assert endpoint.image_encoder.first.out_channels == 128
    assert len(endpoint.image_encoder.body) == 3
    first_dense_conv = endpoint.image_encoder.body[0].blocks[0].layers[0]
    assert first_dense_conv.out_channels == 32

    feature = endpoint.encode_image(torch.randn(1, 3, 32, 64))
    assert feature.shape == (1, 128, 8, 16)


def test_original_train_and_val_are_fixed_size_and_mask_is_nearest():
    config = load_config(REPRESENTATIVE)
    config["dataset"]["image_size"] = [4, 8]
    config["augmentation"]["horizontal_flip"]["enabled"] = False
    config["augmentation"]["color_jitter"]["enabled"] = False
    image = torch.linspace(0, 1, 3 * 8 * 16).reshape(3, 8, 16)
    mask = torch.tensor([
        [0] * 8 + [13] * 8,
        [0] * 8 + [13] * 8,
        [1] * 8 + [18] * 8,
        [1] * 8 + [18] * 8,
        [2] * 8 + [19] * 8,
        [2] * 8 + [19] * 8,
        [3] * 8 + [4] * 8,
        [3] * 8 + [4] * 8,
    ])
    train = _synthetic_dataset(config, augment=True)
    train_image, train_mask = train._train_item(image, mask)
    val = _synthetic_dataset(config, augment=False)
    val_image, val_mask = val._validation_item(image, mask, 0)
    assert train_image.shape == val_image.shape == (3, 4, 8)
    assert train_mask.shape == val_mask.shape == (4, 8)
    assert set(train_mask.unique().tolist()) <= set(mask.unique().tolist())
    assert set(val_mask.unique().tolist()) <= set(mask.unique().tolist())
    torch.testing.assert_close(train_mask, val_mask)


def test_original_train_has_configured_augmentation_but_val_does_not():
    config = load_config(REPRESENTATIVE)
    train = _synthetic_dataset(config, augment=True)
    val = _synthetic_dataset(config, augment=False)
    assert train.augment
    assert not val.augment
    assert config["augmentation"]["horizontal_flip"]["probability"] == 0.5
    assert config["augmentation"]["color_jitter"] == {
        "enabled": True,
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1,
    }


def test_mmseg_wrapper_preserves_existing_protocol_values():
    direct = load_config(MMSEG)
    wrapped = load_config(MMSEG_WRAPPER)
    assert direct["dataset"]["protocol"] == wrapped["dataset"]["protocol"] == "mmseg"
    for section in ("augmentation", "model", "source", "training", "evaluation"):
        assert wrapped[section] == direct[section]


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [(0, "source_pretrain"), (149, "source_pretrain"),
     (150, "flow_training"), (799, "flow_training")],
)
def test_stage_scheduler_boundaries(epoch: int, expected: str):
    config = load_config(REPRESENTATIVE)
    assert training_stage_for_epoch(config, epoch).name == expected


def test_stage_trainability_and_modes_switch_cleanly():
    config = load_config(REPRESENTATIVE)
    endpoint = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
    source = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
    pretrain = training_stage_for_epoch(config, 149)
    set_stage_trainability(endpoint, source, pretrain)
    assert not endpoint.training
    assert not any(parameter.requires_grad for parameter in endpoint.parameters())
    assert source.training
    assert all(parameter.requires_grad for parameter in source.parameters())

    flow = training_stage_for_epoch(config, 150)
    set_stage_trainability(endpoint, source, flow)
    assert endpoint.training and source.training
    assert all(parameter.requires_grad for parameter in endpoint.parameters())
    assert all(parameter.requires_grad for parameter in source.parameters())

    frozen_config = load_config(
        ROOT / "configs/cityscapes/original/psd/"
        "joint_bounded_gaussian_b1_150ep_source_then_frozen.yaml"
    )
    frozen_flow = training_stage_for_epoch(frozen_config, 150)
    set_stage_trainability(endpoint, source, frozen_flow)
    assert endpoint.training and not source.training
    assert not any(parameter.requires_grad for parameter in source.parameters())


def test_epoch_and_step_linear_weight_schedules_and_resume_recompute():
    epoch_schedule = {
        "type": "linear", "unit": "epoch", "start_epoch": 150,
        "duration": 50, "initial": 3.0, "final": 0.2,
    }
    assert schedule_value(
        epoch_schedule, epoch_index=150, optimizer_step=999, fallback=0
    )[0] == pytest.approx(3.0)
    assert schedule_value(
        epoch_schedule, epoch_index=175, optimizer_step=999, fallback=0
    )[0] == pytest.approx(1.6)
    assert schedule_value(
        epoch_schedule, epoch_index=200, optimizer_step=999, fallback=0
    )[0] == pytest.approx(0.2)
    supervision = {
        "weight": 0.2,
        "weight_schedule": {
            "type": "linear", "initial": 3.0, "final": 0.2, "steps": 16000
        },
    }
    assert source_supervision_schedule(supervision, 8000, 3)[0] == pytest.approx(1.6)
    # Resume is stateless: the restored epoch/optimizer step gives the same value.
    assert source_supervision_schedule(supervision, 8000, 200)[0] == pytest.approx(1.6)


def test_flow_warmup_values():
    config = load_config(REPRESENTATIVE)
    schedule = config["training"]["flow_weight_schedule"]
    for epoch, expected in ((150, 0.0), (155, 0.5), (160, 1.0), (799, 1.0)):
        value, _ = schedule_value(
            schedule, epoch_index=epoch, optimizer_step=0, fallback=1.0
        )
        assert value == pytest.approx(expected)


def test_void_gt_is_excluded_and_metric_is_19_class_mean():
    metric = SegmentationMetrics(20, 19, evaluated_class_indices=range(19))
    target = torch.tensor([[0, 1, 19, 2]])
    prediction = torch.tensor([[0, 0, 0, 2]])
    metric.update(prediction, target)
    result = metric.compute()
    assert sum(sum(row) for row in result["confusion_matrix"]) == 3
    assert result["evaluated_class_indices"] == list(range(19))
    expected = (0.5 + 0.0 + 1.0) / 19.0
    assert result["mIoU"] == pytest.approx(expected)


def test_probability_mean_and_majority_vote_are_not_label_averages(monkeypatch):
    probabilities = iter([
        torch.tensor([[[[0.60]], [[0.40]], [[0.00]]]]),
        torch.tensor([[[[0.10]], [[0.90]], [[0.00]]]]),
        torch.tensor([[[[0.55]], [[0.45]], [[0.00]]]]),
    ])
    monkeypatch.setattr(
        inference, "sample_segmentation_probabilities",
        lambda *args, **kwargs: next(probabilities),
    )
    config = {
        "evaluation": {
            "num_samples": 2, "aggregation": "probability_mean",
            "exclude_void_from_prediction": True,
        },
        "dataset": {"void_class_index": 2},
    }
    image = torch.zeros(1, 3, 4, 4)
    assert sample_segmentation_ensemble(None, None, image, config).item() == 1

    probabilities = iter([
        torch.tensor([[[[0.60]], [[0.40]], [[0.00]]]]),
        torch.tensor([[[[0.10]], [[0.90]], [[0.00]]]]),
        torch.tensor([[[[0.55]], [[0.45]], [[0.00]]]]),
    ])
    monkeypatch.setattr(
        inference, "sample_segmentation_probabilities",
        lambda *args, **kwargs: next(probabilities),
    )
    assert sample_segmentation_ensemble(
        None, None, image, config, num_samples=3, aggregation="majority_vote"
    ).item() == 0


def test_checkpoint_records_stage_optimizer_step_and_local_progress():
    config = load_config(REPRESENTATIVE)
    model = nn.Linear(2, 2)
    source = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(
        [{"params": model.parameters(), "name": "model"},
         {"params": source.parameters(), "name": "source"}],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800)
    payload = checkpoint_payload(
        config=config,
        epoch=150,
        global_step=1234,
        model=model,
        source_model=source,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        metrics={"training_stage": "source_pretrain", "stage_local_epoch": 149},
    )
    assert payload["optimizer_step"] == 1234
    assert payload["current_stage"] == "source_pretrain"
    assert payload["stage_local_progress"] == 149


class _TinySource(nn.Module):
    fixed_std = 1.0

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([1.5, -0.5, 0.25, 0.75]))

    def forward_statistics(self, image: torch.Tensor):
        mean = self.logits[None, :, None, None].expand(
            image.shape[0], 4, image.shape[-2] // 4, image.shape[-1] // 4
        )
        return mean, torch.zeros_like(mean)

    def forward(self, image: torch.Tensor):
        mean, logvar = self.forward_statistics(image)
        return mean + torch.randn_like(mean), mean, logvar


class _TinyFlowEndpoint(nn.Module):
    def __init__(self, classes: int = 4) -> None:
        super().__init__()
        self.image_projection = nn.Conv2d(3, classes, 1)
        self.state_projection = nn.Conv2d(classes, classes, 1)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(self.image_projection(image), 4)

    def forward_logits_with_image_feat(self, state, image_feat, s, t):
        del s, t
        return self.state_projection(state) + image_feat

    def forward_logits(self, state, image, s, t):
        return self.forward_logits_with_image_feat(
            state, self.encode_image(image), s, t
        )


def _tiny_stage_setup():
    config = load_config(REPRESENTATIVE)
    config["dataset"]["num_classes"] = 4
    config["model"]["num_classes"] = 4
    config["runtime"]["amp"] = False
    config["loss"]["ignore_index"] = 3
    config["evaluation"]["ignore_index"] = 3
    config["loss"]["consistency"]["enabled"] = False
    config["loss"]["consistency"]["weight"] = 0.0
    config["flow"]["path"] = {"type": "power", "exponent": 1.0}
    config["source"]["bounded_gaussian"]["variance"]["type"] = "fixed"
    endpoint = _TinyFlowEndpoint()
    source = _TinySource()
    return config, DDPCompatibleTrainingModel(endpoint, source, config)


def test_stage1_is_raw_source_ce_only_and_stage2_activates_flow():
    config, adapter = _tiny_stage_setup()
    image = torch.randn(1, 3, 8, 8)
    target = torch.randint(0, 4, (1, 8, 8))

    source_stage = training_stage_for_epoch(config, 0)
    set_stage_trainability(adapter.endpoint_model, adapter.source_model, source_stage)
    stage1 = compute_model_training_objectives(
        adapter, operation=stage_operation(config, source_stage), image=image,
        target=target, epoch_index=0, progress_in_epoch=0.0, optimizer_step=0,
    )
    assert stage1["stats"]["loss_source_ce"] > 0
    assert stage1["stats"]["loss_diagonal"] == 0
    assert stage1["stats"]["loss_consistency"] == 0
    stage1["loss"].backward()
    assert adapter.source_model.logits.grad is not None
    assert all(p.grad is None for p in adapter.endpoint_model.parameters())

    adapter.zero_grad(set_to_none=True)
    flow_stage = training_stage_for_epoch(config, 150)
    set_stage_trainability(adapter.endpoint_model, adapter.source_model, flow_stage)
    stage2 = compute_model_training_objectives(
        adapter, operation=stage_operation(config, flow_stage), image=image,
        target=target, epoch_index=150, progress_in_epoch=0.0, optimizer_step=0,
    )
    assert stage2["stats"]["loss_diagonal"] > 0
    assert stage2["stats"]["flow_effective_weight"] == 0
    stage2["loss"].backward()
    assert adapter.endpoint_model.state_projection.weight.grad is not None


def test_checkpoint_start_configs_keep_load_and_freeze_independent():
    frozen = load_config(
        ROOT / "configs/cityscapes/original/psd/"
        "joint_bounded_gaussian_b1_from_checkpoint_frozen.yaml"
    )
    trainable = load_config(
        ROOT / "configs/cityscapes/original/psd/"
        "joint_bounded_gaussian_b1_from_checkpoint_trainable.yaml"
    )
    assert frozen["source"]["checkpoint"] == trainable["source"]["checkpoint"]
    assert frozen["source"]["freeze"] is True
    assert trainable["source"]["freeze"] is False
    assert training_stage_for_epoch(frozen, 0).train_source is False
    assert training_stage_for_epoch(trainable, 0).train_source is True


def test_resume_epoch_recomputes_stage_without_transition_state():
    config = load_config(REPRESENTATIVE)
    assert training_stage_for_epoch(config, 149).name == "source_pretrain"
    assert training_stage_for_epoch(config, 150).name == "flow_training"
    assert training_stage_for_epoch(config, 200).name == "flow_training"
    weight_150 = schedule_value(
        config["training"]["flow_weight_schedule"],
        epoch_index=150, optimizer_step=123, fallback=1.0,
    )
    weight_200 = schedule_value(
        config["training"]["flow_weight_schedule"],
        epoch_index=200, optimizer_step=999, fallback=1.0,
    )
    assert weight_150[0] == 0.0
    assert weight_200[0] == 1.0


def test_all_mmseg_wrappers_preserve_resolved_semantics():
    import yaml
    for wrapper in (ROOT / "configs/cityscapes/mmseg").rglob("*.yaml"):
        raw = yaml.safe_load(wrapper.read_text())
        target = (wrapper.parent / raw["extends"]).resolve()
        wrapped = load_config(wrapper)
        direct = load_config(target)
        wrapped["runtime"].pop("config_path", None)
        direct["runtime"].pop("config_path", None)
        assert wrapped == direct, wrapper


class _StageDDPComposite(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.endpoint_model = nn.Linear(2, 1)
        self.source_model = nn.Linear(2, 2)

    def forward(self, value: torch.Tensor, source_only: bool) -> torch.Tensor:
        source = self.source_model(value)
        if source_only:
            return source.square().mean()
        return self.endpoint_model(source).square().mean()


def _stage_ddp_worker(rank: int, rendezvous: str) -> None:
    os.environ.update({
        "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "2",
        "GLOO_SOCKET_IFNAME": "lo",
    })
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        model = _StageDDPComposite()
        source_stage = TrainingStage("source_pretrain", 0, 1, True, False)
        flow_stage = TrainingStage("flow_training", 1, 2, True, True)
        set_stage_trainability(model.endpoint_model, model.source_model, source_stage)
        ddp = nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=False
        )
        ddp(torch.ones(2, 2) + rank, True).backward()
        assert model.source_model.weight.grad is not None
        assert model.endpoint_model.weight.grad is None
        model.zero_grad(set_to_none=True)

        # Production uses this same boundary policy: change flags, then rewrap.
        set_stage_trainability(model.endpoint_model, model.source_model, flow_stage)
        ddp = nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=False
        )
        ddp(torch.ones(2, 2) + rank, False).backward()
        assert model.source_model.weight.grad is not None
        assert model.endpoint_model.weight.grad is not None
    finally:
        dist.destroy_process_group()


def test_two_rank_cpu_ddp_stage_boundary_rewrap(tmp_path):
    rendezvous = tmp_path / "stage-ddp-rendezvous"
    mp.spawn(_stage_ddp_worker, args=(str(rendezvous),), nprocs=2, join=True)



def test_required_wandb_stage_namespaces_are_emitted():
    report = {
        "training_stage": "flow_training", "epoch": 151,
        "optimizer_step": 321, "loss_total": 2.0, "loss_primary": 1.0,
        "source_ce_raw": 0.8, "loss_consistency": 0.2,
        "flow_effective_weight": 0.1, "source_ce_effective_weight": 3.0,
        "consistency_weight": 0.5, "source_frozen": 0.0,
        "source_lr": 1.0e-5, "lr": 1.0e-4, "mu_abs": 0.7,
        "mu_min": -2.0, "mu_max": 3.0, "sigma_mean": 1.0,
        "path_difficulty_mean": 0.0, "path_difficulty_std": 0.6,
    }
    payload = _wandb_epoch_payload(report, "psd")
    required = {
        "epoch/training/stage", "epoch/training/epoch", "epoch/training/global_step",
        "epoch/training/optimizer_step", "epoch/loss/total", "epoch/loss/flow",
        "epoch/loss/source_ce", "epoch/loss/consistency", "epoch/weight/flow",
        "epoch/weight/source_ce", "epoch/weight/consistency", "epoch/source/frozen",
        "epoch/source/lr", "epoch/model/lr", "epoch/source/mu_abs_mean", "epoch/source/mu_min",
        "epoch/source/mu_max", "epoch/source/std_mean", "epoch/path/difficulty_mean",
        "epoch/path/difficulty_std",
    }
    assert required <= payload.keys()



def test_probability_extraction_uses_final_pi_without_second_softmax(monkeypatch):
    final_pi = torch.tensor([[[[0.7]], [[0.2]], [[0.1]]]])
    monkeypatch.setattr(
        inference, "sample_prior",
        lambda *args, **kwargs: (torch.zeros_like(final_pi), {}),
    )
    monkeypatch.setattr(
        inference, "run_flow_from_state",
        lambda *args, **kwargs: final_pi,
    )
    config = {
        "evaluation": {"num_steps": 1},
        "flow": {"path": {"type": "power", "exponent": 1.0}},
    }
    result = inference.sample_segmentation_probabilities(
        nn.Identity(), None, torch.zeros(1, 3, 1, 1), config
    )
    torch.testing.assert_close(result, final_pi)
    torch.testing.assert_close(result.sum(dim=1), torch.ones(1, 1, 1))


def test_original_recipes_select_shared_path_and_variance_implementations():
    fixed = load_config(
        ROOT / "configs/cityscapes/original/psd/"
        "joint_bounded_gaussian_b1_800ep.yaml"
    )
    adaptive = load_config(REPRESENTATIVE)
    assert fixed["flow"]["path"]["type"] == "power"
    assert fixed["source"]["bounded_gaussian"]["variance"]["type"] == "fixed"
    assert adaptive["flow"]["path"]["type"] == "entropy_adaptive"
    assert adaptive["source"]["bounded_gaussian"]["variance"] == {
        "type": "entropy_adaptive", "rho": 0.95,
        "normalization": "rank", "eps": 1.0e-8,
    }
