import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

import trainer
from checkpoint import checkpoint_payload, initialize_or_resume, save_checkpoint
from config import load_config, validate_config
from dataset import (
    ADE20KDataset,
    PhotoMetricDistortion,
    _resize_keep_ratio_size,
    build_dataset,
)
from discrete_flow_maps import sample_prior
from inference import terminal_state_to_original_prediction
from losses import compute_consistency_loss, diagonal_cross_entropy, masked_mean
from metrics import SegmentationMetrics
from trainer import _optimizer_step_validation_trigger, build_scheduler


ROOT = Path(__file__).parents[1]
ADE_CONFIG = ROOT / "configs" / "joint_psd_ade20k.yaml"
ADE_ROOT = Path("/home/igarashi_25/datasets/ADEChallengeData2016")


def _config():
    return load_config(ADE_CONFIG)


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (500, 1000, (512, 1024)),
        (1000, 500, (1024, 512)),
    ],
)
def test_mmcv_keep_ratio_resize_semantics(height, width, expected):
    assert _resize_keep_ratio_size(height, width, 2048, 512) == expected


def test_train_ratio_one_resize_uses_mmcv_semantics_and_nearest_mask():
    config = _config()
    config["augmentation"]["random_resize"]["ratio_range"] = [1.0, 1.0]
    dataset = object.__new__(ADE20KDataset)
    dataset.config = config
    image = torch.zeros(3, 1000, 500)
    mask = torch.zeros(1000, 500, dtype=torch.long)
    mask[:, 250:] = 150

    resized_image, resized_mask = dataset._random_resize(image, mask)

    assert resized_image.shape[-2:] == (1024, 512)
    assert resized_mask.shape == (1024, 512)
    assert set(resized_mask.unique().tolist()) == {0, 150}


def test_validation_resize_uses_same_mmcv_semantics_and_keeps_gt_original():
    dataset = object.__new__(ADE20KDataset)
    dataset.config = _config()
    dataset.images = [Path("portrait.jpg")]
    image = torch.zeros(3, 1000, 500)
    mask = torch.zeros(1000, 500, dtype=torch.long)

    sample = dataset._validation_item(image, mask, 0)

    assert sample["model_shape"] == (1024, 512)
    assert sample["image"].shape[-2:] == (1024, 512)
    assert sample["target"].shape == (1000, 500)


def test_ade20k_main_config_is_151_state_and_prevents_double_normalization():
    config = _config()
    assert config["dataset"]["num_classes"] == 151
    assert config["model"]["num_classes"] == 151
    assert config["dataset"]["reduce_zero_label"] is False
    assert config["loss"]["ignore_index"] == 0
    assert config["source"]["input_already_normalized"] is True
    assert config["training"]["max_optimizer_steps"] == 160000
    assert config["training"]["validation_epochs"] == []
    assert config["training"]["scheduler"]["step_unit"] == "optimizer_step"
    assert config["evaluation"]["interval"] == {
        "unit": "optimizer_step", "value": 16000
    }

    invalid = copy.deepcopy(config)
    invalid["source"]["input_already_normalized"] = False
    with pytest.raises(ValueError, match="input_already_normalized"):
        validate_config(invalid)
    with pytest.raises(ValueError, match="positive integer"):
        load_config(ADE_CONFIG, ["evaluation.interval.value=0"])
    with pytest.raises(ValueError, match="unit must be"):
        load_config(ADE_CONFIG, ["evaluation.interval.unit=batch"])


@pytest.mark.skipif(not ADE_ROOT.is_dir(), reason="ADE20K is not installed")
def test_installed_ade20k_has_expected_counts_and_unshifted_labels():
    config = _config()
    training = build_dataset(config, "training", augment=False)
    validation = build_dataset(config, "validation", augment=False)
    assert len(training) == 20210
    assert len(validation) == 2000
    for dataset in (training, validation):
        _, mask = dataset._load(dataset.images[0], dataset.annotations[0])
        assert 0 <= int(mask.min()) <= int(mask.max()) <= 150
        assert bool((mask == 0).any())


def test_ade20k_train_pipeline_keeps_geometry_and_class_zero_state(tmp_path, monkeypatch):
    config = _config()
    config["dataset"]["root"] = str(tmp_path)
    config["augmentation"]["random_resize"].update({
        "base_scale": {"width": 8, "height": 6}, "ratio_range": [1.0, 1.0]
    })
    config["augmentation"]["random_crop"].update({
        "size": [4, 4], "cat_max_ratio": 1.0
    })
    config["augmentation"]["pad"]["size"] = [4, 4]
    image_dir = tmp_path / "images" / "training"
    mask_dir = tmp_path / "annotations" / "training"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    mask = np.tile(np.arange(8, dtype=np.uint8), (6, 1))
    Image.fromarray(image).save(image_dir / "sample.jpg")
    Image.fromarray(mask).save(mask_dir / "sample.png")
    monkeypatch.setitem(ADE20KDataset.EXPECTED_COUNTS, "training", 1)

    torch.manual_seed(9)
    dataset = build_dataset(config, "training", augment=True)
    transformed_image, transformed_mask = dataset[0]
    assert transformed_image.shape == (3, 4, 4)
    assert transformed_mask.shape == (4, 4)
    assert transformed_mask.dtype == torch.long


def test_photo_metric_distortion_never_changes_mask():
    config = _config()["augmentation"]["photometric_distortion"]
    distortion = PhotoMetricDistortion(config)
    image = torch.rand(3, 8, 9)
    mask = torch.randint(0, 151, (8, 9))
    before = mask.clone()
    _ = distortion(image)
    assert torch.equal(mask, before)


def test_masked_losses_ignore_zero_and_all_ignore_is_safe():
    logits = torch.tensor([[[[0.0, -4.0]], [[4.0, 4.0]], [[-4.0, -4.0]]]])
    target = torch.tensor([[[0, 1]]])
    loss = diagonal_cross_entropy(logits, target, ignore_index=0)
    expected = torch.nn.functional.cross_entropy(logits[..., 1:], target[..., 1:])
    assert loss == pytest.approx(float(expected))
    all_ignore = diagonal_cross_entropy(logits, torch.zeros_like(target), ignore_index=0)
    assert all_ignore == 0
    assert torch.isfinite(all_ignore)

    loss_map = torch.tensor([[100.0, 2.0, 4.0]])
    assert masked_mean(loss_map, torch.tensor([[False, True, True]])) == 3


class _Source(nn.Module):
    fixed_std = 1.0

    def forward(self, image):
        mu = torch.arange(12, dtype=image.dtype).reshape(1, 3, 2, 2)
        logvar = torch.zeros_like(mu)
        return mu, mu, logvar


def test_source_alignment_uses_valid_pixel_reduction():
    config = {
        "dataset": {"num_classes": 3},
        "source": {
            "prior_type": "image_gaussian", "use_loss_align": True,
            "align_eps": 1.0e-8, "align_weight": 1.0, "var_weight": 0.0,
        },
    }
    image = torch.zeros(1, 3, 2, 2)
    target_full = torch.tensor([[[0, 1], [2, 1]]])
    target_state = torch.nn.functional.one_hot(
        target_full, 3
    ).permute(0, 3, 1, 2).float()
    _, all_stats = sample_prior(
        config, image, target_state, _Source(), target_full=target_full
    )
    _, masked_stats = sample_prior(
        config, image, target_state, _Source(), target_full=target_full,
        valid_mask_full=torch.tensor([[[False, True], [False, False]]]),
    )
    assert masked_stats["loss_source_align"] != all_stats["loss_source_align"]
    _, ignored_stats = sample_prior(
        config, image, target_state, _Source(), target_full=target_full,
        valid_mask_full=torch.zeros(1, 2, 2, dtype=torch.bool),
    )
    assert ignored_stats["loss_source_align"] == 0


class _Endpoint(nn.Module):
    def forward_logits(self, x, image, s, t):
        return x * (1.0 + t[:, None, None, None]) + image[:, :1]

    def forward_logits_with_image_feat(self, x, image_feat, s, t):
        return self.forward_logits(x, image_feat, s, t)


@pytest.mark.parametrize("loss_type", ["psd", "csd", "ecld", "esd"])
def test_every_consistency_loss_masks_all_ignore_pixels(loss_type):
    x = torch.softmax(torch.randn(1, 3, 2, 2), dim=1)
    image = torch.randn(1, 3, 2, 2)
    precision = {"jvp_dtype": None if loss_type == "psd" else "fp32", "numerical_dtype": "fp32"}
    config = {
        "flow": {"time_eps": 1.0e-5, "probability_eps": 1.0e-8},
        "loss": {"consistency": {
            "precision": precision,
            "ecld": {"ec_weight": 4.0, "td_weight": 2.0, "time_weighting": "none"},
            "adaptive_kl": {"enabled": False},
            "invalid_teacher": {"strategy": "mask_pixel", "log_eps": 1.0e-6, "skip_batch_threshold": None},
        }},
    }
    result = compute_consistency_loss(
        loss_type, model=_Endpoint(), x_s=x, image=image, image_feat=image,
        s=torch.tensor([0.1]), u=torch.tensor([0.4]) if loss_type == "psd" else None,
        t=torch.tensor([0.8]), precision=precision, config=config,
        valid_mask=torch.zeros(1, 2, 2, dtype=torch.bool),
    )
    assert result.loss == 0
    assert torch.isfinite(result.loss)


def test_ade_metric_ignores_gt_zero_but_retains_prediction_zero_as_error():
    metrics = SegmentationMetrics(
        151, 0, evaluated_class_indices=range(1, 151), nanmean=True
    )
    target = torch.tensor([[0, 1, 1, 2]])
    prediction = torch.tensor([[5, 0, 2, 2]])
    metrics.update(prediction, target)
    result = metrics.compute()
    confusion = result["confusion_matrix"]
    assert sum(confusion[0]) == 0
    assert confusion[1][0] == 1
    assert confusion[1][2] == 1
    assert confusion[2][2] == 1
    assert result["pixel_acc"] == pytest.approx(1 / 3)
    assert result["mIoU"] == pytest.approx(0.25)
    assert result["evaluated_class_indices"] == list(range(1, 151))


def test_terminal_state_is_bilinear_resized_before_argmax_and_after_unpadding():
    terminal = torch.tensor([[[[2.0, 0.0, 99.0], [0.0, 2.0, 99.0]],
                              [[0.0, 2.0, 99.0], [2.0, 0.0, 99.0]]]])
    prediction = terminal_state_to_original_prediction(
        terminal, model_shape=(2, 2), original_shape=(5, 5), align_corners=False
    )
    expected = torch.nn.functional.interpolate(
        terminal[..., :2, :2], (5, 5), mode="bilinear", align_corners=False
    ).argmax(1)
    assert prediction.shape == (1, 5, 5)
    assert torch.equal(prediction, expected)
    assert not bool((prediction == 2).any())


@pytest.mark.parametrize(
    (
        "validation_epochs", "start_step", "max_steps", "interval_steps",
        "validation_mious", "expected_best_steps",
    ),
    [
        ([1], 0, 10, 16000, [0.5], [10]),
        ([], 0, 20, 10, [0.5, 0.4], [10]),
        ([], 0, 20, 10, [0.5, 0.6], [10, 20]),
        ([], 159999, 160000, 16000, [0.7], [160000]),
    ],
)
def test_optimizer_step_budget_counts_accumulation_scheduler_and_final_validation(
    tmp_path,
    monkeypatch,
    validation_epochs,
    start_step,
    max_steps,
    interval_steps,
    validation_mious,
    expected_best_steps,
):
    config = _config()
    config["experiment"]["output_dir"] = str(tmp_path / "run")
    config["runtime"].update({"device": "cpu", "amp": False})
    config["training"].update({
        "epochs": 1, "max_optimizer_steps": max_steps, "grad_accum_steps": 4,
        "validation_epochs": validation_epochs, "checkpoint_interval_epochs": 0,
    })
    config["evaluation"]["interval"]["value"] = interval_steps
    config["wandb"]["enabled"] = False
    batches = [
        (torch.zeros(1, 1), torch.zeros(1, dtype=torch.long))
        for _ in range(100)
    ]
    endpoint, source = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
    counts = {"optimizer": 0, "scheduler": 0}
    real_build_scheduler = trainer.build_scheduler

    class CountingAdamW(torch.optim.AdamW):
        def step(self, *args, **kwargs):
            counts["optimizer"] += 1
            return super().step(*args, **kwargs)

    def counting_optimizer(config, adapter):
        optimizer_config = config["training"]["optimizer"]
        return CountingAdamW([
            {"params": adapter.endpoint_model.parameters(), "name": "model",
             "lr": optimizer_config["parameter_groups"]["model"]["lr"]},
            {"params": adapter.source_model.parameters(), "name": "source",
             "lr": optimizer_config["parameter_groups"]["source"]["lr"]},
        ], lr=optimizer_config["lr"], weight_decay=optimizer_config["weight_decay"],
           betas=tuple(optimizer_config["betas"]))

    def counting_scheduler(config, optimizer):
        scheduler = real_build_scheduler(config, optimizer)
        original = scheduler.step
        def step(*args, **kwargs):
            counts["scheduler"] += 1
            return original(*args, **kwargs)
        scheduler.step = step
        return scheduler

    def objectives(model, **kwargs):
        del kwargs
        loss = sum(parameter.square().mean() for parameter in model.parameters())
        value = loss.detach()
        return {"loss": loss, "stats": {
            "loss_total": value, "loss_diagonal": value,
            "loss_consistency": value, "consistency_effective_weight": value,
        }}

    saved = []
    validation_calls = []
    validation_results = iter(validation_mious)
    monkeypatch.setattr(trainer, "_build_loaders", lambda *args: (batches, [], None))
    monkeypatch.setattr(trainer, "build_models", lambda *args: (endpoint, source))
    monkeypatch.setattr(trainer, "build_optimizer", counting_optimizer)
    monkeypatch.setattr(trainer, "build_scheduler", counting_scheduler)
    monkeypatch.setattr(trainer, "run_model_training_objectives", objectives)
    monkeypatch.setattr(
        trainer, "initialize_or_resume",
        lambda *args, **kwargs: SimpleNamespace(
            start_epoch=0,
            global_step=start_step,
            micro_step=0,
            best_miou=float("-inf"),
        ),
    )
    monkeypatch.setattr(trainer, "_save_training_checkpoint", lambda **kwargs: saved.append(kwargs))
    monkeypatch.setattr(
        trainer,
        "validate",
        lambda *args, **kwargs: (
            validation_calls.append(True)
            or {
                "mIoU": next(validation_results),
                "pixel_acc": 0.5,
                "mAcc": 0.3,
            }
        ),
    )
    result = trainer.run_training(config, joint_entrypoint=True)
    expected_updates = max_steps - start_step
    assert counts == {"optimizer": expected_updates, "scheduler": expected_updates}
    best_saves = [
        checkpoint
        for checkpoint in saved
        if checkpoint["filenames"] == ["best.pt"]
    ]
    assert [checkpoint["global_step"] for checkpoint in best_saves] == expected_best_steps
    assert all(
        {
            "training_model", "optimizer", "scheduler", "scaler",
            "micro_step", "metrics",
        } <= checkpoint.keys()
        for checkpoint in best_saves
    )
    assert all(
        checkpoint["metrics"]["best_mIoU"]
        == checkpoint["metrics"]["mIoU"]
        for checkpoint in best_saves
    )
    assert saved[-1]["global_step"] == max_steps
    assert saved[-1]["micro_step"] == expected_updates * 4
    assert result["optimizer_step"] == max_steps
    assert len(validation_calls) == len(validation_mious)


def test_optimizer_step_validation_interval_and_final_trigger():
    config = _config()
    assert _optimizer_step_validation_trigger(config, 15999) is None
    assert _optimizer_step_validation_trigger(config, 16000) == "optimizer_step_interval"
    assert _optimizer_step_validation_trigger(config, 32000) == "optimizer_step_interval"
    assert _optimizer_step_validation_trigger(config, 160000) == "final_optimizer_step"


def test_poly_scheduler_and_optimizer_step_resume_are_continuous(tmp_path):
    config = _config()
    config["training"]["max_optimizer_steps"] = 10
    config["training"]["scheduler"]["warmup_steps"] = 2
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 1.0e-4}])
    scheduler = build_scheduler(config, optimizer)
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    model = nn.Linear(1, 1)
    source = nn.Linear(1, 1)
    payload = checkpoint_payload(
        config=config, epoch=3, global_step=4, micro_step=16,
        model=model, source_model=source, optimizer=optimizer,
        scheduler=scheduler, scaler=None, metrics={"mIoU": 0.1},
    )
    assert payload["scheduler_step_unit"] == "optimizer_step"
    assert payload["micro_step"] == 16
    path = save_checkpoint(payload, tmp_path, "resume.pt")

    resumed_parameter = nn.Parameter(torch.tensor(1.0))
    resumed_optimizer = torch.optim.AdamW([
        {"params": [resumed_parameter], "lr": 1.0e-4}
    ])
    resumed_scheduler = build_scheduler(config, resumed_optimizer)
    config["checkpoint"]["resume"] = str(path)
    state = initialize_or_resume(
        config, model, source, resumed_optimizer, resumed_scheduler, None
    )
    assert state.start_epoch == 3
    assert state.global_step == 4
    assert state.micro_step == 16
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
