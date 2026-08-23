from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from checkpoint import (
    checkpoint_payload,
    initialize_or_resume,
    save_checkpoint,
)
from config import load_config
from inference import sample_segmentation
from model import DiscreteFlowMapModel
from source_model import UNetSourceGenerator
from training_objectives import DDPCompatibleTrainingModel, compute_model_training_objectives


ROOT = Path(__file__).parents[1]


def _make_smoke_components(config: dict):
    classes = config["dataset"]["num_classes"]
    config["model"].update({
        "fusion_channels": 8,
        "rrdb_blocks": 1,
        "rrdb_growth_channels": 2,
        "unet": {
            "base_channels": 8,
            "channel_mults": [1, 2],
            "num_res_blocks": 1,
            "attention_levels": [],
            "num_heads": 1,
            "dropout": 0.0,
            "time_embedding_dim": 16,
        },
    })
    config["source"].update({
        "prior_type": "image_gaussian",
        "backbone": "unet",
        "decoder_channels": 8,
        "learned_logvar": False,
        "fixed_std": 1.0,
        "pretrained": False,
        "freeze": False,
        "freeze_encoder": False,
    })
    endpoint = DiscreteFlowMapModel(config["model"])
    source = UNetSourceGenerator(
        classes, channels=8, learned_logvar=False, fixed_std=1.0,
        state_downsample_factor=4,
    )
    return endpoint, source


@pytest.mark.parametrize(
    "config_name,image_size,supervision",
    [
        ("diagonal_cityscapes.yaml", (32, 64), "align"),
        ("diagonal_ade20k.yaml", (32, 32), "cross_entropy"),
    ],
)
def test_forward_backward_optimizer_evaluation_and_checkpoint_smoke(
    tmp_path, config_name, image_size, supervision
):
    config = load_config(ROOT / "configs" / config_name)
    config = deepcopy(config)
    config["runtime"].update({"amp": False, "device": "cpu"})
    config["distributed"]["enabled"] = False
    config["training"]["label_smoothing"] = 0.0
    config["training"]["batch_size"] = 1
    config["training"]["scheduler"]["step_unit"] = "epoch"
    config["source"]["supervision"] = {"type": supervision, "weight": 0.1}
    endpoint, source = _make_smoke_components(config)
    adapter = DDPCompatibleTrainingModel(endpoint, source, config)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    classes = config["dataset"]["num_classes"]
    image = torch.randn(1, 3, *image_size)
    target = torch.randint(0, classes, (1, *image_size))
    if config["dataset"]["name"] == "ade20k":
        target[:, :4, :4] = 0
    result = compute_model_training_objectives(
        adapter,
        operation="stage1_objectives",
        image=image,
        target=target,
        epoch_index=0,
        progress_in_epoch=0.0,
    )
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    optimizer.step()
    scheduler.step()

    prediction = sample_segmentation(
        endpoint, source, image, config, num_steps=1
    )
    assert prediction.shape == target.shape

    payload = checkpoint_payload(
        config=config,
        epoch=1,
        global_step=1,
        model=endpoint,
        source_model=source,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        metrics={"mIoU": 0.0},
    )
    path = save_checkpoint(payload, tmp_path, f"{config_name}.pt")

    resumed_config = deepcopy(config)
    resumed_config["checkpoint"].update({"resume": str(path), "init_from": None})
    resumed_endpoint, resumed_source = _make_smoke_components(resumed_config)
    resumed_optimizer = torch.optim.AdamW(
        list(resumed_endpoint.parameters()) + list(resumed_source.parameters()), lr=1e-3
    )
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
        resumed_optimizer, lambda _: 1.0
    )
    state = initialize_or_resume(
        resumed_config,
        resumed_endpoint,
        resumed_source,
        resumed_optimizer,
        resumed_scheduler,
        scaler=None,
    )
    assert state.start_epoch == 1
    assert state.global_step == 1
