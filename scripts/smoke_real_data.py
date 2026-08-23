#!/usr/bin/env python3
"""One-sample real-data smoke: train step, inference, checkpoint save/resume."""

from __future__ import annotations

import argparse
import tempfile
from copy import deepcopy
from pathlib import Path

import torch

from checkpoint import checkpoint_payload, initialize_or_resume, save_checkpoint
from config import load_config
from dataset import build_dataset
from inference import sample_segmentation
from model import DiscreteFlowMapModel
from source_model import UNetSourceGenerator
from training_objectives import DDPCompatibleTrainingModel, compute_model_training_objectives


def tiny_components(config: dict, device: torch.device):
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
    endpoint = DiscreteFlowMapModel(config["model"]).to(device)
    source = UNetSourceGenerator(
        classes, 8, False, 1.0, config["model"]["state_downsample_factor"]
    ).to(device)
    return endpoint, source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    config = deepcopy(load_config(arguments.config))
    config["runtime"]["amp"] = device.type == "cuda"
    config["runtime"]["amp_dtype"] = "bf16"
    config["distributed"]["enabled"] = False
    config["training"]["batch_size"] = 1
    config["training"]["scheduler"]["step_unit"] = "epoch"
    config["loss"]["consistency"]["enabled"] = False

    dataset = build_dataset(
        config, config["dataset"]["train_split"], augment=True
    )
    image, target = dataset[0]
    image = image[None].to(device)
    target = target[None].to(device)

    endpoint, source = tiny_components(config, device)
    adapter = DDPCompatibleTrainingModel(endpoint, source, config)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    amp = torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )
    with amp:
        result = compute_model_training_objectives(
            adapter,
            operation="stage1_objectives",
            image=image,
            target=target,
            epoch_index=0,
            progress_in_epoch=0.0,
        )
    result["loss"].backward()
    optimizer.step()
    scheduler.step()

    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        prediction = sample_segmentation(
            endpoint, source, image, config, num_steps=1
        )

    with tempfile.TemporaryDirectory(prefix="dfm-smoke-") as directory:
        payload = checkpoint_payload(
            config=config, epoch=1, global_step=1,
            model=endpoint, source_model=source,
            optimizer=optimizer, scheduler=scheduler, scaler=None,
            metrics={"mIoU": 0.0},
        )
        path = save_checkpoint(payload, directory, "smoke.pt")
        resumed_config = deepcopy(config)
        resumed_config["checkpoint"].update({
            "resume": str(path), "init_from": None
        })
        resumed_endpoint, resumed_source = tiny_components(resumed_config, device)
        resumed_optimizer = torch.optim.AdamW(
            list(resumed_endpoint.parameters()) + list(resumed_source.parameters()),
            lr=1e-3,
        )
        resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
            resumed_optimizer, lambda _: 1.0
        )
        state = initialize_or_resume(
            resumed_config, resumed_endpoint, resumed_source,
            resumed_optimizer, resumed_scheduler, scaler=None,
        )

    print({
        "dataset": config["dataset"]["name"],
        "training_batch": {
            "image": tuple(image.shape),
            "target_full": tuple(target.shape),
        },
        "state": (
            int(result["stats"]["state_height"]),
            int(result["stats"]["state_width"]),
        ),
        "one_hot_state": (
            image.shape[0], config["dataset"]["num_classes"],
            int(result["stats"]["state_height"]),
            int(result["stats"]["state_width"]),
        ),
        "full_target_one_hot_present": False,
        "prediction": tuple(prediction.shape),
        "loss": float(result["loss"].detach()),
        "finite": bool(torch.isfinite(result["loss"])),
        "resumed_epoch": state.start_epoch,
        "resumed_step": state.global_step,
    })


if __name__ == "__main__":
    main()
