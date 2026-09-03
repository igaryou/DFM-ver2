#!/usr/bin/env python3
"""One-step SegFormer source decoder memory/gradient smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from source_model import build_source_model  # noqa: E402
from training_objectives import DDPCompatibleTrainingModel  # noqa: E402
from utils import autocast_context  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", choices=("custom", "standard"), required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    config = load_config(
        ROOT / "configs/cityscapes/diagonal/source_segformer_b1_32k.yaml",
        [
            f"source.segformer_decoder={args.decoder}",
            f"source.pretrained={str(args.pretrained).lower()}",
            f"training.batch_size={args.batch_size}",
        ],
    )
    torch.manual_seed(config["experiment"]["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["experiment"]["seed"])

    source = build_source_model(config).to(device)
    endpoint = nn.Linear(1, 1, bias=False).to(device).requires_grad_(False)
    adapter = DDPCompatibleTrainingModel(endpoint, source, config).to(device).train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in source.parameters() if parameter.requires_grad),
        lr=config["training"]["optimizer"]["parameter_groups"]["source"]["lr"],
        weight_decay=config["training"]["optimizer"]["weight_decay"],
    )
    image = torch.randn(
        args.batch_size, 3, args.height, args.width, device=device
    )
    target = torch.randint(
        0, config["dataset"]["num_classes"],
        (args.batch_size, args.height, args.width), device=device,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    optimizer.zero_grad(set_to_none=True)
    with autocast_context(config, device):
        result = adapter(
            operation="stage1_objectives",
            image=image,
            target=target,
            epoch_index=0,
            progress_in_epoch=0.0,
            optimizer_step=0,
        )
    loss = result["loss"]
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    decoder_modules = (
        [source.decode_head]
        if args.decoder == "standard"
        else [source.projections, source.decoder]
    )
    report = {
        "decoder": args.decoder,
        "batch_size": args.batch_size,
        "input_shape": list(image.shape),
        "loss": float(loss.detach().float().cpu()),
        "forward": True,
        "backward": any(
            parameter.grad is not None for parameter in source.parameters()
        ),
        "optimizer_step": True,
        "source_trainable_parameters": sum(
            parameter.numel() for parameter in source.parameters()
            if parameter.requires_grad
        ),
        "encoder_parameters": sum(
            parameter.numel() for parameter in source.encoder.parameters()
        ),
        "decoder_parameters": sum(
            parameter.numel()
            for module in decoder_modules for parameter in module.parameters()
        ),
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else 0.0
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / 1024**2
            if device.type == "cuda" else 0.0
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
