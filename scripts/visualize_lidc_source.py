#!/usr/bin/env python3
"""Visualize raw LIDC source mu and x0 from a trained checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from checkpoint import _without_module_prefix, validate_source_decoder_checkpoint
from config import load_config
from dataset import build_dataset
from model_factory import build_models
from utils import autocast_context, resolve_device
from visualization import save_lidc_source_mu_x0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize a trained LIDC source distribution without DFM sampling"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-visualizations", type=int, default=32)
    parser.add_argument("--split", default="val")
    return parser


def load_checkpoint_source(
    config: dict, checkpoint_path: str | Path, device: torch.device
):
    """Build the configured models and restore only the source-model weights."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint must contain a mapping: {checkpoint_path}")
    source_state = checkpoint.get("source_model")
    if source_state is None:
        raise RuntimeError(f"Checkpoint has no source_model state: {checkpoint_path}")
    validate_source_decoder_checkpoint(checkpoint, config, checkpoint_path)
    _, source_model = build_models(config, device)
    if source_model is None:
        raise RuntimeError("Configured source does not construct a source_model")
    source_model.load_state_dict(
        _without_module_prefix(source_state),
        strict=config["checkpoint"]["strict_model"],
    )
    source_model.eval()
    return source_model


def save_source_visualizations(
    source_model,
    loader,
    output_dir: str | Path,
    num_visualizations: int,
    config: dict,
    device: torch.device,
) -> int:
    """Run one source forward per batch and save its exact raw mu/x0 tensors."""
    destination = Path(output_dir)
    saved = 0
    source_model.eval()
    with torch.no_grad():
        for batch in loader:
            if saved >= num_visualizations:
                break
            if not isinstance(batch, dict) or "image" not in batch:
                raise ValueError("LIDC visualization batches must contain image")
            image = batch["image"].to(device, non_blocking=True)
            with autocast_context(config, device):
                x0, mu, _ = source_model(image)
            if x0.shape != mu.shape or x0.ndim != 4:
                raise RuntimeError(
                    "source_model must return x0 and mu with shape [B,C,H,W]"
                )
            count = min(image.shape[0], num_visualizations - saved)
            for sample_index in range(count):
                save_lidc_source_mu_x0(
                    mu[sample_index],
                    x0[sample_index],
                    destination / f"sample_{saved:04d}.png",
                    foreground_channel=1,
                    image=image[sample_index],
                )
                saved += 1
    return saved


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.num_visualizations < 0:
        raise ValueError("--num-visualizations must be non-negative")
    config = load_config(arguments.config)
    if config["dataset"]["name"] != "lidc":
        raise ValueError("visualize_lidc_source.py requires dataset.name=lidc")
    device = resolve_device(config["runtime"]["device"])
    source_model = load_checkpoint_source(config, arguments.checkpoint, device)
    dataset = build_dataset(config, arguments.split, augment=False)
    batch_size = max(
        1,
        min(config["evaluation"]["batch_size"], arguments.num_visualizations or 1),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["dataset"]["pin_memory"],
    )
    saved = save_source_visualizations(
        source_model,
        loader,
        arguments.output_dir,
        arguments.num_visualizations,
        config,
        device,
    )
    print(f"Saved {saved} source visualizations to {Path(arguments.output_dir)}")


if __name__ == "__main__":
    main()
