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
from lidc_source_diagnostics import (
    compute_lidc_source_diagnostics,
    save_diagnostics_json,
    summarize_lidc_source_diagnostics,
)
from utils import autocast_context, resolve_device
from visualization import save_lidc_source_multisample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize a trained LIDC source distribution without DFM sampling"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-visualizations", type=int, default=32)
    parser.add_argument("--num-source-samples", type=int, default=16)
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=42)
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
    source_model._visualization_checkpoint_epoch = checkpoint.get("epoch")
    return source_model


def save_source_visualizations(
    source_model,
    loader,
    output_dir: str | Path,
    num_visualizations: int,
    config: dict,
    device: torch.device,
    num_source_samples: int = 16,
    seed: int = 42,
    summary_metadata: dict | None = None,
) -> list[dict]:
    """Compute source statistics once per batch and draw deterministic samples."""
    destination = Path(output_dir)
    saved = 0
    diagnostics = []
    source_model.eval()
    with torch.no_grad():
        for batch in loader:
            if saved >= num_visualizations:
                break
            if (
                not isinstance(batch, dict)
                or "image" not in batch
                or "target" not in batch
            ):
                raise ValueError("LIDC visualization batches require image and target")
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"]
            sample_ids = batch.get("sample_id")
            with autocast_context(config, device):
                mu, logvar = source_model.forward_statistics(image)
            if mu.shape != logvar.shape or mu.ndim != 4:
                raise RuntimeError(
                    "source statistics must return mu/logvar with shape [B,C,H,W]"
                )
            count = min(image.shape[0], num_visualizations - saved)
            for sample_index in range(count):
                x0_samples, _ = sample_source_distribution(
                    mu[sample_index], logvar[sample_index], num_source_samples,
                    seed=seed, sample_index=saved,
                )
                save_lidc_source_multisample(
                    image=image[sample_index],
                    target=target[sample_index],
                    mu=mu[sample_index],
                    x0_samples=x0_samples,
                    path=destination / f"sample_{saved:04d}.png",
                )
                sample_id = (
                    str(sample_ids[sample_index])
                    if sample_ids is not None else None
                )
                sample_metrics = compute_lidc_source_diagnostics(
                    mu[sample_index],
                    logvar[sample_index],
                    x0_samples,
                    target[sample_index],
                    sample_index=saved,
                    sample_id=sample_id,
                )
                save_diagnostics_json(
                    sample_metrics,
                    destination / "metrics" / f"sample_{saved:04d}.json",
                )
                diagnostics.append(sample_metrics)
                saved += 1
    summary = summarize_lidc_source_diagnostics(
        diagnostics,
        {
            **(summary_metadata or {}),
            "num_source_samples": int(num_source_samples),
        },
    )
    save_diagnostics_json(summary, destination / "metrics" / "summary.json")
    return diagnostics


def sample_source_distribution(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    num_source_samples: int,
    *,
    seed: int,
    sample_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample x0 with checkpoint-independent noise for one dataset sample."""
    if num_source_samples <= 0:
        raise ValueError("num_source_samples must be positive")
    if mu.shape != logvar.shape or mu.ndim != 3:
        raise ValueError("mu and logvar must have identical [C,H,W] shapes")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(sample_index))
    epsilon_cpu = torch.randn(
        (num_source_samples, *mu.shape),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    epsilon = epsilon_cpu.to(device=mu.device, dtype=mu.dtype)
    std = torch.exp(0.5 * logvar)
    x0_samples = mu.unsqueeze(0) + std.unsqueeze(0) * epsilon
    return x0_samples, epsilon


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.num_visualizations < 0:
        raise ValueError("--num-visualizations must be non-negative")
    if arguments.num_source_samples <= 0:
        raise ValueError("--num-source-samples must be positive")
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
    summary_metadata = {
        "checkpoint_path": str(Path(arguments.checkpoint).resolve()),
        "config_path": str(Path(arguments.config).resolve()),
        "split": arguments.split,
        "seed": int(arguments.seed),
        "num_visualizations": int(arguments.num_visualizations),
    }
    checkpoint_epoch = getattr(
        source_model, "_visualization_checkpoint_epoch", None
    )
    if checkpoint_epoch is not None:
        summary_metadata["checkpoint_epoch"] = checkpoint_epoch
    diagnostics = save_source_visualizations(
        source_model,
        loader,
        arguments.output_dir,
        arguments.num_visualizations,
        config,
        device,
        num_source_samples=arguments.num_source_samples,
        seed=arguments.seed,
        summary_metadata=summary_metadata,
    )
    print(
        f"Saved {len(diagnostics)} source visualizations and diagnostics to "
        f"{Path(arguments.output_dir)}"
    )


if __name__ == "__main__":
    main()
