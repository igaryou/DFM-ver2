#!/usr/bin/env python3
"""Visualize best-of-N final DFM predictions for all four LIDC annotations."""

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
from inference import sample_lidc_distribution
from lidc_gt_best_diagnostics import (
    compute_lidc_gt_best_diagnostics,
    save_json,
    save_lidc_gt_best_visualization,
    summarize_lidc_gt_best_diagnostics,
)
from model_factory import build_models
from utils import autocast_context, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose best-of-N final LIDC DFM prediction coverage"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-visualizations", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_models(config: dict, checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint must contain a mapping: {checkpoint_path}")
    if checkpoint.get("model") is None or checkpoint.get("source_model") is None:
        raise RuntimeError("Checkpoint requires model and source_model states")
    validate_source_decoder_checkpoint(checkpoint, config, checkpoint_path)
    model, source_model = build_models(config, device)
    if source_model is None:
        raise RuntimeError("LIDC stochastic prediction requires a source_model")
    strict = config["checkpoint"]["strict_model"]
    model.load_state_dict(_without_module_prefix(checkpoint["model"]), strict=strict)
    source_model.load_state_dict(
        _without_module_prefix(checkpoint["source_model"]), strict=strict
    )
    model.eval()
    source_model.eval()
    return model, source_model, checkpoint.get("epoch")


def run_best_prediction_diagnostics(
    model,
    source_model,
    loader,
    output_dir: str | Path,
    *,
    config: dict,
    device: torch.device,
    num_visualizations: int,
    num_samples: int,
    num_steps: int,
    seed: int,
    summary_metadata: dict[str, object] | None = None,
) -> list[dict]:
    """Generate final predictions through the production stochastic helper."""
    destination = Path(output_dir)
    samples = []
    model.eval()
    source_model.eval()
    config = {
        **config,
        "evaluation": {**config["evaluation"], "num_steps": num_steps},
    }
    for batch in loader:
        if len(samples) >= num_visualizations:
            break
        if (
            not isinstance(batch, dict)
            or "image" not in batch
            or "masks" not in batch
        ):
            raise ValueError("LIDC batch requires image and masks")
        batch_size = batch["image"].shape[0]
        for batch_index in range(batch_size):
            if len(samples) >= num_visualizations:
                break
            sample_index = len(samples)
            image = batch["image"][batch_index:batch_index + 1].to(device)
            ground_truths = batch["masks"][batch_index].to(device)
            cuda_devices = []
            if device.type == "cuda":
                cuda_devices = [
                    torch.cuda.current_device()
                    if device.index is None else device.index
                ]
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed + sample_index)
                with torch.no_grad(), autocast_context(config, device):
                    predictions, _ = sample_lidc_distribution(
                        model, source_model, image, config, [num_samples]
                    )
            final_predictions = predictions[0]
            sample_ids = batch.get("sample_id")
            sample_id = None if sample_ids is None else str(sample_ids[batch_index])
            diagnostics = compute_lidc_gt_best_diagnostics(
                final_predictions,
                ground_truths,
                sample_index=sample_index,
                num_steps=num_steps,
                sample_id=sample_id,
            )
            save_lidc_gt_best_visualization(
                image[0], ground_truths, final_predictions, diagnostics,
                destination / f"sample_{sample_index:04d}.png",
            )
            save_json(
                diagnostics,
                destination / "metrics" / f"sample_{sample_index:04d}.json",
            )
            samples.append(diagnostics)
    summary = summarize_lidc_gt_best_diagnostics(
        samples,
        {
            **(summary_metadata or {}),
            "num_samples": num_samples,
            "num_steps": num_steps,
            "seed": seed,
        },
    )
    save_json(summary, destination / "metrics" / "summary.json")
    return samples


def main() -> None:
    arguments = build_parser().parse_args()
    config = load_config(arguments.config)
    if config["dataset"]["name"] != "lidc":
        raise ValueError("This script requires dataset.name=lidc")
    if arguments.num_visualizations < 0:
        raise ValueError("--num-visualizations must be non-negative")
    configured_counts = config["evaluation"]["stochastic_num_samples"]
    num_samples = (
        arguments.num_samples
        if arguments.num_samples is not None
        else (
            configured_counts[-1]
            if configured_counts else config["evaluation"]["num_samples"]
        )
    )
    num_steps = (
        arguments.num_steps
        if arguments.num_steps is not None
        else config["evaluation"]["num_steps"]
    )
    split = arguments.split or config["evaluation"]["split"]
    if num_samples <= 0 or num_steps <= 0:
        raise ValueError("--num-samples and --num-steps must be positive")
    device = resolve_device(config["runtime"]["device"])
    model, source_model, checkpoint_epoch = load_models(
        config, arguments.checkpoint, device
    )
    loader = DataLoader(
        build_dataset(config, split, augment=False),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["dataset"]["pin_memory"],
    )
    metadata: dict[str, object] = {
        "checkpoint_path": str(Path(arguments.checkpoint).resolve()),
        "config_path": str(Path(arguments.config).resolve()),
        "split": split,
        "num_visualizations": arguments.num_visualizations,
    }
    if checkpoint_epoch is not None:
        metadata["checkpoint_epoch"] = checkpoint_epoch
    results = run_best_prediction_diagnostics(
        model, source_model, loader, arguments.output_dir,
        config=config, device=device,
        num_visualizations=arguments.num_visualizations,
        num_samples=num_samples, num_steps=num_steps, seed=arguments.seed,
        summary_metadata=metadata,
    )
    print(
        f"Saved {len(results)} best-of-N LIDC diagnostics to "
        f"{arguments.output_dir}"
    )


if __name__ == "__main__":
    main()
