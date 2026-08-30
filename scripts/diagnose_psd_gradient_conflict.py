#!/usr/bin/env python3
"""Measure gradient conflict between Cityscapes primary CE and raw PSD."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_config
from gradient_conflict_analysis import run_gradient_conflict_analysis
from utils import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training config or config_resolved.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--psd-weight", type=float, default=None, help="Defaults to loss.consistency.weight")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.num_batches <= 0:
        raise ValueError("--num-batches must be positive")
    if arguments.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if arguments.num_workers is not None and arguments.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if arguments.psd_weight is not None and arguments.psd_weight < 0:
        raise ValueError("--psd-weight must be non-negative")
    config = load_config(arguments.config, arguments.set)
    config["runtime"]["device"] = arguments.device
    run_gradient_conflict_analysis(
        config,
        checkpoint_path=arguments.checkpoint,
        output_dir=arguments.output_dir,
        num_batches=arguments.num_batches,
        batch_size=arguments.batch_size,
        psd_weight=arguments.psd_weight,
        seed=arguments.seed,
        device=resolve_device(arguments.device),
        num_workers=arguments.num_workers,
    )


if __name__ == "__main__":
    main()
