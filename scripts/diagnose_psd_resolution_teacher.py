#!/usr/bin/env python3
"""Compare production state-resolution PSD with probability-resized full PSD."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_config
from psd_resolution_teacher_analysis import run_psd_resolution_teacher_analysis
from utils import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--psd-weight", type=float, default=None)
    parser.add_argument("--teacher-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.num_batches <= 0 or arguments.batch_size <= 0:
        raise ValueError("--num-batches and --batch-size must be positive")
    if arguments.psd_weight is not None and arguments.psd_weight < 0:
        raise ValueError("--psd-weight must be non-negative")
    if not 0.0 <= arguments.teacher_confidence_threshold <= 1.0:
        raise ValueError("--teacher-confidence-threshold must be in [0, 1]")
    if arguments.num_workers is not None and arguments.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    config = load_config(arguments.config, arguments.set)
    config["runtime"]["device"] = arguments.device
    run_psd_resolution_teacher_analysis(
        config,
        checkpoint_path=arguments.checkpoint,
        output_dir=arguments.output_dir,
        num_batches=arguments.num_batches,
        batch_size=arguments.batch_size,
        psd_weight=arguments.psd_weight,
        teacher_confidence_threshold=arguments.teacher_confidence_threshold,
        seed=arguments.seed,
        device=resolve_device(arguments.device),
        num_workers=arguments.num_workers,
    )


if __name__ == "__main__":
    main()
