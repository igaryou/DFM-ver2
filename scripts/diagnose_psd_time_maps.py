#!/usr/bin/env python3
"""Run PSD conditional time-bin and direct pi-map diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_config
from psd_time_map_analysis import run_psd_time_map_analysis
from utils import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--time-bin-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-num-batches", type=int, default=0)
    parser.add_argument("--psd-weight", type=float, default=None)
    parser.add_argument("--teacher-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-parameter-details", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    positive = (arguments.batch_size, arguments.time_bin_batches, arguments.eval_batch_size)
    if any(value <= 0 for value in positive) or arguments.eval_num_batches < 0:
        raise ValueError("batch sizes must be positive and eval-num-batches non-negative")
    if arguments.psd_weight is not None and arguments.psd_weight < 0:
        raise ValueError("--psd-weight must be non-negative")
    if not 0 <= arguments.teacher_confidence_threshold <= 1:
        raise ValueError("--teacher-confidence-threshold must be in [0,1]")
    if arguments.num_workers is not None and arguments.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    config = load_config(arguments.config, arguments.set)
    config["runtime"]["device"] = arguments.device
    run_psd_time_map_analysis(
        config,
        checkpoint_path=arguments.checkpoint,
        output_dir=arguments.output_dir,
        batch_size=arguments.batch_size,
        time_bin_batches=arguments.time_bin_batches,
        eval_batch_size=arguments.eval_batch_size,
        eval_num_batches=arguments.eval_num_batches,
        psd_weight=arguments.psd_weight,
        teacher_confidence_threshold=arguments.teacher_confidence_threshold,
        seed=arguments.seed,
        num_workers=arguments.num_workers,
        device=resolve_device(arguments.device),
        save_parameter_details=arguments.save_parameter_details,
    )


if __name__ == "__main__":
    main()
