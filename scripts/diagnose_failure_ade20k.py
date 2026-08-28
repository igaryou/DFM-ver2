#!/usr/bin/env python3
"""Run post-training ADE20K DFM failure analysis without changing weights."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_config
from failure_analysis import (
    DEFAULT_SIGMA_VALUES,
    DEFAULT_STEP_VALUES,
    run_failure_analysis,
)
from utils import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--sigma_values", nargs="+", type=float, default=list(DEFAULT_SIGMA_VALUES)
    )
    parser.add_argument(
        "--step_values", nargs="+", type=int, default=list(DEFAULT_STEP_VALUES)
    )
    parser.add_argument("--num_visualize", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--expected_global_step", type=int, default=160000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.num_visualize < 0:
        raise ValueError("--num_visualize must be non-negative")
    if arguments.max_batches is not None and arguments.max_batches <= 0:
        raise ValueError("--max_batches must be positive")
    config = load_config(arguments.config, arguments.set)
    config["runtime"]["device"] = arguments.device
    run_failure_analysis(
        config,
        checkpoint_path=arguments.checkpoint,
        output_dir=arguments.output_dir,
        sigma_values=arguments.sigma_values,
        step_values=arguments.step_values,
        num_visualize=arguments.num_visualize,
        seed=arguments.seed,
        max_batches=arguments.max_batches,
        expected_global_step=arguments.expected_global_step,
        device=resolve_device(arguments.device),
    )


if __name__ == "__main__":
    main()
