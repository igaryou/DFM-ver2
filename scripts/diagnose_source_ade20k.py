#!/usr/bin/env python3
"""Evaluate source-mean semantics and source-noise ablations without training."""

from __future__ import annotations

import argparse

from config import load_config
from source_diagnostics import (
    DEFAULT_SIGMA_VALUES,
    DEFAULT_STEP_VALUES,
    resolve_diagnostic_checkpoint,
    run_source_diagnostics,
)
from utils import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose a trained ADE20K DFM source without changing weights"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
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
    parser.add_argument("--full_grid", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16"), default=None)
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
    if arguments.amp is not None:
        config["runtime"]["amp"] = arguments.amp
    if arguments.amp_dtype is not None:
        config["runtime"]["amp_dtype"] = arguments.amp_dtype
    checkpoint = resolve_diagnostic_checkpoint(config, arguments.checkpoint)
    run_source_diagnostics(
        config,
        checkpoint_path=checkpoint,
        output_dir=arguments.output_dir,
        sigma_values=arguments.sigma_values,
        step_values=arguments.step_values,
        num_visualize=arguments.num_visualize,
        seed=arguments.seed,
        full_grid=arguments.full_grid,
        max_batches=arguments.max_batches,
        device=resolve_device(arguments.device),
    )


if __name__ == "__main__":
    main()
