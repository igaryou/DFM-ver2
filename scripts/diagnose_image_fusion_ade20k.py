#!/usr/bin/env python3
"""Diagnose ADE20K Swin/FPN image-feature fusion without changing weights."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_config
from image_fusion_analysis import run_image_fusion_analysis
from utils import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_visualize", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--expected_global_step", type=int, default=160000)
    parser.add_argument("--alpha_match", type=float, default=None)
    parser.add_argument("--beta_match", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.num_visualize < 0:
        raise ValueError("--num_visualize must be non-negative")
    if arguments.max_samples is not None and arguments.max_samples <= 0:
        raise ValueError("--max_samples must be positive")
    if arguments.alpha_match is not None and arguments.alpha_match < 0:
        raise ValueError("--alpha_match must be non-negative")
    if arguments.beta_match is not None and arguments.beta_match < 0:
        raise ValueError("--beta_match must be non-negative")
    config = load_config(arguments.config, arguments.set)
    config["runtime"]["device"] = arguments.device
    run_image_fusion_analysis(
        config,
        checkpoint_path=arguments.checkpoint,
        output_dir=arguments.output_dir,
        num_visualize=arguments.num_visualize,
        seed=arguments.seed,
        max_samples=arguments.max_samples,
        expected_global_step=arguments.expected_global_step,
        alpha_match=arguments.alpha_match,
        beta_match=arguments.beta_match,
        device=resolve_device(arguments.device),
    )


if __name__ == "__main__":
    main()
