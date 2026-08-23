from __future__ import annotations

import argparse

from config import load_config
from trainer import (
    build_optimizer as _build_optimizer,
    build_scheduler,
    run_training,
)
from training_objectives import DDPCompatibleTrainingModel


def build_optimizer(config: dict, model, source_model=None):
    """Backward-compatible optimizer factory for the former train.py API."""
    adapter = (
        model
        if getattr(model, "_is_dfm_ddp_adapter", False)
        else DDPCompatibleTrainingModel(model, source_model, config)
    )
    return _build_optimizer(config, adapter)


def train(config: dict) -> dict:
    """Stage 1 / Stage 2 entrypoint. Joint training has a separate executable."""
    return run_training(config, joint_entrypoint=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DFM Stage 1 or Stage 2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    train(load_config(arguments.config, arguments.set))


if __name__ == "__main__":
    main()
