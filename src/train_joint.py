from __future__ import annotations

import argparse

from config import load_config
from trainer import run_training


def train_joint(config: dict) -> dict:
    """Random-initialized diagonal + consistency joint training."""
    return run_training(config, joint_entrypoint=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DFM jointly without Stage 1")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    arguments = parser.parse_args()
    train_joint(load_config(arguments.config, arguments.set))


if __name__ == "__main__":
    main()
