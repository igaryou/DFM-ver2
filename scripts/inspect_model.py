#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from config import load_config
from model import TRANSFORMER_IMAGE_ENCODER_MODEL_IDS
from model_factory import build_models
from model_inspection import inspect_model_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DFM parameter counts")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    arguments = parser.parse_args()
    config = load_config(arguments.config, arguments.set)

    image_encoder = config["model"]["image_encoder"]
    if image_encoder["type"] in TRANSFORMER_IMAGE_ENCODER_MODEL_IDS:
        model_id = TRANSFORMER_IMAGE_ENCODER_MODEL_IDS[
            image_encoder["type"]
        ][image_encoder["variant"]]
        if image_encoder["pretrained"]:
            print(f"Loading/downloading endpoint pretrained weights: {model_id}")
    if config["source"]["type"] == "task_finetuned_segformer":
        print(
            "Loading/downloading task-finetuned source weights: "
            f"{config['source']['model_id']}"
        )
    elif config["source"]["pretrained"]:
        print(
            "Loading/downloading legacy source encoder weights: "
            f"nvidia/mit-{config['source']['segformer_variant']}"
        )
    endpoint, source = build_models(config, torch.device("cpu"))
    print(json.dumps(inspect_model_parameters(endpoint, source), indent=2))


if __name__ == "__main__":
    main()
