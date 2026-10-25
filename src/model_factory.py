from __future__ import annotations

import torch

from model import DiscreteFlowMapModel
from source_model import build_source_model


def build_models(config: dict, device: torch.device):
    model_config = dict(config["model"])
    model_config["in_channels"] = config["dataset"].get("in_channels", 3)
    model_config["cfg"] = config.get("cfg", {})
    factor = model_config.get("state_downsample_factor", 4)
    image_height, image_width = config["dataset"]["image_size"]
    model_config["state_size"] = (
        (int(image_height) + factor - 1) // factor,
        (int(image_width) + factor - 1) // factor,
    )
    model = DiscreteFlowMapModel(model_config).to(device)
    if not config.get("training", {}).get("train_endpoint", True):
        model.requires_grad_(False)
        model.eval()
    source_model = build_source_model(config)
    if source_model is not None:
        source_model = source_model.to(device)
    return model, source_model
