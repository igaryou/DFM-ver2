from __future__ import annotations

import torch

from model import DiscreteFlowMapModel
from source_model import build_source_model


def build_models(config: dict, device: torch.device):
    model = DiscreteFlowMapModel(config["model"]).to(device)
    if not config.get("training", {}).get("train_endpoint", True):
        model.requires_grad_(False)
        model.eval()
    source_model = build_source_model(config)
    if source_model is not None:
        source_model = source_model.to(device)
    return model, source_model
