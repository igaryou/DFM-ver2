from __future__ import annotations

import torch.nn as nn


def _counts(module: nn.Module | None) -> dict[str, int]:
    if module is None:
        return {"total": 0, "trainable": 0}
    parameters = list(module.parameters())
    return {
        "total": sum(parameter.numel() for parameter in parameters),
        "trainable": sum(
            parameter.numel() for parameter in parameters
            if parameter.requires_grad
        ),
    }


def inspect_model_parameters(endpoint: nn.Module, source: nn.Module | None) -> dict:
    image_encoder = endpoint.image_encoder
    backbone = getattr(image_encoder, "backbone", image_encoder)
    neck = getattr(image_encoder, "neck", None)
    projection = getattr(image_encoder, "projection", None)
    endpoint_counts = _counts(endpoint)
    source_counts = _counts(source)
    return {
        "total_parameters": endpoint_counts["total"] + source_counts["total"],
        "trainable_parameters": (
            endpoint_counts["trainable"] + source_counts["trainable"]
        ),
        "endpoint": endpoint_counts,
        "image_encoder": _counts(image_encoder),
        "backbone": _counts(backbone),
        "neck": _counts(neck),
        "image_projection": _counts(projection),
        "dfm_unet": _counts(endpoint.unet),
        "source": source_counts,
    }
