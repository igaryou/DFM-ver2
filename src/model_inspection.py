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


def inspect_source_parameters(source: nn.Module | None) -> dict:
    source_decoder = getattr(source, "decode_head", None)
    if source_decoder is None:
        source_decoder = getattr(source, "decoder", None)
    return {
        "source": _counts(source),
        "source_encoder": _counts(getattr(source, "encoder", None)),
        "source_decoder": _counts(source_decoder),
        "source_projections": _counts(getattr(source, "projections", None)),
    }


def inspect_model_parameters(endpoint: nn.Module, source: nn.Module | None) -> dict:
    image_encoder = endpoint.image_encoder
    backbone = getattr(image_encoder, "backbone", image_encoder)
    neck = getattr(image_encoder, "neck", None)
    projection = getattr(image_encoder, "projection", None)
    endpoint_counts = _counts(endpoint)
    source_report = inspect_source_parameters(source)
    source_counts = source_report["source"]
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
        **source_report,
    }
