"""Diagnostics for conflicts between supervised and PSD gradients.

This module is intentionally independent from the production trainer.  It reuses
the production objective helpers, never calls backward/optimizer.step, and only
operates on gradients returned by ``torch.autograd.grad``.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import DataLoader

import losses
from discrete_flow_maps import (
    linear_path,
    sample_consistency_times,
    sample_prior,
    sample_stage1_times,
)
from model_factory import build_models
from state_space import prepare_state_targets, resize_continuous, state_spatial_size
from dataset import build_dataset
from utils import autocast_context, seed_everything


Gradient = tuple[torch.Tensor | None, ...]
EPS = 1.0e-12


@dataclass(frozen=True)
class DiagnosticGraph:
    primary: torch.Tensor
    source_weighted: torch.Tensor
    psd: torch.Tensor
    primary_ce: torch.Tensor
    source_supervision: torch.Tensor
    s: torch.Tensor
    u: torch.Tensor
    t: torch.Tensor
    valid_pixel_ratio: torch.Tensor
    teacher_prob_state: torch.Tensor
    student_prob_state: torch.Tensor
    target_state: torch.Tensor
    target_full: torch.Tensor
    valid_mask_state: torch.Tensor | None
    valid_mask_full: torch.Tensor | None


def _dot(first: Gradient, second: Gradient, indices: Sequence[int]) -> float:
    total = 0.0
    for index in indices:
        left, right = first[index], second[index]
        if left is not None and right is not None:
            total += float(torch.sum(left.detach().float() * right.detach().float()).cpu())
    return total


def _norm(gradient: Gradient, indices: Sequence[int]) -> float:
    return math.sqrt(max(_dot(gradient, gradient, indices), 0.0))


def add_gradients(first: Gradient, second: Gradient, scale: float = 1.0) -> Gradient:
    """Return ``first + scale * second`` without touching ``parameter.grad``."""
    combined: list[torch.Tensor | None] = []
    for left, right in zip(first, second):
        if left is None:
            combined.append(None if right is None else right.detach() * scale)
        elif right is None:
            combined.append(left.detach())
        else:
            combined.append(left.detach() + scale * right.detach())
    return tuple(combined)


def scale_gradient(gradient: Gradient, scale: float) -> Gradient:
    return tuple(None if value is None else value.detach() * scale for value in gradient)


def gradient_pair_metrics(
    first: Gradient,
    second: Gradient,
    indices: Sequence[int] | None = None,
    *,
    eps: float = EPS,
) -> dict[str, float]:
    """Compute a float32-accumulated dot, norms, cosine, and angle."""
    selected = tuple(range(len(first))) if indices is None else tuple(indices)
    dot = _dot(first, second, selected)
    first_norm = _norm(first, selected)
    second_norm = _norm(second, selected)
    cosine = dot / (first_norm * second_norm + eps)
    cosine = min(max(cosine, -1.0), 1.0)
    return {
        "dot": dot,
        "first_norm": first_norm,
        "second_norm": second_norm,
        "cosine": cosine,
        "angle_degrees": math.degrees(math.acos(cosine)),
        "conflict": bool(cosine < 0.0),
    }


def projection_diagnostics(
    base: Gradient,
    psd: Gradient,
    indices: Sequence[int] | None = None,
    *,
    eps: float = EPS,
) -> dict[str, float | bool]:
    """Evaluate PCGrad-style projection on detached diagnostic tensors only."""
    selected = tuple(range(len(base))) if indices is None else tuple(indices)
    dot = _dot(psd, base, selected)
    base_sq = _dot(base, base, selected)
    raw_coefficient = dot / (base_sq + eps)
    applied = dot < 0.0
    coefficient = raw_coefficient if applied else 0.0
    projected = add_gradients(psd, base, scale=-coefficient)
    removed = scale_gradient(base, coefficient)
    raw_norm = _norm(psd, selected)
    projected_norm = _norm(projected, selected)
    removed_norm = _norm(removed, selected)
    projected_pair = gradient_pair_metrics(base, projected, selected, eps=eps)
    return {
        "projection_applied": applied,
        "projection_coefficient": coefficient,
        "raw_projection_coefficient": raw_coefficient,
        "projected_psd_norm": projected_norm,
        "projected_psd_to_raw_psd_norm_ratio": projected_norm / (raw_norm + eps),
        "removed_component_norm": removed_norm,
        "removed_component_fraction": removed_norm / (raw_norm + eps),
        "base_vs_projected_psd_cosine": projected_pair["cosine"],
    }


def _autograd(loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter], retain: bool) -> Gradient:
    if not loss.requires_grad:
        return tuple(None for _ in parameters)
    return tuple(torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain,
        allow_unused=True,
        create_graph=False,
    ))


def build_diagnostic_graph(
    config: dict[str, Any],
    endpoint_model: torch.nn.Module,
    source_model: torch.nn.Module | None,
    image: torch.Tensor,
    target: torch.Tensor,
    consistency_times: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> DiagnosticGraph:
    """Build one production-equivalent joint graph while bypassing PSD schedule."""
    batch_size = image.shape[0]
    state_size = state_spatial_size(
        image, config.get("model", {}).get("state_downsample_factor", 1)
    )
    ignore_index = config["loss"].get("ignore_index")
    targets = prepare_state_targets(
        target,
        num_classes=config["dataset"]["num_classes"],
        state_size=state_size,
        ignore_index=ignore_index,
        mask_pixel_losses=config["loss"].get("mask_pixel_losses", False),
    )
    # Exactly one source/noise sample and one image feature graph per batch.
    x0, source_stats = sample_prior(
        config,
        image,
        targets.one_hot_state,
        source_model,
        target_full=targets.target_full,
        valid_mask_full=targets.valid_mask_full,
    )
    image_feat = endpoint_model.encode_image(image)
    time_config = config["time_sampling"]
    diagonal_time = sample_stage1_times(
        batch_size,
        image.device,
        time_config["min_time"],
        time_config["max_time"],
    )
    if consistency_times is None:
        s, u, t = sample_consistency_times(
            "psd",
            batch_size,
            image.device,
            time_config["min_time"],
            time_config["max_time"],
            time_config["min_gap"],
        )
        assert u is not None
    else:
        s, u, t = consistency_times
        if any(value.shape != (batch_size,) for value in (s, u, t)):
            raise ValueError("diagnostic consistency_times must each have shape [B]")
        if any(value.device != image.device for value in (s, u, t)):
            raise ValueError("diagnostic consistency_times must be on image.device")
        if not bool(((s < u) & (u < t)).all()):
            raise ValueError("diagnostic PSD consistency_times require s < u < t")

    diagonal_state = linear_path(x0, targets.one_hot_state, diagonal_time)
    diagonal_logits = endpoint_model.forward_logits_with_image_feat(
        diagonal_state, image_feat, diagonal_time, diagonal_time
    )
    diagonal_logits_full = resize_continuous(
        diagonal_logits, targets.target_full.shape[-2:]
    )
    primary_ce = losses.diagonal_cross_entropy(
        diagonal_logits_full,
        targets.target_full,
        config["training"]["label_smoothing"],
        ignore_index=(ignore_index if targets.valid_mask_full is not None else None),
    ).float()
    primary = config["loss"]["primary"]["weight"] * primary_ce

    consistency_state = linear_path(x0, targets.one_hot_state, s)
    consistency = losses.compute_consistency_loss(
        "psd",
        model=endpoint_model,
        x_s=consistency_state,
        image=image,
        image_feat=image_feat,
        s=s,
        u=u,
        t=t,
        precision=config["loss"]["consistency"]["precision"],
        config=config,
        valid_mask=targets.valid_mask_state,
    )
    if consistency.teacher_prob is None or consistency.student_prob is None:
        raise RuntimeError("PSD diagnostic requires teacher and student probabilities")
    source_weighted = source_stats.get(
        "weighted_source_supervision", source_stats["weighted_align"]
    ).float()
    source_supervision = source_stats.get(
        "loss_source_supervision", source_stats["loss_source_align"]
    ).float()
    valid_ratio = (
        targets.valid_mask_full.float().mean()
        if targets.valid_mask_full is not None
        else image.new_ones(())
    )
    return DiagnosticGraph(
        primary=primary.float(),
        source_weighted=source_weighted,
        psd=consistency.loss.float(),
        primary_ce=primary_ce,
        source_supervision=source_supervision,
        s=s,
        u=u,
        t=t,
        valid_pixel_ratio=valid_ratio,
        teacher_prob_state=consistency.teacher_prob,
        student_prob_state=consistency.student_prob,
        target_state=targets.target_state,
        target_full=targets.target_full,
        valid_mask_state=targets.valid_mask_state,
        valid_mask_full=targets.valid_mask_full,
    )


def module_groups(parameter_names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Return overlapping, architecture-aware parameter groups."""
    predicates = {
        "all_trainable_parameters": lambda name: True,
        "endpoint_model": lambda name: name.startswith("endpoint_model."),
        "image_encoder": lambda name: name.startswith("endpoint_model.image_encoder."),
        "swin_backbone": lambda name: name.startswith("endpoint_model.image_encoder.backbone."),
        "ddp_fpn_neck": lambda name: name.startswith("endpoint_model.image_encoder.neck."),
        "image_encoder_projection": lambda name: name.startswith("endpoint_model.image_encoder.projection."),
        "mask_encoder": lambda name: name.startswith("endpoint_model.mask_encoder."),
        "unet": lambda name: name.startswith("endpoint_model.unet."),
        "unet_input": lambda name: name.startswith("endpoint_model.unet.input."),
        "unet_down_path": lambda name: name.startswith(("endpoint_model.unet.down_blocks.", "endpoint_model.unet.downsamples.")),
        "unet_middle": lambda name: name.startswith("endpoint_model.unet.middle."),
        "unet_up_path": lambda name: name.startswith(("endpoint_model.unet.up_blocks.", "endpoint_model.unet.upsamples.")),
        "unet_output": lambda name: name.startswith(("endpoint_model.unet.out_norm.", "endpoint_model.unet.out.")),
        "unet_embed_s": lambda name: name.startswith("endpoint_model.unet.embed_s."),
        "unet_embed_delta": lambda name: name.startswith("endpoint_model.unet.embed_delta."),
        "source_model": lambda name: name.startswith("source_model."),
    }
    return {
        group: tuple(index for index, name in enumerate(parameter_names) if predicate(name))
        for group, predicate in predicates.items()
    }


def parameter_leaf_group(name: str) -> str:
    for group, prefix in (
        ("source_model", "source_model."),
        ("swin_backbone", "endpoint_model.image_encoder.backbone."),
        ("ddp_fpn_neck", "endpoint_model.image_encoder.neck."),
        ("image_encoder_projection", "endpoint_model.image_encoder.projection."),
        ("mask_encoder", "endpoint_model.mask_encoder."),
        ("unet_embed_s", "endpoint_model.unet.embed_s."),
        ("unet_embed_delta", "endpoint_model.unet.embed_delta."),
        ("unet_input", "endpoint_model.unet.input."),
        ("unet_down_path", "endpoint_model.unet.down"),
        ("unet_middle", "endpoint_model.unet.middle."),
        ("unet_up_path", "endpoint_model.unet.up"),
        ("unet_output", "endpoint_model.unet.out"),
        ("image_encoder", "endpoint_model.image_encoder."),
        ("unet", "endpoint_model.unet."),
        ("endpoint_model", "endpoint_model."),
    ):
        if name.startswith(prefix):
            return group
    return "other"


def _complete_metrics(
    primary: Gradient,
    source: Gradient,
    psd: Gradient,
    psd_weight: float,
    indices: Sequence[int],
) -> dict[str, float | bool]:
    # Restrict first so module diagnostics do not allocate full-model temporary
    # gradient tuples for every overlapping group.
    primary = tuple(primary[index] for index in indices)
    source = tuple(source[index] for index in indices)
    psd = tuple(psd[index] for index in indices)
    indices = tuple(range(len(primary)))
    base = add_gradients(primary, source)
    weighted_psd = scale_gradient(psd, psd_weight)
    actual = add_gradients(base, weighted_psd)
    primary_psd = gradient_pair_metrics(primary, psd, indices)
    base_psd = gradient_pair_metrics(base, psd, indices)
    actual_base = gradient_pair_metrics(actual, base, indices)
    primary_norm = _norm(primary, indices)
    base_norm = _norm(base, indices)
    psd_norm = _norm(psd, indices)
    weighted_psd_norm = _norm(weighted_psd, indices)
    result: dict[str, float | bool] = {
        "primary_grad_norm": primary_norm,
        "source_grad_norm": _norm(source, indices),
        "base_grad_norm": base_norm,
        "psd_grad_norm_raw": psd_norm,
        "psd_grad_norm_weighted": weighted_psd_norm,
        "weighted_psd_to_primary_norm_ratio": weighted_psd_norm / (primary_norm + EPS),
        "weighted_psd_to_base_norm_ratio": weighted_psd_norm / (base_norm + EPS),
        "combined_grad_norm": _norm(actual, indices),
        "primary_vs_psd_dot": primary_psd["dot"],
        "primary_vs_psd_cosine": primary_psd["cosine"],
        "primary_vs_psd_angle_degrees": primary_psd["angle_degrees"],
        "primary_vs_psd_conflict": primary_psd["conflict"],
        "base_vs_psd_dot": base_psd["dot"],
        "base_vs_psd_cosine": base_psd["cosine"],
        "base_vs_psd_angle_degrees": base_psd["angle_degrees"],
        "base_vs_psd_conflict": base_psd["conflict"],
        "combined_vs_base_dot": actual_base["dot"],
        "combined_vs_base_cosine": actual_base["cosine"],
        "combined_vs_base_angle_degrees": actual_base["angle_degrees"],
    }
    result.update(projection_diagnostics(base, psd, indices))
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    location = (len(ordered) - 1) * probability
    lower, upper = math.floor(location), math.ceil(location)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def summarize_rows(rows: Sequence[dict[str, Any]], keys: Iterable[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float))]
        if not values:
            continue
        mean = sum(values) / len(values)
        summary[key] = {
            "mean": mean,
            "std": math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
            "median": median(values),
            "min": min(values),
            "max": max(values),
            "p10": _quantile(values, 0.10),
            "p25": _quantile(values, 0.25),
            "p75": _quantile(values, 0.75),
            "p90": _quantile(values, 0.90),
        }
    return summary


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _without_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state.items()}


def _segformer_v5_key_to_v4(key: str) -> str:
    """Translate the known Transformers 5 SegFormer layout to 4.x names."""
    match = re.match(r"^encoder\.stages\.(\d+)\.(.+)$", key)
    if match is None:
        return key
    stage, suffix = match.groups()
    if suffix.startswith("patch_embeddings."):
        return (
            f"encoder.encoder.patch_embeddings.{stage}."
            + suffix.removeprefix("patch_embeddings.")
        )
    if suffix.startswith("layer_norm."):
        return (
            f"encoder.encoder.layer_norm.{stage}."
            + suffix.removeprefix("layer_norm.")
        )
    block = re.match(r"^blocks\.(\d+)\.(.+)$", suffix)
    if block is None:
        return key
    block_index, block_suffix = block.groups()
    replacements = (
        ("layernorm_before.", "layer_norm_1."),
        ("attention.q_proj.", "attention.self.query."),
        ("attention.k_proj.", "attention.self.key."),
        ("attention.v_proj.", "attention.self.value."),
        ("attention.o_proj.", "attention.output.dense."),
        ("attention.sequence_reduction.sequence_reduction.", "attention.self.sr."),
        ("attention.sequence_reduction.layer_norm.", "attention.self.layer_norm."),
        ("layernorm_after.", "layer_norm_2."),
        ("mlp.fc1.", "mlp.dense1."),
        ("mlp.fc2.", "mlp.dense2."),
    )
    for current, legacy in replacements:
        if block_suffix.startswith(current):
            block_suffix = legacy + block_suffix.removeprefix(current)
            break
    return f"encoder.encoder.block.{stage}.{block_index}.{block_suffix}"


def _checkpoint_source_state_for_model(
    state: dict[str, torch.Tensor], source_model: torch.nn.Module
) -> dict[str, torch.Tensor]:
    expected = source_model.state_dict()
    checkpoint_is_v5 = any(key.startswith("encoder.stages.") for key in state)
    model_is_v4 = any(
        key.startswith("encoder.encoder.patch_embeddings.") for key in expected
    )
    if not (checkpoint_is_v5 and model_is_v4):
        return state
    converted = {_segformer_v5_key_to_v4(key): value for key, value in state.items()}
    if len(converted) != len(state):
        raise RuntimeError("SegFormer checkpoint key conversion produced a collision")
    return converted


class _UnusedSwinLayerNormCompatibilityProxy:
    """Accept the production freeze call when old SwinBackbone has no wrapper."""

    @property
    def layernorm(self):
        return self

    def requires_grad_(self, requires_grad: bool = True):
        if requires_grad:
            raise ValueError("The diagnostic Swin compatibility proxy is freeze-only")
        return self


def _build_models_with_swin_api_compatibility(
    config: dict[str, Any], device: torch.device
):
    """Build through production while adapting only the old HF constructor API.

    Transformers 4.46 SwinBackbone directly owns embeddings/encoder and has no
    unused classification-output LayerNorm. Production's freeze call targets that
    redundant norm in the Transformers 5 ``backbone.swin`` wrapper. The temporary
    non-Module proxy makes that call a no-op on v4; it adds no parameter, buffer,
    state_dict key, or forward behavior and is removed immediately after build.
    """
    import transformers

    original_class = transformers.SwinBackbone
    proxy = _UnusedSwinLayerNormCompatibilityProxy()

    class DiagnosticSwinBackbone(original_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not hasattr(self, "swin"):
                self.swin = proxy

    transformers.SwinBackbone = DiagnosticSwinBackbone
    try:
        endpoint, source = build_models(config, device)
    finally:
        transformers.SwinBackbone = original_class
    backbone = getattr(getattr(endpoint, "image_encoder", None), "backbone", None)
    if getattr(backbone, "swin", None) is proxy:
        delattr(backbone, "swin")
    return endpoint, source


def _swin_v5_checkpoint_key_to_v4(key: str) -> str | None:
    prefix = "image_encoder.backbone.swin."
    if not key.startswith(prefix):
        return key
    suffix = key.removeprefix(prefix)
    # This classification-output norm is bypassed by SwinBackbone.forward and
    # was frozen in production. Transformers 4 SwinBackbone does not create it.
    if suffix in {"layernorm.weight", "layernorm.bias"}:
        return None
    key = "image_encoder.backbone." + suffix
    replacements = (
        (".attention.q_proj.", ".attention.self.query."),
        (".attention.k_proj.", ".attention.self.key."),
        (".attention.v_proj.", ".attention.self.value."),
        (".attention.o_proj.", ".attention.output.dense."),
        (
            ".attention.relative_position_bias.relative_position_bias_table",
            ".attention.self.relative_position_bias_table",
        ),
        (".mlp.fc1.", ".intermediate.dense."),
        (".mlp.fc2.", ".output.dense."),
    )
    for current, legacy in replacements:
        key = key.replace(current, legacy)
    return key


def _checkpoint_endpoint_state_for_model(
    state: dict[str, torch.Tensor], endpoint_model: torch.nn.Module
) -> dict[str, torch.Tensor]:
    """Convert only the verified Transformers 5 Swin layout to 4.46 layout."""
    expected = endpoint_model.state_dict()
    checkpoint_is_v5 = any(
        key.startswith("image_encoder.backbone.swin.") for key in state
    )
    model_is_v4 = any(
        key.startswith("image_encoder.backbone.embeddings.") for key in expected
    )
    if checkpoint_is_v5 and model_is_v4:
        converted: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            converted_key = _swin_v5_checkpoint_key_to_v4(key)
            if converted_key is None:
                continue
            if converted_key in converted:
                raise RuntimeError(
                    f"Swin checkpoint key conversion collision: {converted_key}"
                )
            converted[converted_key] = value
        # Transformers 5 treats this fixed window-geometry tensor as
        # non-persistent; 4.46 stores it. It is not a learned parameter.
        expected_buffers = dict(endpoint_model.named_buffers())
        for key, value in expected.items():
            if key in converted:
                continue
            if (
                key.endswith("attention.self.relative_position_index")
                and key in expected_buffers
            ):
                converted[key] = value
        state = converted

    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatches = sorted(
        (key, tuple(state[key].shape), tuple(expected[key].shape))
        for key in set(state) & set(expected)
        if state[key].shape != expected[key].shape
    )
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "Endpoint checkpoint compatibility audit failed before strict load: "
            f"missing={missing}, unexpected={unexpected}, "
            f"shape_mismatches={shape_mismatches}"
        )
    return state


def _load_models(config: dict[str, Any], checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # A full checkpoint is loaded immediately, so pretrained initializers/downloads
    # are deliberately disabled without changing architecture or runtime math.
    import copy
    build_config = copy.deepcopy(config)
    build_config["model"]["image_encoder"]["pretrained"] = False
    build_config["source"]["pretrained"] = False
    build_config["source"]["_load_pretrained"] = False
    build_config["source"]["checkpoint"] = None
    endpoint, source = _build_models_with_swin_api_compatibility(
        build_config, device
    )
    endpoint_state = _without_module_prefix(checkpoint["model"])
    endpoint.load_state_dict(
        _checkpoint_endpoint_state_for_model(endpoint_state, endpoint),
        # The diagnostic always requires exact restoration, independently of a
        # permissive training/import setting in an old resolved config.
        strict=True,
    )
    if source is not None:
        if checkpoint.get("source_model") is None:
            raise RuntimeError("Checkpoint has no source_model state")
        source_state = _without_module_prefix(checkpoint["source_model"])
        source.load_state_dict(
            _checkpoint_source_state_for_model(source_state, source), strict=True
        )
    return checkpoint, endpoint, source


def _heuristic(mean_cosine: float, negative_fraction: float) -> str:
    if mean_cosine < -0.2 or negative_fraction > 0.7:
        return "Strong conflict candidate"
    if mean_cosine < 0.0 or negative_fraction > 0.5:
        return "Moderate conflict"
    return "Little evidence of conflict"


def run_gradient_conflict_analysis(
    config: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path | None,
    num_batches: int,
    batch_size: int,
    psd_weight: float | None,
    seed: int,
    device: torch.device,
    num_workers: int | None = None,
) -> Path:
    if config["loss"]["consistency"]["type"] != "psd":
        raise ValueError("Gradient conflict analysis currently requires consistency.type=psd")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else checkpoint_path.parent / f"gradient_conflict_{checkpoint_path.stem}"
    )
    output.mkdir(parents=True, exist_ok=True)
    weight = (
        float(config["loss"]["consistency"]["weight"])
        if psd_weight is None else float(psd_weight)
    )
    seed_everything(seed)
    checkpoint, endpoint, source = _load_models(config, checkpoint_path, device)
    endpoint.train()
    if source is not None:
        source.train()

    named_parameters = [
        (f"endpoint_model.{name}", parameter)
        for name, parameter in endpoint.named_parameters() if parameter.requires_grad
    ]
    if source is not None:
        named_parameters.extend(
            (f"source_model.{name}", parameter)
            for name, parameter in source.named_parameters() if parameter.requires_grad
        )
    names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]
    groups = module_groups(names)

    dataset = build_dataset(config, config["dataset"]["train_split"], augment=True)
    workers = config["dataset"]["num_workers"] if num_workers is None else num_workers
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        pin_memory=config["dataset"]["pin_memory"],
    )
    batch_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []

    for batch_index, (image, target) in enumerate(loader):
        if batch_index >= num_batches:
            break
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast_context(config, device):
            graph = build_diagnostic_graph(config, endpoint, source, image, target)
        primary_gradient = _autograd(graph.primary, parameters, retain=True)
        psd_gradient = _autograd(graph.psd, parameters, retain=True)
        source_gradient = _autograd(graph.source_weighted, parameters, retain=False)
        all_indices = groups["all_trainable_parameters"]
        global_metrics = _complete_metrics(
            primary_gradient, source_gradient, psd_gradient, weight, all_indices
        )
        batch_rows.append({
            "batch": batch_index,
            "primary_ce": float(graph.primary_ce.detach().cpu()),
            "source_supervision": float(graph.source_supervision.detach().cpu()),
            "source_supervision_weighted": float(graph.source_weighted.detach().cpu()),
            "psd_loss": float(graph.psd.detach().cpu()),
            "s_mean": float(graph.s.detach().float().mean().cpu()),
            "u_mean": float(graph.u.detach().float().mean().cpu()),
            "t_mean": float(graph.t.detach().float().mean().cpu()),
            "valid_pixel_ratio": float(graph.valid_pixel_ratio.detach().cpu()),
            **global_metrics,
        })
        for group, indices in groups.items():
            if indices:
                module_rows.append({
                    "batch": batch_index,
                    "module_group": group,
                    "num_parameters": sum(parameters[index].numel() for index in indices),
                    **_complete_metrics(
                        primary_gradient, source_gradient, psd_gradient, weight, indices
                    ),
                })
        base_gradient = add_gradients(primary_gradient, source_gradient)
        for index, (name, parameter) in enumerate(named_parameters):
            primary_pair = gradient_pair_metrics(primary_gradient, psd_gradient, [index])
            base_pair = gradient_pair_metrics(base_gradient, psd_gradient, [index])
            parameter_rows.append({
                "batch": batch_index,
                "parameter_name": name,
                "module_group": parameter_leaf_group(name),
                "numel": parameter.numel(),
                "primary_grad_norm": _norm(primary_gradient, [index]),
                "source_grad_norm": _norm(source_gradient, [index]),
                "base_grad_norm": _norm(base_gradient, [index]),
                "psd_grad_norm_raw": _norm(psd_gradient, [index]),
                "psd_grad_norm_weighted": weight * _norm(psd_gradient, [index]),
                "primary_vs_psd_cosine": primary_pair["cosine"],
                "base_vs_psd_cosine": base_pair["cosine"],
                "base_vs_psd_dot": base_pair["dot"],
                "conflict": base_pair["conflict"],
            })
        del graph, primary_gradient, psd_gradient, source_gradient, base_gradient

    if len(batch_rows) != num_batches:
        raise RuntimeError(f"Requested {num_batches} batches but loader produced {len(batch_rows)}")
    _write_csv(output / "gradient_conflict_batches.csv", batch_rows)
    _write_csv(output / "gradient_conflict_modules.csv", module_rows)
    _write_csv(output / "gradient_conflict_parameters.csv", parameter_rows)
    arguments = {
        "config": config["runtime"].get("config_path"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "output_dir": str(output),
        "num_batches": num_batches,
        "batch_size": batch_size,
        "psd_weight": weight,
        "seed": seed,
        "device": str(device),
        "num_workers": workers,
    }
    (output / "diagnostic_args.json").write_text(
        json.dumps(arguments, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    numeric_keys = [
        key for key, value in batch_rows[0].items()
        if key != "batch" and isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    global_stats = summarize_rows(batch_rows, numeric_keys)
    global_stats["primary_vs_psd_negative_fraction"] = {
        "mean": sum(row["primary_vs_psd_cosine"] < 0 for row in batch_rows) / len(batch_rows)
    }
    global_stats["base_vs_psd_negative_fraction"] = {
        "mean": sum(row["base_vs_psd_cosine"] < 0 for row in batch_rows) / len(batch_rows)
    }
    # Flat values retain the requested quick-use schema. Full distributions live
    # under ``statistics`` in the same global object.
    global_output = {
        "primary_vs_psd_cosine_mean": global_stats["primary_vs_psd_cosine"]["mean"],
        "base_vs_psd_cosine_mean": global_stats["base_vs_psd_cosine"]["mean"],
        "combined_vs_base_cosine_mean": global_stats["combined_vs_base_cosine"]["mean"],
        "primary_vs_psd_negative_fraction": global_stats[
            "primary_vs_psd_negative_fraction"
        ]["mean"],
        "base_vs_psd_negative_fraction": global_stats[
            "base_vs_psd_negative_fraction"
        ]["mean"],
        "primary_grad_norm_mean": global_stats["primary_grad_norm"]["mean"],
        "base_grad_norm_mean": global_stats["base_grad_norm"]["mean"],
        "psd_grad_norm_mean": global_stats["psd_grad_norm_raw"]["mean"],
        "weighted_psd_grad_norm_mean": global_stats[
            "psd_grad_norm_weighted"
        ]["mean"],
        "weighted_psd_to_primary_ratio_mean": global_stats[
            "weighted_psd_to_primary_norm_ratio"
        ]["mean"],
        "weighted_psd_to_base_ratio_mean": global_stats[
            "weighted_psd_to_base_norm_ratio"
        ]["mean"],
        "statistics": global_stats,
    }
    module_summary: dict[str, Any] = {}
    for group in groups:
        rows = [row for row in module_rows if row["module_group"] == group]
        if rows:
            keys = [
                key for key, value in rows[0].items()
                if key not in {"batch", "module_group", "num_parameters"}
                and isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            module_summary[group] = summarize_rows(rows, keys)
    summary = {
        **arguments,
        "global": global_output,
        "modules": module_summary,
        "heuristic": {
            "classification": _heuristic(
                global_stats["base_vs_psd_cosine"]["mean"],
                global_stats["base_vs_psd_negative_fraction"]["mean"],
            ),
            "note": "Exploratory heuristic only; these are not rigorous research thresholds.",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("===== Global Gradient Conflict =====")
    for key in (
        "primary_vs_psd_cosine", "base_vs_psd_cosine",
        "weighted_psd_to_base_norm_ratio", "combined_vs_base_cosine",
    ):
        print(f"{key}: mean={global_stats[key]['mean']:.6g}")
    print(
        "negative batch fraction: primary/PSD="
        f"{global_stats['primary_vs_psd_negative_fraction']['mean']:.3f}, base/PSD="
        f"{global_stats['base_vs_psd_negative_fraction']['mean']:.3f}"
    )
    print("\n===== Module-wise =====")
    for group in groups:
        if group in module_summary:
            values = module_summary[group]
            print(
                f"{group}: primary_cosine="
                f"{values['primary_vs_psd_cosine']['mean']:.6g}, base_cosine="
                f"{values['base_vs_psd_cosine']['mean']:.6g}, "
                f"weighted_norm_ratio={values['weighted_psd_to_base_norm_ratio']['mean']:.6g}"
            )
    zero_check_groups = (
        "source_model", "swin_backbone", "ddp_fpn_neck",
        "image_encoder_projection",
    )
    for group in zero_check_groups:
        if (
            group in module_summary
            and module_summary[group]["psd_grad_norm_raw"]["max"] == 0.0
        ):
            if group == "source_model":
                reason = (
                    "The PSD student receives x_s built from x0, so a trainable "
                    "image-Gaussian source should normally receive PSD gradient; "
                    "zero suggests frozen/no trainable source or a broken x0 path."
                )
            else:
                reason = (
                    "The PSD student receives the non-detached shared image_feat; "
                    "zero is unexpected unless this group is frozen or absent."
                )
            print(f"WARNING: {group} PSD grad norm is zero. {reason}")
    aggregate: dict[str, list[float]] = defaultdict(list)
    for row in parameter_rows:
        if row["base_grad_norm"] > 0 and row["psd_grad_norm_raw"] > 0:
            aggregate[row["parameter_name"]].append(row["base_vs_psd_cosine"])
    most_conflicting = sorted(
        ((sum(values) / len(values), name) for name, values in aggregate.items())
    )[:20]
    print("\n===== Most Conflicting Parameters =====")
    for cosine, name in most_conflicting:
        print(f"{cosine: .6f}  {name}")
    print(f"\nHeuristic: {summary['heuristic']['classification']}")
    print("(Exploratory diagnostic heuristic, not a rigorous research threshold.)")
    print(f"Output: {output}")
    return output
