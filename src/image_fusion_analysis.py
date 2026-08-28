from __future__ import annotations

import csv
import json
import math
import os
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from dataset import build_dataset
from failure_analysis import (
    BoundedQuantiles,
    DistributionAccumulator,
    original_continuous,
)
from inference import state_to_prediction
from metrics import SegmentationMetrics
from source_diagnostics import (
    _inverse_normalize,
    _json_safe,
    _load_models,
    _new_metrics,
    _save_heatmap,
    _save_mask,
    deterministic_epsilon_like,
)
from state_space import state_spatial_size
from utils import autocast_context, resolve_device, seed_everything


IMAGE_ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
MASK_BETAS = (0.0, 0.25, 0.5, 1.0, 2.0)


@dataclass
class TensorStatistics:
    total: float = 0.0
    square_total: float = 0.0
    absolute_total: float = 0.0
    count: int = 0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    pixel_l2: DistributionAccumulator = field(default_factory=DistributionAccumulator)
    first_shape: list[int] | None = None
    collect_pixel_l2: bool = True

    def update(self, tensor: torch.Tensor) -> None:
        values = tensor.detach().float()
        if self.first_shape is None:
            self.first_shape = list(values.shape)
        self.total += float(values.sum().cpu())
        self.square_total += float(values.square().sum().cpu())
        self.absolute_total += float(values.abs().sum().cpu())
        self.count += values.numel()
        self.minimum = min(self.minimum, float(values.amin().cpu()))
        self.maximum = max(self.maximum, float(values.amax().cpu()))
        if values.ndim == 4 and self.collect_pixel_l2:
            self.pixel_l2.update(torch.linalg.vector_norm(values, dim=1))

    def compute(self) -> dict[str, Any]:
        mean = self.total / max(self.count, 1)
        mean_square = self.square_total / max(self.count, 1)
        return {
            "shape_example": self.first_shape,
            "mean": mean,
            "std": math.sqrt(max(mean_square - mean * mean, 0.0)),
            "abs_mean": self.absolute_total / max(self.count, 1),
            "rms": math.sqrt(mean_square),
            "pixel_l2": self.pixel_l2.compute() if self.collect_pixel_l2 else None,
            "min": self.minimum,
            "max": self.maximum,
            "count": self.count,
            "aggregation_dtype": "float32",
        }


@dataclass
class ChannelRMS:
    square_sum: torch.Tensor | None = None
    values_per_channel: int = 0

    def update(self, tensor: torch.Tensor) -> None:
        values = tensor.detach().float()
        per_channel = values.square().sum(dim=(0, 2, 3)).cpu().double()
        self.square_sum = (
            per_channel if self.square_sum is None else self.square_sum + per_channel
        )
        self.values_per_channel += values.shape[0] * values.shape[2] * values.shape[3]

    def rms(self) -> torch.Tensor:
        if self.square_sum is None:
            return torch.empty(0)
        return (self.square_sum / max(self.values_per_channel, 1)).sqrt().float()


@dataclass
class FractionAccumulator:
    thresholds: tuple[tuple[str, str, float], ...]
    counts: dict[str, int] = field(default_factory=dict)
    total: int = 0

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().float().reshape(-1)
        self.total += values.numel()
        for name, operation, threshold in self.thresholds:
            if operation == "lt":
                count = int((values < threshold).sum().cpu())
            else:
                count = int((values > threshold).sum().cpu())
            self.counts[name] = self.counts.get(name, 0) + count

    def compute(self) -> dict[str, float]:
        return {name: count / max(self.total, 1) for name, count in self.counts.items()}


@dataclass
class RelativeDeltaAccumulator:
    normal_square: float = 0.0
    delta_square: float = 0.0
    count: int = 0

    def update(self, normal: torch.Tensor, ablated: torch.Tensor) -> None:
        normal = normal.detach().float()
        ablated = ablated.detach().float()
        if normal.shape != ablated.shape:
            raise ValueError("activation shapes changed between diagnostic conditions")
        self.normal_square += float(normal.square().sum().cpu())
        self.delta_square += float((normal - ablated).square().sum().cpu())
        self.count += normal.numel()

    def compute(self) -> dict[str, float]:
        normal_rms = math.sqrt(self.normal_square / max(self.count, 1))
        delta_rms = math.sqrt(self.delta_square / max(self.count, 1))
        return {
            "normal_rms": normal_rms,
            "delta_rms": delta_rms,
            "relative_delta": delta_rms / max(normal_rms, 1.0e-12),
        }


@dataclass
class LogitSensitivity:
    absolute_difference: DistributionAccumulator = field(
        default_factory=DistributionAccumulator
    )
    square_difference_sum: float = 0.0
    difference_count: int = 0
    cosine: DistributionAccumulator = field(default_factory=DistributionAccumulator)
    kl: DistributionAccumulator = field(default_factory=DistributionAccumulator)
    changed: int = 0
    valid_pixels: int = 0

    def update(
        self,
        normal_logits: torch.Tensor,
        ablated_logits: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        normal = normal_logits.detach().float()
        ablated = ablated_logits.detach().float()
        difference = normal - ablated
        expanded_valid = valid[:, None].expand_as(difference)
        valid_difference = difference[expanded_valid]
        self.absolute_difference.update(valid_difference.abs())
        self.square_difference_sum += float(valid_difference.square().sum().cpu())
        self.difference_count += valid_difference.numel()
        cosine = F.cosine_similarity(normal, ablated, dim=1)
        self.cosine.update(cosine[valid])
        log_p = F.log_softmax(normal, dim=1)
        log_q = F.log_softmax(ablated, dim=1)
        kl = (log_p.exp() * (log_p - log_q)).sum(dim=1)
        self.kl.update(kl[valid])
        normal_prediction = state_to_prediction(
            normal, void_class_index=0, exclude_void=True
        )
        ablated_prediction = state_to_prediction(
            ablated, void_class_index=0, exclude_void=True
        )
        self.changed += int(
            ((normal_prediction != ablated_prediction) & valid).sum().cpu()
        )
        self.valid_pixels += int(valid.sum().cpu())

    def compute(self) -> dict[str, Any]:
        return {
            "mean_absolute_logit_difference": self.absolute_difference.compute()["mean"],
            "rms_logit_difference": math.sqrt(
                self.square_difference_sum / max(self.difference_count, 1)
            ),
            "cosine_similarity": self.cosine.compute(),
            "softmax_kl_normal_to_ablation": self.kl.compute(),
            "prediction_changed_fraction": self.changed / max(self.valid_pixels, 1),
        }


def diagnostic_fusion(
    mask_feat: torch.Tensor,
    image_feat: torch.Tensor,
    *,
    mask_scale: float = 1.0,
    image_scale: float = 1.0,
) -> torch.Tensor:
    if mask_feat.shape != image_feat.shape:
        raise ValueError("mask_feat and image_feat must have identical shapes")
    return float(mask_scale) * mask_feat + float(image_scale) * image_feat


def deterministic_wrong_image_feature(
    previous_image_feat: torch.Tensor, target_size: tuple[int, int]
) -> torch.Tensor:
    if previous_image_feat.shape[-2:] == target_size:
        return previous_image_feat
    return F.interpolate(
        previous_image_feat.float(), size=target_size,
        mode="bilinear", align_corners=False,
    ).to(previous_image_feat.dtype)


def feature_geometry(
    image_feat: torch.Tensor, mask_feat: torch.Tensor, eps: float = 1.0e-8
) -> dict[str, torch.Tensor]:
    image = image_feat.detach().float()
    mask = mask_feat.detach().float()
    image_norm = torch.linalg.vector_norm(image, dim=1)
    mask_norm = torch.linalg.vector_norm(mask, dim=1)
    cosine = F.cosine_similarity(image, mask, dim=1, eps=eps)
    fused_norm = torch.linalg.vector_norm(image + mask, dim=1)
    return {
        "image_norm": image_norm,
        "mask_norm": mask_norm,
        "per_pixel_ratio": image_norm / (mask_norm + eps),
        "cosine": cosine,
        "cancellation_ratio": fused_norm / (image_norm + mask_norm + eps),
    }


@torch.no_grad()
def extract_image_features(model, image: torch.Tensor) -> dict[str, Any]:
    """Exactly reproduce TransformerImageEncoder and expose neck intermediates."""
    encoder = model.image_encoder
    normalized = image if encoder.input_already_normalized else (
        image - encoder.mean.to(image)
    ) / encoder.std.to(image)
    backbone_input = normalized
    if encoder.backbone_type == "swin":
        minimum = 4 * 2**3 * 7
        pad_height = max(minimum - image.shape[-2], 0)
        pad_width = max(minimum - image.shape[-1], 0)
        if pad_height or pad_width:
            backbone_input = F.pad(backbone_input, (0, pad_width, 0, pad_height))
    hidden_states = encoder._extract_backbone_features(backbone_input)
    stages = []
    for index, (hidden, channels) in enumerate(
        zip(hidden_states, encoder._hidden_sizes)
    ):
        divisor = 4 * 2**index
        feature = encoder._as_spatial_feature(
            hidden, channels, state_spatial_size(backbone_input, divisor)
        )
        original_size = state_spatial_size(image, divisor)
        stages.append(feature[..., :original_size[0], :original_size[1]])

    neck = encoder.neck
    pyramid = [layer(stage) for layer, stage in zip(neck.lateral, stages)]
    for index in range(len(pyramid) - 2, -1, -1):
        pyramid[index] = pyramid[index] + F.interpolate(
            pyramid[index + 1], size=pyramid[index].shape[-2:],
            mode="bilinear", align_corners=False,
        )
    fpn = [layer(feature) for layer, feature in zip(neck.fpn_output, pyramid)]
    target_size = fpn[0].shape[-2:]
    fpn_resized = [
        feature if feature.shape[-2:] == target_size else F.interpolate(
            feature, size=target_size, mode="bilinear", align_corners=False
        )
        for feature in fpn
    ]
    concatenated = torch.cat(fpn_resized, dim=1)
    merge_conv = neck.merge[0](concatenated)
    merge_gn = neck.merge[1](merge_conv)
    image_feat = encoder.projection(merge_gn)
    expected_size = state_spatial_size(image, model.state_downsample_factor)
    if image_feat.shape[-2:] != expected_size:
        image_feat = F.interpolate(
            image_feat, size=expected_size, mode="bilinear", align_corners=False
        )
    return {
        "stages": stages,
        "fpn": fpn,
        "fpn_resized": fpn_resized,
        "concat": concatenated,
        "merge_conv": merge_conv,
        "merge_gn": merge_gn,
        "image_feat": image_feat,
    }


def fusion_logits(
    model,
    mask_feat: torch.Tensor,
    image_feat: torch.Tensor,
    *,
    s_value: float,
    t_value: float,
    mask_scale: float = 1.0,
    image_scale: float = 1.0,
) -> torch.Tensor:
    fused = diagnostic_fusion(
        mask_feat, image_feat, mask_scale=mask_scale, image_scale=image_scale
    )
    batch = fused.shape[0]
    s = torch.full((batch,), s_value, device=fused.device)
    t = torch.full((batch,), t_value, device=fused.device)
    return model.unet(fused, s, t)


def batched_fused_logits(
    model,
    fused_by_name: dict[str, torch.Tensor],
    *,
    s_value: float,
    t_value: float,
    chunk_size: int = 4,
) -> dict[str, torch.Tensor]:
    """Evaluate same-shaped diagnostic conditions in small, no-grad batches."""
    names = list(fused_by_name)
    results: dict[str, torch.Tensor] = {}
    for start in range(0, len(names), chunk_size):
        chunk_names = names[start:start + chunk_size]
        fused = torch.cat([fused_by_name[name] for name in chunk_names], dim=0)
        s = torch.full((fused.shape[0],), s_value, device=fused.device)
        t = torch.full((fused.shape[0],), t_value, device=fused.device)
        logits = model.unet(fused, s, t)
        for offset, name in enumerate(chunk_names):
            results[name] = logits[offset:offset + 1]
    return results


class ActivationCapture(AbstractContextManager):
    def __init__(self, model) -> None:
        unet = model.unet
        self.modules = {
            "input_conv": unet.input,
            **{
                f"down{index + 1}": blocks[-1]
                for index, blocks in enumerate(unet.down_blocks)
            },
            "bottleneck": unet.middle[-1],
            **{
                f"up{index + 1}": blocks[-1]
                for index, blocks in enumerate(unet.up_blocks)
            },
            "output": unet.out,
        }
        self.activations: dict[str, torch.Tensor] = {}
        self.handles = []

    def __enter__(self):
        for name, module in self.modules.items():
            self.handles.append(module.register_forward_hook(self._hook(name)))
        return self

    def _hook(self, name: str):
        def capture(_module, _inputs, output):
            self.activations[name] = output.detach()
        return capture

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        return False


def _metric(config: dict, device: torch.device) -> SegmentationMetrics:
    return _new_metrics(config, device)


def _prediction_from_logits(
    logits: torch.Tensor, sample: dict, config: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    full = original_continuous(logits, sample, config)
    prediction = state_to_prediction(
        full, void_class_index=0, exclude_void=True
    )
    return prediction, full


def _update_metric(
    metric: SegmentationMetrics,
    logits: torch.Tensor,
    target: torch.Tensor,
    sample: dict,
    config: dict,
) -> torch.Tensor:
    prediction, _ = _prediction_from_logits(logits, sample, config)
    metric.update(prediction, target)
    return prediction


def _channel_summary(image_rms: torch.Tensor, mask_rms: torch.Tensor) -> dict[str, Any]:
    ratio = image_rms / mask_rms.clamp_min(1.0e-8)
    order = torch.argsort(ratio)

    def entries(indices: torch.Tensor) -> list[dict[str, float | int]]:
        return [
            {
                "channel": int(index),
                "image_rms": float(image_rms[index]),
                "mask_rms": float(mask_rms[index]),
                "ratio": float(ratio[index]),
            }
            for index in indices
        ]

    histogram, edges = np.histogram(
        ratio.numpy(), bins=[0, 0.25, 0.5, 1, 2, 4, 8, np.inf]
    )
    return {
        "image_rms": image_rms.tolist(),
        "mask_rms": mask_rms.tolist(),
        "ratio": ratio.tolist(),
        "histogram": {
            "edges": [0, 0.25, 0.5, 1, 2, 4, 8, "inf"],
            "counts": histogram.tolist(),
        },
        "smallest_10": entries(order[:10]),
        "largest_10": entries(order[-10:].flip(0)),
        "image_dominant_count": int((ratio > 2).sum()),
        "mask_dominant_count": int((ratio < 0.5).sum()),
        "balanced_count": int(((ratio >= 0.5) & (ratio <= 2)).sum()),
    }


def _per_class_deltas(
    normal: dict[str, Any], ablated: dict[str, Any], limit: int = 20
) -> dict[str, Any]:
    indices = normal["evaluated_class_indices"]
    deltas = [
        float(normal_iou) - float(ablated_iou)
        for normal_iou, ablated_iou in zip(normal["class_iou"], ablated["class_iou"])
    ]
    entries = [
        {"class_index": int(index), "delta_iou": delta}
        for index, delta in zip(indices, deltas)
        if math.isfinite(delta)
    ]
    return {
        "largest_drop": sorted(entries, key=lambda item: item["delta_iou"], reverse=True)[:limit],
        "smallest_absolute_change": sorted(
            entries, key=lambda item: abs(item["delta_iou"])
        )[:limit],
    }


def _write_csv(result: dict[str, Any], path: Path) -> None:
    rows: list[tuple[str, str, str, Any]] = []

    def visit(value: Any, keys: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, (*keys, str(key)))
        elif isinstance(value, (int, float, str, bool)) or value is None:
            rows.append((
                keys[0] if keys else "result",
                keys[1] if len(keys) > 1 else "all",
                ".".join(keys[2:]) if len(keys) > 2 else "value",
                value,
            ))

    for key, value in result.items():
        if key not in {"per_class"}:
            visit(value, (key,))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("section", "variant", "metric", "value"))
        writer.writerows(rows)


def _write_per_class(result: dict[str, Any], path: Path) -> None:
    rows = []
    for comparison, analysis in result["per_class"].items():
        for group, entries in analysis.items():
            for rank, entry in enumerate(entries, 1):
                rows.append((
                    comparison, group, rank,
                    entry["class_index"], entry["delta_iou"],
                ))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("comparison", "group", "rank", "class_index", "delta_iou"))
        writer.writerows(rows)


def _summary(result: dict[str, Any]) -> str:
    scale = result["feature_scale"]
    pi00 = result["pi00_ablation"]
    lines = [
        "ADE20K image fusion analysis",
        f"checkpoint: {result['checkpoint']}",
        f"global_step: {result['global_step']}",
        f"samples: {result['samples']}",
        "",
        "[Feature scale]",
        f"mask RMS: {scale['mask_feat']['rms']:.6f}",
        f"image RMS: {scale['image_feat']['rms']:.6f}",
        f"image/mask RMS: {scale['image_to_mask_rms_ratio']:.6f}",
        f"pixel ratio median: {scale['per_pixel_ratio']['median']:.6f}",
        f"cosine mean: {scale['cosine']['mean']:.6f}",
        f"cancellation mean: {scale['cancellation_ratio']['mean']:.6f}",
        "",
        "[pi00 ablation]",
    ]
    for name, metrics in pi00.items():
        if isinstance(metrics, dict) and "mIoU" in metrics:
            lines.append(f"{name}: {metrics['mIoU']:.6f}")
    lines.append("")
    lines.append("[image scale]")
    for name, metrics in result["image_scale_sweep"].items():
        lines.append(f"{name}: {metrics['mIoU']:.6f}")
    lines.append("")
    lines.append("[conditional decomposition]")
    for name, metrics in result["conditional_decomposition"].items():
        lines.append(f"{name}: {metrics['mIoU']:.6f}")
    lines.append("")
    lines.append("[pi01]")
    for name, metrics in result["pi01_ablation"].items():
        if isinstance(metrics, dict) and "mIoU" in metrics:
            lines.append(f"{name}: {metrics['mIoU']:.6f}")
    return "\n".join(lines) + "\n"


@torch.no_grad()
def run_image_fusion_analysis(
    config: dict,
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    num_visualize: int = 20,
    seed: int = 42,
    max_samples: int | None = None,
    expected_global_step: int = 160000,
    alpha_match: float | None = None,
    beta_match: float | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    if config["dataset"]["name"] != "ade20k":
        raise ValueError("Image fusion analysis is defined for ADE20K")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "feature_statistics").mkdir(exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = (
        resolve_device(config["runtime"]["device"])
        if device is None else torch.device(device)
    )
    seed_everything(seed, deterministic=True)
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint, model, source_model = _load_models(config, checkpoint_path, device)
    global_step = int(checkpoint.get("global_step", -1))
    if global_step != expected_global_step:
        raise ValueError(
            f"Expected checkpoint global_step={expected_global_step}, got {global_step}"
        )
    dataset = build_dataset(config, config["evaluation"]["split"], augment=False)
    sample_count = min(len(dataset), max_samples or len(dataset))

    branch_stats = {
        name: TensorStatistics() for name in ("mask_feat", "image_feat", "fused_feat")
    }
    stage_stats = {
        f"stage{index}": TensorStatistics(collect_pixel_l2=False)
        for index in range(1, 5)
    }
    fpn_stats = {
        f"p{index}": TensorStatistics(collect_pixel_l2=False)
        for index in range(1, 5)
    }
    fpn_resized_stats = {
        f"p{index}_resized_h4": TensorStatistics(collect_pixel_l2=False)
        for index in range(1, 5)
    }
    merge_stats = {
        name: TensorStatistics(collect_pixel_l2=False)
        for name in ("concat", "merge_conv", "merge_gn", "final_image_feat")
    }
    geometry_stats = {
        name: DistributionAccumulator()
        for name in ("per_pixel_ratio", "cosine", "cancellation_ratio")
    }
    cosine_fractions = FractionAccumulator((
        ("fraction_lt_-0.5", "lt", -0.5),
        ("fraction_lt_0", "lt", 0.0),
        ("fraction_gt_0", "gt", 0.0),
        ("fraction_gt_0.5", "gt", 0.5),
    ))
    image_channel_rms = ChannelRMS()
    mask_channel_rms = ChannelRMS()

    pi00_names = (
        "normal", "image_zero", "mask_zero", "both_zero", "wrong_image",
        "image_hflip", "image_roll",
    )
    pi00_metrics = {name: _metric(config, device) for name in pi00_names}
    image_scale_metrics = {
        f"alpha_{alpha:g}": _metric(config, device) for alpha in IMAGE_ALPHAS
    }
    mask_scale_metrics = {
        f"beta_{beta:g}": _metric(config, device) for beta in MASK_BETAS
    }
    matched_metrics = {
        "alpha_match": _metric(config, device) if alpha_match is not None else None,
        "beta_match": _metric(config, device) if beta_match is not None else None,
    }
    conditional_metrics = {
        name: _metric(config, device) for name in (
            "source_on_image_on", "source_on_image_off",
            "source_off_image_on", "source_off_image_off",
        )
    }
    pi01_metrics = {
        name: _metric(config, device) for name in (
            "normal", "image_zero", "wrong_image", "alpha_0.5", "alpha_2",
        )
    }
    sensitivity = {
        "image_zero": LogitSensitivity(),
        "wrong_image": LogitSensitivity(),
    }
    hook_names = [
        "fusion_input", "input_conv", "down1", "down2", "down3", "down4",
        "down5", "bottleneck", "up1", "up2", "up3", "up4", "up5", "output",
    ]
    internal = {
        comparison: {name: RelativeDeltaAccumulator() for name in hook_names}
        for comparison in ("image_zero", "wrong_image")
    }
    production_max_abs = 0.0
    image_extraction_max_abs = 0.0
    production_checked = 0

    last_sample = dataset[len(dataset) - 1]
    last_image = last_sample["image"].unsqueeze(0).to(device)
    with autocast_context(config, device):
        previous_image_feat = model.encode_image(last_image).detach()
    del last_image, last_sample

    for index in range(sample_count):
        sample = dataset[index]
        image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
        target = sample["target"].unsqueeze(0).to(device, non_blocking=True)
        valid = target != 0
        with autocast_context(config, device):
            _, mu, _ = source_model(image)
            epsilon = deterministic_epsilon_like(mu, seed + index)
            extracted = extract_image_features(model, image)
            image_feat = extracted["image_feat"]
            mask_feat = model.mask_encoder(mu + epsilon)
        x0 = mu + epsilon
        wrong_image = deterministic_wrong_image_feature(
            previous_image_feat, image_feat.shape[-2:]
        )
        previous_image_feat = image_feat.detach()
        fused_feat = mask_feat + image_feat

        branch_stats["mask_feat"].update(mask_feat)
        branch_stats["image_feat"].update(image_feat)
        branch_stats["fused_feat"].update(fused_feat)
        image_channel_rms.update(image_feat)
        mask_channel_rms.update(mask_feat)
        geometry = feature_geometry(image_feat, mask_feat)
        for name in geometry_stats:
            geometry_stats[name].update(geometry[name])
        cosine_fractions.update(geometry["cosine"])
        for stage_index, stage in enumerate(extracted["stages"], 1):
            stage_stats[f"stage{stage_index}"].update(stage)
        for fpn_index, feature in enumerate(extracted["fpn"], 1):
            fpn_stats[f"p{fpn_index}"].update(feature)
        for fpn_index, feature in enumerate(extracted["fpn_resized"], 1):
            fpn_resized_stats[f"p{fpn_index}_resized_h4"].update(feature)
        merge_stats["concat"].update(extracted["concat"])
        merge_stats["merge_conv"].update(extracted["merge_conv"])
        merge_stats["merge_gn"].update(extracted["merge_gn"])
        merge_stats["final_image_feat"].update(image_feat)

        def captured_logits(feature: torch.Tensor, s: float, t: float):
            with ActivationCapture(model) as capture:
                with autocast_context(config, device):
                    logits = fusion_logits(
                        model, mask_feat, feature, s_value=s, t_value=t
                    )
            return logits, capture.activations

        normal_logits, normal_activations = captured_logits(image_feat, 0.0, 0.0)
        zero_logits, zero_activations = captured_logits(
            torch.zeros_like(image_feat), 0.0, 0.0
        )
        wrong_logits, wrong_activations = captured_logits(wrong_image, 0.0, 0.0)
        normal_activations["fusion_input"] = fused_feat
        zero_activations["fusion_input"] = mask_feat
        wrong_activations["fusion_input"] = mask_feat + wrong_image
        for name in hook_names:
            internal["image_zero"][name].update(
                normal_activations[name], zero_activations[name]
            )
            internal["wrong_image"][name].update(
                normal_activations[name], wrong_activations[name]
            )

        if production_checked < 4:
            batch = x0.shape[0]
            time = torch.zeros(batch, device=device)
            with autocast_context(config, device):
                production_image_feat = model.encode_image(image)
                production_logits = model.forward_logits_with_image_feat(
                    x0, image_feat, time, time
                )
            image_error = float(
                (image_feat.float() - production_image_feat.float()).abs().amax()
            )
            image_extraction_max_abs = max(image_extraction_max_abs, image_error)
            if not torch.equal(image_feat, production_image_feat):
                raise AssertionError(
                    "diagnostic image extraction != production: "
                    f"max_abs={image_error}"
                )
            error = float((normal_logits.float() - production_logits.float()).abs().amax())
            production_max_abs = max(production_max_abs, error)
            if not torch.equal(normal_logits, production_logits):
                raise AssertionError(f"diagnostic normal != production: max_abs={error}")
            production_checked += 1

        normal_prediction, normal_full = _prediction_from_logits(
            normal_logits, sample, config
        )
        zero_prediction, zero_full = _prediction_from_logits(zero_logits, sample, config)
        wrong_prediction, wrong_full = _prediction_from_logits(
            wrong_logits, sample, config
        )
        pi00_metrics["normal"].update(normal_prediction, target)
        pi00_metrics["image_zero"].update(zero_prediction, target)
        pi00_metrics["wrong_image"].update(wrong_prediction, target)
        sensitivity["image_zero"].update(normal_full, zero_full, valid)
        sensitivity["wrong_image"].update(normal_full, wrong_full, valid)

        with autocast_context(config, device):
            mask_feat_off = model.mask_encoder(epsilon)
            extra_logits = batched_fused_logits(
                model,
                {
                    "mask_zero": image_feat,
                    "both_zero": torch.zeros_like(fused_feat),
                    "image_hflip": mask_feat + torch.flip(image_feat, dims=(-1,)),
                    "image_roll": mask_feat + torch.roll(
                        image_feat,
                        shifts=(max(image_feat.shape[-2] // 4, 1),
                                max(image_feat.shape[-1] // 4, 1)),
                        dims=(-2, -1),
                    ),
                    "alpha_0.25": mask_feat + 0.25 * image_feat,
                    "alpha_0.5": mask_feat + 0.5 * image_feat,
                    "alpha_2": mask_feat + 2.0 * image_feat,
                    "alpha_4": mask_feat + 4.0 * image_feat,
                    "beta_0.25": 0.25 * mask_feat + image_feat,
                    "beta_0.5": 0.5 * mask_feat + image_feat,
                    "beta_2": 2.0 * mask_feat + image_feat,
                    **({
                        "alpha_match": mask_feat + alpha_match * image_feat,
                    } if alpha_match is not None else {}),
                    **({
                        "beta_match": beta_match * mask_feat + image_feat,
                    } if beta_match is not None else {}),
                    "source_off_image_on": mask_feat_off + image_feat,
                    "source_off_image_off": mask_feat_off,
                },
                s_value=0.0,
                t_value=0.0,
            )
        for name in ("mask_zero", "both_zero", "image_hflip", "image_roll"):
            pi00_metrics[name].update(
                _prediction_from_logits(extra_logits[name], sample, config)[0], target
            )

        for alpha in IMAGE_ALPHAS:
            key = f"alpha_{alpha:g}"
            if alpha == 0:
                logits = zero_logits
            elif alpha == 1:
                logits = normal_logits
            else:
                logits = extra_logits[key]
            image_scale_metrics[key].update(
                _prediction_from_logits(logits, sample, config)[0], target
            )

        for beta in MASK_BETAS:
            key = f"beta_{beta:g}"
            if beta == 0:
                logits = extra_logits["mask_zero"]
            elif beta == 1:
                logits = normal_logits
            else:
                logits = extra_logits[key]
            mask_scale_metrics[key].update(
                _prediction_from_logits(logits, sample, config)[0], target
            )

        if alpha_match is not None:
            matched_metrics["alpha_match"].update(
                _prediction_from_logits(
                    extra_logits["alpha_match"], sample, config
                )[0], target
            )
        if beta_match is not None:
            matched_metrics["beta_match"].update(
                _prediction_from_logits(
                    extra_logits["beta_match"], sample, config
                )[0], target
            )

        conditional_metrics["source_on_image_on"].update(normal_prediction, target)
        conditional_metrics["source_on_image_off"].update(zero_prediction, target)
        conditional_metrics["source_off_image_on"].update(
            _prediction_from_logits(
                extra_logits["source_off_image_on"], sample, config
            )[0], target
        )
        conditional_metrics["source_off_image_off"].update(
            _prediction_from_logits(
                extra_logits["source_off_image_off"], sample, config
            )[0], target
        )

        with autocast_context(config, device):
            pi01_logits = batched_fused_logits(
                model,
                {
                    "normal": fused_feat,
                    "image_zero": mask_feat,
                    "wrong_image": mask_feat + wrong_image,
                    "alpha_0.5": mask_feat + 0.5 * image_feat,
                    "alpha_2": mask_feat + 2.0 * image_feat,
                },
                s_value=0.0,
                t_value=1.0,
            )
        for name, logits in pi01_logits.items():
            pi01_metrics[name].update(
                _prediction_from_logits(logits, sample, config)[0], target
            )

        if index < num_visualize:
            directory = output / "visualizations" / f"image_{index:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                np.round(_inverse_normalize(image[0], config) * 255).astype(np.uint8)
            ).save(directory / "input.png")
            _save_mask(target[0].cpu(), directory / "ground_truth.png")
            _save_mask(normal_prediction[0].cpu(), directory / "normal_pi00.png")
            _save_mask(zero_prediction[0].cpu(), directory / "image_zero_pi00.png")
            _save_mask(wrong_prediction[0].cpu(), directory / "wrong_image_pi00.png")
            _save_mask(
                _prediction_from_logits(
                    extra_logits["mask_zero"], sample, config
                )[0][0].cpu(),
                directory / "mask_zero_pi00.png",
            )
            visual_scale_logits = {
                0.5: extra_logits["alpha_0.5"],
                1.0: normal_logits,
                2.0: extra_logits["alpha_2"],
            }
            for alpha, logits in visual_scale_logits.items():
                _save_mask(
                    _prediction_from_logits(logits, sample, config)[0][0].cpu(),
                    directory / f"alpha_{alpha:g}.png",
                )
            for name in (
                "mask_norm", "image_norm", "per_pixel_ratio", "cosine",
                "cancellation_ratio",
            ):
                values = geometry[name]
                full_values = original_continuous(
                    values[:, None], sample, config
                )[:, 0]
                _save_heatmap(
                    full_values[0].cpu(), directory / f"{name}.png", name,
                    cmap="coolwarm" if name == "cosine" else "viridis",
                )

        if (index + 1) % 25 == 0:
            print(
                f"image fusion analysis: {index + 1}/{sample_count} samples",
                flush=True,
            )

    feature_scale = {name: stats.compute() for name, stats in branch_stats.items()}
    feature_scale["image_to_mask_rms_ratio"] = (
        feature_scale["image_feat"]["rms"] / feature_scale["mask_feat"]["rms"]
    )
    feature_scale.update({
        name: stats.compute() for name, stats in geometry_stats.items()
    })
    feature_scale["cosine_fractions"] = cosine_fractions.compute()
    feature_scale["channel_scale"] = _channel_summary(
        image_channel_rms.rms(), mask_channel_rms.rms()
    )
    pi00_result = {name: metric.compute() for name, metric in pi00_metrics.items()}
    image_scale_result = {
        name: metric.compute() for name, metric in image_scale_metrics.items()
    }
    mask_scale_result = {
        name: metric.compute() for name, metric in mask_scale_metrics.items()
    }
    matched_result = {
        name: metric.compute() if metric is not None else None
        for name, metric in matched_metrics.items()
    }
    pi01_result = {name: metric.compute() for name, metric in pi01_metrics.items()}
    best_alpha_name = max(
        image_scale_result, key=lambda key: image_scale_result[key]["mIoU"]
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "global_step": global_step,
        "device": str(device),
        "samples": sample_count,
        "wrong_image_mapping": "dataset index i receives i-1 cyclic feature; bilinear spatial resize",
        "production_equivalence": {
            "samples_checked": production_checked,
            "max_abs_difference": production_max_abs,
            "image_extraction_max_abs_difference": image_extraction_max_abs,
        },
        "feature_scale": feature_scale,
        "swin_stages": {name: stats.compute() for name, stats in stage_stats.items()},
        "fpn": {
            **{name: stats.compute() for name, stats in fpn_stats.items()},
            **{name: stats.compute() for name, stats in fpn_resized_stats.items()},
            "merge": {name: stats.compute() for name, stats in merge_stats.items()},
        },
        "pi00_ablation": pi00_result,
        "image_scale_sweep": image_scale_result,
        "mask_scale_sweep": mask_scale_result,
        "rms_matching": {
            "alpha_match": alpha_match,
            "beta_match": beta_match,
            "metrics": matched_result,
        },
        "conditional_decomposition": {
            name: metric.compute() for name, metric in conditional_metrics.items()
        },
        "logit_sensitivity": {
            name: accumulator.compute() for name, accumulator in sensitivity.items()
        },
        "internal_sensitivity": {
            comparison: {
                name: accumulator.compute() for name, accumulator in layers.items()
            }
            for comparison, layers in internal.items()
        },
        "pi01_ablation": {
            **pi01_result,
            "best_pi00_alpha": best_alpha_name,
            "best_available_scaled_variant": (
                "alpha_0.5" if best_alpha_name == "alpha_0.5"
                else "alpha_2" if best_alpha_name == "alpha_2"
                else "normal"
            ),
        },
        "per_class": {
            "normal_vs_image_zero": _per_class_deltas(
                pi00_result["normal"], pi00_result["image_zero"]
            ),
            "normal_vs_wrong_image": _per_class_deltas(
                pi00_result["normal"], pi00_result["wrong_image"]
            ),
        },
    }
    with (output / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(result), handle, indent=2, ensure_ascii=False, allow_nan=False)
    _write_csv(result, output / "diagnostics.csv")
    _write_per_class(result, output / "per_class.csv")
    (output / "summary.txt").write_text(_summary(result), encoding="utf-8")
    with (output / "feature_statistics" / "channel_ratios.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("channel", "image_rms", "mask_rms", "ratio"))
        for channel, (image_rms, mask_rms, ratio) in enumerate(zip(
            feature_scale["channel_scale"]["image_rms"],
            feature_scale["channel_scale"]["mask_rms"],
            feature_scale["channel_scale"]["ratio"],
        )):
            writer.writerow((channel, image_rms, mask_rms, ratio))
    with (output / "feature_statistics" / "feature_statistics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            _json_safe({
                "feature_scale": feature_scale,
                "swin_stages": result["swin_stages"],
                "fpn": result["fpn"],
            }),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    return result
