from __future__ import annotations

import csv
import copy
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from checkpoint import _without_module_prefix
from dataset import ade20k_eval_collate, build_dataset
from discrete_flow_maps import source_alignment_map_from_indices
from inference import (
    sample_segmentation_from_x0,
    state_to_original_continuous,
    terminal_state_to_original_prediction,
)
from metrics import SegmentationMetrics
from model_factory import build_models
from state_space import prepare_state_targets, resize_continuous
from utils import autocast_context, resolve_device, seed_everything
from visualization import colorize


DEFAULT_SIGMA_VALUES = (1.0, 0.75, 0.5, 0.25, 0.1, 0.0)
DEFAULT_STEP_VALUES = (1, 2, 3, 5)


def deterministic_epsilon_like(mu: torch.Tensor, seed: int) -> torch.Tensor:
    """Generate noise independently of the global RNG and DataLoader batch size."""
    generator = torch.Generator(device=mu.device)
    generator.manual_seed(int(seed))
    return torch.randn(
        mu.shape, device=mu.device, dtype=mu.dtype, generator=generator
    )


def diagnostic_initial_state(
    mu: torch.Tensor,
    epsilon: torch.Tensor,
    sigma: float,
    *,
    mu_zero: bool = False,
) -> torch.Tensor:
    if mu.shape != epsilon.shape:
        raise ValueError("mu and epsilon must have identical shapes")
    noise = float(sigma) * epsilon
    return noise if mu_zero else mu + noise


def mu_gt_cosine(
    mu_full: torch.Tensor,
    target_full: torch.Tensor,
    *,
    eps: float,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather the normalized mean in the GT direction without target one-hot."""
    if valid_mask.shape != target_full.shape:
        raise ValueError("valid_mask and target_full must have identical shapes")
    classes = mu_full.shape[1]
    valid_mask = valid_mask.to(device=target_full.device, dtype=torch.bool)
    in_range = (target_full >= 0) & (target_full < classes)
    if bool((valid_mask & ~in_range).any()):
        raise ValueError("a valid target pixel contains an out-of-range class index")
    safe_target = torch.where(
        valid_mask & in_range, target_full, torch.zeros_like(target_full)
    )
    normalized = F.normalize(mu_full, dim=1, eps=eps)
    return normalized.gather(1, safe_target[:, None]).squeeze(1)


def oracle_state_from_target(
    target_full: torch.Tensor,
    *,
    state_size: tuple[int, int],
    num_classes: int,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest-downsample GT, then one-hot encode only the state resolution."""
    targets = prepare_state_targets(
        target_full,
        num_classes=num_classes,
        state_size=state_size,
        ignore_index=ignore_index,
        mask_pixel_losses=True,
    )
    assert targets.valid_mask_full is not None
    return targets.target_state, targets.one_hot_state, targets.valid_mask_full


def oracle_alignment_map(
    target_full: torch.Tensor,
    *,
    state_size: tuple[int, int],
    num_classes: int,
    ignore_index: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle low-resolution one-hot followed by bilinear full-resolution align."""
    target_state, oracle_state, valid = oracle_state_from_target(
        target_full,
        state_size=state_size,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    oracle_full = F.interpolate(
        oracle_state,
        size=target_full.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    alignment = source_alignment_map_from_indices(
        oracle_full,
        target_full,
        num_classes=num_classes,
        eps=eps,
        valid_mask=valid,
    )
    return alignment, target_state, valid


def colorize_ade20k(mask: torch.Tensor) -> np.ndarray:
    return colorize(mask, dataset_name="ade20k")


@dataclass
class ApproximateQuantiles:
    max_per_update: int = 4096
    max_total: int = 1_000_000
    chunks: list[torch.Tensor] = field(default_factory=list)

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().float().reshape(-1)
        if not values.numel():
            return
        if values.numel() > self.max_per_update:
            indices = torch.linspace(
                0, values.numel() - 1, self.max_per_update,
                device=values.device,
            ).long()
            values = values[indices]
        self.chunks.append(values.cpu())

    def compute(self) -> dict[str, float | int]:
        if not self.chunks:
            return {
                "median": float("nan"), "p10": float("nan"),
                "p25": float("nan"), "p75": float("nan"),
                "p90": float("nan"), "quantile_sample_count": 0,
            }
        values = torch.cat(self.chunks)
        if values.numel() > self.max_total:
            indices = torch.linspace(
                0, values.numel() - 1, self.max_total
            ).long()
            values = values[indices]
        quantiles = torch.quantile(
            values, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
        )
        return {
            "median": float(quantiles[2]),
            "p10": float(quantiles[0]),
            "p25": float(quantiles[1]),
            "p75": float(quantiles[3]),
            "p90": float(quantiles[4]),
            "quantile_sample_count": int(values.numel()),
        }


@dataclass
class MeanAccumulator:
    total: float = 0.0
    count: int = 0

    def update(self, values: torch.Tensor, valid: torch.Tensor | None = None) -> None:
        values = values.detach().float()
        if valid is not None:
            values = values[valid]
        self.total += float(values.sum().cpu())
        self.count += values.numel()

    def compute(self) -> float:
        return self.total / max(self.count, 1)


@dataclass
class MuStatistics:
    abs_sum: float = 0.0
    square_sum: float = 0.0
    element_count: int = 0
    pixel_l2_sum: float = 0.0
    pixel_count: int = 0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    source_variance_sum: float = 0.0
    source_variance_count: int = 0
    pixel_l2_quantiles: ApproximateQuantiles = field(
        default_factory=ApproximateQuantiles
    )

    def update(self, mu: torch.Tensor, logvar: torch.Tensor) -> None:
        values = mu.detach().float()
        self.abs_sum += float(values.abs().sum().cpu())
        self.square_sum += float(values.square().sum().cpu())
        self.element_count += values.numel()
        self.minimum = min(self.minimum, float(values.amin().cpu()))
        self.maximum = max(self.maximum, float(values.amax().cpu()))
        pixel_l2 = torch.linalg.vector_norm(values, dim=1)
        self.pixel_l2_sum += float(pixel_l2.sum().cpu())
        self.pixel_count += pixel_l2.numel()
        self.pixel_l2_quantiles.update(pixel_l2)
        variance = torch.exp(logvar.detach().float())
        self.source_variance_sum += float(variance.sum().cpu())
        self.source_variance_count += variance.numel()

    def compute(self, sigma_values: Iterable[float]) -> dict[str, Any]:
        rms = math.sqrt(self.square_sum / max(self.element_count, 1))
        source_sigma_rms = math.sqrt(
            self.source_variance_sum / max(self.source_variance_count, 1)
        )
        snr = {
            str(float(sigma)): (
                rms / float(sigma) if float(sigma) > 0 else None
            )
            for sigma in sigma_values
        }
        return {
            "abs_mean": self.abs_sum / max(self.element_count, 1),
            "rms": rms,
            "pixel_l2_mean": self.pixel_l2_sum / max(self.pixel_count, 1),
            "min": self.minimum,
            "max": self.maximum,
            "source_sigma_rms_from_logvar": source_sigma_rms,
            "snr_rms_source_sigma": rms / max(source_sigma_rms, 1.0e-12),
            "snr_rms_by_diagnostic_sigma": snr,
            **self.pixel_l2_quantiles.compute(),
        }


def _new_metrics(config: dict, device: torch.device) -> SegmentationMetrics:
    eval_range = config["evaluation"]["eval_class_indices"]
    return SegmentationMetrics(
        config["dataset"]["num_classes"],
        config["dataset"]["void_class_index"],
        device=device,
        evaluated_class_indices=range(eval_range[0], eval_range[1] + 1),
        nanmean=config["evaluation"]["nanmean"],
        prediction_void_retained=not config["evaluation"][
            "exclude_void_from_prediction"
        ],
    )


def resolve_diagnostic_checkpoint(config: dict, checkpoint: str | None) -> Path:
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path.resolve()
    configured = config["evaluation"].get("checkpoint")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"evaluation.checkpoint does not exist: {path}")
        return path.resolve()
    output = Path(config["experiment"]["output_dir"]).expanduser()
    latest = output / "latest.pt"
    if latest.is_file():
        return latest.resolve()
    candidates = sorted(output.glob("*.pt")) if output.is_dir() else []
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise ValueError(
            "--checkpoint is required: no evaluation.checkpoint or latest.pt was found"
        )
    raise ValueError(
        "--checkpoint is required because multiple checkpoint candidates exist: "
        + ", ".join(path.name for path in candidates)
    )


def _segformer_v5_key_to_v4(key: str) -> str:
    """Translate the Transformers 5 SegFormer module layout to 4.x names."""
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
        (
            "attention.sequence_reduction.sequence_reduction.",
            "attention.self.sr.",
        ),
        (
            "attention.sequence_reduction.layer_norm.",
            "attention.self.layer_norm.",
        ),
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
    state: dict[str, torch.Tensor], source_model
) -> dict[str, torch.Tensor]:
    """Keep strict loading while supporting the known Transformers 5/4 rename."""
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


def _load_models(config: dict, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # The complete source state is strict-loaded below. Skip only its pretrained
    # initializer/download; variant, depths, widths, decoder, and runtime math
    # remain those declared by the production config.
    build_config = copy.deepcopy(config)
    build_config["source"]["pretrained"] = False
    build_config["source"]["_load_pretrained"] = False
    build_config["source"]["checkpoint"] = None
    build_config["model"]["image_encoder"]["pretrained"] = False
    model, source_model = build_models(build_config, device)
    model.load_state_dict(
        _without_module_prefix(checkpoint["model"]),
        strict=config["checkpoint"]["strict_model"],
    )
    if source_model is None:
        raise RuntimeError("Source diagnostics require source.prior_type=image_gaussian")
    if checkpoint.get("source_model") is None:
        raise RuntimeError("Checkpoint has no source_model state")
    source_state = _without_module_prefix(checkpoint["source_model"])
    source_model.load_state_dict(
        _checkpoint_source_state_for_model(source_state, source_model),
        strict=config["checkpoint"]["strict_model"],
    )
    model.eval()
    source_model.eval()
    return checkpoint, model, source_model


def _inverse_normalize(image: torch.Tensor, config: dict) -> np.ndarray:
    image = image.detach().float().cpu()
    normalize = config["augmentation"].get("normalize", {})
    if normalize.get("enabled", False):
        mean = image.new_tensor(normalize["mean"])[:, None, None]
        std = image.new_tensor(normalize["std"])[:, None, None]
        image = image * std + mean
    elif config["augmentation"].get("imagenet_normalize", False):
        mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        image = image * std + mean
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


def _save_rgb(array: np.ndarray, path: Path) -> None:
    Image.fromarray(np.round(array * 255).astype(np.uint8)).save(path)


def _save_mask(mask: torch.Tensor, path: Path) -> None:
    Image.fromarray(colorize_ade20k(mask)).save(path)


def _save_heatmap(
    values: torch.Tensor,
    path: Path,
    title: str,
    *,
    valid: torch.Tensor | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> None:
    array = values.detach().float().cpu().numpy()
    if valid is not None:
        array = np.ma.array(array, mask=~valid.detach().cpu().numpy().astype(bool))
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad(color="0.5")
    figure, axis = plt.subplots(figsize=(8, 5))
    plotted = axis.imshow(array, cmap=color_map, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.axis("off")
    figure.colorbar(plotted, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _resize_state_scalar(
    values: torch.Tensor,
    sample: dict,
    config: dict,
) -> torch.Tensor:
    return state_to_original_continuous(
        values[:, None],
        sample["model_shape"],
        sample["original_shape"],
        padded_shape=sample["padded_shape"],
        align_corners=config["evaluation"]["align_corners"],
    )[:, 0]


def _sigma_tag(sigma: float) -> str:
    return f"{float(sigma):g}".replace(".", "p")


def _save_summary(
    path: Path,
    input_rgb: np.ndarray,
    target: torch.Tensor,
    mu_prediction: torch.Tensor,
    mu_norm: torch.Tensor,
    cosine: torch.Tensor,
    confidence: torch.Tensor,
    sigma_predictions: dict[float, torch.Tensor],
    step_predictions: dict[int, torch.Tensor],
    mu_zero_prediction: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    panels: list[tuple[str, Any, str]] = [
        ("Validation input", input_rgb, "rgb"),
        ("Ground truth", colorize_ade20k(target), "rgb"),
        ("mu-only diagnostic", colorize_ade20k(mu_prediction), "rgb"),
        ("||mu||2", mu_norm.cpu().numpy(), "scalar"),
        ("mu-GT cosine", cosine.cpu().numpy(), "cosine"),
        ("diagnostic softmax confidence", confidence.cpu().numpy(), "confidence"),
    ]
    for sigma in (1.0, 0.5, 0.25, 0.0):
        if sigma in sigma_predictions:
            panels.append((
                f"prediction sigma={sigma:g}",
                colorize_ade20k(sigma_predictions[sigma]), "rgb",
            ))
    for step in (1, 3, 5):
        if step in step_predictions:
            panels.append((
                f"prediction steps={step}",
                colorize_ade20k(step_predictions[step]), "rgb",
            ))
    panels.append(("prediction mu=0", colorize_ade20k(mu_zero_prediction), "rgb"))
    columns = 4
    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(20, 5 * rows))
    axes = np.asarray(axes).reshape(-1)
    valid_array = valid.cpu().numpy().astype(bool)
    for axis, (title, data, kind) in zip(axes, panels):
        if kind == "rgb":
            axis.imshow(data)
        elif kind == "cosine":
            axis.imshow(
                np.ma.array(data, mask=~valid_array), cmap="coolwarm",
                vmin=-1, vmax=1,
            )
        elif kind == "confidence":
            axis.imshow(data, cmap="viridis", vmin=0, vmax=1)
        else:
            axis.imshow(data, cmap="magma")
        axis.set_title(title)
        axis.axis("off")
    for axis in axes[len(panels):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _metric_summary(metrics: SegmentationMetrics) -> dict[str, Any]:
    return metrics.compute()


def _write_csv(result: dict[str, Any], path: Path) -> None:
    rows: list[tuple[str, str, str, Any]] = []

    def add_metrics(section: str, variant: str, metrics: dict[str, Any]) -> None:
        for key in ("mIoU", "mAcc", "pixel_acc"):
            rows.append((section, variant, key, metrics[key]))
        indices = metrics.get("evaluated_class_indices", [])
        for class_index, value in zip(indices, metrics.get("class_iou", [])):
            rows.append((section, variant, f"class_{class_index}_iou", value))

    add_metrics("mu_only", "mu", result["mu_only"])
    for sigma, metrics in result["sigma_sweep"].items():
        add_metrics("sigma_sweep", sigma, metrics)
    add_metrics("mu_zero", "sigma_1.0", result["mu_zero"])
    for step, metrics in result["step_sweep"].items():
        add_metrics("step_sweep", step, metrics)
    for key, value in result["mu_statistics"].items():
        if isinstance(value, (int, float)):
            rows.append(("mu_statistics", "all", key, value))
    for key, value in result["gt_cosine"].items():
        if isinstance(value, (int, float)):
            rows.append(("gt_cosine", "all", key, value))
    for key, value in result["align"].items():
        rows.append(("align", "validation", key, value))
    rows.append((
        "mu_zero", "conditional_minus_mu_zero", "delta_mIoU",
        result["conditional_vs_mu_zero_delta_mIoU"],
    ))
    if result.get("full_grid"):
        for sigma, step_results in result["full_grid"].items():
            for step, metrics in step_results.items():
                add_metrics("full_grid", f"sigma={sigma},steps={step}", metrics)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("section", "variant", "metric", "value"))
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _print_console_summary(result: dict[str, Any]) -> None:
    mu_only = result["mu_only"]
    mu_stats = result["mu_statistics"]
    print("\n===== Source diagnostics =====\n")
    print("mu-only diagnostic segmentation")
    print(f"mIoU      : {mu_only['mIoU']:.6f}")
    print(f"mAcc      : {mu_only['mAcc']:.6f}")
    print(f"pixel_acc : {mu_only['pixel_acc']:.6f}\n")
    print("mu statistics")
    print(f"abs mean    : {mu_stats['abs_mean']:.6f}")
    print(f"RMS         : {mu_stats['rms']:.6f}")
    print(f"GT cosine   : {result['gt_cosine']['mean']:.6f}")
    print(f"SNR sigma=1 : {mu_stats['snr_rms_by_diagnostic_sigma'].get('1.0')}\n")
    print("Sigma sweep")
    print("sigma      mIoU       mAcc  pixel_acc")
    for sigma, metrics in result["sigma_sweep"].items():
        print(
            f"{float(sigma):5.2f}  {metrics['mIoU']:9.6f}  "
            f"{metrics['mAcc']:9.6f}  {metrics['pixel_acc']:9.6f}"
        )
    print("\nmu=0 ablation")
    conditional = result["sigma_sweep"].get("1.0")
    if conditional is not None:
        print(f"conditional : {conditional['mIoU']:.6f}")
    print(f"mu_zero     : {result['mu_zero']['mIoU']:.6f}")
    print(f"delta       : {result['conditional_vs_mu_zero_delta_mIoU']:.6f}\n")
    print("Step sweep")
    print("step       mIoU       mAcc  pixel_acc")
    for step, metrics in result["step_sweep"].items():
        print(
            f"{int(step):4d}  {metrics['mIoU']:9.6f}  "
            f"{metrics['mAcc']:9.6f}  {metrics['pixel_acc']:9.6f}"
        )
    align = result["align"]
    print("\nAlign")
    print(f"actual       : {align['actual_validation_align']:.8f}")
    print(f"oracle floor : {align['oracle_align_floor']:.8f}")
    print(f"gap          : {align['gap']:.8f}")
    print(f"ratio        : {align['ratio']:.6f}")


@torch.no_grad()
def run_source_diagnostics(
    config: dict,
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    sigma_values: Iterable[float] = DEFAULT_SIGMA_VALUES,
    step_values: Iterable[int] = DEFAULT_STEP_VALUES,
    num_visualize: int = 20,
    seed: int = 42,
    full_grid: bool = False,
    max_batches: int | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    if config["dataset"]["name"] != "ade20k":
        raise ValueError("This diagnostic is defined for the ADE20K protocol")
    sigma_values = tuple(dict.fromkeys(float(value) for value in sigma_values))
    step_values = tuple(dict.fromkeys(int(value) for value in step_values))
    if not sigma_values or any(value < 0 for value in sigma_values):
        raise ValueError("sigma_values must contain non-negative values")
    if not step_values or any(value <= 0 for value in step_values):
        raise ValueError("step_values must contain positive integers")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = resolve_device(config["runtime"]["device"]) if device is None else torch.device(device)
    print(f"Source diagnostic device: {device}")
    # Diagnostics prioritize exact reproducibility over evaluation throughput,
    # independently of the production training determinism setting.
    seed_everything(seed, deterministic=True)
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint, model, source_model = _load_models(config, checkpoint_path, device)

    dataset = build_dataset(config, config["evaluation"]["split"], augment=False)
    loader = DataLoader(
        dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["dataset"]["pin_memory"],
        collate_fn=ade20k_eval_collate,
    )
    base_step = int(config["evaluation"]["num_steps"])
    default_sigma = 1.0
    conditions = {
        *((sigma, base_step) for sigma in sigma_values),
        *((default_sigma, step) for step in step_values),
        (default_sigma, base_step),
    }
    if full_grid:
        conditions.update(
            (sigma, step) for sigma in sigma_values for step in step_values
        )
    condition_metrics = {
        condition: _new_metrics(config, device) for condition in sorted(conditions)
    }
    mu_only_metrics = _new_metrics(config, device)
    mu_zero_metrics = _new_metrics(config, device)
    mu_statistics = MuStatistics()
    cosine_mean = MeanAccumulator()
    cosine_quantiles = ApproximateQuantiles()
    predicted_align_mean = MeanAccumulator()
    actual_align_mean = MeanAccumulator()
    oracle_align_mean = MeanAccumulator()
    align_difference_mean = MeanAccumulator()
    classes = config["dataset"]["num_classes"]
    ignore_index = config["evaluation"]["ignore_index"]
    align_eps = config["source"]["align_eps"]
    visualized = 0
    sample_index = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        for sample in batch:
            image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
            target = sample["target"].unsqueeze(0).to(device, non_blocking=True)
            valid = target != ignore_index
            with autocast_context(config, device):
                _, mu, logvar = source_model(image)
            epsilon = deterministic_epsilon_like(mu, seed + sample_index)
            mu_statistics.update(mu, logvar)
            mu_full = state_to_original_continuous(
                mu,
                sample["model_shape"],
                sample["original_shape"],
                padded_shape=sample["padded_shape"],
                align_corners=config["evaluation"]["align_corners"],
            )
            mu_prediction = mu_full.argmax(dim=1)
            mu_only_metrics.update(mu_prediction, target)
            # Source supervision itself is a single direct bilinear state-to-GT
            # resize. Keep it distinct from the padding-aware evaluation resize.
            mu_align_full = resize_continuous(mu.float(), target.shape[-2:])
            cosine = mu_gt_cosine(
                mu_align_full, target, eps=align_eps, valid_mask=valid
            )
            cosine_mean.update(cosine, valid)
            cosine_quantiles.update(cosine[valid])
            predicted_align = 2.0 * (1.0 - cosine) / classes
            predicted_align_mean.update(predicted_align, valid)
            actual_map = source_alignment_map_from_indices(
                mu_align_full,
                target,
                num_classes=classes,
                eps=align_eps,
                valid_mask=valid,
            )
            actual_align_mean.update(actual_map, valid)
            align_difference_mean.update((actual_map - predicted_align).abs(), valid)

            _, oracle_state, _ = oracle_state_from_target(
                target,
                state_size=mu.shape[-2:],
                num_classes=classes,
                ignore_index=ignore_index,
            )
            oracle_full = resize_continuous(
                oracle_state, target.shape[-2:]
            )
            oracle_map = source_alignment_map_from_indices(
                oracle_full,
                target,
                num_classes=classes,
                eps=align_eps,
                valid_mask=valid,
            )
            oracle_align_mean.update(oracle_map, valid)

            visual_predictions: dict[tuple[float, int], torch.Tensor] = {}
            for sigma, steps in sorted(conditions):
                x0 = diagnostic_initial_state(mu, epsilon, sigma)
                with autocast_context(config, device):
                    terminal = sample_segmentation_from_x0(
                        model,
                        image,
                        x0,
                        config,
                        num_steps=steps,
                        return_terminal_state=True,
                    )
                prediction = terminal_state_to_original_prediction(
                    terminal,
                    sample["model_shape"],
                    sample["original_shape"],
                    padded_shape=sample["padded_shape"],
                    align_corners=config["evaluation"]["align_corners"],
                    void_class_index=config["dataset"]["void_class_index"],
                    exclude_void=config["evaluation"][
                        "exclude_void_from_prediction"
                    ],
                )
                condition_metrics[(sigma, steps)].update(prediction, target)
                if visualized < num_visualize:
                    visual_predictions[(sigma, steps)] = prediction[0].cpu()

            mu_zero_x0 = diagnostic_initial_state(
                mu, epsilon, default_sigma, mu_zero=True
            )
            with autocast_context(config, device):
                mu_zero_terminal = sample_segmentation_from_x0(
                    model,
                    image,
                    mu_zero_x0,
                    config,
                    num_steps=base_step,
                    return_terminal_state=True,
                )
            mu_zero_prediction = terminal_state_to_original_prediction(
                mu_zero_terminal,
                sample["model_shape"],
                sample["original_shape"],
                padded_shape=sample["padded_shape"],
                align_corners=config["evaluation"]["align_corners"],
                void_class_index=config["dataset"]["void_class_index"],
                exclude_void=config["evaluation"][
                    "exclude_void_from_prediction"
                ],
            )
            mu_zero_metrics.update(mu_zero_prediction, target)

            if visualized < num_visualize:
                directory = output / "visualizations" / f"image_{visualized:03d}"
                directory.mkdir(parents=True, exist_ok=True)
                input_rgb = _inverse_normalize(image[0], config)
                target_cpu = target[0].cpu()
                valid_cpu = valid[0].cpu()
                mu_prediction_cpu = mu_prediction[0].cpu()
                mu_norm_full = torch.linalg.vector_norm(mu_full.float(), dim=1)[0]
                cosine_cpu = cosine[0].cpu()
                confidence = torch.softmax(mu_full.float(), dim=1).amax(dim=1)[0]
                epsilon_magnitude = _resize_state_scalar(
                    torch.linalg.vector_norm(epsilon.float(), dim=1), sample, config
                )[0]
                snr_map = mu_norm_full / math.sqrt(classes)
                _save_rgb(input_rgb, directory / "input.png")
                _save_mask(target_cpu, directory / "ground_truth.png")
                _save_mask(mu_prediction_cpu, directory / "mu_argmax.png")
                _save_heatmap(
                    mu_norm_full, directory / "mu_norm_heatmap.png", "||mu||2"
                )
                _save_heatmap(
                    cosine_cpu,
                    directory / "mu_gt_cosine_heatmap.png",
                    "mu-GT cosine",
                    valid=valid_cpu,
                    vmin=-1,
                    vmax=1,
                    cmap="coolwarm",
                )
                _save_heatmap(
                    confidence,
                    directory / "mu_confidence.png",
                    "diagnostic softmax confidence (not used by inference)",
                    vmin=0,
                    vmax=1,
                )
                _save_heatmap(
                    epsilon_magnitude,
                    directory / "epsilon_magnitude.png",
                    "||epsilon||2",
                )
                _save_heatmap(
                    snr_map,
                    directory / "snr_heatmap_sigma1.png",
                    "SNR map: ||mu||2 / sqrt(C), sigma=1",
                )
                for sigma in sigma_values:
                    x0 = diagnostic_initial_state(mu, epsilon, sigma)
                    x0_label = F.interpolate(
                        x0.argmax(dim=1, keepdim=True).float(),
                        size=target.shape[-2:],
                        mode="nearest",
                    )[0, 0].long().cpu()
                    _save_mask(
                        x0_label,
                        directory / f"x0_argmax_sigma_{_sigma_tag(sigma)}.png",
                    )
                    prediction = visual_predictions[(sigma, base_step)]
                    _save_mask(
                        prediction,
                        directory / f"prediction_sigma_{_sigma_tag(sigma)}.png",
                    )
                for steps in step_values:
                    prediction = visual_predictions[(default_sigma, steps)]
                    _save_mask(
                        prediction, directory / f"prediction_steps_{steps}.png"
                    )
                mu_zero_cpu = mu_zero_prediction[0].cpu()
                _save_mask(mu_zero_cpu, directory / "prediction_mu_zero.png")
                sigma_predictions = {
                    sigma: visual_predictions[(sigma, base_step)]
                    for sigma in sigma_values
                }
                step_predictions = {
                    steps: visual_predictions[(default_sigma, steps)]
                    for steps in step_values
                }
                _save_summary(
                    directory / "summary.png",
                    input_rgb,
                    target_cpu,
                    mu_prediction_cpu,
                    mu_norm_full.cpu(),
                    cosine_cpu,
                    confidence.cpu(),
                    sigma_predictions,
                    step_predictions,
                    mu_zero_cpu,
                    valid_cpu,
                )
                with (directory / "sample.json").open("w", encoding="utf-8") as handle:
                    json.dump({
                        "sample_id": sample["sample_id"],
                        "dataset_index": sample_index,
                        "epsilon_seed": seed + sample_index,
                    }, handle, indent=2)
                visualized += 1
            sample_index += 1

    sigma_results = {
        str(float(sigma)): _metric_summary(condition_metrics[(sigma, base_step)])
        for sigma in sigma_values
    }
    step_results = {
        str(step): _metric_summary(condition_metrics[(default_sigma, step)])
        for step in step_values
    }
    conditional_result = _metric_summary(
        condition_metrics[(default_sigma, base_step)]
    )
    mu_zero_result = _metric_summary(mu_zero_metrics)
    actual_align = actual_align_mean.compute()
    oracle_align = oracle_align_mean.compute()
    full_grid_result = None
    if full_grid:
        full_grid_result = {
            str(float(sigma)): {
                str(step): _metric_summary(condition_metrics[(sigma, step)])
                for step in step_values
            }
            for sigma in sigma_values
        }
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "device": str(device),
        "samples_evaluated": sample_index,
        "batches_evaluated": min(len(loader), max_batches) if max_batches is not None else len(loader),
        "seed": seed,
        "protocol": {
            "dataset": "ade20k",
            "num_classes": classes,
            "ignore_index": ignore_index,
            "evaluated_class_indices": [1, 150],
            "original_resolution": True,
            "continuous_resize": "bilinear",
            "align_corners": False,
            "sigma_sweep_num_steps": base_step,
            "step_sweep_sigma": default_sigma,
            "epsilon_rule": "sample seed + zero-based dataset index",
            "deterministic_algorithms": True,
            "snr_heatmap": "||mu||2 / (sigma * sqrt(C)); sigma=1",
            "quantiles": "deterministic bounded spatial sample",
        },
        "mu_statistics": mu_statistics.compute(sigma_values),
        "gt_cosine": {
            "mean": cosine_mean.compute(),
            **cosine_quantiles.compute(),
            "predicted_align_from_cos_mean": predicted_align_mean.compute(),
            "absolute_difference_from_exact_align_mean": align_difference_mean.compute(),
        },
        "mu_only": _metric_summary(mu_only_metrics),
        "sigma_sweep": sigma_results,
        "mu_zero": mu_zero_result,
        "conditional_vs_mu_zero_delta_mIoU": (
            conditional_result["mIoU"] - mu_zero_result["mIoU"]
        ),
        "step_sweep": step_results,
        "full_grid": full_grid_result,
        "align": {
            "actual_validation_align": actual_align,
            "oracle_align_floor": oracle_align,
            "gap": actual_align - oracle_align,
            "ratio": actual_align / oracle_align if oracle_align > 0 else None,
        },
    }
    with (output / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(result), handle, indent=2, ensure_ascii=False,
            allow_nan=False,
        )
    _write_csv(result, output / "diagnostics.csv")
    _print_console_summary(result)
    return result
