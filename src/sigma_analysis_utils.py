from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from adaptive_path import entropy_adaptive_enabled, source_entropy_difficulty
from checkpoint import _without_module_prefix, validate_source_decoder_checkpoint
from config import load_config
from dataset import ade20k_eval_collate, build_dataset
from inference import (
    run_flow_from_state,
    sample_segmentation_from_x0,
    state_to_original_continuous,
    terminal_state_to_original_prediction,
)
from metrics import SegmentationMetrics
from model_factory import build_models
from state_space import resize_continuous
from utils import resolve_device, seed_everything


SIGMA_PERCENTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
SIGMA_BIN_EDGES = (0.0, 0.01, 0.05, 0.10, float("inf"))
SIGMA_BIN_LABELS = (
    "abs_sigma_minus_1_lt_0.01",
    "abs_sigma_minus_1_0.01_to_0.05",
    "abs_sigma_minus_1_0.05_to_0.10",
    "abs_sigma_minus_1_ge_0.10",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_safe(rows))


def load_sigma_experiment(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    seed: int,
) -> tuple[dict, torch.device, torch.nn.Module, torch.nn.Module, dict]:
    config_path = Path(config_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    config = load_config(config_path)
    if config["dataset"]["name"] != "cityscapes":
        raise ValueError("sigma experiments currently require Cityscapes")
    if config["source"]["prior_type"] != "image_gaussian":
        raise ValueError("sigma experiments require source.prior_type=image_gaussian")
    if config["source"].get("variance_mode") != "pixel_shared":
        raise ValueError("the requested experiments require pixel_shared variance")
    device = resolve_device(config["runtime"]["device"])
    # Ablations need rerun-stable image-to-noise pairing and deterministic flow
    # execution even when the training config allowed faster nondeterministic ops.
    seed_everything(seed, deterministic=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_source_decoder_checkpoint(checkpoint, config, checkpoint_path)
    endpoint, source = build_models(config, device)
    if source is None or checkpoint.get("source_model") is None:
        raise RuntimeError("checkpoint does not contain an image Gaussian source model")
    endpoint.load_state_dict(
        _without_module_prefix(checkpoint["model"]),
        strict=config["checkpoint"]["strict_model"],
    )
    source.load_state_dict(
        _without_module_prefix(checkpoint["source_model"]),
        strict=config["checkpoint"]["strict_model"],
    )
    endpoint.eval()
    source.eval()
    return config, device, endpoint, source, checkpoint


def build_sigma_loader(config: dict, max_images: int | None) -> tuple[DataLoader, int]:
    dataset = build_dataset(config, config["evaluation"]["split"], augment=False)
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("max_images must be positive")
        dataset = Subset(dataset, range(min(max_images, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["dataset"]["pin_memory"],
        persistent_workers=(
            bool(config["dataset"].get("persistent_workers", False))
            and config["dataset"]["num_workers"] > 0
        ),
        collate_fn=(
            ade20k_eval_collate
            if (
                config["dataset"]["name"] == "ade20k"
                or config["evaluation"]["original_resolution"]
            ) else None
        ),
    )
    return loader, len(dataset)


@dataclass(frozen=True)
class EvaluationBatch:
    image: torch.Tensor
    target: torch.Tensor
    image_indices: list[int]
    metadata: dict[str, Any] | None = None


def iter_evaluation_batches(loader: DataLoader) -> Iterator[EvaluationBatch]:
    image_index = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2 and all(
            torch.is_tensor(value) for value in batch
        ):
            image, target = batch
            indices = list(range(image_index, image_index + image.shape[0]))
            image_index += image.shape[0]
            yield EvaluationBatch(image, target, indices)
            continue
        samples = [batch] if isinstance(batch, dict) else list(batch)
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError("unsupported validation batch structure")
            image = sample["image"].unsqueeze(0)
            target = sample["target"].unsqueeze(0)
            yield EvaluationBatch(image, target, [image_index], metadata=sample)
            image_index += 1


def deterministic_epsilon_for_indices(
    reference: torch.Tensor,
    image_indices: Iterable[int],
    *,
    seed: int,
) -> torch.Tensor:
    """Generate one reproducible epsilon stream per dataset image index."""
    indices = list(image_indices)
    if reference.shape[0] != len(indices):
        raise ValueError("one image index is required per reference batch element")
    samples = []
    for sample, image_index in zip(reference, indices, strict=True):
        generator = torch.Generator(device=reference.device)
        sample_seed = (int(seed) + int(image_index) * 1_000_003) % (2**63 - 1)
        generator.manual_seed(sample_seed)
        samples.append(torch.randn(
            sample.shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        ))
    return torch.stack(samples)


def paired_initial_states(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    epsilon: torch.Tensor,
    *,
    fixed_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mu.shape != epsilon.shape:
        raise ValueError("mu and epsilon must have identical shapes")
    if logvar.shape[0] != mu.shape[0] or logvar.shape[-2:] != mu.shape[-2:]:
        raise ValueError("logvar must share mu batch and spatial dimensions")
    if logvar.shape[1] not in {1, mu.shape[1]}:
        raise ValueError("logvar channels must be one or match mu channels")
    if fixed_std < 0:
        raise ValueError("fixed_std must be non-negative")
    sigma = torch.exp(0.5 * logvar.detach().float())
    learned = mu + sigma.to(mu.dtype) * epsilon
    fixed = mu + float(fixed_std) * epsilon
    return learned, fixed, sigma


def target_at_source_resolution(
    target: torch.Tensor, spatial_size: tuple[int, int]
) -> torch.Tensor:
    return F.interpolate(
        target[:, None].float(), size=spatial_size, mode="nearest"
    )[:, 0].long()


def continuous_at_target_resolution(
    state: torch.Tensor,
    target: torch.Tensor,
    metadata: dict[str, Any] | None,
    config: dict,
) -> torch.Tensor:
    if metadata is None:
        return resize_continuous(state.float(), target.shape[-2:])
    return state_to_original_continuous(
        state.float(),
        metadata["model_shape"],
        metadata["original_shape"],
        padded_shape=metadata["padded_shape"],
        align_corners=config["evaluation"]["align_corners"],
    )


def prediction_from_x0(
    endpoint,
    image: torch.Tensor,
    x0: torch.Tensor,
    config: dict,
    *,
    num_steps: int,
    path_difficulty: torch.Tensor | None,
    metadata: dict[str, Any] | None,
) -> torch.Tensor:
    if metadata is None:
        return sample_segmentation_from_x0(
            endpoint,
            image,
            x0,
            config,
            num_steps=num_steps,
            path_difficulty=path_difficulty,
        )
    terminal = run_flow_from_state(
        endpoint,
        image,
        x0,
        config,
        num_steps,
        path_difficulty=path_difficulty,
    )
    return terminal_state_to_original_prediction(
        terminal,
        metadata["model_shape"],
        metadata["original_shape"],
        padded_shape=metadata["padded_shape"],
        align_corners=config["evaluation"]["align_corners"],
        void_class_index=config["dataset"]["void_class_index"],
        exclude_void=config["evaluation"]["exclude_void_from_prediction"],
    )


def source_path_difficulty(mu: torch.Tensor, config: dict) -> torch.Tensor | None:
    if not entropy_adaptive_enabled(config):
        return None
    _, difficulty = source_entropy_difficulty(
        mu, config, spatial_size=mu.shape[-2:]
    )
    return difficulty


def new_segmentation_metrics(config: dict, device: torch.device) -> SegmentationMetrics:
    eval_range = config["evaluation"]["eval_class_indices"]
    return SegmentationMetrics(
        config["dataset"]["num_classes"],
        config["dataset"]["void_class_index"],
        device=device,
        evaluated_class_indices=(
            range(eval_range[0], eval_range[1] + 1)
            if eval_range is not None else None
        ),
        nanmean=config["evaluation"]["nanmean"],
        prediction_void_retained=not config["evaluation"][
            "exclude_void_from_prediction"
        ],
    )


class StreamingDistribution:
    """Exact moments/counts with a bounded deterministic percentile sample."""

    def __init__(
        self,
        *,
        seed: int,
        max_samples: int = 1_000_000,
        exact_values_path: str | Path | None = None,
    ) -> None:
        self.total = 0.0
        self.square_total = 0.0
        self.count = 0
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.abs_deviation_total = 0.0
        self.abs_deviation_square_total = 0.0
        self.threshold_counts = {
            "abs_lt_0.01": 0,
            "abs_lt_0.05": 0,
            "abs_lt_0.10": 0,
            "sigma_lt_0.5": 0,
            "sigma_lt_0.8": 0,
            "sigma_gt_1.2": 0,
            "sigma_gt_1.5": 0,
        }
        self.max_samples = int(max_samples)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.sample_values = torch.empty(0)
        self.sample_priorities = torch.empty(0)
        self.exact_values_path = (
            None if exact_values_path is None else Path(exact_values_path)
        )
        self._exact_handle = None
        self.histogram_counts: np.ndarray | None = None
        self.histogram_edges: np.ndarray | None = None
        if self.exact_values_path is not None:
            self.exact_values_path.parent.mkdir(parents=True, exist_ok=True)
            self._exact_handle = self.exact_values_path.open("wb")

    def update(self, values: torch.Tensor, valid: torch.Tensor | None = None) -> None:
        values = values.detach().float()
        if valid is not None:
            values = values[valid]
        values = values.reshape(-1)
        if not values.numel():
            return
        values_cpu = values.cpu()
        if self._exact_handle is not None:
            values_cpu.numpy().astype(np.float32, copy=False).tofile(self._exact_handle)
        values_double = values.double()
        absolute = (values_double - 1.0).abs()
        self.total += float(values_double.sum().cpu())
        self.square_total += float(values_double.square().sum().cpu())
        self.count += values.numel()
        self.minimum = min(self.minimum, float(values.amin().cpu()))
        self.maximum = max(self.maximum, float(values.amax().cpu()))
        self.abs_deviation_total += float(absolute.sum().cpu())
        self.abs_deviation_square_total += float(absolute.square().sum().cpu())
        self.threshold_counts["abs_lt_0.01"] += int((absolute < 0.01).sum().cpu())
        self.threshold_counts["abs_lt_0.05"] += int((absolute < 0.05).sum().cpu())
        self.threshold_counts["abs_lt_0.10"] += int((absolute < 0.10).sum().cpu())
        self.threshold_counts["sigma_lt_0.5"] += int((values < 0.5).sum().cpu())
        self.threshold_counts["sigma_lt_0.8"] += int((values < 0.8).sum().cpu())
        self.threshold_counts["sigma_gt_1.2"] += int((values > 1.2).sum().cpu())
        self.threshold_counts["sigma_gt_1.5"] += int((values > 1.5).sum().cpu())

        # Random-priority sampling keeps a bounded, order-independent subset.
        candidate_limit = min(values_cpu.numel(), 65_536)
        if candidate_limit < values_cpu.numel():
            indices = torch.randperm(
                values_cpu.numel(), generator=self.generator
            )[:candidate_limit]
            values_cpu = values_cpu[indices]
        priorities = torch.rand(values_cpu.numel(), generator=self.generator)
        combined_values = torch.cat((self.sample_values, values_cpu))
        combined_priorities = torch.cat((self.sample_priorities, priorities))
        if combined_values.numel() > self.max_samples:
            keep = torch.topk(combined_priorities, self.max_samples).indices
            combined_values = combined_values[keep]
            combined_priorities = combined_priorities[keep]
        self.sample_values = combined_values
        self.sample_priorities = combined_priorities

    def compute(
        self,
        prefix: str = "sigma",
        *,
        histogram_range: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        if not self.count:
            return {f"{prefix}_count": 0}
        mean = self.total / self.count
        variance = max(self.square_total / self.count - mean**2, 0.0)
        if self._exact_handle is not None:
            self._exact_handle.flush()
            self._exact_handle.close()
            self._exact_handle = None
            exact_values = np.memmap(
                self.exact_values_path, mode="r", dtype=np.float32,
                shape=(self.count,),
            )
            quantiles = np.quantile(exact_values, SIGMA_PERCENTILES)
            absolute_quantiles = np.quantile(
                np.abs(np.asarray(exact_values) - 1.0), [0.5, 0.95]
            )
            selected_range = histogram_range or (self.minimum, self.maximum)
            if selected_range[1] <= selected_range[0]:
                padding = max(abs(selected_range[0]) * 1.0e-3, 1.0e-4)
                selected_range = (
                    selected_range[0] - padding, selected_range[1] + padding
                )
            self.histogram_counts, self.histogram_edges = np.histogram(
                exact_values, bins=160, range=selected_range
            )
            del exact_values
            self.exact_values_path.unlink(missing_ok=True)
            quantile_sample_count = self.count
            quantile_method = "exact_all_pixels_disk_backed"
        else:
            quantiles = torch.quantile(
                self.sample_values,
                torch.tensor(SIGMA_PERCENTILES, dtype=self.sample_values.dtype),
            ).numpy()
            absolute_sample = (self.sample_values - 1.0).abs()
            absolute_quantiles = torch.quantile(
                absolute_sample, torch.tensor([0.5, 0.95])
            ).numpy()
            self.histogram_counts, self.histogram_edges = np.histogram(
                self.sample_values.numpy(), bins=160,
                range=histogram_range,
            )
            quantile_sample_count = self.sample_values.numel()
            quantile_method = "bounded_random_priority_sample"
        result: dict[str, Any] = {
            f"{prefix}_count": self.count,
            f"{prefix}_mean": mean,
            f"{prefix}_std": math.sqrt(variance),
            f"{prefix}_pixel_variance": variance,
            f"{prefix}_variance_across_pixels": variance,
            f"{prefix}_min": self.minimum,
            f"{prefix}_max": self.maximum,
            f"{prefix}_abs_deviation_mean": self.abs_deviation_total / self.count,
            f"{prefix}_abs_deviation_median": float(absolute_quantiles[0]),
            f"{prefix}_abs_deviation_p95": float(absolute_quantiles[1]),
            f"{prefix}_mse_from_target": self.abs_deviation_square_total / self.count,
            f"{prefix}_quantile_sample_count": quantile_sample_count,
            f"{prefix}_quantile_method": quantile_method,
        }
        for percentile, value in zip(SIGMA_PERCENTILES, quantiles, strict=True):
            result[f"{prefix}_p{round(percentile * 100):02d}"] = float(value)
        result.update({
            f"fraction_abs_{prefix}_minus_1_lt_0.01": self.threshold_counts["abs_lt_0.01"] / self.count,
            f"fraction_abs_{prefix}_minus_1_lt_0.05": self.threshold_counts["abs_lt_0.05"] / self.count,
            f"fraction_abs_{prefix}_minus_1_lt_0.10": self.threshold_counts["abs_lt_0.10"] / self.count,
            f"fraction_{prefix}_lt_0.5": self.threshold_counts["sigma_lt_0.5"] / self.count,
            f"fraction_{prefix}_lt_0.8": self.threshold_counts["sigma_lt_0.8"] / self.count,
            f"fraction_{prefix}_gt_1.2": self.threshold_counts["sigma_gt_1.2"] / self.count,
            f"fraction_{prefix}_gt_1.5": self.threshold_counts["sigma_gt_1.5"] / self.count,
        })
        return result


def per_image_sigma_statistics(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().float().reshape(-1).cpu()
    if not values.numel():
        return {key: float("nan") for key in (
            "sigma_mean", "sigma_std", "sigma_min", "sigma_max",
            "sigma_p05", "sigma_p50", "sigma_p95",
        )} | {"pixel_count": 0}
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.5, 0.95]))
    return {
        "pixel_count": values.numel(),
        "sigma_mean": float(values.mean()),
        "sigma_std": float(values.std(unbiased=False)),
        "sigma_min": float(values.min()),
        "sigma_max": float(values.max()),
        "sigma_p05": float(quantiles[0]),
        "sigma_p50": float(quantiles[1]),
        "sigma_p95": float(quantiles[2]),
    }


def sigma_variation_summary(per_image_rows: list[dict[str, Any]], scope: str) -> dict[str, float]:
    selected = [row for row in per_image_rows if row["scope"] == scope]
    means = torch.tensor([row["sigma_mean"] for row in selected], dtype=torch.float64)
    within = torch.tensor([row["sigma_std"] for row in selected], dtype=torch.float64)
    return {
        "sigma_between_image_std": float(means.std(unbiased=False)) if means.numel() else float("nan"),
        "sigma_within_image_std_mean": float(within.mean()) if within.numel() else float("nan"),
    }


class PredictionComparisonAccumulator:
    def __init__(self) -> None:
        self.valid = 0
        self.disagree = 0
        self.both_correct = 0
        self.learned_only = 0
        self.fixed_only = 0
        self.both_wrong = 0
        self.sigma_agree_sum = 0.0
        self.sigma_disagree_sum = 0.0
        self.abs_agree_sum = 0.0
        self.abs_disagree_sum = 0.0
        self.agree_count = 0
        self.disagree_count = 0
        self.bins = [
            {"count": 0, "disagree": 0, "learned_correct": 0, "fixed_correct": 0}
            for _ in SIGMA_BIN_LABELS
        ]

    def update(
        self,
        learned: torch.Tensor,
        fixed: torch.Tensor,
        target: torch.Tensor,
        sigma: torch.Tensor,
        *,
        void_index: int,
        num_classes: int,
    ) -> None:
        valid = (target >= 0) & (target < num_classes) & (target != void_index)
        disagreement = learned != fixed
        learned_correct = learned == target
        fixed_correct = fixed == target
        self.valid += int(valid.sum())
        self.disagree += int((valid & disagreement).sum())
        self.both_correct += int((valid & learned_correct & fixed_correct).sum())
        self.learned_only += int((valid & learned_correct & ~fixed_correct).sum())
        self.fixed_only += int((valid & ~learned_correct & fixed_correct).sum())
        self.both_wrong += int((valid & ~learned_correct & ~fixed_correct).sum())
        absolute = (sigma - 1.0).abs()
        agree_mask = valid & ~disagreement
        disagree_mask = valid & disagreement
        self.sigma_agree_sum += float(sigma[agree_mask].double().sum().cpu())
        self.sigma_disagree_sum += float(sigma[disagree_mask].double().sum().cpu())
        self.abs_agree_sum += float(absolute[agree_mask].double().sum().cpu())
        self.abs_disagree_sum += float(absolute[disagree_mask].double().sum().cpu())
        self.agree_count += int(agree_mask.sum())
        self.disagree_count += int(disagree_mask.sum())
        for index, (lower, upper) in enumerate(zip(
            SIGMA_BIN_EDGES[:-1], SIGMA_BIN_EDGES[1:], strict=True
        )):
            selected = valid & (absolute >= lower) & (absolute < upper)
            self.bins[index]["count"] += int(selected.sum())
            self.bins[index]["disagree"] += int((selected & disagreement).sum())
            self.bins[index]["learned_correct"] += int((selected & learned_correct).sum())
            self.bins[index]["fixed_correct"] += int((selected & fixed_correct).sum())

    @staticmethod
    def _ratio(numerator: int | float, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    def compute(self) -> dict[str, float | int]:
        return {
            "valid_pixel_count": self.valid,
            "prediction_disagreement_rate": self._ratio(self.disagree, self.valid),
            "both_correct_fraction": self._ratio(self.both_correct, self.valid),
            "learned_only_correct_fraction": self._ratio(self.learned_only, self.valid),
            "fixed_only_correct_fraction": self._ratio(self.fixed_only, self.valid),
            "both_wrong_fraction": self._ratio(self.both_wrong, self.valid),
            "sigma_mean_prediction_agree": self._ratio(self.sigma_agree_sum, self.agree_count),
            "sigma_mean_prediction_disagree": self._ratio(self.sigma_disagree_sum, self.disagree_count),
            "abs_sigma_minus_1_mean_prediction_agree": self._ratio(self.abs_agree_sum, self.agree_count),
            "abs_sigma_minus_1_mean_prediction_disagree": self._ratio(self.abs_disagree_sum, self.disagree_count),
        }

    def bin_rows(self) -> list[dict[str, Any]]:
        rows = []
        for label, values in zip(SIGMA_BIN_LABELS, self.bins, strict=True):
            count = values["count"]
            rows.append({
                "sigma_bin": label,
                "pixel_count": count,
                "pixel_fraction": self._ratio(count, self.valid),
                "prediction_disagreement_rate": self._ratio(values["disagree"], count),
                "learned_pixel_acc": self._ratio(values["learned_correct"], count),
                "fixed_pixel_acc": self._ratio(values["fixed_correct"], count),
            })
        return rows


def parameter_versions(*modules: torch.nn.Module) -> tuple[tuple[str, int], ...]:
    return tuple(
        (f"{module_index}:{name}", parameter._version)
        for module_index, module in enumerate(modules)
        for name, parameter in module.named_parameters()
    )
