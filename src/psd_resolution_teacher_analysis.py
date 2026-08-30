"""Diagnostic-only comparison of state/full PSD and pseudo-target quality."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

import losses
from dataset import build_dataset
from gradient_conflict_analysis import (
    EPS,
    Gradient,
    _autograd,
    _load_models,
    _norm,
    _write_csv,
    add_gradients,
    build_diagnostic_graph,
    gradient_pair_metrics,
    module_groups,
    parameter_leaf_group,
    scale_gradient,
    summarize_rows,
)
from state_space import resize_continuous
from utils import autocast_context, seed_everything


CITYSCAPES_CLASSES = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
)
THIN_RARE_CLASS_IDS = (5, 6, 7, 11, 12, 17, 18)
CONFIDENCE_THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def resize_probability(
    probability: torch.Tensor,
    size: tuple[int, int] | list[int],
    probability_eps: float,
    *,
    detach: bool = False,
) -> torch.Tensor:
    """Bilinearly resize a probability field and renormalize over classes."""
    source = probability.detach() if detach else probability
    resized = resize_continuous(source.float(), size)
    if detach:
        if resized.data_ptr() == source.data_ptr():
            resized = resized.clone()
        resized.clamp_min_(probability_eps)
        resized.div_(resized.sum(dim=1, keepdim=True).clamp_min_(probability_eps))
        return resized
    resized = resized.clamp_min(probability_eps)
    return resized / resized.sum(dim=1, keepdim=True).clamp_min(probability_eps)


def full_resolution_psd(
    teacher_prob_state: torch.Tensor,
    student_prob_state: torch.Tensor,
    *,
    full_size: tuple[int, int] | list[int],
    valid_mask_full: torch.Tensor | None,
    probability_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the existing PSD probabilities; no teacher/student re-forward occurs."""
    teacher_full = resize_probability(
        teacher_prob_state, full_size, probability_eps, detach=True
    )
    student_full = resize_probability(
        student_prob_state, full_size, probability_eps, detach=False
    )
    loss_map = -(
        teacher_full * student_full.clamp_min(probability_eps).log()
    ).sum(dim=1)
    return losses.masked_mean(loss_map, valid_mask_full).float(), teacher_full, student_full


def semantic_prediction(
    probability: torch.Tensor, *, void_class_index: int = 19
) -> tuple[torch.Tensor, torch.Tensor]:
    semantic = probability[:, :void_class_index]
    confidence, prediction = semantic.max(dim=1)
    return prediction, confidence


def _metrics_from_confusion(confusion: torch.Tensor) -> dict[str, Any]:
    confusion = confusion.double()
    true_positive = confusion.diag()
    ground_truth = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    union = ground_truth + predicted - true_positive
    present_iou = union > 0
    present_accuracy = ground_truth > 0
    iou = torch.where(present_iou, true_positive / union.clamp_min(1.0), torch.nan)
    class_accuracy = torch.where(
        present_accuracy, true_positive / ground_truth.clamp_min(1.0), torch.nan
    )
    return {
        "miou": float(torch.nanmean(iou)) if bool(present_iou.any()) else 0.0,
        "pixel_accuracy": float(true_positive.sum() / confusion.sum().clamp_min(1.0)),
        "mean_class_accuracy": (
            float(torch.nanmean(class_accuracy)) if bool(present_accuracy.any()) else 0.0
        ),
        "per_class_iou": [float(value) if torch.isfinite(value) else 0.0 for value in iou],
        "per_class_accuracy": [
            float(value) if torch.isfinite(value) else 0.0 for value in class_accuracy
        ],
        "gt_pixel_count": [int(value) for value in ground_truth],
    }


class StreamingDistribution:
    """Exact moments plus bounded-memory histogram percentile estimates."""

    def __init__(self, minimum: float, maximum: float, bins: int = 4096) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.bins = bins
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.histogram = torch.zeros(bins, dtype=torch.int64)

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().float().reshape(-1)
        if values.numel() == 0:
            return
        self.count += values.numel()
        self.total += float(values.double().sum().cpu())
        self.total_square += float(values.double().square().sum().cpu())
        histogram = torch.histc(
            values.clamp(self.minimum, self.maximum),
            bins=self.bins,
            min=self.minimum,
            max=self.maximum,
        ).to(dtype=torch.int64, device="cpu")
        self.histogram += histogram

    def _quantile(self, probability: float) -> float:
        if self.count == 0:
            return 0.0
        target = max(1, math.ceil(probability * self.count))
        index = int(torch.searchsorted(
            self.histogram.cumsum(0), torch.tensor(target)
        ).clamp_max(self.bins - 1))
        width = (self.maximum - self.minimum) / self.bins
        return self.minimum + (index + 0.5) * width

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {key: 0.0 for key in (
                "mean", "std", "median", "p10", "p25", "p75", "p90",
                "p95", "p99",
            )}
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {
            "mean": mean,
            "std": math.sqrt(variance),
            "median": self._quantile(0.5),
            "p10": self._quantile(0.10),
            "p25": self._quantile(0.25),
            "p75": self._quantile(0.75),
            "p90": self._quantile(0.90),
            "p95": self._quantile(0.95),
            "p99": self._quantile(0.99),
        }


class QualityAccumulator:
    def __init__(self, confidence_threshold: float, probability_eps: float) -> None:
        self.confidence_threshold = confidence_threshold
        self.probability_eps = probability_eps
        self.confusion = torch.zeros(19, 19, dtype=torch.int64)
        self.entropy = StreamingDistribution(0.0, math.log(20.0))
        self.confidence = StreamingDistribution(0.0, 1.0)
        self.valid_count = 0
        self.wrong_count = 0
        self.high_count = 0
        self.high_wrong_count = 0
        self.correct_confidence_sum = 0.0
        self.correct_count = 0
        self.wrong_confidence_sum = 0.0
        self.void_probability_sum = 0.0
        self.void_argmax_count = 0
        self.threshold_wrong_counts = {value: 0 for value in CONFIDENCE_THRESHOLDS}
        self.class_confidence_sum = torch.zeros(19, dtype=torch.float64)
        self.class_confidence_count = torch.zeros(19, dtype=torch.int64)

    @torch.no_grad()
    def update(
        self, probability: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, Any], dict[float, float]]:
        probability = probability.detach().float()
        prediction, confidence = semantic_prediction(probability)
        valid = (target >= 0) & (target < 19)
        prediction_valid = prediction[valid]
        target_valid = target[valid]
        confidence_valid = confidence[valid]
        wrong = prediction_valid != target_valid
        high = confidence_valid > self.confidence_threshold
        # Limit peak metric memory: entropy needs a class-sized temporary, so
        # evaluate one physical sample at a time instead of Bx20xHxW at once.
        entropy_map = torch.empty_like(target, dtype=torch.float32)
        for sample_index in range(probability.shape[0]):
            sample = probability[sample_index]
            entropy_map[sample_index] = -(
                sample.clamp_min(self.probability_eps).log() * sample
            ).sum(dim=0)
        entropy_valid = entropy_map[valid]
        indices = target_valid * 19 + prediction_valid
        confusion = torch.bincount(indices, minlength=19 * 19).reshape(19, 19).cpu()
        self.confusion += confusion
        self.entropy.update(entropy_valid)
        self.confidence.update(confidence_valid)
        valid_count = int(valid.sum())
        wrong_count = int(wrong.sum())
        high_count = int(high.sum())
        high_wrong_count = int((wrong & high).sum())
        self.valid_count += valid_count
        self.wrong_count += wrong_count
        self.high_count += high_count
        self.high_wrong_count += high_wrong_count
        correct = ~wrong
        self.correct_confidence_sum += float(confidence_valid[correct].double().sum().cpu())
        self.correct_count += int(correct.sum())
        self.wrong_confidence_sum += float(confidence_valid[wrong].double().sum().cpu())
        self.void_probability_sum += float(probability[:, 19][valid].double().sum().cpu())
        self.void_argmax_count += int((probability.argmax(dim=1)[valid] == 19).sum())
        threshold_fractions: dict[float, float] = {}
        for threshold in CONFIDENCE_THRESHOLDS:
            count = int((wrong & (confidence_valid > threshold)).sum())
            self.threshold_wrong_counts[threshold] += count
            threshold_fractions[threshold] = count / max(valid_count, 1)
        for class_id in range(19):
            class_mask = target_valid == class_id
            self.class_confidence_sum[class_id] += confidence_valid[class_mask].double().sum().cpu()
            self.class_confidence_count[class_id] += int(class_mask.sum())
        batch_metrics = _metrics_from_confusion(confusion)
        batch_metrics.update({
            "entropy_mean": float(entropy_valid.mean().cpu()) if valid_count else 0.0,
            "confidence_mean": float(confidence_valid.mean().cpu()) if valid_count else 0.0,
            "wrong_pixel_fraction": wrong_count / max(valid_count, 1),
            "high_confidence_pixel_fraction": high_count / max(valid_count, 1),
            "high_confidence_wrong_valid_fraction": high_wrong_count / max(valid_count, 1),
            "high_confidence_wrong_among_wrong_fraction": high_wrong_count / max(wrong_count, 1),
            "teacher_void_probability_mean": (
                float(probability[:, 19][valid].mean().cpu()) if valid_count else 0.0
            ),
            "teacher_void_argmax_ratio_raw20": (
                float((probability.argmax(dim=1)[valid] == 19).float().mean().cpu())
                if valid_count else 0.0
            ),
        })
        return batch_metrics, threshold_fractions

    def compute(self) -> dict[str, Any]:
        result = _metrics_from_confusion(self.confusion)
        result.update({
            "entropy": self.entropy.compute(),
            "confidence": self.confidence.compute(),
            "confidence_mean_correct": self.correct_confidence_sum / max(self.correct_count, 1),
            "confidence_mean_wrong": self.wrong_confidence_sum / max(self.wrong_count, 1),
            "wrong_pixel_fraction": self.wrong_count / max(self.valid_count, 1),
            "high_confidence_pixel_fraction": self.high_count / max(self.valid_count, 1),
            "high_confidence_wrong_valid_fraction": self.high_wrong_count / max(self.valid_count, 1),
            "high_confidence_wrong_among_wrong_fraction": self.high_wrong_count / max(self.wrong_count, 1),
            "teacher_void_probability_mean": self.void_probability_sum / max(self.valid_count, 1),
            "teacher_void_argmax_ratio_raw20": self.void_argmax_count / max(self.valid_count, 1),
            "confidence_threshold": self.confidence_threshold,
            "wrong_confidence_threshold_sweep": {
                str(threshold): count / max(self.valid_count, 1)
                for threshold, count in self.threshold_wrong_counts.items()
            },
            "class_confidence_mean": [
                float(self.class_confidence_sum[index] / self.class_confidence_count[index].clamp_min(1))
                for index in range(19)
            ],
        })
        result["entropy_mean"] = result["entropy"]["mean"]
        result["confidence_mean"] = result["confidence"]["mean"]
        return result


def _resolution_gradient_metrics(
    base: Gradient,
    psd: Gradient,
    psd_weight: float,
    indices: Sequence[int],
) -> dict[str, float]:
    base = tuple(base[index] for index in indices)
    psd = tuple(psd[index] for index in indices)
    selected = tuple(range(len(base)))
    weighted = scale_gradient(psd, psd_weight)
    combined = add_gradients(base, weighted)
    pair = gradient_pair_metrics(base, psd, selected)
    combined_pair = gradient_pair_metrics(combined, base, selected)
    base_norm = _norm(base, selected)
    psd_norm = _norm(psd, selected)
    weighted_norm = _norm(weighted, selected)
    return {
        "base_grad_norm": base_norm,
        "psd_grad_norm_raw": psd_norm,
        "psd_grad_norm_weighted": weighted_norm,
        "base_vs_psd_dot": pair["dot"],
        "base_vs_psd_cosine": pair["cosine"],
        "base_vs_psd_angle_degrees": pair["angle_degrees"],
        "weighted_psd_to_base_norm_ratio": weighted_norm / (base_norm + EPS),
        "combined_grad_norm": _norm(combined, selected),
        "combined_vs_base_cosine": combined_pair["cosine"],
        "combined_vs_base_dot": combined_pair["dot"],
    }


def _prefixed(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def run_psd_resolution_teacher_analysis(
    config: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path | None,
    num_batches: int,
    batch_size: int,
    psd_weight: float | None,
    teacher_confidence_threshold: float,
    seed: int,
    device: torch.device,
    num_workers: int | None = None,
) -> Path:
    if config["loss"]["consistency"]["type"] != "psd":
        raise ValueError("Resolution/teacher analysis requires consistency.type=psd")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else checkpoint_path.parent / f"psd_resolution_teacher_{checkpoint_path.stem}"
    )
    output.mkdir(parents=True, exist_ok=True)
    weight = float(
        config["loss"]["consistency"]["weight"]
        if psd_weight is None else psd_weight
    )
    probability_eps = float(config["flow"]["probability_eps"])
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
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=workers, pin_memory=config["dataset"]["pin_memory"],
    )
    quality = {
        name: QualityAccumulator(teacher_confidence_threshold, probability_eps)
        for name in ("teacher_state", "teacher_full", "student_state", "student_full")
    }
    batch_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    for batch_index, (image, target) in enumerate(loader):
        if batch_index >= num_batches:
            break
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast_context(config, device):
            graph = build_diagnostic_graph(config, endpoint, source, image, target)
        psd_full_loss, teacher_full, student_full = full_resolution_psd(
            graph.teacher_prob_state,
            graph.student_prob_state,
            full_size=tuple(graph.target_full.shape[-2:]),
            valid_mask_full=graph.valid_mask_full,
            probability_eps=probability_eps,
        )
        base_loss = graph.primary + graph.source_weighted
        base_gradient = _autograd(base_loss, parameters, retain=True)
        state_gradient = _autograd(graph.psd, parameters, retain=True)
        full_gradient = _autograd(psd_full_loss, parameters, retain=False)
        all_indices = groups["all_trainable_parameters"]
        state_metrics = _resolution_gradient_metrics(
            base_gradient, state_gradient, weight, all_indices
        )
        full_metrics = _resolution_gradient_metrics(
            base_gradient, full_gradient, weight, all_indices
        )
        row = {
            "batch": batch_index,
            "primary_ce": float(graph.primary_ce.detach().cpu()),
            "source_supervision": float(graph.source_supervision.detach().cpu()),
            "source_supervision_weighted": float(graph.source_weighted.detach().cpu()),
            "psd_state_loss": float(graph.psd.detach().cpu()),
            "psd_full_loss": float(psd_full_loss.detach().cpu()),
            **_prefixed("state", state_metrics),
            **_prefixed("full", full_metrics),
            "delta_base_vs_psd_cosine": (
                full_metrics["base_vs_psd_cosine"] - state_metrics["base_vs_psd_cosine"]
            ),
            "delta_weighted_norm_ratio": (
                full_metrics["weighted_psd_to_base_norm_ratio"]
                - state_metrics["weighted_psd_to_base_norm_ratio"]
            ),
            "s_mean": float(graph.s.detach().float().mean().cpu()),
            "u_mean": float(graph.u.detach().float().mean().cpu()),
            "t_mean": float(graph.t.detach().float().mean().cpu()),
        }
        for group, indices in groups.items():
            if not indices:
                continue
            state_module = _resolution_gradient_metrics(
                base_gradient, state_gradient, weight, indices
            )
            full_module = _resolution_gradient_metrics(
                base_gradient, full_gradient, weight, indices
            )
            module_rows.append({
                "batch": batch_index,
                "module_group": group,
                "num_parameters": sum(parameters[index].numel() for index in indices),
                **_prefixed("state", state_module),
                **_prefixed("full", full_module),
                "delta_base_vs_psd_cosine": (
                    full_module["base_vs_psd_cosine"]
                    - state_module["base_vs_psd_cosine"]
                ),
                "delta_weighted_norm_ratio": (
                    full_module["weighted_psd_to_base_norm_ratio"]
                    - state_module["weighted_psd_to_base_norm_ratio"]
                ),
            })
        for index, (name, parameter) in enumerate(named_parameters):
            state_parameter = _resolution_gradient_metrics(
                base_gradient, state_gradient, weight, [index]
            )
            full_parameter = _resolution_gradient_metrics(
                base_gradient, full_gradient, weight, [index]
            )
            parameter_rows.append({
                "batch": batch_index,
                "parameter_name": name,
                "module_group": parameter_leaf_group(name),
                "numel": parameter.numel(),
                **_prefixed("state", state_parameter),
                **_prefixed("full", full_parameter),
                "delta_base_vs_psd_cosine": (
                    full_parameter["base_vs_psd_cosine"]
                    - state_parameter["base_vs_psd_cosine"]
                ),
                "delta_weighted_norm_ratio": (
                    full_parameter["weighted_psd_to_base_norm_ratio"]
                    - state_parameter["weighted_psd_to_base_norm_ratio"]
                ),
            })
        # Gradient tensors are no longer needed and can dominate GPU memory.
        del base_gradient, state_gradient, full_gradient
        batch_quality: dict[str, dict[str, Any]] = {}

        def update_quality(
            quality_name: str, probability: torch.Tensor, quality_target: torch.Tensor
        ) -> None:
            metrics, thresholds = quality[quality_name].update(
                probability.detach(), quality_target
            )
            batch_quality[quality_name] = metrics
            if quality_name.startswith("teacher"):
                resolution = quality_name.removeprefix("teacher_")
                for threshold, fraction in thresholds.items():
                    threshold_rows.append({
                        "batch": batch_index,
                        "resolution": resolution,
                        "threshold": threshold,
                        "wrong_high_confidence_valid_fraction": fraction,
                    })

        update_quality("teacher_state", graph.teacher_prob_state, graph.target_state)
        update_quality("teacher_full", teacher_full, graph.target_full)
        del teacher_full
        update_quality("student_state", graph.student_prob_state, graph.target_state)
        update_quality("student_full", student_full, graph.target_full)
        del student_full
        for resolution in ("state", "full"):
            teacher_metric = batch_quality[f"teacher_{resolution}"]
            student_metric = batch_quality[f"student_{resolution}"]
            teacher_rows.append({
                "batch": batch_index,
                "resolution": resolution,
                **{f"teacher_{key}": value for key, value in teacher_metric.items() if not isinstance(value, list)},
                **{f"student_{key}": value for key, value in student_metric.items() if not isinstance(value, list)},
            })
        for resolution in ("state", "full"):
            metrics = batch_quality[f"teacher_{resolution}"]
            row.update({
                f"teacher_{resolution}_miou": metrics["miou"],
                f"teacher_{resolution}_pixel_accuracy": metrics["pixel_accuracy"],
                f"teacher_{resolution}_entropy_mean": metrics["entropy_mean"],
                f"teacher_{resolution}_confidence_mean": metrics["confidence_mean"],
                f"teacher_{resolution}_high_confidence_wrong_valid_fraction": metrics[
                    "high_confidence_wrong_valid_fraction"
                ],
            })
        batch_rows.append(row)
        del graph, psd_full_loss

    if len(batch_rows) != num_batches:
        raise RuntimeError(
            f"Requested {num_batches} batches but loader produced {len(batch_rows)}"
        )
    quality_summary = {name: accumulator.compute() for name, accumulator in quality.items()}
    class_rows = []
    for class_id, class_name in enumerate(CITYSCAPES_CLASSES):
        class_rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "gt_pixel_count_state": quality_summary["teacher_state"]["gt_pixel_count"][class_id],
            "gt_pixel_count_full": quality_summary["teacher_full"]["gt_pixel_count"][class_id],
            "teacher_state_iou": quality_summary["teacher_state"]["per_class_iou"][class_id],
            "teacher_full_iou": quality_summary["teacher_full"]["per_class_iou"][class_id],
            "student_state_iou": quality_summary["student_state"]["per_class_iou"][class_id],
            "student_full_iou": quality_summary["student_full"]["per_class_iou"][class_id],
            "teacher_state_confidence_mean": quality_summary["teacher_state"]["class_confidence_mean"][class_id],
            "teacher_full_confidence_mean": quality_summary["teacher_full"]["class_confidence_mean"][class_id],
        })
    _write_csv(output / "psd_resolution_batches.csv", batch_rows)
    _write_csv(output / "psd_resolution_modules.csv", module_rows)
    _write_csv(output / "psd_resolution_parameters.csv", parameter_rows)
    _write_csv(output / "teacher_quality_batches.csv", teacher_rows)
    _write_csv(output / "teacher_quality_classes.csv", class_rows)
    _write_csv(output / "teacher_confidence_thresholds.csv", threshold_rows)
    arguments = {
        "config": config["runtime"].get("config_path"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "output_dir": str(output),
        "num_batches": num_batches,
        "batch_size": batch_size,
        "psd_weight": weight,
        "teacher_confidence_threshold": teacher_confidence_threshold,
        "seed": seed,
        "device": str(device),
        "num_workers": workers,
    }
    (output / "diagnostic_args.json").write_text(
        json.dumps(arguments, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    module_summary: dict[str, Any] = {}
    for group in groups:
        rows = [row for row in module_rows if row["module_group"] == group]
        if rows:
            keys = [
                key for key, value in rows[0].items()
                if key not in {"batch", "module_group", "num_parameters"}
                and isinstance(value, (int, float))
            ]
            module_summary[group] = summarize_rows(rows, keys)
    state_resolution = {
        "loss_mean": _mean(batch_rows, "psd_state_loss"),
        "base_vs_psd_cosine_mean": _mean(batch_rows, "state_base_vs_psd_cosine"),
        "weighted_psd_to_base_ratio_mean": _mean(
            batch_rows, "state_weighted_psd_to_base_norm_ratio"
        ),
        "combined_vs_base_cosine_mean": _mean(
            batch_rows, "state_combined_vs_base_cosine"
        ),
    }
    full_resolution = {
        "loss_mean": _mean(batch_rows, "psd_full_loss"),
        "base_vs_psd_cosine_mean": _mean(batch_rows, "full_base_vs_psd_cosine"),
        "weighted_psd_to_base_ratio_mean": _mean(
            batch_rows, "full_weighted_psd_to_base_norm_ratio"
        ),
        "combined_vs_base_cosine_mean": _mean(
            batch_rows, "full_combined_vs_base_cosine"
        ),
    }
    caution = (
        "This diagnostic only compares loss-resolution effects for the same "
        "teacher/student probability field. Better gradient alignment does not "
        "establish that full-resolution PSD training will improve mIoU."
    )
    interpretation = (
        "Full-resolution PSD is more aligned with the supervised objective."
        if (
            full_resolution["base_vs_psd_cosine_mean"]
            > state_resolution["base_vs_psd_cosine_mean"]
            and full_resolution["combined_vs_base_cosine_mean"]
            > state_resolution["combined_vs_base_cosine_mean"]
        )
        else "No consistent gradient-alignment advantage for full-resolution PSD was observed."
    )
    teacher_error_note = (
        "PSD teacher may exhibit high-confidence confirmation errors."
        if quality_summary["teacher_full"]["high_confidence_wrong_valid_fraction"] > 0.1
        else "No large high-confidence teacher-error fraction under the diagnostic heuristic."
    )
    summary = {
        **arguments,
        "psd_resolution": {
            "state": state_resolution,
            "full": full_resolution,
            "delta": {
                "base_vs_psd_cosine_full_minus_state": _mean(
                    batch_rows, "delta_base_vs_psd_cosine"
                ),
                "weighted_norm_ratio_full_minus_state": _mean(
                    batch_rows, "delta_weighted_norm_ratio"
                ),
            },
            "batch_statistics": summarize_rows(
                batch_rows,
                [key for key, value in batch_rows[0].items() if isinstance(value, (int, float))],
            ),
        },
        "teacher": {
            "state": quality_summary["teacher_state"],
            "full": quality_summary["teacher_full"],
        },
        "student": {
            "state": quality_summary["student_state"],
            "full": quality_summary["student_full"],
        },
        "modules": module_summary,
        "interpretation": {
            "gradient": interpretation,
            "teacher": teacher_error_note,
            "heuristic_only": True,
            "caution": caution,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("===== PSD Resolution Comparison =====")
    for label, values in (("PSD @ 128x256", state_resolution), ("PSD @ 512x1024", full_resolution)):
        print(f"{label}: loss={values['loss_mean']:.6g}, base cosine={values['base_vs_psd_cosine_mean']:.6g}, "
              f"weighted/base={values['weighted_psd_to_base_ratio_mean']:.6g}, "
              f"combined cosine={values['combined_vs_base_cosine_mean']:.6g}")
    print("Delta full-state: cosine="
          f"{summary['psd_resolution']['delta']['base_vs_psd_cosine_full_minus_state']:.6g}, "
          "norm ratio="
          f"{summary['psd_resolution']['delta']['weighted_norm_ratio_full_minus_state']:.6g}")
    print("\n===== Important Modules =====")
    for group in ("unet_output", "unet_embed_s", "unet_embed_delta"):
        values = module_summary[group]
        print(f"{group}: state={values['state_base_vs_psd_cosine']['mean']:.6g}, "
              f"full={values['full_base_vs_psd_cosine']['mean']:.6g}, "
              f"delta={values['delta_base_vs_psd_cosine']['mean']:.6g}")
    print("\n===== PSD Teacher Quality =====")
    for resolution in ("state", "full"):
        values = quality_summary[f"teacher_{resolution}"]
        print(f"Teacher @ {resolution}: mIoU={values['miou']:.6g}, "
              f"pixel_acc={values['pixel_accuracy']:.6g}, "
              f"entropy={values['entropy']['mean']:.6g}, "
              f"confidence={values['confidence']['mean']:.6g}, "
              f"wrong={values['wrong_pixel_fraction']:.6g}, "
              f"wrong_high_conf={values['high_confidence_wrong_valid_fraction']:.6g}")
    print("\n===== Thin / Rare Classes =====")
    for class_id in THIN_RARE_CLASS_IDS:
        values = class_rows[class_id]
        print(f"{values['class_name']}: teacher_state={values['teacher_state_iou']:.6g}, "
              f"teacher_full={values['teacher_full_iou']:.6g}, "
              f"student_state={values['student_state_iou']:.6g}, "
              f"student_full={values['student_full_iou']:.6g}")
    print("\n===== Interpretation =====")
    print(interpretation)
    print(teacher_error_note)
    print(caution)
    print("These are exploratory diagnostic heuristics, not research thresholds.")
    print(f"Output: {output}")
    return output
