"""Diagnostic-only PSD time-bin gradients and direct time-map evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ade20k_eval_collate, build_dataset
from discrete_flow_maps import linear_path, sample_consistency_times, sample_prior
from gradient_conflict_analysis import (
    Gradient,
    _autograd,
    _load_models,
    _write_csv,
    build_diagnostic_graph,
    gradient_pair_metrics,
    module_groups,
    parameter_leaf_group,
    summarize_rows,
)
from inference import state_to_original_continuous
from psd_resolution_teacher_analysis import (
    CITYSCAPES_CLASSES,
    QualityAccumulator,
    THIN_RARE_CLASS_IDS,
    _resolution_gradient_metrics,
)
from state_space import prepare_state_targets, state_spatial_size
from utils import autocast_context, seed_everything


PI0T_GRID = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
PISS_GRID = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90)


@dataclass(frozen=True)
class TimeBin:
    axis: str
    low: float
    high: float
    inclusive_high: bool = False

    @property
    def label(self) -> str:
        bracket = "]" if self.inclusive_high else ")"
        return f"{self.axis}[{self.low:.2f},{self.high:.2f}{bracket}"

    def mask(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        value = s if self.axis == "s" else t - s
        upper = value <= self.high if self.inclusive_high else value < self.high
        return (value >= self.low) & upper


S_BINS = (
    TimeBin("s", 0.00, 0.10),
    TimeBin("s", 0.10, 0.25),
    TimeBin("s", 0.25, 0.50),
    TimeBin("s", 0.50, 0.75),
    TimeBin("s", 0.75, 1.00, True),
)
DELTA_BINS = (
    TimeBin("delta", 0.00, 0.10),
    TimeBin("delta", 0.10, 0.25),
    TimeBin("delta", 0.25, 0.50),
    TimeBin("delta", 0.50, 1.00, True),
)


def conditional_production_times(
    time_bin: TimeBin,
    batch_size: int,
    device: torch.device,
    time_config: dict[str, Any],
    *,
    seed: int,
    max_proposals: int = 100_000,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, float | int]]:
    """Rejection-sample from the unchanged production PSD time distribution."""
    accepted: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    proposed = 0
    devices = [
        device.index if device.index is not None else torch.cuda.current_device()
    ] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        remaining = batch_size
        while remaining > 0 and proposed < max_proposals:
            proposal_size = min(remaining, max_proposals - proposed)
            s, u, t = sample_consistency_times(
                "psd", proposal_size, device,
                time_config["min_time"], time_config["max_time"],
                time_config["min_gap"],
            )
            assert u is not None
            proposed += proposal_size
            keep = time_bin.mask(s, t)
            if bool(keep.any()):
                accepted.append((s[keep], u[keep], t[keep]))
                remaining -= int(keep.sum())
        if remaining:
            raise RuntimeError(
                f"Conditional production time sampling failed for {time_bin.label}: "
                f"accepted={batch_size - remaining}, proposed={proposed}, "
                f"max_proposals={max_proposals}"
            )
    s = torch.cat([values[0] for values in accepted])[:batch_size]
    u = torch.cat([values[1] for values in accepted])[:batch_size]
    t = torch.cat([values[2] for values in accepted])[:batch_size]
    if not bool(time_bin.mask(s, t).all()) or not bool(((s < u) & (u < t)).all()):
        raise AssertionError("Conditional production sampler returned invalid times")
    stats: dict[str, float | int] = {
        "accepted_count": batch_size,
        "proposed_count": proposed,
        "acceptance_rate": batch_size / proposed,
        "mean_s": float(s.float().mean().cpu()),
        "mean_u": float(u.float().mean().cpu()),
        "mean_t": float(t.float().mean().cpu()),
        "mean_delta": float((t - s).float().mean().cpu()),
    }
    return (s, u, t), stats


def _capture_rng(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return torch.random.get_rng_state(), cuda_state


def _restore_rng(
    state: tuple[torch.Tensor, torch.Tensor | None], device: torch.device
) -> None:
    torch.random.set_rng_state(state[0])
    if device.type == "cuda":
        assert state[1] is not None
        torch.cuda.set_rng_state(state[1], device)


def _quality_scalars(metrics: dict[str, Any], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_miou": metrics["miou"],
        f"{prefix}_pixel_accuracy": metrics["pixel_accuracy"],
        f"{prefix}_macc": metrics["mean_class_accuracy"],
        f"{prefix}_entropy": metrics["entropy_mean"],
        f"{prefix}_confidence": metrics["confidence_mean"],
        f"{prefix}_wrong_fraction": metrics["wrong_pixel_fraction"],
        f"{prefix}_high_confidence_wrong_fraction": metrics[
            "high_confidence_wrong_valid_fraction"
        ],
        f"{prefix}_void_probability_mean": metrics["teacher_void_probability_mean"],
        f"{prefix}_void_argmax_ratio_raw20": metrics[
            "teacher_void_argmax_ratio_raw20"
        ],
    }


def _aggregate_module_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    identities = sorted({
        (row["axis"], row["bin_low"], row["bin_high"], row["bin_label"], row["module_group"])
        for row in rows
    })
    for axis, low, high, label, group in identities:
        selected = [
            row for row in rows
            if row["axis"] == axis and row["bin_label"] == label
            and row["module_group"] == group
        ]
        stats = summarize_rows(selected, (
            "base_vs_psd_cosine", "weighted_psd_to_base_norm_ratio",
            "combined_vs_base_cosine",
        ))
        result.append({
            "axis": axis, "bin_low": low, "bin_high": high,
            "bin_label": label, "module_group": group,
            "base_vs_psd_cosine_mean": stats["base_vs_psd_cosine"]["mean"],
            "base_vs_psd_cosine_std": stats["base_vs_psd_cosine"]["std"],
            "weighted_psd_to_base_norm_ratio_mean": stats[
                "weighted_psd_to_base_norm_ratio"
            ]["mean"],
            "combined_vs_base_cosine_mean": stats["combined_vs_base_cosine"]["mean"],
        })
    return result


def _phase_a(
    config: dict[str, Any], endpoint, source, *, loader: DataLoader,
    time_bin_batches: int, weight: float, confidence_threshold: float,
    seed: int, device: torch.device, save_parameter_details: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
           list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
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
    bins = S_BINS + DELTA_BINS
    quality = {
        (time_bin.label, role): QualityAccumulator(
            confidence_threshold, float(config["flow"]["probability_eps"])
        )
        for time_bin in bins for role in ("teacher", "student")
    }
    batch_rows: list[dict[str, Any]] = []
    raw_module_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    warning_count = 0
    for batch_index, (image, target) in enumerate(loader):
        if batch_index >= time_bin_batches:
            break
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        model_rng = _capture_rng(device)
        post_reference_rng = None
        reference_base: Gradient | None = None
        for bin_index, time_bin in enumerate(bins):
            times, sampling = conditional_production_times(
                time_bin, image.shape[0], device, config["time_sampling"],
                seed=seed + batch_index * 10_000 + bin_index,
            )
            _restore_rng(model_rng, device)
            with autocast_context(config, device):
                graph = build_diagnostic_graph(
                    config, endpoint, source, image, target,
                    consistency_times=times,
                )
            if post_reference_rng is None:
                post_reference_rng = _capture_rng(device)
            base_gradient = _autograd(
                graph.primary + graph.source_weighted, parameters, retain=True
            )
            psd_gradient = _autograd(graph.psd, parameters, retain=False)
            if reference_base is None:
                reference_base = tuple(
                    None if value is None else value.detach() for value in base_gradient
                )
            reproducibility = gradient_pair_metrics(reference_base, base_gradient)
            reference_norm = reproducibility["first_norm"]
            norm_error = abs(
                reproducibility["second_norm"] - reference_norm
            ) / (reference_norm + 1.0e-12)
            if reproducibility["cosine"] < 0.999:
                warning_count += 1
                print(
                    f"WARNING: base gradient reproducibility for {time_bin.label} "
                    f"is {reproducibility['cosine']:.8f}"
                )
            global_metrics = _resolution_gradient_metrics(
                base_gradient, psd_gradient, weight,
                groups["all_trainable_parameters"],
            )
            teacher_metrics, _ = quality[(time_bin.label, "teacher")].update(
                graph.teacher_prob_state, graph.target_state
            )
            student_metrics, _ = quality[(time_bin.label, "student")].update(
                graph.student_prob_state, graph.target_state
            )
            common = {
                "batch": batch_index,
                "axis": time_bin.axis,
                "bin_low": time_bin.low,
                "bin_high": time_bin.high,
                "bin_label": time_bin.label,
            }
            batch_rows.append({
                **common, **sampling,
                "psd_loss": float(graph.psd.detach().cpu()),
                **global_metrics,
                "base_reproducibility_cosine": reproducibility["cosine"],
                "base_reproducibility_norm_relative_error": norm_error,
                **_quality_scalars(teacher_metrics, "teacher"),
                **_quality_scalars(student_metrics, "student"),
            })
            for group, indices in groups.items():
                if indices:
                    raw_module_rows.append({
                        **common, "module_group": group,
                        **_resolution_gradient_metrics(
                            base_gradient, psd_gradient, weight, indices
                        ),
                    })
            if save_parameter_details:
                for parameter_index, (name, parameter) in enumerate(named_parameters):
                    parameter_rows.append({
                        **common,
                        "parameter_name": name,
                        "module_group": parameter_leaf_group(name),
                        "numel": parameter.numel(),
                        **_resolution_gradient_metrics(
                            base_gradient, psd_gradient, weight, [parameter_index]
                        ),
                    })
            del graph, base_gradient, psd_gradient
        assert post_reference_rng is not None
        _restore_rng(post_reference_rng, device)
        del reference_base
    if len(batch_rows) != time_bin_batches * len(bins):
        raise RuntimeError("Training loader did not provide all requested time-bin batches")
    summary_rows = []
    quality_rows = []
    for time_bin in bins:
        selected = [row for row in batch_rows if row["bin_label"] == time_bin.label]
        teacher = quality[(time_bin.label, "teacher")].compute()
        student = quality[(time_bin.label, "student")].compute()
        accepted = sum(int(row["accepted_count"]) for row in selected)
        proposed = sum(int(row["proposed_count"]) for row in selected)
        summary_rows.append({
            "axis": time_bin.axis, "bin_low": time_bin.low,
            "bin_high": time_bin.high, "bin_label": time_bin.label,
            "accepted_count": accepted, "proposed_count": proposed,
            "acceptance_rate": accepted / proposed,
            "mean_s": sum(row["mean_s"] for row in selected) / len(selected),
            "mean_u": sum(row["mean_u"] for row in selected) / len(selected),
            "mean_t": sum(row["mean_t"] for row in selected) / len(selected),
            "mean_delta": sum(row["mean_delta"] for row in selected) / len(selected),
            "psd_loss": sum(row["psd_loss"] for row in selected) / len(selected),
            "base_vs_psd_cosine": sum(row["base_vs_psd_cosine"] for row in selected) / len(selected),
            "weighted_psd_to_base_norm_ratio": sum(
                row["weighted_psd_to_base_norm_ratio"] for row in selected
            ) / len(selected),
            "combined_vs_base_cosine": sum(
                row["combined_vs_base_cosine"] for row in selected
            ) / len(selected),
            "teacher_miou": teacher["miou"],
            "teacher_pixel_accuracy": teacher["pixel_accuracy"],
            "teacher_entropy": teacher["entropy_mean"],
            "teacher_confidence": teacher["confidence_mean"],
            "teacher_wrong_fraction": teacher["wrong_pixel_fraction"],
            "teacher_high_confidence_wrong_fraction": teacher[
                "high_confidence_wrong_valid_fraction"
            ],
            "student_miou": student["miou"],
            "student_pixel_accuracy": student["pixel_accuracy"],
        })
        quality_rows.append({
            "axis": time_bin.axis, "bin_low": time_bin.low,
            "bin_high": time_bin.high, "bin_label": time_bin.label,
            **_quality_scalars(teacher, "teacher"),
            **_quality_scalars(student, "student"),
        })
    module_rows = _aggregate_module_rows(raw_module_rows)
    metadata = {
        "base_reproducibility_warning_count": warning_count,
        "teacher_quality_at_large_s_contains_more_ground_truth_information_via_x_s": True,
        "teacher_quality_caution": (
            "Teacher quality at large s contains more ground-truth information "
            "through x_s=(1-s)x0+s*x1 and is not an inference-only quality measure."
        ),
    }
    return batch_rows, summary_rows, module_rows, quality_rows, parameter_rows, metadata


def _model_target_state(
    sample: dict[str, Any], image: torch.Tensor, target: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    model_shape = tuple(sample["model_shape"])
    target_model = F.interpolate(
        target[:, None].float(), size=model_shape, mode="nearest"
    )[:, 0].long()
    pad_height = image.shape[-2] - target_model.shape[-2]
    pad_width = image.shape[-1] - target_model.shape[-1]
    if pad_height < 0 or pad_width < 0:
        raise ValueError("Model target is larger than the padded validation image")
    target_model = F.pad(target_model, (0, pad_width, 0, pad_height), value=19)
    return prepare_state_targets(
        target_model,
        num_classes=20,
        state_size=state_spatial_size(image, config["model"]["state_downsample_factor"]),
        ignore_index=19,
        mask_pixel_losses=True,
    ).one_hot_state


@torch.no_grad()
def _original_probability(
    endpoint, x: torch.Tensor, image_feat: torch.Tensor,
    s_value: float, t_value: float, sample: dict[str, Any], config: dict[str, Any],
) -> torch.Tensor:
    batch = x.shape[0]
    s = x.new_full((batch,), s_value, dtype=torch.float32)
    t = x.new_full((batch,), t_value, dtype=torch.float32)
    with autocast_context(config, x.device):
        logits = endpoint.forward_logits_with_image_feat(x, image_feat, s, t)
    logits_original = state_to_original_continuous(
        logits.float(), sample["model_shape"], sample["original_shape"],
        padded_shape=sample["padded_shape"],
        align_corners=config["evaluation"]["align_corners"],
    )
    return torch.softmax(logits_original, dim=1)


def _map_row(
    map_type: str,
    s: float,
    t: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "map_type": map_type,
        "s": s,
        "t": t,
        "miou": metrics["miou"],
        "pixel_accuracy": metrics["pixel_accuracy"],
        "macc": metrics["mean_class_accuracy"],
        "entropy": metrics["entropy_mean"],
        "confidence": metrics["confidence_mean"],
        "off_diagonal_gap_miou": "",
    }


@torch.no_grad()
def _phase_b(
    config: dict[str, Any], endpoint, source, *, loader: DataLoader,
    eval_num_batches: int, confidence_threshold: float, device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
           list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    endpoint.eval()
    if source is not None:
        source.eval()
    probability_eps = float(config["flow"]["probability_eps"])
    accumulators: dict[tuple[str, float], QualityAccumulator] = {}
    for t in PI0T_GRID:
        accumulators[("pi0t", t)] = QualityAccumulator(
            confidence_threshold, probability_eps
        )
    for s in PISS_GRID:
        accumulators[("piss", s)] = QualityAccumulator(
            confidence_threshold, probability_eps
        )
        accumulators[("pis1", s)] = QualityAccumulator(
            confidence_threshold, probability_eps
        )
    direct = {
        "pi00": QualityAccumulator(confidence_threshold, probability_eps),
        "pi01": QualityAccumulator(confidence_threshold, probability_eps),
    }
    processed = 0
    for batch_index, batch in enumerate(loader):
        if eval_num_batches > 0 and batch_index >= eval_num_batches:
            break
        samples = batch if isinstance(batch, list) else [batch]
        for sample in samples:
            image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
            target = sample["target"].unsqueeze(0).to(device, non_blocking=True)
            x1 = _model_target_state(sample, image, target, config)
            with autocast_context(config, device):
                x0, _ = sample_prior(config, image, None, source)
                image_feat = endpoint.encode_image(image)
            for t_value in PI0T_GRID:
                probability = _original_probability(
                    endpoint, x0, image_feat, 0.0, t_value, sample, config
                )
                accumulators[("pi0t", t_value)].update(probability, target)
                if t_value == 0.0:
                    direct["pi00"].update(probability, target)
                    accumulators[("piss", 0.0)].update(probability, target)
                if t_value == 1.0:
                    direct["pi01"].update(probability, target)
                    accumulators[("pis1", 0.0)].update(probability, target)
                del probability
            for s_value in PISS_GRID[1:]:
                s = x0.new_full((x0.shape[0],), s_value, dtype=torch.float32)
                x_s = linear_path(x0, x1, s)
                probability_ss = _original_probability(
                    endpoint, x_s, image_feat, s_value, s_value, sample, config
                )
                accumulators[("piss", s_value)].update(probability_ss, target)
                del probability_ss
                probability_s1 = _original_probability(
                    endpoint, x_s, image_feat, s_value, 1.0, sample, config
                )
                accumulators[("pis1", s_value)].update(probability_s1, target)
                del probability_s1, x_s
            processed += 1
    pi0t_rows = [
        _map_row("pi0t", 0.0, t, accumulators[("pi0t", t)].compute())
        for t in PI0T_GRID
    ]
    piss_rows = [
        _map_row("piss", s, s, accumulators[("piss", s)].compute())
        for s in PISS_GRID
    ]
    pis1_rows = [
        {
            **_map_row("pis1", s, 1.0, accumulators[("pis1", s)].compute()),
            "off_diagonal_gap_miou": (
                accumulators[("pis1", s)].compute()["miou"]
                - accumulators[("piss", s)].compute()["miou"]
            ),
        }
        for s in PISS_GRID
    ]
    pi00 = direct["pi00"].compute()
    pi01 = direct["pi01"].compute()
    if pi0t_rows[0]["miou"] != pi00["miou"] or pi0t_rows[-1]["miou"] != pi01["miou"]:
        raise AssertionError("pi0t endpoints do not match pi00/pi01")
    if piss_rows[0]["miou"] != pi00["miou"] or pis1_rows[0]["miou"] != pi01["miou"]:
        raise AssertionError("s=0 diagonal/off-diagonal endpoints do not match direct maps")
    direct_rows = [_map_row("pi00", 0.0, 0.0, pi00), _map_row("pi01", 0.0, 1.0, pi01)]
    map_rows = direct_rows + pi0t_rows + piss_rows + pis1_rows
    class_rows = [
        {
            "class_id": class_id,
            "class_name": name,
            "pi00_iou": pi00["per_class_iou"][class_id],
            "pi01_iou": pi01["per_class_iou"][class_id],
            "pi01_minus_pi00_iou": (
                pi01["per_class_iou"][class_id] - pi00["per_class_iou"][class_id]
            ),
            "gt_pixel_count": pi00["gt_pixel_count"][class_id],
        }
        for class_id, name in enumerate(CITYSCAPES_CLASSES)
    ]
    metadata = {
        "evaluated_images": processed,
        "pi01_minus_pi00_miou": pi01["miou"] - pi00["miou"],
        "pi01_minus_pi00_pixel_accuracy": (
            pi01["pixel_accuracy"] - pi00["pixel_accuracy"]
        ),
        "pi01_minus_pi00_macc": (
            pi01["mean_class_accuracy"] - pi00["mean_class_accuracy"]
        ),
        "pi00": pi00,
        "pi01": pi01,
        "gt_leakage_caution": (
            "pi_ss and pi_s1 for s>0 are GT-interpolated diagnostic probes; "
            "x_s contains ground-truth x1 information and is not pure inference."
        ),
    }
    if not math.isclose(
        pis1_rows[0]["off_diagonal_gap_miou"],
        metadata["pi01_minus_pi00_miou"],
        abs_tol=1.0e-12,
    ):
        raise AssertionError("s=0 off-diagonal gap does not equal pi01-pi00")
    return map_rows, class_rows, pi0t_rows, piss_rows, pis1_rows, metadata


def run_psd_time_map_analysis(
    config: dict[str, Any], *, checkpoint_path: str | Path,
    output_dir: str | Path | None, batch_size: int, time_bin_batches: int,
    eval_batch_size: int, eval_num_batches: int, psd_weight: float | None,
    teacher_confidence_threshold: float, seed: int, num_workers: int | None,
    device: torch.device, save_parameter_details: bool = False,
) -> Path:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not config["evaluation"]["original_resolution"]:
        raise ValueError(
            "Direct pi-map evaluation currently requires original_resolution=true"
        )
    output = (
        Path(output_dir).expanduser().resolve() if output_dir is not None
        else checkpoint_path.parent / f"psd_time_maps_{checkpoint_path.stem}"
    )
    output.mkdir(parents=True, exist_ok=True)
    weight = float(
        config["loss"]["consistency"]["weight"]
        if psd_weight is None else psd_weight
    )
    workers = config["dataset"]["num_workers"] if num_workers is None else num_workers
    seed_everything(seed)
    checkpoint, endpoint, source = _load_models(config, checkpoint_path, device)
    train_loader = DataLoader(
        build_dataset(config, config["dataset"]["train_split"], augment=True),
        batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=workers, pin_memory=config["dataset"]["pin_memory"],
    )
    phase_a = _phase_a(
        config, endpoint, source, loader=train_loader,
        time_bin_batches=time_bin_batches, weight=weight,
        confidence_threshold=teacher_confidence_threshold,
        seed=seed, device=device,
        save_parameter_details=save_parameter_details,
    )
    batch_rows, summary_rows, module_rows, quality_rows, parameter_rows, phase_a_meta = phase_a
    eval_dataset = build_dataset(
        config, config["evaluation"]["split"], augment=False
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=workers, pin_memory=config["dataset"]["pin_memory"],
        collate_fn=(
            ade20k_eval_collate
            if config["evaluation"]["original_resolution"] else None
        ),
    )
    phase_b = _phase_b(
        config, endpoint, source, loader=eval_loader,
        eval_num_batches=eval_num_batches,
        confidence_threshold=teacher_confidence_threshold, device=device,
    )
    map_rows, class_rows, pi0t_rows, piss_rows, pis1_rows, phase_b_meta = phase_b
    checkpoint_step = checkpoint.get("global_step")
    for rows in (
        batch_rows, summary_rows, module_rows, quality_rows, parameter_rows,
        map_rows, class_rows, pi0t_rows, piss_rows, pis1_rows,
    ):
        for row in rows:
            row["checkpoint_global_step"] = checkpoint_step
    _write_csv(output / "time_bin_gradient_batches.csv", batch_rows)
    _write_csv(output / "time_bin_gradient_summary.csv", summary_rows)
    _write_csv(output / "time_bin_modules.csv", module_rows)
    _write_csv(output / "time_bin_teacher_quality.csv", quality_rows)
    if save_parameter_details:
        _write_csv(output / "time_bin_parameters.csv", parameter_rows)
    _write_csv(output / "pi_map_summary.csv", map_rows)
    _write_csv(output / "pi_map_classes.csv", class_rows)
    _write_csv(output / "pi0t_curve.csv", pi0t_rows)
    _write_csv(output / "piss_curve.csv", piss_rows)
    _write_csv(output / "pis1_curve.csv", pis1_rows)
    arguments = {
        "config": config["runtime"].get("config_path"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "output_dir": str(output),
        "batch_size": batch_size,
        "time_bin_batches": time_bin_batches,
        "eval_batch_size": eval_batch_size,
        "eval_num_batches": eval_num_batches,
        "psd_weight": weight,
        "teacher_confidence_threshold": teacher_confidence_threshold,
        "seed": seed, "num_workers": workers, "device": str(device),
        "save_parameter_details": save_parameter_details,
    }
    (output / "diagnostic_args.json").write_text(
        json.dumps(arguments, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    early = next(row for row in summary_rows if row["bin_label"] == S_BINS[0].label)
    late = next(row for row in summary_rows if row["bin_label"] == S_BINS[-1].label)
    early_delta_module = next(
        row for row in module_rows
        if row["bin_label"] == S_BINS[0].label
        and row["module_group"] == "unet_embed_delta"
    )
    pi00, pi01 = phase_b_meta["pi00"], phase_b_meta["pi01"]
    off_diagonal_evidence = (
        early_delta_module["base_vs_psd_cosine_mean"] < 0
        and phase_b_meta["pi01_minus_pi00_miou"] < 0
    )
    interpretation = (
        "Evidence is consistent with an off-diagonal/time-conditioning optimization issue."
        if off_diagonal_evidence else
        "The selected heuristics do not jointly isolate an off-diagonal/time-conditioning issue."
    )
    early_time_interpretation = (
        "PSD problems may be concentrated near the inference-relevant early-time region."
        if (
            early["base_vs_psd_cosine"] < 0
            and late["base_vs_psd_cosine"] > 0
            and early["teacher_miou"] < late["teacher_miou"]
        ) else
        "The early-vs-late s-bin heuristic does not isolate an early-time problem."
    )
    summary = {
        **arguments,
        "phase_a_time_bins": summary_rows,
        "phase_a_metadata": phase_a_meta,
        "phase_b_direct_maps": phase_b_meta,
        "pi0t_curve": pi0t_rows,
        "piss_curve": piss_rows,
        "pis1_curve": pis1_rows,
        "joint_diagnostic": {
            "early_s_teacher_miou": early["teacher_miou"],
            "early_s_global_cosine": early["base_vs_psd_cosine"],
            "early_s_embed_delta_cosine": early_delta_module[
                "base_vs_psd_cosine_mean"
            ],
            "pi01_minus_pi00_miou": phase_b_meta["pi01_minus_pi00_miou"],
            "interpretation": interpretation,
            "early_time_interpretation": early_time_interpretation,
            "heuristic_only": True,
            "causality_not_established": True,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("===== Direct Map Evaluation =====")
    print(f"pi00: mIoU={pi00['miou']:.6g}")
    print(f"pi01: mIoU={pi01['miou']:.6g}")
    print(f"pi01-pi00: {phase_b_meta['pi01_minus_pi00_miou']:.6g}")
    print("\n===== pi0t Curve =====")
    for row in pi0t_rows:
        print(f"t={row['t']:.2f}: mIoU={row['miou']:.6g}")
    print("\n===== Diagonal vs Off-Diagonal =====")
    for diagonal, off_diagonal in zip(piss_rows, pis1_rows):
        print(f"s={diagonal['s']:.2f}: piss={diagonal['miou']:.6g}, "
              f"pis1={off_diagonal['miou']:.6g}, "
              f"gap={off_diagonal['off_diagonal_gap_miou']:.6g}")
    for axis, title in (("s", "PSD s-bin"), ("delta", "PSD delta-bin")):
        print(f"\n===== {title} =====")
        for row in (value for value in summary_rows if value["axis"] == axis):
            important = {
                group: next(
                    value for value in module_rows
                    if value["bin_label"] == row["bin_label"]
                    and value["module_group"] == group
                )["base_vs_psd_cosine_mean"]
                for group in ("unet_output", "unet_embed_s", "unet_embed_delta")
            }
            print(f"{row['bin_label']}: teacher_mIoU={row['teacher_miou']:.6g}, "
                  f"cosine={row['base_vs_psd_cosine']:.6g}, "
                  f"unet_output={important['unet_output']:.6g}, "
                  f"embed_s={important['unet_embed_s']:.6g}, "
                  f"embed_delta={important['unet_embed_delta']:.6g}")
    print("\n===== Thin / Rare Classes: pi00 vs pi01 =====")
    for class_id in THIN_RARE_CLASS_IDS:
        row = class_rows[class_id]
        print(f"{row['class_name']}: pi00={row['pi00_iou']:.6g}, "
              f"pi01={row['pi01_iou']:.6g}, delta={row['pi01_minus_pi00_iou']:.6g}")
    print("\n===== Interpretation =====")
    print(interpretation)
    print(early_time_interpretation)
    print(phase_a_meta["teacher_quality_caution"])
    print(phase_b_meta["gt_leakage_caution"])
    print("These are diagnostic heuristics; they do not establish causality.")
    print(f"Output: {output}")
    return output
