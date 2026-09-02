from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ade20k_eval_collate, build_dataset
from discrete_flow_maps import flow_map, make_time_grid
from inference import (
    state_to_original_continuous,
    state_to_prediction,
)
from metrics import SegmentationMetrics
from source_diagnostics import (
    _inverse_normalize,
    _json_safe,
    _load_models,
    _new_metrics,
    _save_heatmap,
    _save_mask,
    deterministic_epsilon_like,
    diagnostic_initial_state,
)
from utils import autocast_context, resolve_device, seed_everything


DEFAULT_SIGMA_VALUES = (1.0, 0.75, 0.5, 0.25, 0.1, 0.0)
DEFAULT_STEP_VALUES = (1, 2, 3, 5, 10)
QUANTILE_LEVELS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


@dataclass
class BoundedQuantiles:
    max_per_update: int = 4096
    max_total: int = 1_000_000
    chunks: list[torch.Tensor] = field(default_factory=list)

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().float().reshape(-1)
        if values.numel() == 0:
            return
        if values.numel() > self.max_per_update:
            # Float32 linspace can round the final value up to ``numel`` once
            # tensors exceed 2**24 elements, producing an out-of-bounds CUDA
            # index. Build the same evenly spaced sample with integer math.
            indices = (
                torch.arange(
                    self.max_per_update, device=values.device, dtype=torch.int64
                )
                * (values.numel() - 1)
                // (self.max_per_update - 1)
            )
            values = values[indices]
        self.chunks.append(values.cpu())

    def compute(self) -> dict[str, float | int]:
        if not self.chunks:
            return {
                **{_quantile_name(level): float("nan") for level in QUANTILE_LEVELS},
                "quantile_sample_count": 0,
            }
        values = torch.cat(self.chunks)
        if values.numel() > self.max_total:
            indices = (
                torch.arange(self.max_total, dtype=torch.int64)
                * (values.numel() - 1)
                // (self.max_total - 1)
            )
            values = values[indices]
        quantiles = torch.quantile(
            values.float(), torch.tensor(QUANTILE_LEVELS, dtype=torch.float32)
        )
        return {
            **{
                _quantile_name(level): float(value)
                for level, value in zip(QUANTILE_LEVELS, quantiles)
            },
            "quantile_sample_count": int(values.numel()),
        }


def _quantile_name(level: float) -> str:
    return "median" if level == 0.5 else f"p{round(level * 100)}"


@dataclass
class DistributionAccumulator:
    total: float = 0.0
    square_total: float = 0.0
    count: int = 0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    quantiles: BoundedQuantiles = field(default_factory=BoundedQuantiles)

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().float().reshape(-1)
        if values.numel() == 0:
            return
        self.total += float(values.sum().cpu())
        self.square_total += float(values.square().sum().cpu())
        self.count += values.numel()
        self.minimum = min(self.minimum, float(values.amin().cpu()))
        self.maximum = max(self.maximum, float(values.amax().cpu()))
        self.quantiles.update(values)

    def compute(self) -> dict[str, float | int]:
        mean = self.total / max(self.count, 1)
        variance = max(self.square_total / max(self.count, 1) - mean * mean, 0.0)
        return {
            "mean": mean,
            "std": math.sqrt(variance),
            **self.quantiles.compute(),
            "min": self.minimum if self.count else float("nan"),
            "max": self.maximum if self.count else float("nan"),
            "count": self.count,
            "aggregation_dtype": "float32",
        }


@dataclass
class MarginAccumulator:
    correct: DistributionAccumulator = field(default_factory=DistributionAccumulator)
    wrong_max: DistributionAccumulator = field(default_factory=DistributionAccumulator)
    margin: DistributionAccumulator = field(default_factory=DistributionAccumulator)
    positive_count: int = 0
    total_count: int = 0
    threshold_counts: dict[float, int] = field(
        default_factory=lambda: {value: 0 for value in (0, 0.5, 1, 2, 3, 4)}
    )

    def update(
        self, correct: torch.Tensor, wrong_max: torch.Tensor, margin: torch.Tensor
    ) -> None:
        self.correct.update(correct)
        self.wrong_max.update(wrong_max)
        self.margin.update(margin)
        margin = margin.detach().float()
        self.total_count += margin.numel()
        self.positive_count += int((margin > 0).sum().cpu())
        for threshold in self.threshold_counts:
            self.threshold_counts[threshold] += int((margin > threshold).sum().cpu())

    def compute(self) -> dict[str, Any]:
        return {
            "correct_score_stats": self.correct.compute(),
            "wrong_max_stats": self.wrong_max.compute(),
            "margin_stats": self.margin.compute(),
            "top1_accuracy": self.positive_count / max(self.total_count, 1),
            "margin_threshold_fractions": {
                f"margin_gt_{threshold:g}": count / max(self.total_count, 1)
                for threshold, count in self.threshold_counts.items()
            },
        }


@dataclass
class RetentionAccumulator:
    top1_count: int = 0
    total_count: int = 0
    retained_count: int = 0
    mu_correct_count: int = 0

    def update(self, mu_margin: torch.Tensor, x0_margin: torch.Tensor) -> None:
        mu_correct = mu_margin.detach().float() > 0
        x0_correct = x0_margin.detach().float() > 0
        self.total_count += x0_correct.numel()
        self.top1_count += int(x0_correct.sum().cpu())
        self.mu_correct_count += int(mu_correct.sum().cpu())
        self.retained_count += int((mu_correct & x0_correct).sum().cpu())

    def compute(self) -> dict[str, float | int]:
        return {
            "top1_accuracy": self.top1_count / max(self.total_count, 1),
            "top1_retention_given_mu_correct": (
                self.retained_count / max(self.mu_correct_count, 1)
            ),
            "total_valid_pixels": self.total_count,
            "mu_correct_pixels": self.mu_correct_count,
        }


def class_score_maps(
    scores: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return correct, maximum-wrong, and margin maps in float32."""
    if scores.ndim != 4 or target.shape != scores.shape[:1] + scores.shape[-2:]:
        raise ValueError("scores/target shapes must be [B,C,H,W] and [B,H,W]")
    classes = scores.shape[1]
    valid = (target != ignore_index) & (target >= 0) & (target < classes)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    values = scores.detach().float()
    correct = values.gather(1, safe_target[:, None]).squeeze(1)
    top_values, top_indices = values.topk(2, dim=1)
    wrong_max = torch.where(
        top_indices[:, 0] == safe_target, top_values[:, 1], top_values[:, 0]
    )
    return correct, wrong_max, correct - wrong_max, valid


def valid_class_scores(
    scores: torch.Tensor, target: torch.Tensor, *, ignore_index: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    correct, wrong_max, margin, valid = class_score_maps(
        scores, target, ignore_index=ignore_index
    )
    return correct[valid], wrong_max[valid], margin[valid]


@torch.no_grad()
def run_flow_with_image_feat(
    model,
    image_feat: torch.Tensor,
    initial_state: torch.Tensor,
    config: dict,
    num_steps: int,
) -> torch.Tensor:
    """Production-equivalent Flow Map with a caller-cached image feature."""
    state = initial_state
    for scalar_s, scalar_t in make_time_grid(num_steps, state.device):
        batch = state.shape[0]
        s = scalar_s.expand(batch)
        t = scalar_t.expand(batch)
        logits = model.forward_logits_with_image_feat(state, image_feat, s, t)
        probability = torch.softmax(logits.float(), dim=1).to(state.dtype)
        state = flow_map(
            state, probability, s, t,
            config["flow"]["time_eps"], config["flow"],
        )
    return state


@torch.no_grad()
def endpoint_probability(
    model,
    image_feat: torch.Tensor,
    state: torch.Tensor,
    *,
    s_value: float,
    t_value: float,
) -> torch.Tensor:
    batch = state.shape[0]
    s = torch.full((batch,), s_value, device=state.device)
    t = torch.full((batch,), t_value, device=state.device)
    logits = model.forward_logits_with_image_feat(state, image_feat, s, t)
    return torch.softmax(logits.float(), dim=1).to(state.dtype)


def original_continuous(
    state: torch.Tensor, sample: dict, config: dict
) -> torch.Tensor:
    return state_to_original_continuous(
        state,
        sample["model_shape"],
        sample["original_shape"],
        padded_shape=sample["padded_shape"],
        align_corners=config["evaluation"]["align_corners"],
    )


def original_prediction(
    state: torch.Tensor,
    sample: dict,
    config: dict,
    *,
    exclude_void: bool,
) -> torch.Tensor:
    return state_to_prediction(
        original_continuous(state, sample, config),
        void_class_index=config["dataset"]["void_class_index"],
        exclude_void=exclude_void,
    )


def state_resolution_oracle(
    target: torch.Tensor,
    sample: dict,
    *,
    state_size: tuple[int, int],
    num_classes: int,
) -> torch.Tensor:
    """Map original GT through the production model/padding/H/4 geometry."""
    model_target = F.interpolate(
        target[:, None].float(), size=sample["model_shape"], mode="nearest"
    )[:, 0].long()
    padded_height, padded_width = sample["padded_shape"]
    pad_height = padded_height - model_target.shape[-2]
    pad_width = padded_width - model_target.shape[-1]
    model_target = F.pad(model_target, (0, pad_width, 0, pad_height), value=0)
    state_target = F.interpolate(
        model_target[:, None].float(), size=state_size, mode="nearest"
    )[:, 0].long()
    return F.one_hot(state_target, num_classes).permute(0, 3, 1, 2).float()


def _full_state_metrics(config: dict, device: torch.device) -> SegmentationMetrics:
    eval_range = config["evaluation"]["eval_class_indices"]
    return SegmentationMetrics(
        config["dataset"]["num_classes"],
        config["dataset"]["void_class_index"],
        device=device,
        evaluated_class_indices=range(eval_range[0], eval_range[1] + 1),
        nanmean=config["evaluation"]["nanmean"],
        prediction_void_retained=True,
    )


def _save_rgb(array: np.ndarray, path: Path) -> None:
    Image.fromarray(np.round(array * 255).astype(np.uint8)).save(path)


def _sigma_key(sigma: float) -> str:
    return f"sigma_{float(sigma):.2f}"


def _sigma_tag(sigma: float) -> str:
    return f"{float(sigma):g}".replace(".", "p")


def _write_long_csv(result: dict[str, Any], path: Path) -> None:
    rows: list[tuple[str, str, str, Any]] = []

    def visit(value: Any, keys: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, (*keys, str(key)))
        elif isinstance(value, (int, float, str, bool)) or value is None:
            section = keys[0] if keys else "result"
            variant = keys[1] if len(keys) > 1 else "all"
            metric = ".".join(keys[2:]) if len(keys) > 2 else "value"
            rows.append((section, variant, metric, value))

    for key in (
        "source_mu", "x0", "dfm_sigma_sweep", "mu_zero", "pi00", "pi01",
        "pi00_minus_pi01_mIoU", "pi01_production_equivalence", "step_sweep",
        "state_resolution_oracle", "full_resolution_sanity",
    ):
        visit(result[key], (key,))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("section", "variant", "metric", "value"))
        writer.writerows(rows)


def _summary_text(result: dict[str, Any]) -> str:
    source = result["source_mu"]
    margin = source["margin_stats"]
    fractions = source["margin_threshold_fractions"]
    lines = [
        "ADE20K DFM failure analysis",
        f"checkpoint: {result['checkpoint']}",
        f"global_step: {result['global_step']}",
        f"samples: {result['samples_evaluated']}",
        "",
        "[Source]",
        f"mu-only semantic mIoU: {source['mu_only_semantic_only']['mIoU']:.6f}",
        f"mu-only full-state mIoU: {source['mu_only_full_state']['mIoU']:.6f}",
        f"mu top1 accuracy: {source['top1_accuracy']:.6f}",
        f"mu margin mean: {margin['mean']:.6f}",
        f"mu margin median: {margin['median']:.6f}",
        f"fraction margin>1: {fractions['margin_gt_1']:.6f}",
        f"fraction margin>2: {fractions['margin_gt_2']:.6f}",
        "",
        "[Noise]",
        "sigma,x0_top1,retention,x0_mIoU,final_mIoU",
    ]
    for sigma, x0 in result["x0"].items():
        final = result["dfm_sigma_sweep"][sigma]
        lines.append(
            f"{sigma},{x0['top1_accuracy']:.6f},"
            f"{x0['top1_retention_given_mu_correct']:.6f},"
            f"{x0['metrics']['mIoU']:.6f},{final['mIoU']:.6f}"
        )
    lines.extend([
        "",
        "[Flow Map]",
        f"pi00 mIoU: {result['pi00']['mIoU']:.6f}",
        f"pi01 mIoU: {result['pi01']['mIoU']:.6f}",
    ])
    for step, metrics in result["step_sweep"].items():
        lines.append(f"production {step}-step mIoU: {metrics['mIoU']:.6f}")
    lines.extend([
        "",
        "[Resolution]",
        f"full-resolution sanity mIoU: {result['full_resolution_sanity']['mIoU']:.6f}",
        f"H/4 semantic oracle mIoU: "
        f"{result['state_resolution_oracle']['semantic_only']['mIoU']:.6f}",
        "",
        "[Conditional prior]",
        f"mu + epsilon mIoU: {result['dfm_sigma_sweep']['sigma_1.00']['mIoU']:.6f}",
        f"epsilon only mIoU: {result['mu_zero']['mIoU']:.6f}",
        f"delta mIoU: {result['conditional_minus_mu_zero_mIoU']:.6f}",
    ])
    return "\n".join(lines) + "\n"


@torch.no_grad()
def run_failure_analysis(
    config: dict,
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    sigma_values: Iterable[float] = DEFAULT_SIGMA_VALUES,
    step_values: Iterable[int] = DEFAULT_STEP_VALUES,
    num_visualize: int = 20,
    seed: int = 42,
    max_batches: int | None = None,
    expected_global_step: int | None = 160000,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    if config["dataset"]["name"] != "ade20k":
        raise ValueError("Failure analysis is defined for ADE20K")
    sigma_values = tuple(dict.fromkeys(float(value) for value in sigma_values))
    step_values = tuple(dict.fromkeys(int(value) for value in step_values))
    if 1.0 not in sigma_values or 0.0 not in sigma_values:
        raise ValueError("sigma_values must include 1.0 and 0.0")
    if 1 not in step_values:
        raise ValueError("step_values must include 1")
    if any(value < 0 for value in sigma_values):
        raise ValueError("sigma values must be non-negative")
    if any(value <= 0 for value in step_values):
        raise ValueError("step values must be positive")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = (
        resolve_device(config["runtime"]["device"])
        if device is None else torch.device(device)
    )
    seed_everything(seed, deterministic=True)
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint, model, source_model = _load_models(config, checkpoint_path, device)
    global_step = int(checkpoint.get("global_step", -1))
    if expected_global_step is not None and global_step != expected_global_step:
        raise ValueError(
            f"Expected checkpoint global_step={expected_global_step}, got {global_step}"
        )

    dataset = build_dataset(config, config["evaluation"]["split"], augment=False)
    loader = DataLoader(
        dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["dataset"]["pin_memory"],
        collate_fn=ade20k_eval_collate,
    )
    classes = config["dataset"]["num_classes"]
    ignore_index = config["evaluation"]["ignore_index"]

    mu_full_metrics = _full_state_metrics(config, device)
    mu_semantic_metrics = _new_metrics(config, device)
    x0_metrics = {sigma: _new_metrics(config, device) for sigma in sigma_values}
    dfm_metrics = {sigma: _new_metrics(config, device) for sigma in sigma_values}
    step_metrics = {step: _new_metrics(config, device) for step in step_values}
    pi00_metrics = _new_metrics(config, device)
    pi01_metrics = _new_metrics(config, device)
    mu_zero_metrics = _new_metrics(config, device)
    oracle_full_metrics = _full_state_metrics(config, device)
    oracle_semantic_metrics = _new_metrics(config, device)
    sanity_metrics = _new_metrics(config, device)

    mu_margins = MarginAccumulator()
    mu_norm = DistributionAccumulator()
    x0_margins = {sigma: MarginAccumulator() for sigma in sigma_values}
    retentions = {sigma: RetentionAccumulator() for sigma in sigma_values}
    pi01_match_pixels = 0
    pi01_total_pixels = 0
    pi01_terminal_max_abs = 0.0
    sample_index = 0
    visualized = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        for sample in batch:
            image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
            target = sample["target"].unsqueeze(0).to(device, non_blocking=True)
            with autocast_context(config, device):
                _, mu, logvar = source_model(image)
                image_feat = model.encode_image(image)
            del logvar
            epsilon = deterministic_epsilon_like(mu, seed + sample_index)
            mu_full = original_continuous(mu, sample, config)
            mu_correct_map, mu_wrong_map, mu_margin_map, valid = class_score_maps(
                mu_full, target, ignore_index=ignore_index
            )
            mu_margins.update(
                mu_correct_map[valid], mu_wrong_map[valid], mu_margin_map[valid]
            )
            mu_norm.update(torch.linalg.vector_norm(mu_full.float(), dim=1)[valid])

            mu_full_prediction = state_to_prediction(mu_full, exclude_void=False)
            mu_semantic_prediction = state_to_prediction(
                mu_full,
                void_class_index=0,
                exclude_void=True,
            )
            mu_full_metrics.update(mu_full_prediction, target)
            mu_semantic_metrics.update(mu_semantic_prediction, target)
            sanity_metrics.update(target, target)

            oracle_state = state_resolution_oracle(
                target,
                sample,
                state_size=mu.shape[-2:],
                num_classes=classes,
            ).to(device)
            oracle_full = original_continuous(oracle_state, sample, config)
            oracle_full_prediction = state_to_prediction(oracle_full, exclude_void=False)
            oracle_semantic_prediction = state_to_prediction(
                oracle_full, void_class_index=0, exclude_void=True
            )
            oracle_full_metrics.update(oracle_full_prediction, target)
            oracle_semantic_metrics.update(oracle_semantic_prediction, target)
            del oracle_state, oracle_full

            sigma_visuals: dict[float, dict[str, torch.Tensor]] = {}
            sigma_terminals: dict[float, torch.Tensor] = {}
            for sigma in sigma_values:
                x0 = diagnostic_initial_state(mu, epsilon, sigma)
                x0_full = original_continuous(x0, sample, config)
                x0_correct, x0_wrong, x0_margin, x0_valid = class_score_maps(
                    x0_full, target, ignore_index=ignore_index
                )
                if not torch.equal(valid, x0_valid):
                    raise RuntimeError("valid pixel mask changed across diagnostics")
                x0_margins[sigma].update(
                    x0_correct[valid], x0_wrong[valid], x0_margin[valid]
                )
                retentions[sigma].update(mu_margin_map[valid], x0_margin[valid])
                x0_prediction = state_to_prediction(
                    x0_full, void_class_index=0, exclude_void=True
                )
                x0_metrics[sigma].update(x0_prediction, target)
                with autocast_context(config, device):
                    terminal = run_flow_with_image_feat(
                        model, image_feat, x0, config, num_steps=1
                    )
                if not torch.isfinite(terminal).all():
                    raise FloatingPointError(f"non-finite terminal state at sigma={sigma}")
                final_prediction = original_prediction(
                    terminal, sample, config, exclude_void=True
                )
                dfm_metrics[sigma].update(final_prediction, target)
                sigma_terminals[sigma] = terminal
                if visualized < num_visualize:
                    sigma_visuals[sigma] = {
                        "x0_prediction": x0_prediction[0].cpu(),
                        "x0_margin": x0_margin[0].cpu(),
                        "final_prediction": final_prediction[0].cpu(),
                    }
                del x0_full

            x0_sigma1 = diagnostic_initial_state(mu, epsilon, 1.0)
            with autocast_context(config, device):
                pi00 = endpoint_probability(
                    model, image_feat, x0_sigma1, s_value=0.0, t_value=0.0
                )
                pi01 = endpoint_probability(
                    model, image_feat, x0_sigma1, s_value=0.0, t_value=1.0
                )
            production_one_step = sigma_terminals[1.0]
            terminal_error = float(
                (pi01.float() - production_one_step.float()).abs().amax().cpu()
            )
            pi01_terminal_max_abs = max(pi01_terminal_max_abs, terminal_error)
            pi00_prediction = original_prediction(
                pi00, sample, config, exclude_void=True
            )
            pi01_prediction = original_prediction(
                pi01, sample, config, exclude_void=True
            )
            production_one_prediction = original_prediction(
                production_one_step, sample, config, exclude_void=True
            )
            pi00_metrics.update(pi00_prediction, target)
            pi01_metrics.update(pi01_prediction, target)
            matching = int(
                (pi01_prediction == production_one_prediction).sum().cpu()
            )
            sample_agreement = matching / pi01_prediction.numel()
            # In bf16, x + (pi - x) can differ from pi by one or two ULPs.
            # Require numerical equivalence and near-perfect final labels while
            # recording the exact discrepancy instead of hiding it.
            if terminal_error > 0.02 or sample_agreement < 0.995:
                raise AssertionError(
                    "pi01 and production 1-step predictions differ: "
                    f"{matching}/{pi01_prediction.numel()} pixels; "
                    f"terminal max_abs={terminal_error}"
                )
            pi01_match_pixels += matching
            pi01_total_pixels += pi01_prediction.numel()

            for step in step_values:
                if step == 1:
                    terminal = production_one_step
                    prediction = production_one_prediction
                else:
                    with autocast_context(config, device):
                        terminal = run_flow_with_image_feat(
                            model, image_feat, x0_sigma1, config, num_steps=step
                        )
                    prediction = original_prediction(
                        terminal, sample, config, exclude_void=True
                    )
                step_metrics[step].update(prediction, target)

            with autocast_context(config, device):
                mu_zero_terminal = run_flow_with_image_feat(
                    model, image_feat, epsilon, config, num_steps=1
                )
            mu_zero_prediction = original_prediction(
                mu_zero_terminal, sample, config, exclude_void=True
            )
            mu_zero_metrics.update(mu_zero_prediction, target)

            if visualized < num_visualize:
                directory = output / "visualizations" / f"image_{visualized:03d}"
                directory.mkdir(parents=True, exist_ok=True)
                _save_rgb(_inverse_normalize(image[0], config), directory / "input.png")
                _save_mask(target[0].cpu(), directory / "ground_truth.png")
                _save_mask(
                    mu_semantic_prediction[0].cpu(),
                    directory / "mu_semantic_argmax.png",
                )
                _save_mask(
                    mu_full_prediction[0].cpu(), directory / "mu_full_argmax.png"
                )
                _save_heatmap(
                    mu_margin_map[0].cpu(), directory / "mu_margin.png", "mu margin",
                    valid=valid[0].cpu(), cmap="coolwarm",
                )
                _save_heatmap(
                    mu_correct_map[0].cpu(), directory / "mu_correct.png", "mu correct",
                    valid=valid[0].cpu(), cmap="viridis",
                )
                _save_heatmap(
                    mu_wrong_map[0].cpu(), directory / "mu_wrong_max.png", "mu wrong max",
                    valid=valid[0].cpu(), cmap="viridis",
                )
                for sigma, visuals in sigma_visuals.items():
                    tag = _sigma_tag(sigma)
                    _save_mask(
                        visuals["x0_prediction"], directory / f"x0_sigma_{tag}.png"
                    )
                    _save_heatmap(
                        visuals["x0_margin"],
                        directory / f"x0_margin_sigma_{tag}.png",
                        f"x0 margin sigma={sigma:g}", valid=valid[0].cpu(),
                        cmap="coolwarm",
                    )
                    _save_mask(
                        visuals["final_prediction"],
                        directory / f"final_sigma_{tag}.png",
                    )
                _save_mask(pi00_prediction[0].cpu(), directory / "pi00.png")
                _save_mask(pi01_prediction[0].cpu(), directory / "pi01.png")
                _save_mask(
                    production_one_prediction[0].cpu(),
                    directory / "production_1step.png",
                )
                with (directory / "sample.json").open("w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "sample_id": sample["sample_id"],
                            "dataset_index": sample_index,
                            "epsilon_seed": seed + sample_index,
                        },
                        handle,
                        indent=2,
                    )
                visualized += 1

            sample_index += 1
            if sample_index % 25 == 0:
                print(f"failure analysis: {sample_index}/{len(dataset)} samples")

    mu_result = mu_margins.compute()
    mu_result.update({
        "mu_only_full_state": mu_full_metrics.compute(),
        "mu_only_semantic_only": mu_semantic_metrics.compute(),
        "norm_stats": mu_norm.compute(),
        "correct_over_sigma_1_stats": mu_margins.correct.compute(),
        "margin_over_sigma_1_stats": mu_margins.margin.compute(),
    })
    x0_result = {}
    for sigma in sigma_values:
        margin_result = x0_margins[sigma].compute()
        retention_result = retentions[sigma].compute()
        x0_result[_sigma_key(sigma)] = {
            "margin_stats": margin_result["margin_stats"],
            "correct_score_stats": margin_result["correct_score_stats"],
            "wrong_max_stats": margin_result["wrong_max_stats"],
            **retention_result,
            "metrics": x0_metrics[sigma].compute(),
        }

    dfm_result = {
        _sigma_key(sigma): dfm_metrics[sigma].compute() for sigma in sigma_values
    }
    mu_zero_result = mu_zero_metrics.compute()
    pi00_result = pi00_metrics.compute()
    pi01_result = pi01_metrics.compute()
    result = {
        "checkpoint": str(checkpoint_path),
        "global_step": global_step,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "samples_evaluated": sample_index,
        "batches_evaluated": math.ceil(sample_index / config["evaluation"]["batch_size"]),
        "evaluation_protocol": {
            "dataset": "ade20k",
            "num_classes": classes,
            "valid_gt_classes": [1, 150],
            "void_gt_excluded": 0,
            "semantic_only_final_prediction": True,
            "continuous_original_resolution_resize": "bilinear",
            "align_corners": config["evaluation"]["align_corners"],
            "amp": config["runtime"]["amp"],
            "amp_dtype": config["runtime"]["amp_dtype"],
            "seed": seed,
            "epsilon_rule": "42 + zero-based dataset index",
            "quantiles": "bounded deterministic sampling, float32",
            "cached_image_feature": True,
        },
        "source_mu": mu_result,
        "x0": x0_result,
        "dfm_sigma_sweep": dfm_result,
        "mu_zero": mu_zero_result,
        "conditional_minus_mu_zero_mIoU": (
            dfm_result["sigma_1.00"]["mIoU"] - mu_zero_result["mIoU"]
        ),
        "pi00": pi00_result,
        "pi01": pi01_result,
        "pi00_minus_pi01_mIoU": pi00_result["mIoU"] - pi01_result["mIoU"],
        "pi01_production_equivalence": {
            "prediction_pixel_agreement": pi01_match_pixels / max(pi01_total_pixels, 1),
            "terminal_max_abs_difference": pi01_terminal_max_abs,
            "pixels_compared": pi01_total_pixels,
        },
        "step_sweep": {
            str(step): step_metrics[step].compute() for step in step_values
        },
        "state_resolution_oracle": {
            "full_state": oracle_full_metrics.compute(),
            "semantic_only": oracle_semantic_metrics.compute(),
        },
        "full_resolution_sanity": sanity_metrics.compute(),
        "checkpoint_96000": "not_found",
    }
    with (output / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(result), handle, indent=2, ensure_ascii=False, allow_nan=False)
    _write_long_csv(result, output / "diagnostics.csv")
    (output / "summary.txt").write_text(_summary_text(result), encoding="utf-8")
    return result
