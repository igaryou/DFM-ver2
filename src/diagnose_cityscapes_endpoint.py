from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adaptive_path import source_entropy_difficulty
from checkpoint import _without_module_prefix
from config import load_config
from dataset import ade20k_eval_collate, build_dataset
from discrete_flow_maps import linear_path
from inference import state_to_original_continuous, terminal_state_to_original_prediction
from metrics import SegmentationMetrics
from model_factory import build_models
from source_diagnostics import _checkpoint_source_state_for_model
from state_space import prepare_state_targets, state_spatial_size, target_state_from_config
from utils import autocast_context, resolve_device, seed_everything
from visualization import colorize


DEFAULT_T_VALUES = (0.0, 0.1, 0.25, 0.5, 0.75)
CITYSCAPES_CLASSES = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
)
ARCHITECTURE_FIELDS = (
    ("source", "segformer_variant"), ("source", "segformer_decoder"),
    ("model", "endpoint", "type"),
    ("model", "endpoint", "segformer_variant"),
    ("model", "state_downsample_factor"),
)


def _nested(config: dict, path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def validate_checkpoint_architecture(config: dict, checkpoint: dict) -> None:
    """Fail before loading when saved and requested architectures disagree."""
    saved = checkpoint.get("config")
    if not isinstance(saved, dict):
        raise RuntimeError("Checkpoint has no saved config; architecture cannot be verified")
    mismatches = []
    for path in ARCHITECTURE_FIELDS:
        current, previous = _nested(config, path), _nested(saved, path)
        if current != previous:
            mismatches.append(f"{'.'.join(path)}: checkpoint={previous!r}, config={current!r}")
    if mismatches:
        raise RuntimeError("Checkpoint/config architecture mismatch:\n" + "\n".join(mismatches))


def deterministic_epsilon_like(reference: torch.Tensor, seed: int, sample_id: str) -> torch.Tensor:
    digest = hashlib.sha256(f"{int(seed)}:{sample_id}".encode()).digest()
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
    return torch.randn(reference.shape, dtype=reference.dtype,
                       device=reference.device, generator=generator)


def zero_image_feature(image_feature: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(image_feature)


def fixed_t_state(x0: torch.Tensor, x1: torch.Tensor, t: float, config: dict,
                  difficulty: torch.Tensor | None = None) -> torch.Tensor:
    time = x0.new_full((x0.shape[0],), float(t), dtype=torch.float32)
    return linear_path(x0, x1, time, config, difficulty)


def model_target_state(sample: dict, image: torch.Tensor, target: torch.Tensor,
                       config: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Map original GT through production validation geometry into state space."""
    target_model = F.interpolate(
        target[:, None].float(), size=tuple(sample["model_shape"]), mode="nearest"
    )[:, 0].long()
    pad_h = image.shape[-2] - target_model.shape[-2]
    pad_w = image.shape[-1] - target_model.shape[-1]
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Model target is larger than the padded validation image")
    ignore = int(config["dataset"]["void_class_index"])
    target_model = F.pad(target_model, (0, pad_w, 0, pad_h), value=ignore)
    targets = prepare_state_targets(
        target_model, num_classes=int(config["dataset"]["num_classes"]),
        state_size=state_spatial_size(image, int(config["model"]["state_downsample_factor"])),
        ignore_index=ignore, mask_pixel_losses=True,
    )
    return target_state_from_config(targets.one_hot_state, config), targets.valid_mask_state


def _metrics(config: dict, device: torch.device) -> SegmentationMetrics:
    indices = config["evaluation"]["eval_class_indices"]
    evaluated = range(indices[0], indices[1] + 1) if indices is not None else None
    return SegmentationMetrics(
        config["dataset"]["num_classes"], config["dataset"]["void_class_index"],
        device=device, evaluated_class_indices=evaluated,
        nanmean=config["evaluation"]["nanmean"],
        prediction_void_retained=not config["evaluation"]["exclude_void_from_prediction"],
    )


def _prediction(state: torch.Tensor, sample: dict, config: dict) -> torch.Tensor:
    return terminal_state_to_original_prediction(
        state, sample["model_shape"], sample["original_shape"],
        padded_shape=sample["padded_shape"],
        align_corners=config["evaluation"]["align_corners"],
        void_class_index=config["dataset"]["void_class_index"],
        exclude_void=config["evaluation"]["exclude_void_from_prediction"],
    )


class SignalNoiseStatistics:
    def __init__(self) -> None:
        self.mu_abs = self.mu_sq = self.noise_abs = self.noise_sq = 0.0
        self.count = 0
        self.mu_min, self.mu_max = float("inf"), float("-inf")
        self.sigma_sum = self.sigma_count = 0.0
        self.sigma_min, self.sigma_max = float("inf"), float("-inf")
        self.x0_abs = 0.0
        self.pixel_snr: list[torch.Tensor] = []

    def update(self, mu: torch.Tensor, sigma: torch.Tensor, noise: torch.Tensor) -> None:
        mu, sigma, noise = mu.float(), sigma.float(), noise.float()
        self.mu_abs += float(mu.abs().sum()); self.mu_sq += float(mu.square().sum())
        self.noise_abs += float(noise.abs().sum()); self.noise_sq += float(noise.square().sum())
        self.x0_abs += float((mu + noise).abs().sum()); self.count += mu.numel()
        self.mu_min = min(self.mu_min, float(mu.amin())); self.mu_max = max(self.mu_max, float(mu.amax()))
        self.sigma_sum += float(sigma.sum()); self.sigma_count += sigma.numel()
        self.sigma_min = min(self.sigma_min, float(sigma.amin())); self.sigma_max = max(self.sigma_max, float(sigma.amax()))
        snr = torch.linalg.vector_norm(mu, dim=1) / (
            torch.linalg.vector_norm(noise, dim=1) + 1e-12)
        # Deterministic bounded collection prevents validation-scale host-memory growth.
        flat = snr.flatten()
        if flat.numel() > 8192:
            flat = flat[torch.linspace(0, flat.numel() - 1, 8192, device=flat.device).long()]
        self.pixel_snr.append(flat.cpu())

    def compute(self) -> dict[str, float]:
        values = torch.cat(self.pixel_snr) if self.pixel_snr else torch.tensor([float("nan")])
        q = torch.quantile(values, torch.tensor([.1, .25, .5, .75, .9]))
        mu_rms = math.sqrt(self.mu_sq / max(self.count, 1))
        noise_rms = math.sqrt(self.noise_sq / max(self.count, 1))
        return {"mu_abs_mean": self.mu_abs / max(self.count, 1), "mu_rms": mu_rms,
                "noise_abs_mean": self.noise_abs / max(self.count, 1), "noise_rms": noise_rms,
                "global_snr": mu_rms / max(noise_rms, 1e-12),
                "pixel_snr_mean": float(values.mean()), "pixel_snr_median": float(q[2]),
                "pixel_snr_p10": float(q[0]), "pixel_snr_p25": float(q[1]),
                "pixel_snr_p75": float(q[3]), "pixel_snr_p90": float(q[4]),
                "mu_min": self.mu_min, "mu_max": self.mu_max,
                "sigma_mean": self.sigma_sum / max(self.sigma_count, 1),
                "sigma_min": self.sigma_min, "sigma_max": self.sigma_max,
                "x0_abs_mean": self.x0_abs / max(self.count, 1)}


def _load_models(config: dict, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_architecture(config, checkpoint)
    build_config = copy.deepcopy(config)
    build_config["source"].update({"pretrained": False, "_load_pretrained": False, "checkpoint": None})
    build_config["model"]["image_encoder"]["pretrained"] = False
    model, source = build_models(build_config, device)
    # Diagnostics deliberately require exact keys and shapes, regardless of a permissive YAML.
    model.load_state_dict(_without_module_prefix(checkpoint["model"]), strict=True)
    if source is None or checkpoint.get("source_model") is None:
        raise RuntimeError("Checkpoint must contain both model and source_model")
    source.load_state_dict(_checkpoint_source_state_for_model(
        _without_module_prefix(checkpoint["source_model"]), source), strict=True)
    model.eval(); source.eval()
    return checkpoint, model, source


def _saved_and_requested_config(config: dict, checkpoint: dict) -> dict:
    saved = checkpoint["config"]
    paths = ARCHITECTURE_FIELDS + (("flow", "path", "type"),
        ("flow", "path", "exponent"), ("source", "fixed_std"))
    result = {}
    for path in paths:
        key = ".".join(path)
        result[key] = {"checkpoint": _nested(saved, path), "config": _nested(config, path)}
        print(f"{key}: checkpoint={result[key]['checkpoint']!r}, config={result[key]['config']!r}")
    return result


def _visualize(directory: Path, image: torch.Tensor, target: torch.Tensor,
               predictions: dict[str, torch.Tensor], mu: torch.Tensor,
               noise: torch.Tensor, sample: dict, config: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    normalize = config["augmentation"].get("normalize", {})
    shown = image[0].detach().float().cpu()
    if normalize.get("enabled", False):
        shown = shown * shown.new_tensor(normalize["std"])[:, None, None] + shown.new_tensor(normalize["mean"])[:, None, None]
    shown = shown[:, :sample["model_shape"][0], :sample["model_shape"][1]]
    shown = F.interpolate(shown[None], sample["original_shape"], mode="bilinear", align_corners=False)[0]
    Image.fromarray((shown.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(directory / "input.png")
    Image.fromarray(colorize(target[0].cpu(), "cityscapes")).save(directory / "ground_truth.png")
    for name, prediction in predictions.items():
        Image.fromarray(colorize(prediction[0].cpu(), "cityscapes")).save(directory / f"{name}.png")
    for name, field in (("mu_norm", torch.linalg.vector_norm(mu.float(), dim=1)),
                        ("noise_norm", torch.linalg.vector_norm(noise.float(), dim=1)),
                        ("pixel_snr", torch.linalg.vector_norm(mu.float(), dim=1) /
                         (torch.linalg.vector_norm(noise.float(), dim=1) + 1e-12))):
        values = state_to_original_continuous(field[:, None], sample["model_shape"],
            sample["original_shape"], padded_shape=sample["padded_shape"],
            align_corners=config["evaluation"]["align_corners"])[0, 0].cpu()
        fig, ax = plt.subplots(figsize=(10, 5)); plot = ax.imshow(values, cmap="magma")
        ax.set_title(name); ax.axis("off"); fig.colorbar(plot, ax=ax); fig.tight_layout()
        fig.savefig(directory / f"{name}.png", dpi=120); plt.close(fig)


def _metric_result(metric: SegmentationMetrics) -> dict:
    return metric.compute()


@torch.no_grad()
def run_diagnostics(config: dict, *, checkpoint_path: str | Path, output_dir: str | Path,
                    t_values: Iterable[float] = DEFAULT_T_VALUES,
                    num_visualizations: int = 16, seed: int = 42,
                    max_batches: int | None = None, device: str | torch.device | None = None,
                    dataset=None, models=None) -> dict[str, Any]:
    if config["dataset"]["name"] != "cityscapes":
        raise ValueError("This diagnostic requires Cityscapes")
    times = tuple(dict.fromkeys(float(t) for t in t_values))
    if not times or any(t < 0 or t > 1 for t in times):
        raise ValueError("t-values must be within [0, 1]")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config["runtime"]["device"]) if device is None else torch.device(device)
    seed_everything(seed, deterministic=True)
    if models is None:
        checkpoint, model, source = _load_models(config, Path(checkpoint_path).resolve(), device)
    else:
        checkpoint, model, source = models
        validate_checkpoint_architecture(config, checkpoint)
        model.eval(); source.eval()
    config_comparison = _saved_and_requested_config(config, checkpoint)
    dataset = build_dataset(config, config["evaluation"]["split"], augment=False) if dataset is None else dataset
    loader = DataLoader(dataset, batch_size=config["evaluation"]["batch_size"], shuffle=False,
        num_workers=0 if models is not None else config["dataset"]["num_workers"],
        pin_memory=False if models is not None else config["dataset"]["pin_memory"],
        collate_fn=ade20k_eval_collate if config["evaluation"]["original_resolution"] else None)
    source_metric = _metrics(config, device)
    xt_metrics = {t: _metrics(config, device) for t in times}
    normal_metrics = {t: _metrics(config, device) for t in times}
    zero_metrics = {t: _metrics(config, device) for t in times}
    statistics = SignalNoiseStatistics(); sample_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches: break
        samples = batch if isinstance(batch, list) else [{"image": batch[0][i], "target": batch[1][i],
            "model_shape": tuple(batch[1][i].shape), "original_shape": tuple(batch[1][i].shape),
            "padded_shape": tuple(batch[1][i].shape), "sample_id": str(sample_count + i)} for i in range(len(batch[0]))]
        for sample in samples:
            image = sample["image"].unsqueeze(0).to(device); target = sample["target"].unsqueeze(0).to(device)
            with autocast_context(config, device):
                mu, logvar = source.forward_statistics(image)
                image_feature = model.encode_image(image)
            epsilon = deterministic_epsilon_like(mu, seed, str(sample["sample_id"]))
            sigma = torch.exp(0.5 * logvar); noise = sigma * epsilon; x0 = mu + noise
            statistics.update(mu, sigma, noise)
            source_metric.update(_prediction(mu, sample, config), target)
            x1, valid_state = model_target_state(sample, image, target, config)
            difficulty = None
            if config["flow"]["path"].get("type") == "entropy_adaptive":
                _, difficulty = source_entropy_difficulty(mu, config,
                    valid_mask=valid_state, spatial_size=x0.shape[-2:])
            visual_predictions = {"mu_argmax": _prediction(mu, sample, config),
                                  "x0_argmax": _prediction(x0, sample, config)}
            for t in times:
                # Diagnostic-only: x_t contains validation GT and is not normal inference.
                xt = fixed_t_state(x0, x1, t, config, difficulty)
                time = xt.new_full((xt.shape[0],), t, dtype=torch.float32)
                with autocast_context(config, device):
                    normal = model.forward_logits_with_image_feat(xt, image_feature, time, time)
                    zero = model.forward_logits_with_image_feat(xt, zero_image_feature(image_feature), time, time)
                if normal.shape != xt.shape or zero.shape != xt.shape:
                    raise RuntimeError(f"Endpoint output shape mismatch at t={t}: state={xt.shape}, normal={normal.shape}, zero={zero.shape}")
                xt_pred, normal_pred, zero_pred = (_prediction(v, sample, config) for v in (xt, normal, zero))
                xt_metrics[t].update(xt_pred, target); normal_metrics[t].update(normal_pred, target); zero_metrics[t].update(zero_pred, target)
                visual_predictions[f"t_{t:g}_normal"] = normal_pred
                visual_predictions[f"t_{t:g}_zero_image"] = zero_pred
            if sample_count < num_visualizations:
                _visualize(output / "visualizations" / f"sample_{sample_count:03d}", image,
                           target, visual_predictions, mu, noise, sample, config)
            sample_count += 1
    source_result = _metric_result(source_metric)
    fixed = {}
    for t in times:
        xt, normal, zero = map(_metric_result, (xt_metrics[t], normal_metrics[t], zero_metrics[t]))
        fixed[str(t)] = {"t": t, "x_t_argmax": xt, "endpoint_normal": normal,
                         "endpoint_zero_image": zero,
                         "normal_zero_delta_mIoU": normal["mIoU"] - zero["mIoU"],
                         "endpoint_x_t_delta_mIoU": normal["mIoU"] - xt["mIoU"]}
    result = {"checkpoint": str(checkpoint_path), "checkpoint_global_step": checkpoint.get("global_step"),
              "samples_evaluated": sample_count, "seed": seed,
              "diagnostic_warning": "Fixed-t states contain validation GT; these results are diagnostic-only, not normal inference.",
              "config_comparison": config_comparison, "source": source_result,
              "signal_noise": statistics.compute(), "fixed_t": fixed}
    _write_outputs(result, output)
    return result


def _write_outputs(result: dict, output: Path) -> None:
    with (output / "diagnostics.json").open("w", encoding="utf-8") as f: json.dump(result, f, indent=2, allow_nan=False)
    with (output / "diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("t", "x_t_argmax_mIoU", "endpoint_normal_mIoU", "endpoint_zero_image_mIoU", "normal_zero_delta_mIoU", "endpoint_x_t_delta_mIoU")); writer.writeheader()
        for row in result["fixed_t"].values(): writer.writerow({"t": row["t"], "x_t_argmax_mIoU": row["x_t_argmax"]["mIoU"], "endpoint_normal_mIoU": row["endpoint_normal"]["mIoU"], "endpoint_zero_image_mIoU": row["endpoint_zero_image"]["mIoU"], "normal_zero_delta_mIoU": row["normal_zero_delta_mIoU"], "endpoint_x_t_delta_mIoU": row["endpoint_x_t_delta_mIoU"]})
    with (output / "per_class_iou.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(("condition", "t", "class_index", "class_name", "iou"))
        indices = result["source"]["evaluated_class_indices"]
        for index, iou in zip(indices, result["source"]["class_iou"]): writer.writerow(("source_mu", "", index, CITYSCAPES_CLASSES[index], iou))
        for row in result["fixed_t"].values():
            for condition in ("x_t_argmax", "endpoint_normal", "endpoint_zero_image"):
                for index, iou in zip(indices, row[condition]["class_iou"]): writer.writerow((condition, row["t"], index, CITYSCAPES_CLASSES[index], iou))
    s, lines = result["signal_noise"], ["WARNING: fixed-t evaluation uses validation GT and is diagnostic-only (not normal inference).", "", "[Source]", f"mu mIoU: {result['source']['mIoU']:.6f}", f"mu mAcc: {result['source']['mAcc']:.6f}", f"mu pixel accuracy: {result['source']['pixel_acc']:.6f}", "", "[Signal / Noise]"]
    lines += [f"{key}: {value:.6f}" for key, value in s.items()]
    lines += ["", "[Fixed t]", "t | x_t argmax mIoU | endpoint normal mIoU | endpoint zero-image mIoU | normal-zero delta | endpoint-x_t delta"]
    for row in result["fixed_t"].values(): lines.append(f"{row['t']:.2f} | {row['x_t_argmax']['mIoU']:.6f} | {row['endpoint_normal']['mIoU']:.6f} | {row['endpoint_zero_image']['mIoU']:.6f} | {row['normal_zero_delta_mIoU']:.6f} | {row['endpoint_x_t_delta_mIoU']:.6f}")
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostic-only Cityscapes fixed-t endpoint evaluation")
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--t-values", nargs="+", type=float, default=DEFAULT_T_VALUES)
    parser.add_argument("--num-visualizations", type=int, default=16); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int); return parser.parse_args()


def main() -> None:
    args = parse_args(); config = load_config(args.config)
    run_diagnostics(config, checkpoint_path=args.checkpoint, output_dir=args.output_dir,
                    t_values=args.t_values, num_visualizations=args.num_visualizations,
                    seed=args.seed, max_batches=args.max_batches)


if __name__ == "__main__": main()
