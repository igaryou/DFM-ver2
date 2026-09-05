from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import load_config
from checkpoint import validate_source_decoder_checkpoint
from dataset import ade20k_eval_collate, build_dataset
from distributed import (
    DistributedEvalSampler,
    all_reduce_confusion_matrix,
    assert_config_equal_across_ranks,
    barrier,
    cleanup_distributed,
    seed_data_loader_worker,
    setup_distributed,
    validate_global_batch_size,
)
from inference import (
    run_flow_from_state,
    sample_segmentation_from_x0,
    state_to_original_continuous,
    terminal_state_to_original_prediction,
)
from metrics import SegmentationMetrics
from source_diagnostics import _load_models
from utils import autocast_context, seed_everything
from visualization import colorize


CITYSCAPES_CLASSES = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
)
CHECKPOINT_PRIORITY = ("best.pt", "latest.pt", "last.pt")


def resolve_ablation_checkpoint(
    checkpoint: str | Path | None,
    checkpoint_dir: str | Path | None,
) -> Path:
    """Resolve one checkpoint using an explicit, documented directory priority."""
    if checkpoint is not None and checkpoint_dir is not None:
        raise ValueError("Specify only one of --checkpoint and --checkpoint-dir")
    if checkpoint is not None:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path.resolve()
    if checkpoint_dir is None:
        raise ValueError("--checkpoint or --checkpoint-dir is required")
    directory = Path(checkpoint_dir).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")
    for name in CHECKPOINT_PRIORITY:
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    candidates = sorted(directory.glob("*.pt"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found in: {directory}")
    raise ValueError(
        "Checkpoint directory has no preferred best.pt/latest.pt/last.pt and "
        "contains multiple candidates; specify --checkpoint explicitly: "
        + ", ".join(path.name for path in candidates)
    )


def assert_power2_path(config: dict, checkpoint: dict | None = None) -> None:
    path = config["flow"]["path"]
    if path["type"] != "power" or float(path["exponent"]) != 2.0:
        raise ValueError(
            "Source/Flow ablation requires flow.path.type=power and exponent=2.0"
        )
    if checkpoint is None or not isinstance(checkpoint.get("config"), dict):
        return
    saved_path = checkpoint["config"].get("flow", {}).get("path", {})
    if (
        saved_path.get("type") != "power"
        or float(saved_path.get("exponent", float("nan"))) != 2.0
    ):
        raise RuntimeError(
            "Checkpoint was not trained with flow.path.type=power, exponent=2.0"
        )


def stable_sample_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def source_pair_from_one_forward(
    source_model,
    image: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call the production Gaussian source once under a sample-local RNG."""
    devices = []
    if image.device.type == "cuda":
        devices = [
            image.device.index
            if image.device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if image.device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        z0, mu, logvar = source_model(image)
    if z0.shape != mu.shape or mu.shape != logvar.shape:
        raise AssertionError("z0, mu, and logvar must have identical shapes")
    return z0, mu, logvar


def _new_metrics(config: dict, device: torch.device) -> SegmentationMetrics:
    eval_range = config["evaluation"]["eval_class_indices"]
    evaluated = (
        range(eval_range[0], eval_range[1] + 1)
        if eval_range is not None else None
    )
    return SegmentationMetrics(
        config["dataset"]["num_classes"],
        config["dataset"]["void_class_index"],
        device=device,
        evaluated_class_indices=evaluated,
        nanmean=config["evaluation"]["nanmean"],
        prediction_void_retained=not config["evaluation"][
            "exclude_void_from_prediction"
        ],
    )


def original_prediction(
    state: torch.Tensor, sample: dict, config: dict
) -> torch.Tensor:
    return terminal_state_to_original_prediction(
        state,
        sample["model_shape"],
        sample["original_shape"],
        padded_shape=sample["padded_shape"],
        align_corners=config["evaluation"]["align_corners"],
        void_class_index=config["dataset"]["void_class_index"],
        exclude_void=config["evaluation"]["exclude_void_from_prediction"],
    )


def run_flow_conditions(
    model,
    image: torch.Tensor,
    mu: torch.Tensor,
    z0: torch.Tensor,
    config: dict,
    *,
    num_steps: int,
    return_trajectory: bool = False,
):
    """Run both ablations through the same production Flow Map implementation."""
    if return_trajectory:
        mu_terminal, mu_trajectory = run_flow_from_state(
            model, image, mu, config, num_steps, return_trajectory=True
        )
        z0_terminal, z0_trajectory = run_flow_from_state(
            model, image, z0, config, num_steps, return_trajectory=True
        )
        return mu_terminal, z0_terminal, mu_trajectory, z0_trajectory
    mu_terminal = sample_segmentation_from_x0(
        model, image, mu, config, num_steps=num_steps,
        return_terminal_state=True,
    )
    z0_terminal = sample_segmentation_from_x0(
        model, image, z0, config, num_steps=num_steps,
        return_terminal_state=True,
    )
    return mu_terminal, z0_terminal, None, None


def _original_scalar(
    values: torch.Tensor, sample: dict, config: dict
) -> torch.Tensor:
    return state_to_original_continuous(
        values[:, None],
        sample["model_shape"],
        sample["original_shape"],
        padded_shape=sample["padded_shape"],
        align_corners=config["evaluation"]["align_corners"],
    )[:, 0]


def _display_image(image: torch.Tensor, sample: dict, config: dict) -> torch.Tensor:
    displayed = image.detach().float().cpu()
    normalize = config["augmentation"].get("normalize", {})
    if normalize.get("enabled", False):
        mean = displayed.new_tensor(normalize["mean"])[None, :, None, None]
        std = displayed.new_tensor(normalize["std"])[None, :, None, None]
        displayed = displayed * std + mean
    elif config["augmentation"].get("imagenet_normalize", False):
        mean = displayed.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = displayed.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        displayed = displayed * std + mean
    model_height, model_width = (int(v) for v in sample["model_shape"])
    displayed = displayed[..., :model_height, :model_width]
    displayed = F.interpolate(
        displayed,
        size=tuple(int(v) for v in sample["original_shape"]),
        mode="bilinear",
        align_corners=False,
    )
    return displayed[0].clamp(0, 1)


def _save_comparison(
    path: Path,
    *,
    image: torch.Tensor,
    target: torch.Tensor,
    source_mu: torch.Tensor,
    source_z0: torch.Tensor,
    flow_mu: torch.Tensor,
    flow_z0: torch.Tensor,
    confidence: torch.Tensor,
    noise_magnitude: torch.Tensor,
) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(22, 9))
    top = (
        ("Input", image.permute(1, 2, 0).numpy()),
        ("GT", colorize(target, "cityscapes")),
        ("argmax(mu)", colorize(source_mu, "cityscapes")),
        ("argmax(z0)", colorize(source_z0, "cityscapes")),
        ("Flow(mu)", colorize(flow_mu, "cityscapes")),
    )
    for axis, (title, values) in zip(axes[0], top, strict=True):
        axis.imshow(values)
        axis.set_title(title)
        axis.axis("off")
    axes[1, 0].imshow(colorize(flow_z0, "cityscapes"))
    axes[1, 0].set_title("Flow(z0)")
    confidence_image = axes[1, 1].imshow(
        confidence.numpy(), cmap="viridis", vmin=0.0, vmax=1.0
    )
    axes[1, 1].set_title("source confidence")
    figure.colorbar(confidence_image, ax=axes[1, 1], fraction=0.046, pad=0.04)
    noise_image = axes[1, 2].imshow(noise_magnitude.numpy(), cmap="magma")
    axes[1, 2].set_title("||z0 - mu||2")
    figure.colorbar(noise_image, ax=axes[1, 2], fraction=0.046, pad=0.04)
    axes[1, 3].imshow((source_mu != flow_mu).numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1, 3].set_title("mu -> Flow changed")
    axes[1, 4].imshow((source_z0 != flow_z0).numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1, 4].set_title("z0 -> Flow changed")
    for axis in axes[1]:
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _save_trajectory(
    trajectory: torch.Tensor,
    output_dir: Path,
    sample_id: str,
) -> None:
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    for step, mask in enumerate(trajectory[0]):
        plt.imsave(
            sample_dir / f"step_{step:03d}.png",
            colorize(mask, "cityscapes"),
        )


def _metric_payload(metrics: SegmentationMetrics) -> dict[str, Any]:
    result = metrics.compute()
    return {"miou": result["mIoU"], **result}


def build_result(
    source_mu: dict[str, Any],
    flow_mu: dict[str, Any],
    flow_z0: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_mu": source_mu,
        "flow_from_mu": flow_mu,
        "flow_from_z0": flow_z0,
        "source_mu_miou": source_mu["miou"],
        "flow_from_mu_miou": flow_mu["miou"],
        "flow_from_z0_miou": flow_z0["miou"],
        "delta_flow_mu_vs_source": flow_mu["miou"] - source_mu["miou"],
        "delta_flow_z0_vs_source": flow_z0["miou"] - source_mu["miou"],
        "delta_flow_mu_vs_z0": flow_mu["miou"] - flow_z0["miou"],
        "delta_noise": flow_z0["miou"] - flow_mu["miou"],
    }


def _write_per_class_csv(result: dict[str, Any], path: Path) -> None:
    indices = result["source_mu"]["evaluated_class_indices"]
    rows = []
    for offset, class_id in enumerate(indices):
        rows.append({
            "class_id": class_id,
            "class_name": CITYSCAPES_CLASSES[class_id],
            "source_mu": result["source_mu"]["class_iou"][offset],
            "flow_from_mu": result["flow_from_mu"]["class_iou"][offset],
            "flow_from_z0": result["flow_from_z0"]["class_iou"][offset],
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_source_flow_ablation(
    config: dict,
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    num_steps: int | None = None,
    num_visualizations: int = 16,
    seed: int = 42,
    save_trajectory: bool = False,
    max_batches: int | None = None,
) -> dict[str, Any]:
    if config["dataset"]["name"] != "cityscapes":
        raise ValueError("Source/Flow ablation currently supports Cityscapes")
    if not config["evaluation"]["original_resolution"]:
        raise ValueError("Source/Flow ablation requires original-resolution evaluation")
    if config["source"]["prior_type"] != "image_gaussian":
        raise ValueError("Source/Flow ablation requires source.prior_type=image_gaussian")
    assert_power2_path(config)
    steps = config["evaluation"]["num_steps"] if num_steps is None else int(num_steps)
    if steps <= 0:
        raise ValueError("num_steps must be a positive integer")
    output = Path(output_dir).expanduser().resolve()
    context = setup_distributed(config)
    try:
        assert_config_equal_across_ranks(config, context)
        seed_everything(seed, config["runtime"]["deterministic"])
        checkpoint, model, source_model = _load_models(
            config, Path(checkpoint_path).resolve(), context.device
        )
        validate_source_decoder_checkpoint(checkpoint, config, checkpoint_path)
        assert_power2_path(config, checkpoint)
        if context.is_main_process:
            output.mkdir(parents=True, exist_ok=True)
            print(
                f"checkpoint={Path(checkpoint_path).resolve()} "
                f"path=power exponent=2.0 num_steps={steps} seed={seed}"
            )
        barrier(context)

        dataset = build_dataset(config, config["evaluation"]["split"], augment=False)
        sampler = (
            DistributedEvalSampler(
                dataset, rank=context.rank, world_size=context.world_size
            )
            if context.distributed else None
        )
        local_batch = validate_global_batch_size(
            config["evaluation"]["batch_size"], context.world_size
        )
        loader = DataLoader(
            dataset,
            batch_size=local_batch,
            sampler=sampler,
            shuffle=False,
            num_workers=config["dataset"]["num_workers"],
            pin_memory=config["dataset"]["pin_memory"],
            worker_init_fn=seed_data_loader_worker,
            collate_fn=ade20k_eval_collate,
        )
        metrics_source_mu = _new_metrics(config, context.device)
        metrics_flow_mu = _new_metrics(config, context.device)
        metrics_flow_z0 = _new_metrics(config, context.device)
        visualized = 0
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            for sample in batch:
                image = sample["image"].unsqueeze(0).to(
                    context.device, non_blocking=True
                )
                target = sample["target"].unsqueeze(0).to(
                    context.device, non_blocking=True
                )
                sample_id = str(sample["sample_id"])
                sample_seed = stable_sample_seed(seed, sample_id)
                with autocast_context(config, context.device):
                    z0, mu, logvar = source_pair_from_one_forward(
                        source_model, image, seed=sample_seed
                    )
                    (
                        mu_terminal,
                        z0_terminal,
                        mu_trajectory,
                        z0_trajectory,
                    ) = run_flow_conditions(
                        model,
                        image,
                        mu,
                        z0,
                        config,
                        num_steps=steps,
                        return_trajectory=save_trajectory,
                    )

                source_mu_prediction = original_prediction(mu, sample, config)
                source_z0_prediction = original_prediction(z0, sample, config)
                flow_mu_prediction = original_prediction(mu_terminal, sample, config)
                flow_z0_prediction = original_prediction(z0_terminal, sample, config)
                metrics_source_mu.update(source_mu_prediction, target)
                metrics_flow_mu.update(flow_mu_prediction, target)
                metrics_flow_z0.update(flow_z0_prediction, target)

                if context.is_main_process and visualized < num_visualizations:
                    mu_full = state_to_original_continuous(
                        mu,
                        sample["model_shape"],
                        sample["original_shape"],
                        padded_shape=sample["padded_shape"],
                        align_corners=config["evaluation"]["align_corners"],
                    )
                    confidence = torch.softmax(mu_full.float(), dim=1).amax(dim=1)
                    noise_magnitude = _original_scalar(
                        torch.linalg.vector_norm((z0 - mu).float(), dim=1),
                        sample,
                        config,
                    )
                    _save_comparison(
                        output / "visualizations" / f"{sample_id}.png",
                        image=_display_image(image, sample, config),
                        target=target[0].cpu(),
                        source_mu=source_mu_prediction[0].cpu(),
                        source_z0=source_z0_prediction[0].cpu(),
                        flow_mu=flow_mu_prediction[0].cpu(),
                        flow_z0=flow_z0_prediction[0].cpu(),
                        confidence=confidence[0].cpu(),
                        noise_magnitude=noise_magnitude[0].cpu(),
                    )
                    if save_trajectory:
                        _save_trajectory(
                            mu_trajectory,
                            output / "trajectory_mu",
                            sample_id,
                        )
                        _save_trajectory(
                            z0_trajectory,
                            output / "trajectory_z0",
                            sample_id,
                        )
                    visualized += 1

        for metrics in (metrics_source_mu, metrics_flow_mu, metrics_flow_z0):
            metrics.confusion_matrix = all_reduce_confusion_matrix(
                metrics.confusion_matrix, context
            )
        result = build_result(
            _metric_payload(metrics_source_mu),
            _metric_payload(metrics_flow_mu),
            _metric_payload(metrics_flow_z0),
        )
        result["metadata"] = {
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "seed": int(seed),
            "num_steps": steps,
            "flow_path_type": "power",
            "flow_path_exponent": 2.0,
            "source_definition": "z0 = mu + exp(0.5 * logvar) * epsilon",
            "fixed_std": config["source"]["fixed_std"],
        }
        if context.is_main_process:
            with (output / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
            _write_per_class_csv(result, output / "per_class_iou.csv")
            with (output / "config.json").open("w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)
            print(json.dumps(result, indent=2))
        barrier(context)
        return result
    finally:
        cleanup_distributed(context)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare source mu, Flow(mu), and Flow(sampled Gaussian z0)"
    )
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--checkpoint-dir")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--num-visualizations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-trajectory", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    if args.split is not None:
        config["evaluation"]["split"] = args.split
    checkpoint = resolve_ablation_checkpoint(args.checkpoint, args.checkpoint_dir)
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir is not None
        else checkpoint.parent / "source_flow_ablation"
    )
    evaluate_source_flow_ablation(
        config,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        num_steps=args.num_steps,
        num_visualizations=args.num_visualizations,
        seed=args.seed,
        save_trajectory=args.save_trajectory,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
