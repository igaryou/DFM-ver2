from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from sigma_analysis_utils import (
    StreamingDistribution,
    build_sigma_loader,
    continuous_at_target_resolution,
    iter_evaluation_batches,
    load_sigma_experiment,
    per_image_sigma_statistics,
    sigma_variation_summary,
    target_at_source_resolution,
    write_csv,
    write_json,
)
from utils import autocast_context
from visualization import colorize


def _display_image(image: torch.Tensor, config: dict) -> torch.Tensor:
    image = image.detach().float().cpu()
    normalize = config["augmentation"]["normalize"]
    if config["augmentation"]["imagenet_normalize"] or normalize["enabled"]:
        mean = image.new_tensor(normalize["mean"])[:, None, None]
        std = image.new_tensor(normalize["std"])[:, None, None]
        image = image * std + mean
    return image.clamp(0, 1)


def save_sigma_panel(
    image: torch.Tensor,
    target: torch.Tensor,
    sigma: torch.Tensor,
    path: Path,
    config: dict,
    *,
    scale: str,
    contrast_limits: tuple[float, float],
) -> None:
    if scale == "absolute":
        vmin, vmax = 0.5, 1.5
        suffix = "absolute scale [0.5, 1.5]"
    elif scale == "contrast":
        vmin, vmax = contrast_limits
        if vmax <= vmin:
            padding = max(abs(vmin) * 1.0e-3, 1.0e-4)
            vmin, vmax = vmin - padding, vmax + padding
        suffix = f"contrast scale [p05={vmin:.4g}, p95={vmax:.4g}]"
    else:
        raise ValueError("scale must be absolute or contrast")
    image = _display_image(image, config)
    target = target.detach().cpu()
    sigma = sigma.detach().float().cpu()
    deviation = (sigma - 1.0).abs()
    figure, axes = plt.subplots(1, 4, figsize=(22, 5))
    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title("input RGB")
    axes[1].imshow(colorize(target, dataset_name=config["dataset"]["name"]))
    axes[1].set_title("ground truth")
    sigma_artist = axes[2].imshow(sigma, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[2].set_title(
        f"sigma(x): {suffix}\nmin={sigma.min():.4g} "
        f"mean={sigma.mean():.4g} max={sigma.max():.4g}"
    )
    figure.colorbar(sigma_artist, ax=axes[2], fraction=0.046, pad=0.04)
    deviation_max = max(float(torch.quantile(deviation, 0.95)), 1.0e-8)
    deviation_artist = axes[3].imshow(
        deviation.clamp(max=deviation_max), cmap="magma", vmin=0.0,
        vmax=deviation_max,
    )
    axes[3].set_title(
        f"abs(sigma - 1)\nmean={deviation.mean():.4g}, "
        f"display max=p95={deviation_max:.4g}"
    )
    figure.colorbar(deviation_artist, ax=axes[3], fraction=0.046, pad=0.04)
    for axis in axes:
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_sigma_histogram(
    all_values: torch.Tensor,
    valid_values: torch.Tensor,
    all_stats: dict,
    path: Path,
    *,
    all_histogram: tuple[object, object] | None = None,
    valid_histogram: tuple[object, object] | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    if all_histogram is None or valid_histogram is None:
        axis.hist(all_values.numpy(), bins=160, density=True, alpha=0.60, label="all pixels")
        axis.hist(
            valid_values.numpy(), bins=160, density=True, alpha=0.45,
            label="GT non-void pixels",
        )
    else:
        for (counts, edges), label, alpha in (
            (all_histogram, "all pixels", 0.75),
            (valid_histogram, "GT non-void pixels", 0.60),
        ):
            widths = edges[1:] - edges[:-1]
            density = counts / max(counts.sum(), 1) / widths
            axis.stairs(density, edges, label=label, linewidth=1.5, alpha=alpha)
    axis.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="sigma=1")
    axis.set_xlabel("sigma(x)")
    axis.set_ylabel("density")
    axis.set_title(
        "Learned sigma distribution\n"
        f"mean={all_stats['sigma_mean']:.6g}, std={all_stats['sigma_std']:.6g}, "
        f"median={all_stats['sigma_p50']:.6g}, "
        f"p05={all_stats['sigma_p05']:.6g}, p95={all_stats['sigma_p95']:.6g}"
    )
    axis.legend()
    axis.grid(alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_image_variation_plot(rows: list[dict], path: Path) -> None:
    selected = [row for row in rows if row["scope"] == "all"]
    indices = [row["image_index"] for row in selected]
    means = [row["sigma_mean"] for row in selected]
    stds = [row["sigma_std"] for row in selected]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(indices, means, linewidth=0.8, marker=".", markersize=2)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("image index")
    axes[0].set_ylabel("sigma mean")
    axes[0].set_title("Per-image sigma mean")
    axes[1].scatter(means, stds, s=10, alpha=0.65)
    axes[1].set_xlabel("per-image sigma mean")
    axes[1].set_ylabel("per-image sigma std")
    axes[1].set_title("Within-image vs between-image variation")
    for axis in axes:
        axis.grid(alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


@torch.no_grad()
def analyze(args: argparse.Namespace) -> dict:
    config, device, endpoint, source, checkpoint = load_sigma_experiment(
        args.config, args.checkpoint, seed=args.seed
    )
    checkpoint_epoch = checkpoint.get("epoch")
    del checkpoint
    # Experiment 1 only uses the source, but loading through build_models above
    # verifies that the single checkpoint is architecture-compatible end-to-end.
    del endpoint
    loader, evaluated_images = build_sigma_loader(config, args.max_images)
    output_root = Path(args.output_dir).expanduser() if args.output_dir else (
        Path(config["experiment"]["output_dir"]) / "sigma_analysis"
    )
    output_dir = output_root / "sigma_distribution"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_distribution = StreamingDistribution(
        seed=args.seed, exact_values_path=output_dir / ".sigma_all.float32"
    )
    valid_distribution = StreamingDistribution(
        seed=args.seed + 1,
        exact_values_path=output_dir / ".sigma_gt_non_void.float32",
    )
    per_image_rows: list[dict] = []
    visualization_records: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    void_index = int(config["dataset"]["void_class_index"])

    progress = tqdm(iter_evaluation_batches(loader), total=len(loader), desc="sigma analysis")
    for batch in progress:
        image = batch.image.to(device, non_blocking=True)
        target = batch.target.to(device, non_blocking=True)
        with autocast_context(config, device):
            mu, logvar = source.forward_statistics(image)
        sigma = torch.exp(0.5 * logvar.detach().float())
        if sigma.shape[1] != 1:
            raise AssertionError(f"expected pixel_shared sigma, got {tuple(sigma.shape)}")
        sigma_map = sigma[:, 0]
        target_state = target_at_source_resolution(target, sigma_map.shape[-2:])
        valid_state = target_state != void_index
        all_distribution.update(sigma_map)
        valid_distribution.update(sigma_map, valid_state)

        for local_index, image_index in enumerate(batch.image_indices):
            for scope, values in (
                ("all", sigma_map[local_index]),
                ("gt_non_void", sigma_map[local_index][valid_state[local_index]]),
            ):
                per_image_rows.append({
                    "image_index": image_index,
                    "scope": scope,
                    **per_image_sigma_statistics(values),
                })
            if len(visualization_records) < args.max_visualizations:
                sigma_full = continuous_at_target_resolution(
                    sigma[local_index:local_index + 1],
                    target[local_index:local_index + 1],
                    batch.metadata,
                    config,
                )[0, 0]
                visualization_records.append((
                    image_index,
                    image[local_index].detach().cpu(),
                    target[local_index].detach().cpu(),
                    sigma_full.detach().cpu(),
                ))

    all_stats = all_distribution.compute("sigma")
    histogram_range = (all_stats["sigma_min"], all_stats["sigma_max"])
    valid_stats = valid_distribution.compute(
        "sigma", histogram_range=histogram_range
    )
    all_stats.update(sigma_variation_summary(per_image_rows, "all"))
    valid_stats.update(sigma_variation_summary(per_image_rows, "gt_non_void"))
    summary = {
        "experiment": "learned_sigma_distribution",
        "config_path": str(Path(args.config).expanduser().resolve()),
        "checkpoint_path": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "seed": args.seed,
        "evaluated_images": evaluated_images,
        "split": config["evaluation"]["split"],
        "source_architecture": type(source).__name__,
        "mu_model": config["source"]["segformer_variant"],
        "logvar_model": config["source"]["logvar_model"]["segformer_variant"],
        "variance_mode": config["source"]["variance_mode"],
        "target_std": config["source"]["variance_regularization"]["target_std"],
        "all_pixels": all_stats,
        "gt_non_void_pixels": valid_stats,
    }
    write_json(output_dir / "sigma_stats.json", summary)
    write_csv(output_dir / "sigma_per_image.csv", per_image_rows)
    save_sigma_histogram(
        all_distribution.sample_values,
        valid_distribution.sample_values,
        all_stats,
        output_dir / "sigma_histogram.png",
        all_histogram=(
            all_distribution.histogram_counts, all_distribution.histogram_edges
        ),
        valid_histogram=(
            valid_distribution.histogram_counts, valid_distribution.histogram_edges
        ),
    )
    save_image_variation_plot(per_image_rows, output_dir / "sigma_image_means.png")
    contrast_limits = (all_stats["sigma_p05"], all_stats["sigma_p95"])
    for image_index, image, target, sigma in visualization_records:
        for scale in ("absolute", "contrast"):
            save_sigma_panel(
                image, target, sigma,
                output_dir / "visualizations"
                / f"image_{image_index:04d}_{scale}.png",
                config,
                scale=scale,
                contrast_limits=contrast_limits,
            )
    print(f"Saved sigma analysis to {output_dir}")
    print(summary["all_pixels"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze learned pixel-shared sigma")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-visualizations", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
