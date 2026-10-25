from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

from sigma_analysis_utils import (
    PredictionComparisonAccumulator,
    build_sigma_loader,
    continuous_at_target_resolution,
    deterministic_epsilon_for_indices,
    iter_evaluation_batches,
    load_sigma_experiment,
    new_segmentation_metrics,
    paired_initial_states,
    parameter_versions,
    prediction_from_x0,
    source_path_difficulty,
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


def save_ablation_panel(
    image: torch.Tensor,
    target: torch.Tensor,
    mu_prediction: torch.Tensor,
    sigma: torch.Tensor,
    learned_prediction: torch.Tensor,
    fixed_prediction: torch.Tensor,
    learned_initial_prediction: torch.Tensor,
    fixed_initial_prediction: torch.Tensor,
    path: Path,
    config: dict,
) -> None:
    image = _display_image(image, config)
    target = target.detach().cpu()
    mu_prediction = mu_prediction.detach().cpu()
    sigma = sigma.detach().float().cpu()
    learned_prediction = learned_prediction.detach().cpu()
    fixed_prediction = fixed_prediction.detach().cpu()
    learned_initial_prediction = learned_initial_prediction.detach().cpu()
    fixed_initial_prediction = fixed_initial_prediction.detach().cpu()
    void = int(config["dataset"]["void_class_index"])
    valid = target != void
    disagreement = valid & (learned_prediction != fixed_prediction)
    learned_correct = valid & (learned_prediction == target)
    fixed_correct = valid & (fixed_prediction == target)
    learned_only = learned_correct & ~fixed_correct
    fixed_only = fixed_correct & ~learned_correct
    deviation = (sigma - 1.0).abs()

    figure, axes = plt.subplots(3, 4, figsize=(22, 15))
    axes = axes.reshape(-1)
    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title("input RGB")
    axes[1].imshow(colorize(target, dataset_name=config["dataset"]["name"]))
    axes[1].set_title("ground truth")
    axes[2].imshow(colorize(mu_prediction, dataset_name=config["dataset"]["name"]))
    axes[2].set_title("argmax(mu)")
    sigma_artist = axes[3].imshow(sigma, cmap="viridis", vmin=0.5, vmax=1.5)
    axes[3].set_title(f"sigma(x)\nmean={sigma.mean():.4g}, std={sigma.std(unbiased=False):.4g}")
    figure.colorbar(sigma_artist, ax=axes[3], fraction=0.046, pad=0.04)
    axes[4].imshow(colorize(learned_prediction, dataset_name=config["dataset"]["name"]))
    axes[4].set_title("DFM prediction: learned sigma")
    axes[5].imshow(colorize(fixed_prediction, dataset_name=config["dataset"]["name"]))
    axes[5].set_title("DFM prediction: fixed sigma")
    axes[6].imshow(disagreement, cmap="gray", vmin=0, vmax=1)
    axes[6].set_title(f"prediction disagreement\nrate={disagreement.float().mean():.4g}")
    axes[7].imshow(learned_only, cmap="Greens", vmin=0, vmax=1)
    axes[7].set_title(f"learned only correct\nfraction={learned_only.float().mean():.4g}")
    axes[8].imshow(fixed_only, cmap="Reds", vmin=0, vmax=1)
    axes[8].set_title(f"fixed only correct\nfraction={fixed_only.float().mean():.4g}")
    deviation_max = max(float(torch.quantile(deviation, 0.95)), 1.0e-8)
    deviation_artist = axes[9].imshow(
        deviation.clamp(max=deviation_max), cmap="magma", vmin=0, vmax=deviation_max
    )
    axes[9].set_title(f"abs(sigma - 1)\nmean={deviation.mean():.4g}")
    figure.colorbar(deviation_artist, ax=axes[9], fraction=0.046, pad=0.04)
    axes[10].imshow(colorize(
        learned_initial_prediction, dataset_name=config["dataset"]["name"]
    ))
    axes[10].set_title("argmax(x0): learned sigma")
    axes[11].imshow(colorize(
        fixed_initial_prediction, dataset_name=config["dataset"]["name"]
    ))
    axes[11].set_title("argmax(x0): fixed sigma")
    for axis in axes:
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _condition_metrics(prefix: str, result: dict) -> dict:
    return {
        f"{prefix}_mIoU": result["mIoU"],
        f"{prefix}_pixel_acc": result["pixel_acc"],
        f"{prefix}_mAcc": result["mAcc"],
        f"{prefix}_class_iou": result["class_iou"],
        f"{prefix}_class_accuracy": result["class_accuracy"],
    }


def comparison_result(
    learned_result: dict,
    fixed_result: dict,
    comparison: PredictionComparisonAccumulator,
) -> dict:
    return {
        **_condition_metrics("learned", learned_result),
        **_condition_metrics("fixed", fixed_result),
        "delta_mIoU": learned_result["mIoU"] - fixed_result["mIoU"],
        "delta_pixel_acc": learned_result["pixel_acc"] - fixed_result["pixel_acc"],
        "delta_mAcc": learned_result["mAcc"] - fixed_result["mAcc"],
        "delta_class_iou": [
            learned - fixed for learned, fixed in zip(
                learned_result["class_iou"], fixed_result["class_iou"], strict=True
            )
        ],
        **comparison.compute(),
    }


@torch.no_grad()
def evaluate_ablation(args: argparse.Namespace) -> dict:
    config, device, endpoint, source, checkpoint = load_sigma_experiment(
        args.config, args.checkpoint, seed=args.seed
    )
    checkpoint_epoch = checkpoint.get("epoch")
    del checkpoint
    steps_values = (
        sorted(set(args.num_steps))
        if args.num_steps else [int(config["evaluation"]["num_steps"])]
    )
    if any(value <= 0 for value in steps_values):
        raise ValueError("all num_steps values must be positive")
    loader, evaluated_images = build_sigma_loader(config, args.max_images)
    output_root = Path(args.output_dir).expanduser() if args.output_dir else (
        Path(config["experiment"]["output_dir"]) / "sigma_analysis"
    )
    output_dir = output_root / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    learned_metrics = {
        steps: new_segmentation_metrics(config, device) for steps in steps_values
    }
    fixed_metrics = {
        steps: new_segmentation_metrics(config, device) for steps in steps_values
    }
    comparisons = {
        steps: PredictionComparisonAccumulator() for steps in steps_values
    }
    visualization_counts = {steps: 0 for steps in steps_values}
    per_image_rows: list[dict] = []
    void_index = int(config["dataset"]["void_class_index"])
    num_classes = int(config["dataset"]["num_classes"])
    versions_before = parameter_versions(endpoint, source)

    progress = tqdm(iter_evaluation_batches(loader), total=len(loader), desc="sigma ablation")
    for batch in progress:
        image = batch.image.to(device, non_blocking=True)
        target = batch.target.to(device, non_blocking=True)
        # One source forward and one epsilon construction are shared by every
        # condition and every NFE value for this image batch.
        with autocast_context(config, device):
            mu, logvar = source.forward_statistics(image)
        epsilon = deterministic_epsilon_for_indices(
            mu, batch.image_indices, seed=args.seed
        )
        x0_learned, x0_fixed, sigma = paired_initial_states(
            mu, logvar, epsilon, fixed_std=args.fixed_std
        )
        path_difficulty = source_path_difficulty(mu, config)
        sigma_full = continuous_at_target_resolution(
            sigma, target, batch.metadata, config
        )[:, 0]
        need_visualization = any(
            visualization_counts[steps] < args.max_visualizations
            for steps in steps_values
        )
        if need_visualization:
            mu_full = continuous_at_target_resolution(mu, target, batch.metadata, config)
            learned_initial_full = continuous_at_target_resolution(
                x0_learned, target, batch.metadata, config
            )
            fixed_initial_full = continuous_at_target_resolution(
                x0_fixed, target, batch.metadata, config
            )

        for steps in steps_values:
            with autocast_context(config, device):
                learned_prediction = prediction_from_x0(
                    endpoint, image, x0_learned, config,
                    num_steps=steps, path_difficulty=path_difficulty,
                    metadata=batch.metadata,
                )
                fixed_prediction = prediction_from_x0(
                    endpoint, image, x0_fixed, config,
                    num_steps=steps, path_difficulty=path_difficulty,
                    metadata=batch.metadata,
                )
            learned_metrics[steps].update(learned_prediction, target)
            fixed_metrics[steps].update(fixed_prediction, target)
            comparisons[steps].update(
                learned_prediction, fixed_prediction, target, sigma_full,
                void_index=void_index, num_classes=num_classes,
            )

            for local_index, image_index in enumerate(batch.image_indices):
                learned_one = new_segmentation_metrics(config, device)
                fixed_one = new_segmentation_metrics(config, device)
                learned_one.update(
                    learned_prediction[local_index:local_index + 1],
                    target[local_index:local_index + 1],
                )
                fixed_one.update(
                    fixed_prediction[local_index:local_index + 1],
                    target[local_index:local_index + 1],
                )
                comparison_one = PredictionComparisonAccumulator()
                comparison_one.update(
                    learned_prediction[local_index:local_index + 1],
                    fixed_prediction[local_index:local_index + 1],
                    target[local_index:local_index + 1],
                    sigma_full[local_index:local_index + 1],
                    void_index=void_index, num_classes=num_classes,
                )
                learned_one_result = learned_one.compute()
                fixed_one_result = fixed_one.compute()
                per_image_rows.append({
                    "num_steps": steps,
                    "image_index": image_index,
                    "learned_mIoU": learned_one_result["mIoU"],
                    "fixed_mIoU": fixed_one_result["mIoU"],
                    "delta_mIoU": (
                        learned_one_result["mIoU"] - fixed_one_result["mIoU"]
                    ),
                    "learned_pixel_acc": learned_one_result["pixel_acc"],
                    "fixed_pixel_acc": fixed_one_result["pixel_acc"],
                    **comparison_one.compute(),
                })

                if visualization_counts[steps] < args.max_visualizations:
                    save_ablation_panel(
                        image[local_index], target[local_index],
                        mu_full[local_index].argmax(dim=0),
                        sigma_full[local_index],
                        learned_prediction[local_index],
                        fixed_prediction[local_index],
                        learned_initial_full[local_index].argmax(dim=0),
                        fixed_initial_full[local_index].argmax(dim=0),
                        output_dir / f"nfe_{steps}" / "visualizations"
                        / f"image_{image_index:04d}.png",
                        config,
                    )
                    visualization_counts[steps] += 1

    if parameter_versions(endpoint, source) != versions_before:
        raise RuntimeError("evaluation mutated endpoint or source model parameters")

    result_by_steps: dict[str, dict] = {}
    per_class_rows: list[dict] = []
    sigma_bin_rows: list[dict] = []
    for steps in steps_values:
        learned = learned_metrics[steps].compute()
        fixed = fixed_metrics[steps].compute()
        result = comparison_result(learned, fixed, comparisons[steps])
        result_by_steps[f"nfe_{steps}"] = result
        for class_index, learned_iou, fixed_iou, learned_accuracy, fixed_accuracy in zip(
            learned["evaluated_class_indices"],
            learned["class_iou"], fixed["class_iou"],
            learned["class_accuracy"], fixed["class_accuracy"], strict=True,
        ):
            per_class_rows.append({
                "num_steps": steps,
                "class_index": class_index,
                "learned_iou": learned_iou,
                "fixed_iou": fixed_iou,
                "delta_iou": learned_iou - fixed_iou,
                "learned_accuracy": learned_accuracy,
                "fixed_accuracy": fixed_accuracy,
                "delta_accuracy": learned_accuracy - fixed_accuracy,
            })
        for row in comparisons[steps].bin_rows():
            sigma_bin_rows.append({"num_steps": steps, **row})

    summary = {
        "experiment": "learned_sigma_vs_fixed_sigma",
        "difference_convention": "learned_minus_fixed",
        "config_path": str(Path(args.config).expanduser().resolve()),
        "checkpoint_path": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "num_steps": steps_values,
        "fixed_std": args.fixed_std,
        "seed": args.seed,
        "evaluated_images": evaluated_images,
        "split": config["evaluation"]["split"],
        "source_architecture": type(source).__name__,
        "epsilon_policy": "one deterministic tensor per dataset image index, shared by all conditions and NFE values",
        "mu_policy": "one source forward per image batch, shared by all conditions and NFE values",
        "results": result_by_steps,
    }
    write_json(output_dir / "comparison_summary.json", summary)
    write_csv(output_dir / "comparison_per_class.csv", per_class_rows)
    write_csv(output_dir / "comparison_per_image.csv", per_image_rows)
    write_csv(output_dir / "comparison_sigma_bins.csv", sigma_bin_rows)
    for steps in steps_values:
        write_json(output_dir / f"nfe_{steps}" / "metrics.json", {
            "num_steps": steps,
            **result_by_steps[f"nfe_{steps}"],
        })
    print(f"Saved sigma ablation to {output_dir}")
    for steps in steps_values:
        result = result_by_steps[f"nfe_{steps}"]
        print(
            f"NFE={steps}: learned_mIoU={result['learned_mIoU']:.6g} "
            f"fixed_mIoU={result['fixed_mIoU']:.6g} "
            f"delta={result['delta_mIoU']:.6g} "
            f"disagreement={result['prediction_disagreement_rate']:.6g}"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare learned pixel-shared sigma with a fixed sigma"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-steps", type=int, nargs="+", default=None)
    parser.add_argument("--fixed-std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-visualizations", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_ablation(parse_args())
