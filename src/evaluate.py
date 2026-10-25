from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader,Subset

from checkpoint import _without_module_prefix
from config import load_config
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
    sample_lidc_distribution, sample_segmentation, sample_segmentation_ensemble,
    terminal_state_to_original_prediction,
)
from metrics import LIDCStochasticMetrics, SegmentationMetrics
from model_factory import build_models
from source_noise_diagnostics import (
    DIAGNOSTIC_RESULT_KEYS,
    SourceNoiseDiagnosticsAccumulator,
    source_noise_diagnostic_maps,
    source_noise_diagnostics_enabled,
)
from utils import autocast_context, init_wandb, seed_everything
from visualization import save_prediction, save_source_noise_diagnostics


@torch.no_grad()
def evaluate(config: dict, checkpoint_path: str | Path) -> dict:
    context = setup_distributed(config)
    wandb_run = None
    try:
        assert_config_equal_across_ranks(config, context)
        device = context.device
        if (
            context.is_main_process
            and config["source"]["prior_type"] == "image_simplex_mixture"
        ):
            simplex = config["source"]["simplex_prior"]["inference"]
            print(
                "Simplex source prior: mode=inference "
                f"lambda={simplex['lambda']} temperature={simplex['temperature']} "
                f"dirichlet_alpha={simplex['dirichlet_alpha']}"
            )
        seed_everything(
            config["experiment"]["seed"], config["runtime"]["deterministic"]
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model, source_model = build_models(config, device)
        model.load_state_dict(
            _without_module_prefix(checkpoint["model"]),
            strict=config["checkpoint"]["strict_model"],
        )
        if source_model is not None:
            if checkpoint.get("source_model") is None:
                raise RuntimeError("Checkpoint has no source_model state")
            source_model.load_state_dict(
                _without_module_prefix(checkpoint["source_model"]),
                strict=config["checkpoint"]["strict_model"],
            )
            source_model.eval()
        model.eval()

        dataset = build_dataset(
            config, config["evaluation"]["split"], augment=False
        )
        subset_size = config["evaluation"]["random_subset_size"]
        if subset_size is not None:
            if subset_size <= 0:
                raise ValueError("evaluation.random_subset_size must be positive")
            if subset_size > len(dataset):
                raise ValueError(
                    f"evaluation.random_subset_size={subset_size} exceeds "
                    f"dataset size {len(dataset)}"
                    )

            generator = torch.Generator().manual_seed(
                config["evaluation"]["random_subset_seed"]
            )
            indices = torch.randperm(
                len(dataset), generator=generator
            )[:subset_size].tolist()

            dataset = Subset(dataset, indices)

            if context.is_main_process:
                print(
                    f"Evaluation random subset: "
                    f"{subset_size}/{len(indices) if False else len(dataset)} samples, "
                    f"seed={config['evaluation']['random_subset_seed']}"
                )
        sampler = (
            DistributedEvalSampler(
                dataset, rank=context.rank, world_size=context.world_size
            )
            if context.distributed else None
        )
        local_batch_size = validate_global_batch_size(
            config["evaluation"]["batch_size"], context.world_size
        )
        loader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=config["dataset"]["num_workers"],
            pin_memory=config["dataset"]["pin_memory"],
            worker_init_fn=seed_data_loader_worker,
            collate_fn=(
                ade20k_eval_collate
                if (
                    config["dataset"]["name"] == "ade20k"
                    or config["evaluation"]["original_resolution"]
                ) else None
            ),
        )
        eval_range = config["evaluation"]["eval_class_indices"]
        metrics = SegmentationMetrics(
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
        stochastic_counts = config["evaluation"]["stochastic_num_samples"]
        lidc_metrics = (
            LIDCStochasticMetrics(stochastic_counts, device=device)
            if stochastic_counts else None
        )
        checkpoint_stem = Path(checkpoint_path).stem
        output_dir = (
            Path(config["experiment"]["output_dir"])
            / f"evaluation_{checkpoint_stem}_{config['evaluation']['num_steps']}steps"
        )
        if context.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        barrier(context)
        visualized = 0
        source_diagnostic_visualized = 0
        source_diagnostic_config = config["source"]["diagnostics"]
        source_diagnostic_enabled = source_noise_diagnostics_enabled(config)
        source_diagnostic_accumulator = (
            SourceNoiseDiagnosticsAccumulator() if source_diagnostic_enabled else None
        )
        source_diagnostic_kwargs = (
            {"return_source_diagnostics": True} if source_diagnostic_enabled else {}
        )
        for batch_index, batch in enumerate(loader):
            maximum_batches = config["evaluation"]["max_batches"]
            if maximum_batches is not None and batch_index >= maximum_batches:
                break
            if (
                config["dataset"]["name"] == "ade20k"
                or config["evaluation"]["original_resolution"]
            ):
                items = []
                for sample in batch:
                    image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
                    target = sample["target"].unsqueeze(0).to(device, non_blocking=True)
                    with autocast_context(config, device):
                        terminal_result = sample_segmentation(
                            model, source_model, image, config,
                            return_terminal_state=True,
                            **source_diagnostic_kwargs,
                        )
                    if source_diagnostic_enabled:
                        terminal, source_diagnostic_tensors = terminal_result
                        source_diagnostic_maps = source_noise_diagnostic_maps(
                            **{key: source_diagnostic_tensors[key] for key in ("x0", "mu", "sigma")}
                        )
                        assert source_diagnostic_accumulator is not None
                        source_diagnostic_accumulator.update(source_diagnostic_maps)
                        if (
                            context.is_main_process
                            and source_diagnostic_config["visualization"]
                            and source_diagnostic_visualized
                            < source_diagnostic_config["max_visualizations"]
                        ):
                            save_source_noise_diagnostics(
                                image[0], target[0],
                                {key: value[0] for key, value in source_diagnostic_maps.items()},
                                output_dir / "source_diagnostics"
                                / f"sample_{source_diagnostic_visualized:04d}.png",
                                imagenet_normalize=(
                                    config["augmentation"]["imagenet_normalize"]
                                    or config["augmentation"]["normalize"]["enabled"]
                                ),
                                dataset_name=config["dataset"]["name"],
                            )
                            source_diagnostic_visualized += 1
                    else:
                        terminal = terminal_result
                    prediction = terminal_state_to_original_prediction(
                        terminal, sample["model_shape"], sample["original_shape"],
                        padded_shape=sample["padded_shape"],
                        align_corners=config["evaluation"]["align_corners"],
                        void_class_index=config["dataset"]["void_class_index"],
                        exclude_void=config["evaluation"][
                            "exclude_void_from_prediction"
                        ],
                    )
                    metrics.update(prediction, target)
                    items.append((image, target, prediction))
            else:
                if isinstance(batch, dict):
                    image, target = batch["image"], batch["target"]
                else:
                    image, target = batch
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                source_diagnostic_tensors = None
                with autocast_context(config, device):
                    if lidc_metrics is not None:
                        lidc_result = sample_lidc_distribution(
                            model, source_model, image, config, stochastic_counts,
                            **source_diagnostic_kwargs,
                        )
                        if source_diagnostic_enabled:
                            prediction_samples, prediction, source_diagnostic_tensors = lidc_result
                        else:
                            prediction_samples, prediction = lidc_result
                    elif config["evaluation"]["num_samples"] == 1:
                        segmentation_result = sample_segmentation(
                            model, source_model, image, config,
                            **source_diagnostic_kwargs,
                        )
                        if source_diagnostic_enabled:
                            prediction, source_diagnostic_tensors = segmentation_result
                        else:
                            prediction = segmentation_result
                    else:
                        ensemble_result = sample_segmentation_ensemble(
                            model, source_model, image, config,
                            **source_diagnostic_kwargs,
                        )
                        if source_diagnostic_enabled:
                            prediction, source_diagnostic_tensors = ensemble_result
                        else:
                            prediction = ensemble_result
                if source_diagnostic_tensors is not None:
                    source_diagnostic_maps = source_noise_diagnostic_maps(
                        **{key: source_diagnostic_tensors[key] for key in ("x0", "mu", "sigma")}
                    )
                    assert source_diagnostic_accumulator is not None
                    source_diagnostic_accumulator.update(source_diagnostic_maps)
                    if (
                        context.is_main_process
                        and source_diagnostic_config["visualization"]
                        and source_diagnostic_visualized
                        < source_diagnostic_config["max_visualizations"]
                    ):
                        remaining = (
                            source_diagnostic_config["max_visualizations"]
                            - source_diagnostic_visualized
                        )
                        for index in range(min(image.shape[0], remaining)):
                            save_source_noise_diagnostics(
                                image[index], target[index],
                                {key: value[index] for key, value in source_diagnostic_maps.items()},
                                output_dir / "source_diagnostics"
                                / f"sample_{source_diagnostic_visualized:04d}.png",
                                imagenet_normalize=(
                                    config["augmentation"]["imagenet_normalize"]
                                    or config["augmentation"]["normalize"]["enabled"]
                                ),
                                dataset_name=config["dataset"]["name"],
                            )
                            source_diagnostic_visualized += 1
                metrics.update(prediction, target)
                if lidc_metrics is not None:
                    lidc_metrics.update(
                        prediction_samples, batch["masks"], prediction
                    )
                items = [(image, target, prediction)]
            if context.is_main_process and config["evaluation"]["save_predictions"]:
                remaining = config["evaluation"]["max_visualizations"] - visualized
                for item_index, (images, targets, predictions) in enumerate(items):
                    for index in range(min(images.shape[0], max(remaining, 0))):
                        display_image = images[index]
                        if display_image.shape[-2:] != targets[index].shape:
                            display_image = torch.nn.functional.interpolate(
                                display_image[None].float(), size=targets[index].shape,
                                mode="bilinear", align_corners=False,
                            )[0]
                        save_prediction(
                            display_image, targets[index], predictions[index],
                            output_dir / "predictions"
                            / f"{batch_index:04d}_{item_index:02d}_{index:02d}.png",
                            (
                                config["augmentation"]["imagenet_normalize"]
                                or config["augmentation"]["normalize"]["enabled"]
                            ),
                            dataset_name=config["dataset"]["name"],
                        )
                        visualized += 1
                        remaining -= 1
        metrics.confusion_matrix = all_reduce_confusion_matrix(
            metrics.confusion_matrix, context
        )
        result = metrics.compute()
        if source_diagnostic_accumulator is not None:
            result.update(source_diagnostic_accumulator.compute(
                device=device, distributed=context.distributed
            ))
        if lidc_metrics is not None:
            lidc_metrics.synchronize()
            result.update(lidc_metrics.compute())
        result.update({
            "dataset_protocol": config["dataset"].get("protocol", "mmseg"),
            "evaluation_resolution": list(config["dataset"]["image_size"]),
            "original_resolution": bool(config["evaluation"]["original_resolution"]),
            "num_samples": int(config["evaluation"]["num_samples"]),
            "aggregation": config["evaluation"]["aggregation"],
        })
        if context.is_main_process:
            with (output_dir / "metrics.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(result, handle, indent=2)
            print(json.dumps(result, indent=2))
            wandb_run = (
                init_wandb(config)
                if (
                    config["dataset"].get("protocol") == "lidc"
                    or source_diagnostic_enabled
                ) else None
            )
            if wandb_run is not None:
                split = config["evaluation"]["split"]
                wandb_run.log({
                    f"{split}/{key}": value
                    for key, value in result.items()
                    if isinstance(value, (int, float))
                })
                wandb_run.log({
                    f"epoch/source/{key.removeprefix('source_')}": result[key]
                    for key in DIAGNOSTIC_RESULT_KEYS if key in result
                })
        barrier(context)
        return result
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        cleanup_distributed(context)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a DFM checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    arguments = parser.parse_args()
    config = load_config(arguments.config, arguments.set)
    checkpoint = arguments.checkpoint or config["evaluation"]["checkpoint"]
    if not checkpoint:
        raise ValueError("--checkpoint or evaluation.checkpoint must be specified")
    evaluate(config, checkpoint)


if __name__ == "__main__":
    main()
