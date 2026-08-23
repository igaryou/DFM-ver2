from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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
from inference import sample_segmentation, terminal_state_to_original_prediction
from metrics import SegmentationMetrics
from model_factory import build_models
from utils import autocast_context, seed_everything
from visualization import save_prediction


@torch.no_grad()
def evaluate(config: dict, checkpoint_path: str | Path) -> dict:
    context = setup_distributed(config)
    try:
        assert_config_equal_across_ranks(config, context)
        device = context.device
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
                if config["dataset"]["name"] == "ade20k" else None
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
        for batch_index, batch in enumerate(loader):
            maximum_batches = config["evaluation"]["max_batches"]
            if maximum_batches is not None and batch_index >= maximum_batches:
                break
            if config["dataset"]["name"] == "ade20k":
                items = []
                for sample in batch:
                    image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
                    target = sample["target"].unsqueeze(0).to(device, non_blocking=True)
                    with autocast_context(config, device):
                        terminal = sample_segmentation(
                            model, source_model, image, config,
                            return_terminal_state=True,
                        )
                    prediction = terminal_state_to_original_prediction(
                        terminal, sample["model_shape"], sample["original_shape"],
                        padded_shape=sample["padded_shape"],
                        align_corners=config["evaluation"]["align_corners"],
                    )
                    metrics.update(prediction, target)
                    items.append((image, target, prediction))
            else:
                image, target = batch
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with autocast_context(config, device):
                    prediction = sample_segmentation(model, source_model, image, config)
                metrics.update(prediction, target)
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
                        )
                        visualized += 1
                        remaining -= 1
        metrics.confusion_matrix = all_reduce_confusion_matrix(
            metrics.confusion_matrix, context
        )
        result = metrics.compute()
        if context.is_main_process:
            with (output_dir / "metrics.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(result, handle, indent=2)
            print(json.dumps(result, indent=2))
        barrier(context)
        return result
    finally:
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
