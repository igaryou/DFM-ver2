from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from config import load_config
from dataset import build_dataset
from inference import sample_segmentation
from model_factory import build_models
from training_objectives import (
    DDPCompatibleTrainingModel,
    compute_model_training_objectives,
)


def _gradient_norm(parameters) -> float:
    squared = sum(
        float(parameter.grad.detach().float().square().sum())
        for parameter in parameters
        if parameter.grad is not None
    )
    return squared**0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real LIDC joint PSD step")
    parser.add_argument(
        "--config",
        default="configs/lidc/psd/segformer_source_align_joint.yaml",
    )
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    device = torch.device(arguments.device)
    sample = build_dataset(
        config, config["dataset"]["train_split"], augment=False
    )[0]
    image = sample["image"][None].to(device)
    target = sample["target"][None].to(device)
    masks = sample["masks"][None].to(device)
    endpoint, source = build_models(config, device)
    adapter = DDPCompatibleTrainingModel(endpoint, source, config)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1.0e-4)

    with torch.no_grad():
        x0, mu, logvar = source(image)
        endpoint_logits, _ = endpoint(
            x0, image, torch.tensor([0.25], device=device),
            torch.tensor([0.75], device=device),
        )
    result = compute_model_training_objectives(
        adapter,
        operation="joint_objectives",
        image=image,
        target=target,
        spatial_valid_mask=torch.ones_like(target, dtype=torch.bool),
        epoch_index=0,
        progress_in_epoch=0.0,
        optimizer_step=0,
    )
    optimizer.zero_grad(set_to_none=True)
    result["loss"].backward()
    source_grad = _gradient_norm(source.parameters())
    rrdb_grad = _gradient_norm(endpoint.image_encoder.parameters())
    endpoint_grad = _gradient_norm(
        parameter for name, parameter in endpoint.named_parameters()
        if not name.startswith("image_encoder.")
    )
    optimizer.step()
    terminal = sample_segmentation(
        endpoint, source, image, config, num_steps=1,
        return_terminal_state=True,
    )
    prediction = terminal.argmax(dim=1)
    report = {
        "shapes": {
            "image": list(image.shape),
            "gt": list(target.shape),
            "gt_all": list(masks.shape),
            "source_mu": list(mu.shape),
            "source_logvar": list(logvar.shape),
            "source_x0": list(x0.shape),
            "endpoint": list(endpoint_logits.shape),
            "terminal": list(terminal.shape),
            "prediction": list(prediction.shape),
        },
        "losses": {
            "primary": float(result["diagonal_objective"].detach()),
            "psd": float(result["psd_objective"].detach()),
            "align": float(result["source_objective"].detach()),
            "total": float(result["loss"].detach()),
        },
        "gradient_norms": {
            "source_segformer": source_grad,
            "rrdb": rrdb_grad,
            "endpoint_unet": endpoint_grad,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
