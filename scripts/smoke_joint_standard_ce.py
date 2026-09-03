#!/usr/bin/env python3
"""One-step bf16 Joint PSD + standard SegFormer source CE smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from model_factory import build_models  # noqa: E402
from trainer import build_optimizer  # noqa: E402
from training_objectives import DDPCompatibleTrainingModel  # noqa: E402
from utils import autocast_context  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    config = load_config(
        ROOT
        / "configs/cityscapes/psd/joint_swin_t_segformer_b1_standard_ce_160k.yaml",
        [f"training.batch_size={args.batch_size}"],
    )
    torch.manual_seed(config["experiment"]["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["experiment"]["seed"])
    endpoint, source = build_models(config, device)
    adapter = DDPCompatibleTrainingModel(endpoint, source, config).to(device).train()
    optimizer = build_optimizer(config, adapter)
    captured = {}

    def capture_source_output(_module, _inputs, output):
        x0, mu, logvar = output
        captured.update({
            "x0_shape": list(x0.shape),
            "mu_shape": list(mu.shape),
            "logvar_shape": list(logvar.shape),
            "sample_diff_abs": float((x0 - mu).detach().float().abs().mean().cpu()),
        })

    hook = source.register_forward_hook(capture_source_output)
    image = torch.randn(
        args.batch_size, 3, args.height, args.width, device=device
    )
    target = torch.randint(
        0,
        config["dataset"]["num_classes"],
        (args.batch_size, args.height, args.width),
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(config, device):
        result = adapter(
            operation="joint_objectives",
            image=image,
            target=target,
            epoch_index=0,
            progress_in_epoch=0.0,
            optimizer_step=0,
        )
    hook.remove()
    loss = result["loss"]
    loss.backward()

    endpoint_gradient = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in endpoint.parameters() if parameter.requires_grad
    )
    encoder_gradient = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in source.encoder.parameters() if parameter.requires_grad
    )
    head_gradient = source.decode_head.classifier.weight.grad
    head_gradient_ok = (
        head_gradient is not None and torch.isfinite(head_gradient).all().item()
    )
    finite_scalars = all(
        torch.isfinite(value.detach().float()).all().item()
        for value in result["stats"].values()
        if torch.is_tensor(value)
    )
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    stats = result["stats"]
    report = {
        "batch_size": args.batch_size,
        "input_shape": list(image.shape),
        **captured,
        "loss_total": float(stats["loss_total"].float().cpu()),
        "loss_diagonal": float(stats["loss_diagonal"].float().cpu()),
        "loss_consistency": float(stats["loss_consistency"].float().cpu()),
        "loss_source_ce": float(stats["loss_source_ce"].float().cpu()),
        "weighted_source_supervision": float(
            stats["weighted_source_supervision"].float().cpu()
        ),
        "source_mu_abs": float(stats["source_mu_abs"].float().cpu()),
        "source_mu_min": float(stats["source_mu_min"].float().cpu()),
        "source_mu_max": float(stats["source_mu_max"].float().cpu()),
        "source_sigma_mean": float(stats["source_sigma_mean"].float().cpu()),
        "endpoint_gradient": endpoint_gradient,
        "source_encoder_gradient": encoder_gradient,
        "standard_head_classifier_gradient": head_gradient_ok,
        "all_scalar_stats_finite": finite_scalars,
        "optimizer_step": True,
        "standard_head_parameters": sum(
            parameter.numel() for parameter in source.decode_head.parameters()
        ),
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else 0.0
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / 1024**2
            if device.type == "cuda" else 0.0
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
