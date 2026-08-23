from __future__ import annotations

import json
import logging
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch


class _ConsoleVisibilityFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "console", True))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not deterministic


def autocast_context(config: dict[str, Any], device: torch.device):
    if not config["runtime"]["amp"] or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if config["runtime"]["amp_dtype"] == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_grad_scaler(config: dict[str, Any], device: torch.device):
    enabled = (
        config["runtime"]["amp"]
        and config["runtime"]["amp_dtype"] == "fp16"
        and device.type == "cuda"
    )
    return torch.amp.GradScaler("cuda", enabled=enabled)


def setup_logger(output_dir: str | Path) -> logging.Logger:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"dfm.{output.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        stream.addFilter(_ConsoleVisibilityFilter())
        file_handler = logging.FileHandler(output / "train_log.txt", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(file_handler)
    return logger


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    serializable = {
        key: (float(value.detach().cpu()) if torch.is_tensor(value) and value.numel() == 1 else value)
        for key, value in payload.items()
    }
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(serializable, ensure_ascii=False, default=str) + "\n")


class AverageMeter:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                value = value.detach().float().cpu().item()
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                self.sums[key] = self.sums.get(key, 0.0) + float(value)
                self.counts[key] = self.counts.get(key, 0) + 1

    def compute(self) -> dict[str, float]:
        return {key: total / self.counts[key] for key, total in self.sums.items()}


def init_wandb(config: dict[str, Any]):
    if not config["wandb"]["enabled"]:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb.enabled=true but wandb is not installed") from exc
    return wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        name=config["wandb"]["name"] or config["experiment"]["name"],
        mode=config["wandb"]["mode"],
        tags=config["wandb"]["tags"],
        config=config,
    )
