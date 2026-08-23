from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sized

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_main_process: bool
    backend: str | None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def setup_distributed(
    config: dict | None = None,
    *,
    allow_cpu_distributed: bool = False,
    backend: str | None = None,
) -> DistributedContext:
    """Initialize env:// DDP when requested and WORLD_SIZE is greater than one."""
    distributed_config = (config or {}).get("distributed", {})
    enabled = distributed_config.get("enabled", "auto")
    init_method = distributed_config.get("init_method", "env://")
    requested_backend = backend or distributed_config.get("backend")
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)

    if enabled not in {"auto", True, False}:
        raise ValueError("distributed.enabled must be auto, true, or false")
    if world_size < 1:
        raise RuntimeError(f"WORLD_SIZE must be >= 1, got {world_size}")
    if not (0 <= rank < world_size):
        raise RuntimeError(f"RANK={rank} is invalid for WORLD_SIZE={world_size}")
    if local_rank < 0:
        raise RuntimeError(f"LOCAL_RANK must be >= 0, got {local_rank}")
    if enabled is False and world_size > 1:
        raise RuntimeError("WORLD_SIZE > 1 conflicts with distributed.enabled=false")
    use_distributed = world_size > 1 and enabled in {"auto", True}

    runtime_device = (config or {}).get("runtime", {}).get("device", "auto")
    if not use_distributed:
        if runtime_device == "auto":
            device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = torch.device(runtime_device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            torch.cuda.set_device(device)
        return DistributedContext(False, 0, 0, 1, device, True, None)

    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    selected_backend = requested_backend or (
        "nccl" if torch.cuda.is_available() else "gloo"
    )
    use_cpu_gloo = allow_cpu_distributed and selected_backend == "gloo"
    if torch.cuda.is_available() and not use_cpu_gloo:
        if selected_backend == "nccl" and not dist.is_nccl_available():
            raise RuntimeError("NCCL is unavailable")
        device_count = torch.cuda.device_count()
        if world_size > device_count:
            raise RuntimeError(
                f"WORLD_SIZE={world_size} exceeds visible CUDA devices={device_count}"
            )
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is invalid for {device_count} CUDA devices"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if not allow_cpu_distributed:
            raise RuntimeError("Distributed training requires CUDA/NCCL")
        if selected_backend != "gloo":
            raise RuntimeError("CPU distributed tests require backend=gloo")
        device = torch.device("cpu")

    init_kwargs = {}
    if device.type == "cuda" and selected_backend == "nccl":
        init_kwargs["device_id"] = device
    try:
        dist.init_process_group(
            backend=selected_backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            **init_kwargs,
        )
    except Exception:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    return DistributedContext(
        True, rank, local_rank, world_size, device, rank == 0, selected_backend
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    if not context.distributed:
        return
    if not dist.is_initialized():
        raise RuntimeError("barrier requested without an initialized process group")
    if context.backend == "nccl":
        dist.barrier(device_ids=[context.local_rank])
    else:
        dist.barrier()


def validate_global_batch_size(global_batch_size: int, world_size: int) -> int:
    if global_batch_size <= 0:
        raise ValueError("training.batch_size must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if global_batch_size % world_size:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"world size {world_size}"
        )
    return global_batch_size // world_size


def wrap_ddp(module: torch.nn.Module, context: DistributedContext, config: dict):
    if not context.distributed:
        return module
    options = config["distributed"]
    cuda_kwargs = {}
    if context.device.type == "cuda":
        cuda_kwargs = {
            "device_ids": [context.local_rank],
            "output_device": context.local_rank,
        }
    return DistributedDataParallel(
        module,
        broadcast_buffers=options["broadcast_buffers"],
        find_unused_parameters=options["find_unused_parameters"],
        gradient_as_bucket_view=options["gradient_as_bucket_view"],
        **cuda_kwargs,
    )


def unwrap_model(module):
    current = module
    while isinstance(current, DistributedDataParallel):
        current = current.module
    return current


def reduce_scalar(
    value: torch.Tensor | float,
    context: DistributedContext,
    reduction: str = "mean",
) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.detach().clone()
        if tensor.dtype != torch.float64:
            tensor = tensor.float()
    else:
        tensor = torch.tensor(float(value), device=context.device)
    if not context.distributed:
        return tensor
    if reduction in {"sum", "mean"}:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if reduction == "mean":
            tensor /= context.world_size
    elif reduction == "max":
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    elif reduction == "min":
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    else:
        raise ValueError(f"Unknown distributed reduction: {reduction}")
    return tensor


def reduce_metric_dict(
    metrics: Mapping[str, torch.Tensor | float],
    context: DistributedContext,
    *,
    min_keys: set[str] | None = None,
    max_keys: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    min_keys = min_keys or set()
    max_keys = max_keys or set()
    overlap = min_keys & max_keys
    if overlap:
        raise ValueError(
            "Metric keys cannot use both min and max reduction: "
            f"{sorted(overlap)}"
        )
    reduced = {}
    for key, value in metrics.items():
        if key in min_keys:
            reduction = "min"
        elif key in max_keys:
            reduction = "max"
        else:
            reduction = "mean"
        reduced[key] = reduce_scalar(value, context, reduction)
    return reduced


class EpochMetricMeter:
    """Accumulate detached scalar tensors without synchronizing every batch."""

    def __init__(
        self,
        *,
        min_keys: set[str] | None = None,
        max_keys: set[str] | None = None,
    ) -> None:
        self.min_keys = min_keys or set()
        self.max_keys = max_keys or set()
        overlap = self.min_keys & self.max_keys
        if overlap:
            raise ValueError(
                "Metric keys cannot use both min and max reduction: "
                f"{sorted(overlap)}"
            )
        self.sums: dict[str, Any] = {}
        self.counts: dict[str, Any] = {}
        self.minima: dict[str, Any] = {}
        self.maxima: dict[str, Any] = {}

    def update(self, metrics: Mapping[str, Any]) -> None:
        for key, value in metrics.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                scalar = value.detach().float()
                finite = torch.isfinite(scalar)
                valid_count = finite.to(dtype=torch.float64)
            elif isinstance(value, (int, float)):
                scalar = float(value)
                if not math.isfinite(scalar):
                    continue
                valid_count = 1
            else:
                continue
            if key in self.min_keys:
                if torch.is_tensor(scalar):
                    scalar = torch.where(
                        finite, scalar, torch.full_like(scalar, float("inf"))
                    )
                    previous = self.minima.get(key)
                    self.minima[key] = (
                        scalar.clone()
                        if previous is None
                        else torch.minimum(previous, scalar)
                    )
                else:
                    self.minima[key] = min(self.minima.get(key, scalar), scalar)
            elif key in self.max_keys:
                if torch.is_tensor(scalar):
                    scalar = torch.where(
                        finite, scalar, torch.full_like(scalar, float("-inf"))
                    )
                    previous = self.maxima.get(key)
                    self.maxima[key] = (
                        scalar.clone()
                        if previous is None
                        else torch.maximum(previous, scalar)
                    )
                else:
                    self.maxima[key] = max(self.maxima.get(key, scalar), scalar)
            else:
                if torch.is_tensor(scalar):
                    scalar = torch.where(finite, scalar, torch.zeros_like(scalar))
                    previous_sum = self.sums.get(key)
                    previous_count = self.counts.get(key)
                    self.sums[key] = (
                        scalar.clone()
                        if previous_sum is None
                        else previous_sum + scalar
                    )
                    self.counts[key] = (
                        valid_count.clone()
                        if previous_count is None
                        else previous_count + valid_count
                    )
                else:
                    self.sums[key] = self.sums.get(key, 0.0) + scalar
                    self.counts[key] = self.counts.get(key, 0) + valid_count

    def compute(self) -> dict[str, float]:
        def scalar_float(value: Any) -> float:
            if torch.is_tensor(value):
                return float(value.detach().cpu())
            return float(value)

        means = {}
        for key, total in self.sums.items():
            count = scalar_float(self.counts[key])
            if count > 0:
                means[key] = scalar_float(total) / count
        minima = {
            key: scalar_float(value)
            for key, value in self.minima.items()
            if math.isfinite(scalar_float(value))
        }
        maxima = {
            key: scalar_float(value)
            for key, value in self.maxima.items()
            if math.isfinite(scalar_float(value))
        }
        return means | minima | maxima


def _reduction_tensor(value: Any, context: DistributedContext) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device=context.device, dtype=torch.float64)
    return torch.tensor(float(value), device=context.device, dtype=torch.float64)


def reduce_epoch_metric_meter(
    meter: EpochMetricMeter,
    context: DistributedContext,
) -> dict[str, torch.Tensor]:
    """Reduce all epoch means in one SUM, plus one MIN/MAX for extrema."""
    result: dict[str, torch.Tensor] = {}
    mean_keys = sorted(meter.sums)
    if mean_keys:
        packed = torch.stack(
            [_reduction_tensor(meter.sums[key], context) for key in mean_keys]
            + [_reduction_tensor(meter.counts[key], context) for key in mean_keys]
        )
        if context.distributed:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        sums, counts = packed[:len(mean_keys)], packed[len(mean_keys):]
        for key, total, count in zip(mean_keys, sums, counts, strict=True):
            if float(count.detach().cpu()) > 0:
                result[key] = total / count

    min_keys = sorted(meter.minima)
    if min_keys:
        minima = torch.stack([
            _reduction_tensor(meter.minima[key], context) for key in min_keys
        ])
        if context.distributed:
            dist.all_reduce(minima, op=dist.ReduceOp.MIN)
        for key, value in zip(min_keys, minima, strict=True):
            if bool(torch.isfinite(value).detach().cpu()):
                result[key] = value

    max_keys = sorted(meter.maxima)
    if max_keys:
        maxima = torch.stack([
            _reduction_tensor(meter.maxima[key], context) for key in max_keys
        ])
        if context.distributed:
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        for key, value in zip(max_keys, maxima, strict=True):
            if bool(torch.isfinite(value).detach().cpu()):
                result[key] = value
    return result


def reduce_max_values(
    values: list[float],
    context: DistributedContext,
) -> torch.Tensor:
    packed = torch.tensor(values, device=context.device, dtype=torch.float64)
    if context.distributed:
        dist.all_reduce(packed, op=dist.ReduceOp.MAX)
    return packed


def all_reduce_confusion_matrix(
    confusion_matrix: torch.Tensor, context: DistributedContext
) -> torch.Tensor:
    result = confusion_matrix.clone()
    if context.distributed:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def assert_config_equal_across_ranks(config: dict, context: DistributedContext) -> None:
    if not context.distributed:
        return
    sanitized = json.loads(json.dumps(config, sort_keys=True, default=str))
    sanitized.get("runtime", {}).pop("config_path", None)
    digest = hashlib.sha256(
        json.dumps(sanitized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    gathered: list[str | None] = [None] * context.world_size
    dist.all_gather_object(gathered, digest)
    if len(set(gathered)) != 1:
        raise RuntimeError(f"Resolved config differs across ranks: {gathered}")


def parameter_checksum(module: torch.nn.Module, context: DistributedContext) -> dict[str, float]:
    checksum = torch.zeros((), device=context.device, dtype=torch.float64)
    for parameter in module.parameters():
        checksum += parameter.detach().double().sum()
    minimum = reduce_scalar(checksum, context, "min")
    maximum = reduce_scalar(checksum, context, "max")
    difference = float((maximum - minimum).abs().cpu())
    if difference > 1.0e-7:
        raise RuntimeError(f"DDP parameter checksum mismatch: max-min={difference}")
    return {"parameter_checksum": float(checksum.cpu()), "checksum_max_diff": difference}


def seed_data_loader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class DistributedEvalSampler(Sampler[int]):
    """Non-padding sampler: each evaluation index occurs on exactly one rank."""

    def __init__(self, dataset: Sized, *, rank: int, world_size: int) -> None:
        if not (0 <= rank < world_size):
            raise ValueError(f"Invalid rank={rank} for world_size={world_size}")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        return math.ceil(max(len(self.dataset) - self.rank, 0) / self.world_size)
