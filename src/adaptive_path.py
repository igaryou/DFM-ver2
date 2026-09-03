from __future__ import annotations

import torch
import torch.nn.functional as F


def path_config(config: dict | None) -> dict:
    """Resolve either a complete config, a flow block, or a path block."""
    if config is None:
        return {"type": "power", "exponent": 1.0}
    if "flow" in config:
        config = config["flow"]
    if "path" in config:
        config = config["path"]
    return config


def entropy_adaptive_enabled(config: dict | None) -> bool:
    return path_config(config).get("type", "power") == "entropy_adaptive"


def shannon_entropy(
    source_state: torch.Tensor,
    *,
    representation: str = "logits",
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Compute class-wise Shannon entropy in fp32 and return ``[B,H,W]``."""
    if source_state.ndim != 4:
        raise ValueError("source_state must have shape [B,C,H,W]")
    values = source_state.float()
    if representation == "logits":
        probability = torch.softmax(values, dim=1)
    elif representation == "probability":
        probability = values.clamp_min(0.0)
        probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(eps)
    else:
        raise ValueError("entropy representation must be logits or probability")
    return -(probability * probability.clamp_min(eps).log()).sum(dim=1)


def _average_rank(values: torch.Tensor) -> torch.Tensor:
    """Zero-based average ranks for one 1-D tensor, including stable ties."""
    if values.numel() <= 1:
        return torch.full_like(values, 0.5)
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    ends = counts.cumsum(0)
    starts = ends - counts
    average = (starts + ends - 1).to(values.dtype) * 0.5
    sorted_ranks = torch.repeat_interleave(average, counts)
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks / float(values.numel() - 1)


def normalize_entropy(
    entropy: torch.Tensor,
    normalization: str,
    *,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1.0e-8,
    zscore_clip: float = 3.0,
    num_classes: int = 20,
) -> torch.Tensor:
    """Normalize entropy image-wise to mean-zero difficulty in ``[-1,1]``."""
    if entropy.ndim != 3:
        raise ValueError("entropy must have shape [B,H,W]")
    if normalization not in {"mean", "zscore", "minmax", "rank"}:
        raise ValueError(f"Unknown entropy normalization: {normalization}")
    if eps <= 0.0:
        raise ValueError("entropy eps must be positive")
    if zscore_clip <= 0.0:
        raise ValueError("entropy zscore_clip must be positive")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    valid_mask = (
        torch.ones_like(entropy, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.to(device=entropy.device, dtype=torch.bool)
    )
    if valid_mask.shape != entropy.shape:
        raise ValueError("valid_mask and entropy must have identical shapes")

    output = torch.zeros_like(entropy, dtype=torch.float32)
    entropy32 = entropy.float()
    for index in range(entropy.shape[0]):
        valid = valid_mask[index]
        values = entropy32[index][valid]
        if values.numel() == 0:
            continue
        if normalization == "mean":
            normalized = (values - values.mean()) / torch.log(
                values.new_tensor(float(num_classes))
            )
        elif normalization == "zscore":
            standard_deviation = values.std(unbiased=False)
            if standard_deviation <= eps:
                normalized = torch.zeros_like(values)
            else:
                zscore = (values - values.mean()) / (standard_deviation + eps)
                clipped = zscore.clamp(-zscore_clip, zscore_clip)
                clipped_mean = clipped.mean()
                normalized = (clipped - clipped_mean) / (
                    zscore_clip + clipped_mean.abs() + eps
                )
        elif normalization == "minmax":
            value_range = values.max() - values.min()
            if value_range <= eps:
                normalized = torch.zeros_like(values)
            else:
                percentile = (values - values.min()) / (value_range + eps)
                normalized = percentile - percentile.mean()
        else:
            normalized = 2.0 * _average_rank(values) - 1.0
            # Average ranks are analytically centered; remove fp roundoff only.
            normalized = normalized - normalized.mean()
        output[index][valid] = normalized.clamp(-1.0, 1.0)
    return output


def source_entropy_difficulty(
    source_state: torch.Tensor,
    config: dict,
    *,
    valid_mask: torch.Tensor | None = None,
    spatial_size: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build raw entropy and normalized difficulty at state resolution."""
    path = path_config(config)
    entropy_config = path["entropy"]
    source = config.get("source", {})
    representation = (
        source.get("representation", "probability")
        if source.get("type") == "task_finetuned_segformer"
        else "logits"
    )
    entropy = shannon_entropy(
        source_state, representation=representation, eps=float(entropy_config["eps"])
    )
    if spatial_size is not None and entropy.shape[-2:] != spatial_size:
        entropy = F.interpolate(
            entropy[:, None], size=spatial_size, mode="bilinear", align_corners=False
        )[:, 0]
    mask = valid_mask if entropy_config.get("exclude_ignore", True) else None
    if entropy_config.get("exclude_predicted_void", False):
        semantic_mask = source_predicted_semantic_mask(source_state, config)
        if semantic_mask.shape[-2:] != entropy.shape[-2:]:
            semantic_mask = F.interpolate(
                semantic_mask[:, None].float(), entropy.shape[-2:], mode="nearest"
            )[:, 0].bool()
        mask = semantic_mask if mask is None else mask & semantic_mask
    difficulty = normalize_entropy(
        entropy,
        entropy_config["normalization"],
        valid_mask=mask,
        eps=float(entropy_config["eps"]),
        zscore_clip=float(entropy_config["zscore_clip"]),
        num_classes=int(config["dataset"]["num_classes"]),
    )
    return entropy, difficulty


def source_predicted_semantic_mask(
    source_state: torch.Tensor, config: dict
) -> torch.Tensor:
    """Mask pixels whose source argmax is not the configured void class."""
    void_index = config["dataset"].get("void_class_index")
    if void_index is None:
        return torch.ones(
            source_state.shape[0], *source_state.shape[-2:],
            dtype=torch.bool, device=source_state.device,
        )
    return source_state.argmax(dim=1) != int(void_index)


def adaptive_lambda(
    time: torch.Tensor, difficulty: torch.Tensor, *, beta: float
) -> torch.Tensor:
    if time.ndim != 1 or difficulty.ndim != 3 or time.shape[0] != difficulty.shape[0]:
        raise ValueError("time must be [B] and difficulty must be [B,H,W]")
    t = time.float()[:, None, None]
    return t - beta * t * (1.0 - t) * difficulty.float()


def adaptive_lambda_derivative(
    time: torch.Tensor, difficulty: torch.Tensor, *, beta: float
) -> torch.Tensor:
    if time.ndim != 1 or difficulty.ndim != 3 or time.shape[0] != difficulty.shape[0]:
        raise ValueError("time must be [B] and difficulty must be [B,H,W]")
    t = time.float()[:, None, None]
    return 1.0 - beta * (1.0 - 2.0 * t) * difficulty.float()


def adaptive_path_stats(
    entropy: torch.Tensor,
    difficulty: torch.Tensor,
    time: torch.Tensor,
    coefficient: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    valid = (
        torch.ones_like(difficulty, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.bool()
    )
    zero = difficulty.sum() * 0.0

    def stat(values: torch.Tensor, operation: str) -> torch.Tensor:
        selected = values[valid]
        if not selected.numel():
            return zero
        if operation == "mean":
            return selected.mean()
        if operation == "std":
            return selected.std(unbiased=False)
        return getattr(selected, operation)()

    target = time.float()[:, None, None].expand_as(coefficient)
    easy, hard = valid & (difficulty < 0), valid & (difficulty > 0)
    return {
        "path_entropy_mean": stat(entropy, "mean"),
        "path_entropy_std": stat(entropy, "std"),
        "path_entropy_min": stat(entropy, "min"),
        "path_entropy_max": stat(entropy, "max"),
        "path_difficulty_mean": stat(difficulty, "mean"),
        "path_difficulty_std": stat(difficulty, "std"),
        "path_difficulty_min": stat(difficulty, "min"),
        "path_difficulty_max": stat(difficulty, "max"),
        "path_lambda_mean": stat(coefficient, "mean"),
        "path_lambda_min": stat(coefficient, "min"),
        "path_lambda_max": stat(coefficient, "max"),
        "path_lambda_mean_minus_t": stat(coefficient - target, "mean"),
        "path_lambda_easy_mean": coefficient[easy].mean() if easy.any() else zero,
        "path_lambda_hard_mean": coefficient[hard].mean() if hard.any() else zero,
    }
