from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn.functional as F


DIAGNOSTIC_RESULT_KEYS = (
    "source_noise_rms_mean",
    "source_noise_rms_std",
    "source_mu_rms_mean",
    "source_noise_signal_ratio_mean",
    "source_noise_signal_ratio_std",
    "source_noise_signal_ratio_median",
    "source_noise_signal_ratio_p95",
    "source_mu_x0_cosine_mean",
    "source_direction_change_mean",
    "source_direction_change_std",
    "source_argmax_flip_rate",
    "source_margin_mean",
    "source_margin_sigma_ratio_mean",
    "source_margin_sigma_ratio_median",
    "source_margin_sigma_ratio_p05",
    "source_sigma_mean_flip",
    "source_sigma_mean_no_flip",
    "source_margin_mean_flip",
    "source_margin_mean_no_flip",
    "source_noise_signal_ratio_mean_flip",
    "source_noise_signal_ratio_mean_no_flip",
)


def source_noise_diagnostics_enabled(config: dict) -> bool:
    return bool(config.get("source", {}).get("diagnostics", {}).get("enabled", False)) and (
        config.get("source", {}).get("prior_type")
        in {"gaussian", "image_gaussian"}
    )


def source_noise_diagnostic_maps(
    x0: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    """Measure the exact sampled perturbation ``x0 - mu`` at source resolution."""
    if x0.ndim != 4 or mu.shape != x0.shape:
        raise ValueError("x0 and mu must have identical [B,C,H,W] shapes")
    if x0.shape[1] < 2:
        raise ValueError("source diagnostics require at least two classes")
    if sigma.ndim == 0:
        sigma = sigma.reshape(1, 1, 1, 1).expand(x0.shape[0], 1, *x0.shape[-2:])
    if sigma.ndim != 4 or sigma.shape[0] != x0.shape[0] or sigma.shape[-2:] != x0.shape[-2:]:
        raise ValueError("sigma must be scalar or have shape [B,1|C,H,W]")
    if sigma.shape[1] not in {1, x0.shape[1]}:
        raise ValueError("sigma channels must be one or match x0 channels")

    x0_fp32 = x0.detach().float()
    mu_fp32 = mu.detach().float()
    sigma_fp32 = sigma.detach().float()
    # Do not reconstruct epsilon or draw diagnostic noise. This is exactly the
    # perturbation consumed by Flow Map inference.
    noise = x0_fp32 - mu_fp32
    noise_rms = noise.square().mean(dim=1).sqrt()
    mu_rms = mu_fp32.square().mean(dim=1).sqrt()
    ratio = noise_rms / (mu_rms + float(eps))
    cosine = F.cosine_similarity(mu_fp32, x0_fp32, dim=1, eps=float(eps))
    identical = noise.square().sum(dim=1) == 0
    cosine = torch.where(identical, torch.ones_like(cosine), cosine)
    direction_change = 1.0 - cosine
    mu_prediction = mu_fp32.argmax(dim=1)
    x0_prediction = x0_fp32.argmax(dim=1)
    flip_map = mu_prediction != x0_prediction
    top2 = mu_fp32.topk(2, dim=1)
    margin = top2.values[:, 0] - top2.values[:, 1]

    if sigma_fp32.shape[1] == 1:
        sigma_pixel = sigma_fp32[:, 0]
        difference_std = math.sqrt(2.0) * sigma_pixel
    else:
        # For channelwise independent Gaussian noise, the top1-minus-top2
        # perturbation has std sqrt(sigma_top1^2 + sigma_top2^2).
        top_sigma = sigma_fp32.gather(1, top2.indices)
        difference_std = top_sigma.square().sum(dim=1).sqrt()
        sigma_pixel = sigma_fp32.square().mean(dim=1).sqrt()
    margin_sigma_ratio = margin / (difference_std + float(eps))

    return {
        "noise": noise,
        "noise_rms": noise_rms,
        "mu_rms": mu_rms,
        "noise_signal_ratio": ratio,
        "mu_x0_cosine": cosine,
        "direction_change": direction_change,
        "mu_prediction": mu_prediction,
        "x0_prediction": x0_prediction,
        "flip_map": flip_map,
        "margin": margin,
        "sigma_pixel": sigma_pixel,
        "margin_sigma_ratio": margin_sigma_ratio,
    }


@dataclass
class _Moments:
    total: float = 0.0
    square_total: float = 0.0
    count: int = 0

    def update(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        values = values.detach().float()
        if mask is not None:
            values = values[mask]
        if not values.numel():
            return
        self.total += float(values.double().sum().cpu())
        self.square_total += float(values.double().square().sum().cpu())
        self.count += values.numel()


@dataclass
class _BoundedSamples:
    max_per_update: int = 2048
    max_total: int = 32768
    chunks: list[torch.Tensor] = field(default_factory=list)
    count: int = 0

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().float().reshape(-1)
        if not values.numel() or self.count >= self.max_total:
            return
        take = min(values.numel(), self.max_per_update, self.max_total - self.count)
        if take < values.numel():
            indices = torch.linspace(0, values.numel() - 1, take, device=values.device).long()
            values = values[indices]
        self.chunks.append(values.cpu())
        self.count += values.numel()

    def tensor(self) -> torch.Tensor:
        return torch.cat(self.chunks) if self.chunks else torch.empty(0)


class SourceNoiseDiagnosticsAccumulator:
    """Exact distributed means plus bounded, pooled percentile samples."""

    _MOMENT_NAMES = (
        "noise_rms", "mu_rms", "noise_signal_ratio", "mu_x0_cosine",
        "direction_change", "flip", "margin", "margin_sigma_ratio",
        "sigma_flip", "sigma_no_flip", "margin_flip", "margin_no_flip",
        "ratio_flip", "ratio_no_flip",
    )

    def __init__(self) -> None:
        self.moments = {name: _Moments() for name in self._MOMENT_NAMES}
        self.ratio_samples = _BoundedSamples()
        self.margin_sigma_samples = _BoundedSamples()

    def update(self, maps: dict[str, torch.Tensor]) -> None:
        flip = maps["flip_map"]
        no_flip = ~flip
        for name in (
            "noise_rms", "mu_rms", "noise_signal_ratio", "mu_x0_cosine",
            "direction_change", "margin", "margin_sigma_ratio",
        ):
            self.moments[name].update(maps[name])
        self.moments["flip"].update(flip.float())
        self.moments["sigma_flip"].update(maps["sigma_pixel"], flip)
        self.moments["sigma_no_flip"].update(maps["sigma_pixel"], no_flip)
        self.moments["margin_flip"].update(maps["margin"], flip)
        self.moments["margin_no_flip"].update(maps["margin"], no_flip)
        self.moments["ratio_flip"].update(maps["noise_signal_ratio"], flip)
        self.moments["ratio_no_flip"].update(maps["noise_signal_ratio"], no_flip)
        self.ratio_samples.update(maps["noise_signal_ratio"])
        self.margin_sigma_samples.update(maps["margin_sigma_ratio"])

    @staticmethod
    def _pooled_samples(samples: torch.Tensor, distributed: bool) -> torch.Tensor:
        if not distributed:
            return samples
        gathered: list[torch.Tensor | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, samples)
        return torch.cat([value for value in gathered if value is not None])

    def compute(self, *, device: torch.device, distributed: bool = False) -> dict[str, float]:
        packed = torch.tensor(
            [value for name in self._MOMENT_NAMES for value in (
                self.moments[name].total,
                self.moments[name].square_total,
                self.moments[name].count,
            )],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        reduced: dict[str, tuple[float, float, int]] = {}
        for index, name in enumerate(self._MOMENT_NAMES):
            total, square_total, count = packed[index * 3:index * 3 + 3].tolist()
            reduced[name] = (total, square_total, int(count))

        def mean(name: str) -> float:
            total, _, count = reduced[name]
            return total / count if count else float("nan")

        def std(name: str) -> float:
            total, square_total, count = reduced[name]
            if not count:
                return float("nan")
            variance = max(square_total / count - (total / count) ** 2, 0.0)
            return math.sqrt(variance)

        ratio = self._pooled_samples(self.ratio_samples.tensor(), distributed)
        margin_sigma = self._pooled_samples(
            self.margin_sigma_samples.tensor(), distributed
        )
        ratio_quantiles = (
            torch.quantile(ratio, torch.tensor([0.5, 0.95]))
            if ratio.numel() else torch.full((2,), float("nan"))
        )
        margin_quantiles = (
            torch.quantile(margin_sigma, torch.tensor([0.05, 0.5]))
            if margin_sigma.numel() else torch.full((2,), float("nan"))
        )
        return {
            "source_noise_rms_mean": mean("noise_rms"),
            "source_noise_rms_std": std("noise_rms"),
            "source_mu_rms_mean": mean("mu_rms"),
            "source_noise_signal_ratio_mean": mean("noise_signal_ratio"),
            "source_noise_signal_ratio_std": std("noise_signal_ratio"),
            "source_noise_signal_ratio_median": float(ratio_quantiles[0]),
            "source_noise_signal_ratio_p95": float(ratio_quantiles[1]),
            "source_mu_x0_cosine_mean": mean("mu_x0_cosine"),
            "source_direction_change_mean": mean("direction_change"),
            "source_direction_change_std": std("direction_change"),
            "source_argmax_flip_rate": mean("flip"),
            "source_margin_mean": mean("margin"),
            "source_margin_sigma_ratio_mean": mean("margin_sigma_ratio"),
            "source_margin_sigma_ratio_median": float(margin_quantiles[1]),
            "source_margin_sigma_ratio_p05": float(margin_quantiles[0]),
            "source_sigma_mean_flip": mean("sigma_flip"),
            "source_sigma_mean_no_flip": mean("sigma_no_flip"),
            "source_margin_mean_flip": mean("margin_flip"),
            "source_margin_mean_no_flip": mean("margin_no_flip"),
            "source_noise_signal_ratio_mean_flip": mean("ratio_flip"),
            "source_noise_signal_ratio_mean_no_flip": mean("ratio_no_flip"),
        }
