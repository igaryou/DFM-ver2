from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from state_space import state_spatial_size


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=time.device, dtype=torch.float32) * -scale
        )
        values = time.float()[:, None] * frequencies[None]
        embedding = torch.cat((values.sin(), values.cos()), dim=1)
        return F.pad(embedding, (0, self.dim - embedding.shape[1]))


class DenseResidualBlock(nn.Module):
    def __init__(self, channels: int, growth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Conv2d(channels + index * growth, growth, 3, padding=1)
            for index in range(4)
        )
        self.final = nn.Conv2d(channels + 4 * growth, channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = [inputs]
        for layer in self.layers:
            features.append(F.leaky_relu(layer(torch.cat(features, dim=1)), 0.2))
        return inputs + 0.2 * self.final(torch.cat(features, dim=1))


class RRDB(nn.Module):
    def __init__(self, channels: int, growth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(DenseResidualBlock(channels, growth) for _ in range(3))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for block in self.blocks:
            output = block(output)
        return inputs + 0.2 * output


class ImageEncoder(nn.Module):
    """Downsample inside the encoder, then run the original RRDB trunk."""

    def __init__(
        self, channels: int, blocks: int, growth: int, downsample_factor: int = 4
    ) -> None:
        super().__init__()
        stages = int(math.log2(downsample_factor))
        if 2**stages != downsample_factor:
            raise ValueError("ImageEncoder downsample_factor must be a power of two")
        self.downsample_factor = downsample_factor
        self.first = nn.Conv2d(
            3, channels, 3, stride=2 if stages else 1, padding=1
        )
        self.downsample = nn.ModuleList(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1)
            for _ in range(max(stages - 1, 0))
        )
        self.body = nn.Sequential(*(RRDB(channels, growth) for _ in range(blocks)))
        self.body_out = nn.Conv2d(channels, channels, 3, padding=1)
        self.out = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        first = F.leaky_relu(self.first(image), 0.2)
        for layer in self.downsample:
            first = F.leaky_relu(layer(first), 0.2)
        return self.out(F.leaky_relu(first + self.body_out(self.body(first)), 0.2))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float):
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.norm2 = group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = hidden + self.time(time_embedding)[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + self.skip(inputs)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("attention channels must be divisible by num_heads")
        self.heads = heads
        self.norm = group_norm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        q, k, v = self.qkv(self.norm(inputs).flatten(2)).chunk(3, dim=1)
        head_dim = channels // self.heads
        q = q.view(batch, self.heads, head_dim, -1)
        k = k.view(batch, self.heads, head_dim, -1)
        v = v.view(batch, self.heads, head_dim, -1)
        attention = torch.einsum("bnci,bncj->bnij", q * head_dim**-0.5, k).softmax(-1)
        output = torch.einsum("bnij,bncj->bnci", attention, v).reshape(batch, channels, -1)
        return inputs + self.proj(output).view(batch, channels, height, width)


class UNetBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, time_dim: int,
        attention: bool, dropout: float, heads: int,
    ) -> None:
        super().__init__()
        self.residual = ResidualBlock(in_channels, out_channels, time_dim, dropout)
        self.attention = AttentionBlock(out_channels, heads) if attention else nn.Identity()

    def forward(self, inputs: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        return self.attention(self.residual(inputs, time_embedding))


class DFMUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        channel_mults: Iterable[int],
        num_res_blocks: int,
        attention_levels: Iterable[int],
        num_heads: int,
        dropout: float,
        time_dim: int,
    ) -> None:
        super().__init__()
        channel_mults = tuple(channel_mults)
        attention_levels = set(attention_levels)
        self.embed_s = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels), nn.Linear(base_channels, time_dim),
            nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        self.embed_delta = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels), nn.Linear(base_channels, time_dim),
            nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        self.input = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = base_channels
        skip_channels: list[int] = []
        for level, multiplier in enumerate(channel_mults):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                output_channels = base_channels * multiplier
                blocks.append(UNetBlock(
                    channels, output_channels, time_dim, level in attention_levels,
                    dropout, num_heads,
                ))
                channels = output_channels
                skip_channels.append(channels)
            self.down_blocks.append(blocks)
            self.downsamples.append(
                nn.Conv2d(channels, channels, 3, stride=2, padding=1)
                if level < len(channel_mults) - 1 else nn.Identity()
            )
        self.middle = nn.ModuleList((
            UNetBlock(
                channels, channels, time_dim,
                len(channel_mults) - 1 in attention_levels, dropout, num_heads,
            ),
            UNetBlock(channels, channels, time_dim, False, dropout, num_heads),
        ))
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        reversed_skips = list(reversed(skip_channels))
        for reverse_level, multiplier in enumerate(reversed(channel_mults)):
            blocks = nn.ModuleList()
            original_level = len(channel_mults) - 1 - reverse_level
            for _ in range(num_res_blocks):
                output_channels = base_channels * multiplier
                blocks.append(UNetBlock(
                    channels + reversed_skips.pop(0), output_channels, time_dim,
                    original_level in attention_levels, dropout, num_heads,
                ))
                channels = output_channels
            self.up_blocks.append(blocks)
            self.upsamples.append(
                nn.Conv2d(channels, channels, 3, padding=1)
                if reverse_level < len(channel_mults) - 1 else nn.Identity()
            )
        self.out_norm = group_norm(channels)
        self.out = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_embedding = self.embed_s(s) + self.embed_delta(t - s)
        hidden = self.input(inputs)
        skips = []
        for blocks, downsample in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                hidden = block(hidden, time_embedding)
                skips.append(hidden)
            hidden = downsample(hidden)
        for block in self.middle:
            hidden = block(hidden, time_embedding)
        for blocks, upsample in zip(self.up_blocks, self.upsamples):
            for block in blocks:
                skip = skips.pop()
                if hidden.shape[-2:] != skip.shape[-2:]:
                    hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
                hidden = block(torch.cat((hidden, skip), dim=1), time_embedding)
            if not isinstance(upsample, nn.Identity):
                hidden = upsample(F.interpolate(hidden, scale_factor=2, mode="nearest"))
        return self.out(F.silu(self.out_norm(hidden)))


class DiscreteFlowMapModel(nn.Module):
    """Image-conditioned mean-denoiser logits z_theta(x_s, I, s, t)."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        channels = config["fusion_channels"]
        num_classes = config["num_classes"]
        unet = config["unet"]
        self.num_classes = num_classes
        self.state_downsample_factor = config.get("state_downsample_factor", 4)
        self.mask_encoder = nn.Conv2d(num_classes, channels, 3, padding=1)
        self.image_encoder = ImageEncoder(
            channels,
            config["rrdb_blocks"],
            config["rrdb_growth_channels"],
            self.state_downsample_factor,
        )
        self.unet = DFMUNet(
            in_channels=channels,
            out_channels=num_classes,
            base_channels=unet["base_channels"],
            channel_mults=unet["channel_mults"],
            num_res_blocks=unet["num_res_blocks"],
            attention_levels=unet["attention_levels"],
            num_heads=unet["num_heads"],
            dropout=unet["dropout"],
            time_dim=unet["time_embedding_dim"],
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.image_encoder(image)
        expected = state_spatial_size(image, self.state_downsample_factor)
        assert feature.shape[-2:] == expected, (
            f"image feature {feature.shape[-2:]} != state size {expected}"
        )
        return feature

    def forward_logits_with_image_feat(
        self, x_s: torch.Tensor, image_feat: torch.Tensor,
        s: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        assert x_s.shape[0] == image_feat.shape[0]
        assert x_s.shape[1] == self.num_classes
        assert x_s.shape[-2:] == image_feat.shape[-2:], (
            f"state {x_s.shape[-2:]} != image feature {image_feat.shape[-2:]}"
        )
        logits = self.unet(self.mask_encoder(x_s) + image_feat, s, t)
        assert logits.shape == x_s.shape, (
            f"endpoint logits {tuple(logits.shape)} != state {tuple(x_s.shape)}"
        )
        return logits

    def forward_logits(
        self, x_s: torch.Tensor, image: torch.Tensor,
        s: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_logits_with_image_feat(x_s, self.encode_image(image), s, t)

    def forward(
        self, x_s: torch.Tensor, image: torch.Tensor,
        s: torch.Tensor, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward_logits(x_s, image, s, t)
        return logits, torch.softmax(logits, dim=1)
