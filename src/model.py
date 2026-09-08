from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from state_space import state_spatial_size
from segformer_architecture import build_segformer_config


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


class StateResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.conv2(F.silu(self.conv1(F.silu(inputs))))


class FullResolutionStateEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: int, blocks: int) -> None:
        super().__init__()
        self.input = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.blocks = nn.Sequential(*(
            StateResidualBlock(channels) for _ in range(blocks)
        ))
        self.output_channels = channels

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.blocks(F.silu(self.input(state)))


class TimeConditionedSegFormer(nn.Module):
    """MiT encoder with stage-wise conditioning on s and t-s."""

    def __init__(
        self, variant: str, in_channels: int, time_dim: int, pretrained: bool = False,
    ) -> None:
        super().__init__()
        if pretrained:
            raise ValueError(
                "Endpoint SegFormer consumes fused features and cannot load RGB pretrained weights"
            )
        try:
            from transformers import SegformerModel
        except ImportError as exc:
            raise RuntimeError(
                "model.endpoint.type=segformer requires transformers"
            ) from exc
        config = build_segformer_config(variant, in_channels)
        self.encoder = SegformerModel(config)
        self.hidden_sizes = list(config.hidden_sizes)
        base_dim = self.hidden_sizes[0]
        self.embed_s = nn.Sequential(
            SinusoidalTimeEmbedding(base_dim), nn.Linear(base_dim, time_dim),
            nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        self.embed_delta = nn.Sequential(
            SinusoidalTimeEmbedding(base_dim), nn.Linear(base_dim, time_dim),
            nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        self.stage_time = nn.ModuleList(
            nn.Linear(time_dim, channels) for channels in self.hidden_sizes
        )

    def forward(
        self, inputs: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
    ) -> list[torch.Tensor]:
        time_embedding = self.embed_s(s) + self.embed_delta(t - s)
        hidden = inputs
        features = []
        for stage, projection in zip(self.encoder.stages, self.stage_time):
            hidden = stage(hidden)
            hidden = hidden + projection(time_embedding)[:, :, None, None].to(
                hidden.dtype
            )
            features.append(hidden)
        return features


TRANSFORMER_IMAGE_ENCODER_MODEL_IDS = {
    "swin": {
        "tiny": "microsoft/swin-tiny-patch4-window7-224",
        "small": "microsoft/swin-small-patch4-window7-224",
        "base": "microsoft/swin-base-patch4-window7-224",
        "large": "microsoft/swin-large-patch4-window7-224",
    },
    "convnext": {
        "tiny": "facebook/convnext-tiny-224",
        "small": "facebook/convnext-small-224",
        "base": "facebook/convnext-base-224",
        "large": "facebook/convnext-large-224",
    },
}

_SWIN_SPECS = {
    "tiny": ([96, 192, 384, 768], [2, 2, 6, 2], [3, 6, 12, 24]),
    "small": ([96, 192, 384, 768], [2, 2, 18, 2], [3, 6, 12, 24]),
    "base": ([128, 256, 512, 1024], [2, 2, 18, 2], [4, 8, 16, 32]),
    "large": ([192, 384, 768, 1536], [2, 2, 18, 2], [6, 12, 24, 48]),
}

_CONVNEXT_SPECS = {
    "tiny": ([96, 192, 384, 768], [3, 3, 9, 3]),
    "small": ([96, 192, 384, 768], [3, 3, 27, 3]),
    "base": ([128, 256, 512, 1024], [3, 3, 27, 3]),
    "large": ([192, 384, 768, 1536], [3, 3, 27, 3]),
}


def load_transformer_image_backbone(
    backbone_type: str, variant: str, pretrained: bool,
) -> tuple[nn.Module, list[int]]:
    """Build an HF backbone. Kept as a function so tests can replace the loader."""
    try:
        from transformers import (
            ConvNextConfig,
            ConvNextModel,
            SwinBackbone,
            SwinConfig,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"model.image_encoder.type={backbone_type} requires transformers"
        ) from exc

    model_id = TRANSFORMER_IMAGE_ENCODER_MODEL_IDS[backbone_type][variant]
    if backbone_type == "swin":
        hidden_sizes, depths, heads = _SWIN_SPECS[variant]
        if pretrained:
            backbone = SwinBackbone.from_pretrained(
                model_id,
                out_features=["stage1", "stage2", "stage3", "stage4"],
                attn_implementation="eager",
            )
        else:
            swin_config = SwinConfig(
                num_channels=3,
                patch_size=4,
                window_size=7,
                embed_dim=hidden_sizes[0],
                depths=depths,
                num_heads=heads,
                out_features=["stage1", "stage2", "stage3", "stage4"],
            )
            swin_config._attn_implementation = "eager"
            backbone = SwinBackbone(swin_config)
        hidden_sizes = list(backbone.channels)
        # SwinBackbone returns its own normalized stage4 feature and does not
        # consume SwinModel's classification-output LayerNorm. Keep those two
        # redundant parameters out of DDP/optimizer rather than reporting them
        # as trainable-but-unused.
        swin_wrapper = getattr(backbone, "swin", None)
        redundant_layernorm = getattr(swin_wrapper, "layernorm", None)
        if redundant_layernorm is not None:
            redundant_layernorm.requires_grad_(False)
    else:
        hidden_sizes, depths = _CONVNEXT_SPECS[variant]
        if pretrained:
            backbone = ConvNextModel.from_pretrained(model_id)
        else:
            backbone = ConvNextModel(ConvNextConfig(
                num_channels=3,
                hidden_sizes=hidden_sizes,
                depths=depths,
                output_hidden_states=True,
            ))
        # Multi-scale hidden states bypass the classification-output norm.
        # Preserve the existing feature path while excluding only this
        # structurally unused pair from DDP/optimizer.
        backbone.layernorm.requires_grad_(False)
    configured_sizes = list(
        backbone.channels
        if backbone_type == "swin"
        else getattr(backbone.config, "hidden_sizes", hidden_sizes)
    )
    if len(configured_sizes) != 4:
        raise ValueError(
            f"{backbone_type} backbone must expose four hidden sizes, got "
            f"{configured_sizes}"
        )
    return backbone, configured_sizes


class DDPFPNMultiStageMerging(nn.Module):
    """Lightweight FPN followed by DDP-style H/4 multi-stage merging."""

    def __init__(self, input_channels: Iterable[int], channels: int) -> None:
        super().__init__()
        input_channels = tuple(input_channels)
        if len(input_channels) != 4:
            raise ValueError("DDP FPN requires exactly four backbone stages")
        self.lateral = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(input_channel, channels, 1),
                group_norm(channels),
            )
            for input_channel in input_channels
        )
        self.fpn_output = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                group_norm(channels),
            )
            for _ in input_channels
        )
        self.merge = nn.Sequential(
            nn.Conv2d(channels * len(input_channels), channels, 1),
            group_norm(channels),
        )

    def forward(self, features: Iterable[torch.Tensor]) -> torch.Tensor:
        features = tuple(features)
        if len(features) != 4:
            raise ValueError("DDP FPN requires exactly four feature tensors")
        pyramid = [layer(feature) for layer, feature in zip(self.lateral, features)]
        for index in range(len(pyramid) - 2, -1, -1):
            pyramid[index] = pyramid[index] + F.interpolate(
                pyramid[index + 1],
                size=pyramid[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        pyramid = [layer(feature) for layer, feature in zip(self.fpn_output, pyramid)]
        target_size = pyramid[0].shape[-2:]
        merged = torch.cat([
            feature if feature.shape[-2:] == target_size else F.interpolate(
                feature, size=target_size, mode="bilinear", align_corners=False
            )
            for feature in pyramid
        ], dim=1)
        return self.merge(merged)


class TransformerImageEncoder(nn.Module):
    """Swin/ConvNeXt multi-scale encoder with an optional DDP-style neck."""

    def __init__(
        self,
        backbone_type: str,
        variant: str,
        pretrained: bool,
        freeze_backbone: bool,
        neck_type: str,
        neck_channels: int,
        fusion_channels: int,
        state_downsample_factor: int,
        input_already_normalized: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_type = backbone_type
        self.variant = variant
        self.freeze_backbone = freeze_backbone
        self.state_downsample_factor = state_downsample_factor
        self.input_already_normalized = input_already_normalized
        self.backbone, hidden_sizes = load_transformer_image_backbone(
            backbone_type, variant, pretrained
        )
        if neck_type == "ddp_fpn_merge":
            self.neck = DDPFPNMultiStageMerging(hidden_sizes, neck_channels)
            projection_input_channels = neck_channels
        else:
            self.neck = nn.Identity()
            projection_input_channels = hidden_sizes[0]
        self.projection = nn.Conv2d(projection_input_channels, fusion_channels, 1)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None],
            persistent=False,
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None],
            persistent=False,
        )
        self._hidden_sizes = hidden_sizes
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @staticmethod
    def _as_spatial_feature(
        feature: torch.Tensor,
        channels: int,
        expected_size: tuple[int, int],
    ) -> torch.Tensor:
        if feature.ndim == 4:
            if feature.shape[1] == channels:
                return feature
            if feature.shape[-1] == channels:
                return feature.permute(0, 3, 1, 2).contiguous()
        elif feature.ndim == 3 and feature.shape[-1] == channels:
            if feature.shape[1] != expected_size[0] * expected_size[1]:
                raise ValueError(
                    f"Cannot reshape hidden state {tuple(feature.shape)} to "
                    f"spatial size {expected_size}"
                )
            return feature.transpose(1, 2).reshape(
                feature.shape[0], channels, *expected_size
            )
        raise ValueError(
            f"Unsupported backbone hidden state shape {tuple(feature.shape)} "
            f"for {channels} channels"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _extract_backbone_features(
        self, backbone_input: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        if self.backbone_type == "swin":
            outputs = self.backbone(pixel_values=backbone_input, return_dict=True)
            features = tuple(outputs.feature_maps)
        else:
            outputs = self.backbone(
                pixel_values=backbone_input,
                output_hidden_states=True,
                return_dict=True,
            )
            features = tuple(outputs.hidden_states[-4:])
        if len(features) != 4:
            raise ValueError(
                f"{self.backbone_type} must return four stage features, got "
                f"{len(features)}"
            )
        return features

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        normalized = image if self.input_already_normalized else (
            image - self.mean.to(image)
        ) / self.std.to(image)
        backbone_input = normalized
        if self.backbone_type == "swin":
            # Every Swin stage needs at least one complete 7x7 window in the
            # current Transformers implementation. Production crops already
            # satisfy this; padding keeps small debug/smoke images supported.
            minimum = 4 * 2**3 * 7
            pad_height = max(minimum - image.shape[-2], 0)
            pad_width = max(minimum - image.shape[-1], 0)
            if pad_height or pad_width:
                backbone_input = F.pad(
                    backbone_input, (0, pad_width, 0, pad_height)
                )
        hidden_states = self._extract_backbone_features(backbone_input)
        features = []
        for index, (hidden, channels) in enumerate(
            zip(hidden_states, self._hidden_sizes)
        ):
            divisor = 4 * 2**index
            feature = self._as_spatial_feature(
                hidden,
                channels,
                state_spatial_size(backbone_input, divisor),
            )
            original_size = state_spatial_size(image, divisor)
            feature = feature[..., :original_size[0], :original_size[1]]
            if feature.shape[1] != channels or feature.shape[-2:] != original_size:
                raise AssertionError(
                    f"{self.backbone_type} stage {index + 1} must have shape "
                    f"[B,{channels},{original_size[0]},{original_size[1]}], got "
                    f"{tuple(feature.shape)}"
                )
            features.append(feature)
        merged = self.neck(features) if not isinstance(self.neck, nn.Identity) else features[0]
        output = self.projection(merged)
        target_size = state_spatial_size(image, self.state_downsample_factor)
        if output.shape[-2:] != target_size:
            output = F.interpolate(
                output, size=target_size, mode="bilinear", align_corners=False
            )
        return output


def build_image_encoder(config: dict) -> nn.Module:
    encoder = config.get("image_encoder")
    if encoder is None or encoder.get("type", "rrdb") == "rrdb":
        return ImageEncoder(
            config["fusion_channels"],
            config["rrdb_blocks"],
            config["rrdb_growth_channels"],
            config.get("state_downsample_factor", 4),
        )
    neck = encoder["neck"]
    return TransformerImageEncoder(
        encoder["type"],
        encoder["variant"],
        encoder["pretrained"],
        encoder["freeze"],
        neck["type"],
        neck["channels"],
        config["fusion_channels"],
        config.get("state_downsample_factor", 4),
        encoder["input_already_normalized"],
    )


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
        self.num_classes = num_classes
        self.state_downsample_factor = config.get("state_downsample_factor", 4)
        endpoint = config.get("endpoint", {})
        self.endpoint_type = endpoint.get("type", "unet")
        if self.endpoint_type == "segformer":
            image_config = endpoint["image_encoder"]
            state_config = endpoint["state_encoder"]
            fusion_config = endpoint["fusion"]
            self.segformer_variant = endpoint["segformer_variant"]
            self.image_encoder = ImageEncoder(
                image_config["channels"], image_config["blocks"],
                image_config["growth_channels"], downsample_factor=1,
            )
            self.state_encoder = FullResolutionStateEncoder(
                num_classes, state_config["channels"], state_config["blocks"]
            )
            if fusion_config["type"] == "concat":
                projection_channels = (
                    image_config["channels"] + state_config["channels"]
                )
            else:
                if image_config["channels"] != state_config["channels"]:
                    raise ValueError("add fusion requires equal image/state encoder channels")
                projection_channels = image_config["channels"]
            fused_channels = fusion_config["channels"]
            self.fusion_type = fusion_config["type"]
            self.fusion_projection = nn.Conv2d(projection_channels, fused_channels, 1)
            self.segformer_encoder = TimeConditionedSegFormer(
                self.segformer_variant, fused_channels,
                endpoint["time_embedding_dim"], endpoint["pretrained"],
            )
            decoder_channels = endpoint["decoder_channels"]
            self.decode_projections = nn.ModuleList(
                nn.Conv2d(size, decoder_channels, 1)
                for size in self.segformer_encoder.hidden_sizes
            )
            self.decode_fusion = nn.Sequential(
                nn.Conv2d(decoder_channels * 4, decoder_channels, 1),
                group_norm(decoder_channels), nn.SiLU(),
            )
            self.classifier = nn.Conv2d(decoder_channels, num_classes, 1)
            return

        unet = config["unet"]
        self.mask_encoder = nn.Conv2d(num_classes, channels, 3, padding=1)
        self.image_encoder = build_image_encoder(config)
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
        expected = self.expected_image_feature_size(image)
        assert feature.shape[-2:] == expected, (
            f"image feature {feature.shape[-2:]} != state size {expected}"
        )
        return feature

    def expected_image_feature_size(self, image: torch.Tensor) -> tuple[int, int]:
        if self.endpoint_type == "segformer":
            return tuple(image.shape[-2:])
        return state_spatial_size(image, self.state_downsample_factor)

    def forward_logits_with_image_feat(
        self, x_s: torch.Tensor, image_feat: torch.Tensor,
        s: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        assert x_s.shape[0] == image_feat.shape[0]
        assert x_s.shape[1] == self.num_classes
        if self.endpoint_type == "segformer":
            full_state = F.interpolate(
                x_s, size=image_feat.shape[-2:], mode="bilinear", align_corners=False
            )
            state_feat = self.state_encoder(full_state)
            fused = (
                torch.cat((image_feat, state_feat), dim=1)
                if self.fusion_type == "concat" else image_feat + state_feat
            )
            features = self.segformer_encoder(self.fusion_projection(fused), s, t)
            target_size = features[0].shape[-2:]
            decoded = [
                projection(feature) if feature.shape[-2:] == target_size else F.interpolate(
                    projection(feature), size=target_size,
                    mode="bilinear", align_corners=False,
                )
                for feature, projection in zip(features, self.decode_projections)
            ]
            logits = self.classifier(self.decode_fusion(torch.cat(decoded, dim=1)))
            if logits.shape[-2:] != x_s.shape[-2:]:
                logits = F.interpolate(
                    logits, size=x_s.shape[-2:], mode="bilinear", align_corners=False
                )
            assert logits.shape == x_s.shape
            return logits
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
