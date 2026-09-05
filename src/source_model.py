from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import group_norm
from state_space import state_spatial_size


class UNetSourceGenerator(nn.Module):
    """Small source option for offline/debug use."""

    def __init__(
        self,
        num_classes: int,
        channels: int,
        learned_logvar: bool,
        fixed_std,
        state_downsample_factor: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fixed_std = None if learned_logvar else fixed_std
        output_channels = num_classes * 2 if self.fixed_std is None else num_classes
        stages = int(math.log2(state_downsample_factor))
        if 2**stages != state_downsample_factor:
            raise ValueError("source state_downsample_factor must be a power of two")
        self.state_downsample_factor = state_downsample_factor
        layers: list[nn.Module] = []
        in_channels = 3
        for _ in range(stages):
            layers.extend((
                nn.Conv2d(in_channels, channels, 3, stride=2, padding=1),
                nn.SiLU(),
            ))
            in_channels = channels
        if not stages:
            layers.extend((nn.Conv2d(3, channels, 3, padding=1), nn.SiLU()))
        layers.extend((
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, output_channels, 1),
        ))
        self.network = nn.Sequential(*layers)

    def forward_statistics(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.network(image)
        assert output.shape[-2:] == state_spatial_size(
            image, self.state_downsample_factor
        )
        if self.fixed_std is None:
            mu, logvar = output.chunk(2, dim=1)
        else:
            mu = output
            logvar = torch.full_like(mu, math.log(float(self.fixed_std) ** 2))
        return mu, logvar

    def forward(self, image: torch.Tensor):
        mu, logvar = self.forward_statistics(image)
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu), mu, logvar


class SegFormerSourceGenerator(nn.Module):
    """SegFormer image-conditioned Gaussian source ported from CFM/segv4."""

    MODEL_NAMES = {f"b{i}": f"nvidia/mit-b{i}" for i in range(6)}
    DEPTHS = {
        "b0": [2, 2, 2, 2], "b1": [2, 2, 2, 2], "b2": [3, 4, 6, 3],
        "b3": [3, 4, 18, 3], "b4": [3, 8, 27, 3], "b5": [3, 6, 40, 3],
    }
    HIDDEN = {
        "b0": [32, 64, 160, 256], "b1": [64, 128, 320, 512],
        "b2": [64, 128, 320, 512], "b3": [64, 128, 320, 512],
        "b4": [64, 128, 320, 512], "b5": [64, 128, 320, 512],
    }
    DECODER_HIDDEN = {
        "b0": 256, "b1": 256, "b2": 768, "b3": 768,
        "b4": 768, "b5": 768,
    }

    def __init__(
        self, num_classes: int, variant: str, pretrained: bool, decoder_channels: int,
        freeze_encoder: bool, learned_logvar: bool, fixed_std, mu_tanh_scale: float,
        input_already_normalized: bool = False,
        state_downsample_factor: int = 4,
        decoder_type: str = "custom",
    ) -> None:
        super().__init__()
        if variant not in self.MODEL_NAMES:
            raise ValueError(f"Unknown SegFormer variant: {variant}")
        if decoder_type not in {"custom", "standard"}:
            raise ValueError("decoder_type must be custom or standard")
        try:
            from transformers import SegformerConfig, SegformerModel
        except ImportError as exc:
            raise RuntimeError("source.backbone=segformer requires transformers") from exc
        if pretrained:
            self.encoder = SegformerModel.from_pretrained(self.MODEL_NAMES[variant])
        else:
            heads = [1, 2, 5, 8]
            self.encoder = SegformerModel(SegformerConfig(
                num_channels=3, num_encoder_blocks=4, depths=self.DEPTHS[variant],
                sr_ratios=[8, 4, 2, 1], hidden_sizes=self.HIDDEN[variant],
                patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
                num_attention_heads=heads, mlp_ratios=[4, 4, 4, 4],
                hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                drop_path_rate=0.1,
            ))
        self.num_classes = num_classes
        self.fixed_std = None if learned_logvar else fixed_std
        self.mu_tanh_scale = mu_tanh_scale
        self.input_already_normalized = input_already_normalized
        self.state_downsample_factor = state_downsample_factor
        self.segformer_decoder = decoder_type
        output_channels = num_classes * 2 if self.fixed_std is None else num_classes
        if decoder_type == "custom":
            hidden_sizes = list(self.encoder.config.hidden_sizes)
            self.projections = nn.ModuleList(
                nn.Conv2d(size, decoder_channels, 1) for size in hidden_sizes
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(decoder_channels * 4, decoder_channels, 3, padding=1),
                group_norm(decoder_channels), nn.SiLU(),
                nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
                group_norm(decoder_channels), nn.SiLU(),
                nn.Conv2d(decoder_channels, output_channels, 1),
            )
        else:
            try:
                from transformers.models.segformer.modeling_segformer import (
                    SegformerDecodeHead,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "The installed transformers version does not expose "
                    "SegformerDecodeHead"
                ) from exc
            head_config = copy.deepcopy(self.encoder.config)
            head_config.decoder_hidden_size = self.DECODER_HIDDEN[variant]
            head_config.num_labels = output_channels
            self.decode_head = SegformerDecodeHead(head_config)
            # Match SegformerForSemanticSegmentation.post_init() while avoiding
            # construction of a second, immediately discarded MiT encoder.
            self.decode_head.apply(self.encoder._init_weights)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None],
            persistent=False,
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None],
            persistent=False,
        )
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward_statistics(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = image if self.input_already_normalized else (
            image - self.mean.to(image)
        ) / self.std.to(image)
        hidden_states = self.encoder(
            pixel_values=normalized, output_hidden_states=True, return_dict=True
        ).hidden_states[-4:]
        target_size = state_spatial_size(image, self.state_downsample_factor)
        if getattr(self, "segformer_decoder", "custom") == "standard":
            output = self.decode_head(hidden_states)
            if output.shape[-2:] != target_size:
                output = F.interpolate(
                    output, size=target_size, mode="bilinear", align_corners=False
                )
        else:
            features = [
                F.interpolate(
                    projection(hidden), target_size,
                    mode="bilinear", align_corners=False,
                )
                for hidden, projection in zip(hidden_states, self.projections)
            ]
            output = self.decoder(torch.cat(features, dim=1))
        if self.fixed_std is None:
            mu, logvar = output.chunk(2, dim=1)
        else:
            mu = output
            logvar = torch.full_like(mu, math.log(float(self.fixed_std) ** 2))
        if self.mu_tanh_scale > 0:
            mu = torch.tanh(mu) * self.mu_tanh_scale
        return mu, logvar

    def forward(self, image: torch.Tensor):
        mu, logvar = self.forward_statistics(image)
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu), mu, logvar


class TaskFinetunedSegFormerSourceGenerator(nn.Module):
    """Frozen/trainable semantic SegFormer decode head used as Gaussian mean."""

    def __init__(
        self,
        num_classes: int,
        void_class_index: int,
        model_id: str,
        representation: str,
        freeze: bool,
        fixed_std: float,
        void_channel_value: float = 0.0,
        input_already_normalized: bool = False,
        state_downsample_factor: int = 4,
        segmentation_model: nn.Module | None = None,
        load_pretrained: bool = True,
    ) -> None:
        super().__init__()
        if representation not in {"probability", "logits"}:
            raise ValueError("representation must be probability or logits")
        if not 0 <= void_class_index < num_classes:
            raise ValueError("void_class_index must identify a DFM state channel")
        if segmentation_model is None:
            try:
                from transformers import (
                    SegformerConfig,
                    SegformerForSemanticSegmentation,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "task_finetuned_segformer requires transformers"
                ) from exc
            if load_pretrained:
                segmentation_model = (
                    SegformerForSemanticSegmentation.from_pretrained(model_id)
                )
            else:
                segmentation_model = SegformerForSemanticSegmentation(
                    SegformerConfig.from_pretrained(model_id)
                )
        semantic_classes = int(segmentation_model.config.num_labels)
        if semantic_classes != num_classes - 1:
            raise ValueError(
                f"Task SegFormer has {semantic_classes} semantic channels, but the "
                f"{num_classes}-state protocol requires {num_classes - 1}"
            )
        self.segmentation_model = segmentation_model
        self.num_classes = num_classes
        self.void_class_index = void_class_index
        self.representation = representation
        self.freeze_source = freeze
        self.fixed_std = float(fixed_std)
        self.void_channel_value = float(void_channel_value)
        self.input_already_normalized = input_already_normalized
        self.state_downsample_factor = state_downsample_factor
        self.register_buffer(
            "semantic_indices",
            torch.tensor([
                index for index in range(num_classes)
                if index != void_class_index
            ], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None],
            persistent=False,
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None],
            persistent=False,
        )
        if freeze:
            self.requires_grad_(False)
            self.train(False)

    def train(self, mode: bool = True):
        if self.freeze_source:
            return super().train(False)
        return super().train(mode)

    def forward_statistics(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = image if self.input_already_normalized else (
            image - self.mean.to(image)
        ) / self.std.to(image)
        logits = self.segmentation_model(
            pixel_values=normalized, return_dict=True
        ).logits
        if logits.ndim != 4 or logits.shape[1] != self.num_classes - 1:
            raise AssertionError(
                f"Task SegFormer logits must have shape [B,{self.num_classes - 1},H,W], "
                f"got {tuple(logits.shape)}"
            )
        semantic = (
            torch.softmax(logits.float(), dim=1).to(logits.dtype)
            if self.representation == "probability"
            else logits
        )
        target_size = state_spatial_size(image, self.state_downsample_factor)
        semantic = F.interpolate(
            semantic, size=target_size, mode="bilinear", align_corners=False
        )
        mu = semantic.new_full(
            (semantic.shape[0], self.num_classes, *target_size),
            self.void_channel_value,
        ).index_copy(1, self.semantic_indices, semantic)
        logvar = torch.full_like(mu, math.log(self.fixed_std**2))
        assert mu.shape[1:] == (self.num_classes, *target_size)
        return mu, logvar

    def forward(self, image: torch.Tensor):
        mu, logvar = self.forward_statistics(image)
        x0 = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        assert x0.shape == mu.shape == logvar.shape
        return x0, mu, logvar


def source_statistics(source_model: nn.Module, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source mean/log-variance without sampling x0 when supported."""
    forward_statistics = getattr(source_model, "forward_statistics", None)
    if forward_statistics is not None:
        return forward_statistics(image)
    _, mean, log_variance = source_model(image)
    return mean, log_variance


def build_source_model(config: dict):
    source = config["source"]
    if source["prior_type"] not in {
        "image_gaussian", "image_bounded_gaussian", "image_simplex_mixture"
    }:
        return None
    fixed_std = source["fixed_std"]
    if not source["learned_logvar"] and fixed_std is None:
        fixed_std = 1.0
    if source.get("type", "trainable_segformer") == "task_finetuned_segformer":
        model = TaskFinetunedSegFormerSourceGenerator(
            config["dataset"]["num_classes"],
            config["dataset"]["void_class_index"],
            source["model_id"],
            source["representation"],
            source["freeze"],
            fixed_std,
            source["void_channel_value"],
            source["input_already_normalized"],
            config["model"].get("state_downsample_factor", 4),
            load_pretrained=source.get("_load_pretrained", True),
        )
    elif source["backbone"] == "unet":
        model = UNetSourceGenerator(
            config["dataset"]["num_classes"], source["decoder_channels"],
            source["learned_logvar"], fixed_std,
            config["model"].get("state_downsample_factor", 4),
        )
    else:
        model = SegFormerSourceGenerator(
            config["dataset"]["num_classes"], source["segformer_variant"],
            source["pretrained"], source["decoder_channels"], source["freeze_encoder"],
            source["learned_logvar"], fixed_std, source["mu_tanh_scale"],
            source["input_already_normalized"],
            config["model"].get("state_downsample_factor", 4),
            source["segformer_decoder"],
        )
    if source["checkpoint"]:
        checkpoint = torch.load(source["checkpoint"], map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            from checkpoint import validate_source_decoder_checkpoint
            validate_source_decoder_checkpoint(
                checkpoint, config, source["checkpoint"]
            )
        state = checkpoint.get("source_model", checkpoint.get("model", checkpoint))
        model.load_state_dict(state, strict=True)
    if source["freeze"]:
        model.requires_grad_(False)
    return model
