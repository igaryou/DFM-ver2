from __future__ import annotations

import copy
import math
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import group_norm
from segformer_architecture import (
    SEGFORMER_DECODER_HIDDEN,
    SEGFORMER_DEPTHS,
    SEGFORMER_HIDDEN_SIZES,
    SEGFORMER_MODEL_NAMES,
    build_segformer_config,
)
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
        in_channels: int = 3,
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
        current_channels = in_channels
        for _ in range(stages):
            layers.extend((
                nn.Conv2d(current_channels, channels, 3, stride=2, padding=1),
                nn.SiLU(),
            ))
            current_channels = channels
        if not stages:
            layers.extend((nn.Conv2d(in_channels, channels, 3, padding=1), nn.SiLU()))
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

    MODEL_NAMES = SEGFORMER_MODEL_NAMES
    DEPTHS = SEGFORMER_DEPTHS
    HIDDEN = SEGFORMER_HIDDEN_SIZES
    DECODER_HIDDEN = SEGFORMER_DECODER_HIDDEN

    def __init__(
        self, num_classes: int, variant: str, pretrained: bool, decoder_channels: int,
        freeze_encoder: bool, learned_logvar: bool, fixed_std, mu_tanh_scale: float,
        input_already_normalized: bool = False,
        state_downsample_factor: int = 4,
        decoder_type: str = "custom",
        in_channels: int = 3,
        variance_mode: str = "channelwise",
        variance_target_std: float = 1.0,
        init_logvar_at_target: bool = False,
    ) -> None:
        super().__init__()
        if variant not in self.MODEL_NAMES:
            raise ValueError(f"Unknown SegFormer variant: {variant}")
        if decoder_type not in {"custom", "standard"}:
            raise ValueError("decoder_type must be custom or standard")
        if variance_mode not in {"channelwise", "pixel_shared"}:
            raise ValueError("variance_mode must be channelwise or pixel_shared")
        if variance_mode == "pixel_shared" and not learned_logvar:
            raise ValueError("variance_mode=pixel_shared requires learned_logvar=true")
        try:
            from transformers import SegformerModel
        except ImportError as exc:
            raise RuntimeError("source.backbone=segformer requires transformers") from exc
        if pretrained and in_channels != 3:
            raise ValueError(
                "pretrained SegFormer source currently requires a 3-channel input; "
                "use source.pretrained=false for 1-channel LIDC CT images"
            )
        if pretrained:
            self.encoder = SegformerModel.from_pretrained(self.MODEL_NAMES[variant])
        else:
            self.encoder = SegformerModel(build_segformer_config(variant, in_channels))
        self.num_classes = num_classes
        self.fixed_std = None if learned_logvar else fixed_std
        self.mu_tanh_scale = mu_tanh_scale
        self.input_already_normalized = input_already_normalized
        self.state_downsample_factor = state_downsample_factor
        self.segformer_decoder = decoder_type
        self.variance_mode = variance_mode
        output_channels = (
            num_classes + 1
            if self.fixed_std is None and variance_mode == "pixel_shared"
            else num_classes * 2 if self.fixed_std is None else num_classes
        )
        if decoder_type == "custom":
            hidden_sizes = list(self.encoder.config.hidden_sizes)
            self.projections = nn.ModuleList(
                nn.Conv2d(size, decoder_channels, 1) for size in hidden_sizes
            )
            decoder_layers = [
                nn.Conv2d(decoder_channels * 4, decoder_channels, 3, padding=1),
                group_norm(decoder_channels), nn.SiLU(),
                nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
                group_norm(decoder_channels), nn.SiLU(),
            ]
            if self.fixed_std is None and variance_mode == "pixel_shared":
                self.decoder = nn.Sequential(*decoder_layers)
                self.mu_head = nn.Conv2d(decoder_channels, num_classes, 1)
                self.logvar_head = nn.Conv2d(decoder_channels, 1, 1)
                if init_logvar_at_target:
                    nn.init.zeros_(self.logvar_head.weight)
                    nn.init.constant_(
                        self.logvar_head.bias,
                        math.log(float(variance_target_std) ** 2),
                    )
            else:
                # Preserve the legacy module layout and state-dict names.
                self.decoder = nn.Sequential(
                    *decoder_layers, nn.Conv2d(decoder_channels, output_channels, 1)
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
            if (
                self.fixed_std is None
                and variance_mode == "pixel_shared"
                and init_logvar_at_target
            ):
                with torch.no_grad():
                    self.decode_head.classifier.weight[-1].zero_()
                    self.decode_head.classifier.bias[-1].fill_(
                        math.log(float(variance_target_std) ** 2)
                    )
        normalization = (
            ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            if in_channels == 3 else ([0.5], [0.5])
        )
        self.register_buffer(
            "mean", torch.tensor(normalization[0])[None, :, None, None],
            persistent=False,
        )
        self.register_buffer(
            "std", torch.tensor(normalization[1])[None, :, None, None],
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
            if self.fixed_std is None and self.variance_mode == "pixel_shared":
                mu = self.mu_head(output)
                logvar = self.logvar_head(output)
                output = None
        if self.fixed_std is None and not (
            self.segformer_decoder == "custom"
            and self.variance_mode == "pixel_shared"
        ):
            if self.variance_mode == "pixel_shared":
                mu, logvar = output[:, :self.num_classes], output[:, self.num_classes:]
            else:
                mu, logvar = output.chunk(2, dim=1)
        else:
            if self.fixed_std is not None:
                mu = output
                logvar = torch.full_like(mu, math.log(float(self.fixed_std) ** 2))
        if self.mu_tanh_scale > 0:
            mu = torch.tanh(mu) * self.mu_tanh_scale
        return mu, logvar

    def forward(self, image: torch.Tensor):
        mu, logvar = self.forward_statistics(image)
        sigma = torch.exp(0.5 * logvar.float())
        if not torch.isfinite(sigma).all():
            raise FloatingPointError("source sigma contains NaN or Inf")
        x0 = mu + sigma.to(mu.dtype) * torch.randn_like(mu)
        if not torch.isfinite(x0.float()).all():
            raise FloatingPointError("sampled source x0 contains NaN or Inf")
        return x0, mu, logvar


def _initialize_predictor_output(model: nn.Module, value: float) -> None:
    """Initialize the final predictor to emit one spatially constant value."""
    if isinstance(model, UNetSourceGenerator):
        head = model.network[-1]
    elif isinstance(model, SegFormerSourceGenerator):
        head = (
            model.decode_head.classifier
            if model.segformer_decoder == "standard"
            else model.decoder[-1]
        )
    else:  # pragma: no cover - guarded by the source factory.
        raise TypeError(f"Unsupported source predictor: {type(model).__name__}")
    if not isinstance(head, nn.Conv2d):
        raise TypeError("source predictor output head must be Conv2d")
    nn.init.zeros_(head.weight)
    nn.init.constant_(head.bias, value)


class IndependentGaussianSourceGenerator(nn.Module):
    """Gaussian source with independently configured mu and logvar networks."""

    def __init__(
        self,
        mu_model: nn.Module,
        logvar_model: nn.Module,
        *,
        num_classes: int,
        variance_mode: str,
        target_std: float,
        init_logvar_at_target: bool,
    ) -> None:
        super().__init__()
        self.mu_model = mu_model
        self.logvar_model = logvar_model
        self.num_classes = num_classes
        self.variance_mode = variance_mode
        # sample_prior uses this attribute to decide whether variance is learned.
        self.fixed_std = None
        if init_logvar_at_target:
            _initialize_predictor_output(
                self.logvar_model, math.log(float(target_std) ** 2)
            )

    def forward_statistics(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu, _ = self.mu_model.forward_statistics(image)
        logvar, _ = self.logvar_model.forward_statistics(image)
        expected_channels = 1 if self.variance_mode == "pixel_shared" else self.num_classes
        if mu.shape[1] != self.num_classes or logvar.shape[1] != expected_channels:
            raise AssertionError(
                "independent Gaussian source produced invalid channels: "
                f"mu={tuple(mu.shape)}, logvar={tuple(logvar.shape)}"
            )
        if mu.shape[0] != logvar.shape[0] or mu.shape[-2:] != logvar.shape[-2:]:
            raise AssertionError(
                "mu and logvar networks must produce the same batch/spatial shape"
            )
        return mu, logvar

    def forward(self, image: torch.Tensor):
        mu, logvar = self.forward_statistics(image)
        sigma = torch.exp(0.5 * logvar.float())
        if not torch.isfinite(sigma).all():
            raise FloatingPointError("source sigma contains NaN or Inf")
        x0 = mu + sigma.to(mu.dtype) * torch.randn_like(mu)
        if not torch.isfinite(x0.float()).all():
            raise FloatingPointError("sampled source x0 contains NaN or Inf")
        return x0, mu, logvar


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


def unconditional_source_mean(
    source_model: nn.Module,
    *,
    batch_size: int,
    spatial_size: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Broadcast the learned image-free source mean without allocating copies."""
    position = getattr(source_model, "pos_source", None)
    if position is None:
        raise RuntimeError(
            "CFG unconditional source is unavailable; the model was not built "
            "with cfg.enabled=true"
        )
    mean = position
    if mean.shape[-2:] != spatial_size:
        mean = F.interpolate(
            mean.unsqueeze(0), size=spatial_size,
            mode="bilinear", align_corners=False,
        ).squeeze(0)
    return mean.to(device=device, dtype=dtype).unsqueeze(0).expand(
        batch_size, -1, -1, -1
    )


def freeze_source_model(model: nn.Module) -> nn.Module:
    """Completely freeze a source and keep it in eval mode across train calls."""
    model.requires_grad_(False)
    model.eval()

    def frozen_train(instance: nn.Module, mode: bool = True) -> nn.Module:
        del mode
        nn.Module.train(instance, False)
        return instance

    model.train = MethodType(frozen_train, model)
    model._source_frozen = True
    return model


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
    elif source["logvar_model"]["mode"] == "separate":
        state_factor = config["model"].get("state_downsample_factor", 4)
        in_channels = config["dataset"].get("in_channels", 3)

        def build_predictor(settings: dict, outputs: int, mu_tanh_scale: float):
            if settings["backbone"] == "unet":
                return UNetSourceGenerator(
                    outputs, settings["decoder_channels"], False, 1.0,
                    state_factor, in_channels,
                )
            return SegFormerSourceGenerator(
                outputs, settings["segformer_variant"], settings["pretrained"],
                settings["decoder_channels"], settings["freeze_encoder"],
                False, 1.0, mu_tanh_scale,
                settings["input_already_normalized"], state_factor,
                settings["segformer_decoder"], in_channels,
            )

        mu_model = build_predictor(source, config["dataset"]["num_classes"], source["mu_tanh_scale"])
        logvar_channels = (
            1 if source["variance_mode"] == "pixel_shared"
            else config["dataset"]["num_classes"]
        )
        logvar_model = build_predictor(
            source["logvar_model"], logvar_channels, 0.0
        )
        regularization = source["variance_regularization"]
        model = IndependentGaussianSourceGenerator(
            mu_model, logvar_model,
            num_classes=config["dataset"]["num_classes"],
            variance_mode=source["variance_mode"],
            target_std=regularization["target_std"],
            init_logvar_at_target=regularization["init_at_target"],
        )
    elif source["backbone"] == "unet":
        model = UNetSourceGenerator(
            config["dataset"]["num_classes"], source["decoder_channels"],
            source["learned_logvar"], fixed_std,
            config["model"].get("state_downsample_factor", 4),
            config["dataset"].get("in_channels", 3),
        )
    else:
        model = SegFormerSourceGenerator(
            config["dataset"]["num_classes"], source["segformer_variant"],
            source["pretrained"], source["decoder_channels"], source["freeze_encoder"],
            source["learned_logvar"], fixed_std, source["mu_tanh_scale"],
            source["input_already_normalized"],
            config["model"].get("state_downsample_factor", 4),
            source["segformer_decoder"],
            config["dataset"].get("in_channels", 3),
            source.get("variance_mode", "channelwise"),
            source.get("variance_regularization", {}).get("target_std", 1.0),
            source.get("variance_regularization", {}).get("init_at_target", False),
        )
    checkpoint_loaded = False
    if source["checkpoint"]:
        checkpoint = torch.load(source["checkpoint"], map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            from checkpoint import validate_source_decoder_checkpoint
            validate_source_decoder_checkpoint(
                checkpoint, config, source["checkpoint"]
            )
        state = checkpoint.get("source_model", checkpoint.get("model", checkpoint))
        model.load_state_dict(state, strict=True)
        checkpoint_loaded = True
    cfg = config.get("cfg", {})
    if cfg.get("enabled", False) and cfg.get("uncond_source", {}).get(
        "enabled", True
    ):
        state_size = state_spatial_size(
            config["dataset"]["image_size"],
            config["model"].get("state_downsample_factor", 4),
        )
        model.register_parameter(
            "pos_source",
            nn.Parameter(torch.zeros(
                config["dataset"]["num_classes"], *state_size
            )),
        )
    model._source_checkpoint_loaded = checkpoint_loaded
    model._source_frozen = False
    if source["freeze"]:
        freeze_source_model(model)
    return model
