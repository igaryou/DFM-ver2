from __future__ import annotations

import sys
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from checkpoint import checkpoint_payload, validate_cfg_checkpoint
from config import load_config
from discrete_flow_maps import sample_prior
from inference import run_flow_from_state, sample_segmentation_ensemble
from model_factory import build_models
from training_objectives import DDPCompatibleTrainingModel


CFG_CONFIG = (
    ROOT / "configs/cityscapes/original/psd/joint_cfg_256x512_800ep.yaml"
)


class TinySource(nn.Module):
    def __init__(self, classes: int, size: tuple[int, int]):
        super().__init__()
        self.mean = nn.Conv2d(3, classes, 1)
        self.pos_source = nn.Parameter(torch.zeros(classes, *size))
        self.fixed_std = 1.0
        self._source_checkpoint_loaded = False

    def forward_statistics(self, image):
        mean = self.mean(image)
        return mean, torch.zeros_like(mean)

    def forward(self, image):
        mean, logvar = self.forward_statistics(image)
        return mean + torch.randn_like(mean), mean, logvar


class TinyEndpoint(nn.Module):
    endpoint_type = "unet"
    cfg_enabled = True

    def __init__(self, classes: int, size: tuple[int, int]):
        super().__init__()
        self.num_classes = classes
        self.image_encoder = nn.Conv2d(3, 4, 1)
        self.mask_encoder = nn.Conv2d(classes, 4, 1)
        self.classifier = nn.Conv2d(4, classes, 1)
        self.pos_UNet = nn.Parameter(torch.zeros(2, *size))
        self.uncond_image_projection = nn.Conv2d(2, 4, 1)
        self.seen_features: list[torch.Tensor] = []

    def encode_image(self, image):
        return self.image_encoder(image)

    def expected_image_feature_size(self, image):
        return tuple(image.shape[-2:])

    def unconditional_image_feature(self, batch_size, spatial_size, *, dtype, device):
        feature = self.uncond_image_projection(self.pos_UNet.unsqueeze(0))
        assert feature.shape[-2:] == spatial_size
        return feature.to(device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

    def condition_image_feature(self, image_feat, condition_drop_mask):
        unconditional = self.unconditional_image_feature(
            image_feat.shape[0], tuple(image_feat.shape[-2:]),
            dtype=image_feat.dtype, device=image_feat.device,
        )
        self._last_condition_drop_mask = condition_drop_mask.detach().clone()
        return torch.where(
            condition_drop_mask[:, None, None, None], unconditional, image_feat
        )

    def forward_logits_with_image_feat(self, x, image_feat, s, t):
        del s, t
        self.seen_features.append(image_feat.detach().clone())
        return self.classifier(self.mask_encoder(x) + image_feat)

    def forward_logits(self, x, image, s, t):
        return self.forward_logits_with_image_feat(x, self.encode_image(image), s, t)


def tiny_training_setup(dropout: float):
    config = load_config(CFG_CONFIG)
    config["dataset"]["image_size"] = [8, 8]
    config["cfg"]["dropout_prob"] = dropout
    classes = config["dataset"]["num_classes"]
    endpoint = TinyEndpoint(classes, (8, 8))
    source = TinySource(classes, (8, 8))
    adapter = DDPCompatibleTrainingModel(endpoint, source, config)
    image = torch.randn(3, 3, 8, 8)
    target = torch.randint(0, classes, (3, 8, 8))
    spatial = torch.ones_like(target, dtype=torch.bool)
    return config, adapter, endpoint, source, image, target, spatial


@pytest.mark.parametrize("dropout,expected", [(0.0, False), (1.0, True)])
def test_cfg_training_extremes_and_gradients(dropout, expected):
    _, adapter, endpoint, source, image, target, spatial = tiny_training_setup(
        dropout
    )
    result = adapter(
        operation="stage1_objectives", image=image, target=target,
        spatial_valid_mask=spatial, epoch_index=0, progress_in_epoch=0.0,
    )
    result["loss"].backward()

    assert adapter._last_condition_drop_mask.tolist() == [expected] * 3
    assert torch.equal(
        source._last_condition_drop_mask, endpoint._last_condition_drop_mask
    )
    assert source.pos_source.grad is not None
    assert endpoint.pos_UNet.grad is not None
    assert endpoint.uncond_image_projection.weight.grad is not None
    assert endpoint.uncond_image_projection.bias.grad is not None
    assert result["stats"]["loss_source_align"].item() == 0.0


def test_cfg_dropout_is_samplewise_and_shared(monkeypatch):
    _, adapter, endpoint, source, image, target, spatial = tiny_training_setup(0.5)
    real_rand = torch.rand
    calls = 0

    def controlled_rand(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return torch.tensor([0.1, 0.9, 0.2], device=kwargs.get("device"))
        return real_rand(*args, **kwargs)

    monkeypatch.setattr(torch, "rand", controlled_rand)
    result = adapter(
        operation="stage1_objectives", image=image, target=target,
        spatial_valid_mask=spatial, epoch_index=0, progress_in_epoch=0.0,
    )
    expected = torch.tensor([True, False, True])
    assert torch.equal(adapter._last_condition_drop_mask.cpu(), expected)
    assert torch.equal(source._last_condition_drop_mask.cpu(), expected)
    assert torch.equal(endpoint._last_condition_drop_mask.cpu(), expected)
    assert result["stats"]["cfg_dropout_count"].item() == 2


def test_psd_teacher_student_reuse_the_same_condition_feature():
    config, adapter, endpoint, _, image, target, spatial = tiny_training_setup(1.0)
    config["loss"]["consistency"]["start"] = {"unit": "epoch", "value": 0}
    config["loss"]["consistency"]["warmup_epochs"] = 0
    adapter(
        operation="joint_objectives", image=image, target=target,
        spatial_valid_mask=spatial, epoch_index=0, progress_in_epoch=1.0,
    )
    assert len(endpoint.seen_features) == 4  # diagonal + 2 teacher + student
    assert all(
        torch.equal(endpoint.seen_features[0], feature)
        for feature in endpoint.seen_features[1:]
    )


class GuidedEndpoint(nn.Module):
    cfg_enabled = True

    def __init__(self):
        super().__init__()
        self.unconditional_calls = 0
        self.normal_calls = 0

    def encode_image(self, image):
        return image[:, :2]

    def unconditional_image_feature(self, batch_size, spatial_size, *, dtype, device):
        self.unconditional_calls += 1
        values = torch.tensor([2.0, -1.0], dtype=dtype, device=device)
        return values[None, :, None, None].expand(batch_size, -1, *spatial_size)

    def forward_logits_with_image_feat(self, x, image_feat, s, t):
        del x, s, t
        return image_feat

    def forward_logits(self, x, image, s, t):
        self.normal_calls += 1
        return self.forward_logits_with_image_feat(x, self.encode_image(image), s, t)


def inference_config(enabled: bool, scale: float):
    return {
        "model": {"state_downsample_factor": 1},
        "flow": {"time_eps": 1.0e-5, "path": {"type": "linear"}},
        "evaluation": {"cfg": {"enabled": enabled, "guidance_scale": scale}},
    }


@pytest.mark.parametrize("scale", [0.0, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("steps", [1, 3, 5])
def test_cfg_inference_scales_and_steps(scale, steps):
    model = GuidedEndpoint()
    image = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]])
    x0 = torch.zeros(1, 2, 1, 1)
    result = run_flow_from_state(
        model, image, x0, inference_config(True, scale), steps
    )
    assert result.shape == x0.shape
    assert torch.isfinite(result).all()


def test_cfg_scale_one_matches_normal_and_off_skips_unconditional_forward():
    image = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]])
    x0 = torch.zeros(1, 2, 1, 1)
    normal_model = GuidedEndpoint()
    guided_model = GuidedEndpoint()
    normal = run_flow_from_state(
        normal_model, image, x0, inference_config(False, 1.0), 3
    )
    guided = run_flow_from_state(
        guided_model, image, x0, inference_config(True, 1.0), 3
    )
    assert torch.allclose(normal, guided)
    assert normal_model.normal_calls == 3
    assert normal_model.unconditional_calls == 0


def test_cfg_multisample_inference_preserves_void_policy():
    config, _, endpoint, source, image, _, _ = tiny_training_setup(0.5)
    config["evaluation"].update({
        "num_steps": 3,
        "num_samples": 3,
        "aggregation": "probability_mean",
        "exclude_void_from_prediction": True,
    })
    prediction = sample_segmentation_ensemble(
        endpoint, source, image, config, num_samples=3, num_steps=3
    )
    assert prediction.shape == (image.shape[0], *image.shape[-2:])
    assert not bool((prediction == config["dataset"]["void_class_index"]).any())


def test_cfg_config_shapes_and_legacy_checkpoint_error():
    config = load_config(CFG_CONFIG)
    assert config["dataset"]["image_size"] == [256, 512]
    source_parameters = 20 * 256 * 512
    unet_parameters = 64 * 256 * 512
    projection_parameters = 128 * 64 + 128
    assert source_parameters + unet_parameters + projection_parameters == 11_018_368

    with pytest.raises(RuntimeError, match="CFG-free checkpoint"):
        validate_cfg_checkpoint(
            {"model": {}, "source_model": {}}, config, "legacy.pt"
        )


def test_real_cfg_modules_build_and_mix_at_state_resolution():
    config = load_config(CFG_CONFIG)
    config["dataset"]["image_size"] = [8, 8]
    config["source"].update({
        "backbone": "unet", "pretrained": False, "decoder_channels": 8,
    })
    config["model"]["fusion_channels"] = 8
    config["model"]["rrdb_blocks"] = 1
    config["model"]["rrdb_growth_channels"] = 4
    config["model"]["unet"].update({
        "base_channels": 8,
        "channel_mults": [1],
        "num_res_blocks": 1,
        "attention_levels": [],
        "num_heads": 1,
        "time_embedding_dim": 16,
    })
    config["cfg"]["uncond_image"]["channels"] = 4
    endpoint, source = build_models(config, torch.device("cpu"))
    assert source.pos_source.shape == (20, 8, 8)
    assert endpoint.pos_UNet.shape == (4, 8, 8)
    assert endpoint.uncond_image_projection.weight.shape == (8, 4, 1, 1)

    image = torch.randn(2, 3, 8, 8)
    drop = torch.tensor([True, False])
    conditional = endpoint.encode_image(image)
    mixed = endpoint.condition_image_feature(conditional, drop)
    mixed.sum().backward()
    assert endpoint.pos_UNet.grad is not None
    assert endpoint.uncond_image_projection.weight.grad is not None

    x0, _ = sample_prior(
        config, image, None, source,
        sampling_mode="training", condition_drop_mask=drop,
    )
    x0.mean().backward()
    assert source.pos_source.grad is not None


def test_evaluation_cfg_requires_cfg_trained_model():
    with pytest.raises(ValueError, match="CFG-trained model"):
        load_config(
            CFG_CONFIG,
            ["cfg.enabled=false", "evaluation.cfg.enabled=true"],
        )


def test_cfg_bf16_forward_backward():
    _, adapter, endpoint, source, image, target, spatial = tiny_training_setup(1.0)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = adapter(
            operation="stage1_objectives", image=image, target=target,
            spatial_valid_mask=spatial, epoch_index=0, progress_in_epoch=0.0,
        )
    result["loss"].backward()
    for parameter in (
        source.pos_source,
        endpoint.pos_UNet,
        endpoint.uncond_image_projection.weight,
        endpoint.uncond_image_projection.bias,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_cfg_checkpoint_and_optimizer_round_trip():
    config, adapter, endpoint, source, image, target, spatial = tiny_training_setup(1.0)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    result = adapter(
        operation="stage1_objectives", image=image, target=target,
        spatial_valid_mask=spatial, epoch_index=0, progress_in_epoch=0.0,
    )
    result["loss"].backward()
    optimizer.step()
    payload = checkpoint_payload(
        config=config, epoch=1, global_step=1,
        model=endpoint, source_model=source,
        optimizer=optimizer, scheduler=scheduler, scaler=None, metrics={},
    )
    assert "pos_UNet" in payload["model"]
    assert "uncond_image_projection.weight" in payload["model"]
    assert "uncond_image_projection.bias" in payload["model"]
    assert "pos_source" in payload["source_model"]
    assert payload["optimizer"]["state"]
    assert payload["scheduler"] is not None

    _, restored, restored_endpoint, restored_source, *_ = tiny_training_setup(1.0)
    restored_endpoint.load_state_dict(payload["model"], strict=True)
    restored_source.load_state_dict(payload["source_model"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.0e-3)
    restored_optimizer.load_state_dict(payload["optimizer"])
    assert torch.equal(restored_endpoint.pos_UNet, endpoint.pos_UNet)
    assert torch.equal(restored_source.pos_source, source.pos_source)


def _cfg_ddp_worker(rank: int, world_size: int, rendezvous: str):
    os.environ.update({
        "GLOO_SOCKET_IFNAME": "lo",
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
    })
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}",
        rank=rank, world_size=world_size,
    )
    try:
        _, adapter, endpoint, source, image, target, spatial = tiny_training_setup(
            float(rank)
        )
        ddp = nn.parallel.DistributedDataParallel(
            adapter, find_unused_parameters=False
        )
        optimizer = torch.optim.SGD(ddp.parameters(), lr=1.0e-3)
        result = ddp(
            operation="stage1_objectives", image=image, target=target,
            spatial_valid_mask=spatial, epoch_index=0, progress_in_epoch=0.0,
        )
        result["loss"].backward()
        assert source.pos_source.grad is not None
        assert endpoint.pos_UNet.grad is not None
        assert endpoint.uncond_image_projection.weight.grad is not None
        optimizer.step()
        checksum = sum(parameter.detach().sum() for parameter in adapter.parameters())
        gathered = [torch.zeros_like(checksum) for _ in range(world_size)]
        dist.all_gather(gathered, checksum)
        assert all(torch.allclose(gathered[0], value) for value in gathered[1:])
    finally:
        dist.destroy_process_group()


def test_cfg_two_rank_ddp_has_no_unused_parameter_deadlock(tmp_path):
    mp.spawn(
        _cfg_ddp_worker,
        args=(2, str(tmp_path / "cfg-gloo-rendezvous")),
        nprocs=2,
        join=True,
    )
