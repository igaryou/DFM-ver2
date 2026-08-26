import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from distributed import (
    DistributedContext,
    DistributedEvalSampler,
    EpochMetricMeter,
    all_reduce_confusion_matrix,
    cleanup_distributed,
    reduce_epoch_metric_meter,
    reduce_max_values,
    reduce_metric_dict,
    reduce_scalar,
    setup_distributed,
    validate_global_batch_size,
)
from metrics import SegmentationMetrics


class _Composite(nn.Module):
    def __init__(self, *, freeze_source: bool = False):
        super().__init__()
        self.endpoint_model = nn.Sequential(
            nn.Linear(2, 4), nn.SiLU(), nn.Linear(4, 1)
        )
        self.source_model = nn.Linear(2, 2)
        if freeze_source:
            self.source_model.requires_grad_(False)

    def forward(self, value):
        source = self.source_model(value)
        return self.endpoint_model(source)


def _gloo_worker(
    rank: int, world_size: int, rendezvous_file: str, output_dir: str
):
    os.environ.update({
        "GLOO_SOCKET_IFNAME": "lo",
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
    })
    config = {
        "runtime": {"device": "auto"},
        "distributed": {
            "enabled": "auto",
            "backend": "gloo",
            "init_method": f"file://{rendezvous_file}",
        },
    }
    context = setup_distributed(
        config, allow_cpu_distributed=True, backend="gloo"
    )
    try:
        assert context.rank == rank
        assert context.local_rank == rank
        assert context.world_size == world_size
        assert context.device.type == "cpu"
        assert float(reduce_scalar(rank + 1.0, context, "mean")) == 1.5
        assert float(reduce_scalar(rank + 1.0, context, "max")) == 2.0
        local_negative = -3.0 if rank == 0 else -1.0
        assert float(reduce_scalar(local_negative, context, "min")) == -3.0
        assert float(reduce_scalar(local_negative, context, "max")) == -1.0
        assert float(reduce_scalar(local_negative, context, "mean")) == -2.0
        reduced = reduce_metric_dict(
            {
                "esd_log_arg_min": local_negative,
                "esd_teacher_max": 0.25 if rank == 0 else 0.75,
                "esd_clamp_ratio": 0.2 if rank == 0 else 0.6,
            },
            context,
            min_keys={"esd_log_arg_min"},
            max_keys={"esd_teacher_max"},
        )
        assert float(reduced["esd_log_arg_min"]) == -3.0
        assert float(reduced["esd_teacher_max"]) == 0.75
        assert float(reduced["esd_clamp_ratio"]) == pytest.approx(0.4)

        epoch_meter = EpochMetricMeter(
            min_keys={"esd_log_arg_min"},
            max_keys={"esd_teacher_max"},
        )
        if rank == 0:
            epoch_meter.update({
                "esd_log_arg_min": -3.0,
                "esd_teacher_max": 0.4,
                "loss_esd": 0.2,
            })
            epoch_meter.update({
                "esd_log_arg_min": -2.0,
                "esd_teacher_max": 0.5,
                "loss_esd": 0.4,
            })
        else:
            epoch_meter.update({
                "esd_log_arg_min": -1.0,
                "esd_teacher_max": 0.7,
                "loss_esd": 0.6,
            })
            epoch_meter.update({
                "esd_log_arg_min": -4.0,
                "esd_teacher_max": 0.6,
                "loss_esd": 0.8,
            })
        local_epoch = epoch_meter.compute()
        assert local_epoch["esd_log_arg_min"] == (-3.0 if rank == 0 else -4.0)
        assert local_epoch["esd_teacher_max"] == (0.5 if rank == 0 else 0.7)
        assert local_epoch["loss_esd"] == pytest.approx(
            0.3 if rank == 0 else 0.7
        )
        global_epoch = reduce_epoch_metric_meter(epoch_meter, context)
        assert float(global_epoch["esd_log_arg_min"]) == -4.0
        assert float(global_epoch["esd_teacher_max"]) == pytest.approx(0.7)
        assert float(global_epoch["loss_esd"]) == pytest.approx(0.5)

        weighted_meter = EpochMetricMeter()
        if rank == 0:
            weighted_meter.update({"metric": 1.0})
            weighted_meter.update({"metric": 3.0})
        else:
            weighted_meter.update({"metric": 9.0})
        weighted = reduce_epoch_metric_meter(weighted_meter, context)
        assert float(weighted["metric"]) == pytest.approx(13.0 / 3.0)
        maxima = reduce_max_values(
            [float(rank + 1), float(10 - rank)],
            context,
        )
        torch.testing.assert_close(
            maxima.cpu(), torch.tensor([2.0, 10.0], dtype=torch.float64)
        )

        local_confusion = torch.zeros(3, 3, dtype=torch.int64)
        local_confusion[rank, rank] = rank + 1
        global_confusion = all_reduce_confusion_matrix(local_confusion, context)
        assert torch.equal(
            global_confusion,
            torch.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 0]]),
        )
        metrics = SegmentationMetrics(3, 2)
        metrics.confusion_matrix = global_confusion
        assert metrics.compute()["mIoU"] == 1.0

        model = _Composite()
        ddp = nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.1)
        with ddp.no_sync():
            ddp(torch.tensor([[float(rank), 1.0]])).sum().backward()
        ddp(torch.tensor([[float(rank + 1), 1.0]])).sum().backward()
        endpoint_gradient = model.endpoint_model[0].weight.grad.detach().clone()
        source_gradient = model.source_model.weight.grad.detach().clone()
        assert torch.isfinite(endpoint_gradient).all()
        assert torch.isfinite(source_gradient).all()
        optimizer.step()
        checksum = sum(parameter.detach().sum() for parameter in model.parameters())
        gathered = [torch.zeros_like(checksum) for _ in range(world_size)]
        dist.all_gather(gathered, checksum)
        assert all(torch.equal(gathered[0], value) for value in gathered[1:])

        frozen_model = _Composite(freeze_source=True)
        frozen_ddp = nn.parallel.DistributedDataParallel(frozen_model)
        frozen_ddp(torch.ones(1, 2)).sum().backward()
        assert frozen_model.source_model.weight.grad is None
        assert frozen_model.endpoint_model[0].weight.grad is not None

        if context.is_main_process:
            Path(output_dir, "rank0-only.pt").write_bytes(b"one writer")
    finally:
        cleanup_distributed(context)


def _swin_two_iteration_worker(
    rank: int, world_size: int, rendezvous_file: str
) -> None:
    os.environ.update({
        "GLOO_SOCKET_IFNAME": "lo",
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
    })
    context = setup_distributed(
        {
            "runtime": {"device": "cpu"},
            "distributed": {
                "enabled": True,
                "backend": "gloo",
                "init_method": f"file://{rendezvous_file}",
            },
        },
        allow_cpu_distributed=True,
        backend="gloo",
    )
    try:
        import model as model_module
        from transformers import SwinBackbone, SwinConfig

        def tiny_swin_loader(backbone_type, variant, pretrained):
            assert (backbone_type, variant, pretrained) == ("swin", "tiny", False)
            config = SwinConfig(
                image_size=224,
                patch_size=4,
                window_size=7,
                embed_dim=8,
                depths=[1, 1, 1, 1],
                num_heads=[1, 2, 4, 8],
                out_features=["stage1", "stage2", "stage3", "stage4"],
            )
            config._attn_implementation = "eager"
            backbone = SwinBackbone(config)
            backbone.swin.layernorm.requires_grad_(False)
            return backbone, list(backbone.channels)

        model_module.load_transformer_image_backbone = tiny_swin_loader
        encoder = model_module.TransformerImageEncoder(
            "swin", "tiny", False, False,
            "ddp_fpn_merge", 8, 4, 4, True,
        )
        ddp = nn.parallel.DistributedDataParallel(
            encoder, find_unused_parameters=False
        )
        optimizer = torch.optim.SGD(ddp.parameters(), lr=1.0e-3)
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            image = torch.randn(1, 3, 64, 64) + rank + iteration
            output = ddp(image)
            assert output.shape == (1, 4, 16, 16)
            output.float().square().mean().backward()
            missing = [
                name for name, parameter in encoder.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            assert missing == []
            optimizer.step()
    finally:
        cleanup_distributed(context)


def test_world_size_one_fallback(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    context = setup_distributed({
        "runtime": {"device": "cpu"},
        "distributed": {"enabled": "auto"},
    })
    assert not context.distributed
    assert context.rank == context.local_rank == 0
    assert context.world_size == 1
    assert context.is_main_process


def test_gloo_two_process_reductions_gradients_no_sync_and_rank0_write(tmp_path):
    rendezvous_file = tmp_path / "gloo-rendezvous"
    mp.spawn(
        _gloo_worker,
        args=(2, str(rendezvous_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    assert (tmp_path / "rank0-only.pt").read_bytes() == b"one writer"


def test_swin_two_rank_two_iteration_ddp_has_no_unused_parameters(tmp_path):
    rendezvous_file = tmp_path / "swin-gloo-rendezvous"
    mp.spawn(
        _swin_two_iteration_worker,
        args=(2, str(rendezvous_file)),
        nprocs=2,
        join=True,
    )


def test_distributed_eval_sampler_has_no_duplicates():
    dataset = list(range(11))
    shards = [
        list(DistributedEvalSampler(dataset, rank=rank, world_size=3))
        for rank in range(3)
    ]
    flattened = [index for shard in shards for index in shard]
    assert sorted(flattened) == list(range(11))
    assert len(flattened) == len(set(flattened))


def test_global_batch_size_divisibility():
    assert validate_global_batch_size(4, 2) == 2
    with pytest.raises(ValueError, match="must be divisible"):
        validate_global_batch_size(5, 2)


def test_reduction_keys_cannot_be_both_min_and_max():
    context = DistributedContext(
        distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        is_main_process=True,
        backend=None,
    )
    with pytest.raises(ValueError, match="both min and max"):
        reduce_metric_dict(
            {"metric": 1.0},
            context,
            min_keys={"metric"},
            max_keys={"metric"},
        )
    with pytest.raises(ValueError, match="both min and max"):
        EpochMetricMeter(
            min_keys={"metric"},
            max_keys={"metric"},
        )
