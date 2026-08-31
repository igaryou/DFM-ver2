import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from dfm_stabilization import uncertainty_weighted_esd_loss
from training_objectives import DDPCompatibleTrainingModel


class TinyEndpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(4, 4, 1)
        self.image_encoder = nn.Conv2d(3, 4, 1)
        self.time_scale = nn.Parameter(torch.linspace(-0.1, 0.1, 4))

    def encode_image(self, image):
        return self.image_encoder(image)

    def forward_logits_with_image_feat(self, x, image_feat, s, t):
        return (
            self.projection(x) + image_feat
            + (s + 0.5 * t)[:, None, None, None]
            * self.time_scale[None, :, None, None]
        )

    def forward_logits(self, x, image, s, t):
        return self.forward_logits_with_image_feat(
            x, self.encode_image(image), s, t
        )


def _config():
    return {
        "experiment": {"stage": "joint_training"},
        "runtime": {"amp": False, "amp_dtype": "bf16"},
        "dataset": {"num_classes": 4},
        "model": {"state_downsample_factor": 1},
        "source": {
            "prior_type": "gaussian", "prior_noise_std": 1.0,
            "var_weight": 0.0, "align_weight": 0.0,
            "use_loss_align": False,
        },
        "flow": {"time_eps": 1.0e-5, "probability_eps": 1.0e-8},
        "time_sampling": {
            "min_time": 0.0, "max_time": 0.9, "min_gap": 1.0e-3,
        },
        "training": {"label_smoothing": 0.0},
        "loss": {
            "ignore_index": None, "mask_pixel_losses": False,
            "primary": {
                "weight": 1.0,
                "adaptive_weighting": {"enabled": True, "r": 0.5, "c": 0.01},
            },
            "consistency": {
                "enabled": True, "type": "esd", "weight": 1.0,
                "max_weight": 1.0, "start_epoch": 0,
                "start": {"unit": "optimizer_step", "value": 0},
                "warmup_epochs": 0, "warmup_steps": 0,
                "precision": {
                    "jvp_dtype": "fp32", "numerical_dtype": "fp32",
                    "debug_assertions": True,
                },
                "adaptive_kl": {
                    "enabled": True, "c": 1.0e-6, "r": 0.5,
                    "normalize_mean": False, "max_weight": None,
                },
                "invalid_teacher": {
                    "strategy": "mask_pixel", "log_eps": 1.0e-6,
                    "skip_batch_threshold": None,
                },
                "learnable_weight": {
                    "enabled": True, "dependency": "st",
                    "type": "uncertainty", "time_embedding_dim": 8,
                    "hidden_dim": 8, "init_effective_weight": 1.0,
                    "lr": None, "weight_decay": 0.0,
                },
            },
        },
    }


def _worker(rank, init_file):
    dist.init_process_group(
        "gloo", rank=rank, world_size=2,
        init_method=f"file://{init_file}",
    )
    try:
        torch.manual_seed(101 + rank)
        adapter = DDPCompatibleTrainingModel(TinyEndpoint(), None, _config())
        model = DistributedDataParallel(adapter)
        image = torch.randn(2, 3, 3, 4)
        target = torch.randint(0, 4, (2, 3, 4))
        result = model(
            operation="joint_objectives", image=image, target=target,
            epoch_index=0, progress_in_epoch=0.0, optimizer_step=0,
        )
        assert torch.isfinite(result["loss"])
        result["loss"].backward()
        assert adapter.endpoint_model.projection.weight.grad is not None
        assert torch.isfinite(adapter.endpoint_model.projection.weight.grad).all()
        assert adapter.consistency_weight_model.output.bias.grad is not None
        assert torch.isfinite(adapter.consistency_weight_model.output.bias.grad).all()
        checksum = adapter.consistency_weight_model.output.bias.grad.detach().clone()
        gathered = [torch.zeros_like(checksum) for _ in range(2)]
        dist.all_gather(gathered, checksum)
        torch.testing.assert_close(gathered[0], gathered[1])
    finally:
        dist.destroy_process_group()


def test_two_process_esd_learnable_backward():
    if not dist.is_available() or not dist.is_gloo_available():
        return
    with tempfile.TemporaryDirectory() as directory:
        init_file = str(Path(directory) / "init")
        mp.spawn(_worker, args=(init_file,), nprocs=2, join=True)


def _unequal_count_worker(rank, init_file):
    dist.init_process_group(
        "gloo", rank=rank, world_size=2,
        init_method=f"file://{init_file}",
    )
    try:
        network = DistributedDataParallel(nn.Linear(1, 1, bias=False))
        with torch.no_grad():
            network.module.weight.fill_(0.2)
        if rank == 0:
            sample_loss = torch.tensor([1.0, 50.0])
            valid = torch.tensor([True, False])
        else:
            sample_loss = torch.tensor([3.0, 5.0, 70.0])
            valid = torch.tensor([True, True, False])
        weight_logit = network(torch.ones(sample_loss.numel(), 1)).squeeze(1)
        objective, _ = uncertainty_weighted_esd_loss(
            sample_loss, valid, weight_logit
        )
        objective.backward()
        expected = torch.tensor(
            1.0 - torch.exp(torch.tensor(-0.2)) * (1.0 + 3.0 + 5.0) / 3.0
        )
        torch.testing.assert_close(network.module.weight.grad.squeeze(), expected)
    finally:
        dist.destroy_process_group()


def test_unequal_rank_valid_counts_match_global_mean_gradient():
    if not dist.is_available() or not dist.is_gloo_available():
        return
    with tempfile.TemporaryDirectory() as directory:
        init_file = str(Path(directory) / "init")
        mp.spawn(_unequal_count_worker, args=(init_file,), nprocs=2, join=True)
