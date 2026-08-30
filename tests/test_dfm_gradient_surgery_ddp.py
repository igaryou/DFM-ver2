import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from dfm_stabilization import apply_global_gradient_surgery


class _DistributedToyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.endpoint_model = nn.Linear(2, 1, bias=False)
        self.source_model = None
        self.psd_weight_model = None


def _worker(rank: int, rendezvous: str, output: str):
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        adapter = _DistributedToyAdapter()
        endpoint = adapter.endpoint_model.weight.flatten()
        diagonal_vectors = (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
        psd_vectors = (torch.tensor([-2.0, 1.0]), torch.tensor([1.0, -2.0]))
        diagonal = (endpoint * diagonal_vectors[rank]).sum()
        psd = (endpoint * psd_vectors[rank]).sum()
        zero = endpoint.sum() * 0.0
        objectives = {
            "diagonal_objective": diagonal,
            "psd_objective": psd,
            "source_objective": zero,
            "loss": diagonal + psd,
        }
        apply_global_gradient_surgery(
            adapter=adapter,
            objectives=objectives,
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            world_size=2,
        )
        if rank == 0:
            torch.save(adapter.endpoint_model.weight.grad, output)
    finally:
        dist.destroy_process_group()


def test_two_process_projects_global_gradients_not_local_projections(tmp_path):
    rendezvous = tmp_path / "surgery-rendezvous"
    output = tmp_path / "gradient.pt"
    try:
        mp.spawn(_worker, args=(str(rendezvous), str(output)), nprocs=2, join=True)
    except Exception as exc:
        if "gloo" in str(exc).lower() or "socket" in str(exc).lower():
            pytest.skip(f"Gloo unavailable in this environment: {exc}")
        raise
    gradient = torch.load(output, weights_only=True)
    # Global Gd=[.5,.5], Gp=[-.5,-.5], so global projection removes Gp.
    torch.testing.assert_close(gradient, torch.tensor([[0.5, 0.5]]))
    # Local projection then averaging would instead yield [1,1].
    assert not torch.allclose(gradient, torch.tensor([[1.0, 1.0]]))
