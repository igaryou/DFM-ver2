import math

import pytest
import torch
import torch.nn as nn

from dfm_stabilization import (
    PSDTimeWeightNetwork,
    apply_global_gradient_surgery,
    project_conflicting_gradient,
    uncertainty_weighted_psd_loss,
)
from losses import adaptive_diagonal_cross_entropy


def test_adaptive_diagonal_closed_form_detached_void_and_normalization():
    logits = torch.tensor(
        [[[[3.0, -1.0, 0.0]], [[-1.0, 3.0, 0.0]], [[0.0, 0.0, 4.0]]]],
        requires_grad=True,
    )
    target = torch.tensor([[[0, 1, 19]]])
    result = adaptive_diagonal_cross_entropy(
        logits, target, r=0.5, c=0.01, ignore_index=19
    )
    q = logits.softmax(1)
    mismatch = q.square().sum(1) - 2 * q.gather(1, target.clamp_max(2)[:, None]).squeeze(1) + 1
    weights = (mismatch + 0.01).pow(-0.5)
    ce = torch.nn.functional.cross_entropy(logits, target, ignore_index=19, reduction="none")
    expected = (weights[:, :, :2] * ce[:, :, :2]).sum() / 2
    torch.testing.assert_close(result.loss, expected)
    assert not result.adaptive_weight.requires_grad
    assert result.valid_mask.sum() == 2
    result.loss.backward()
    assert logits.grad is not None


def test_adaptive_weight_is_larger_near_target_and_limit_is_ten():
    target = torch.tensor([[[0, 0]]])
    logits = torch.tensor([[[[30.0, 0.0]], [[-30.0, 0.0]]]])
    result = adaptive_diagonal_cross_entropy(logits, target, r=0.5, c=0.01)
    assert result.adaptive_weight[0, 0, 0] == pytest.approx(10.0, rel=1e-5)
    assert result.adaptive_weight[0, 0, 0] > result.adaptive_weight[0, 0, 1]


def test_adaptive_diagonal_empty_valid_pixels_and_full_resolution_shape():
    logits = torch.randn(1, 20, 8, 12, requires_grad=True)
    target = torch.full((1, 8, 12), 19)
    result = adaptive_diagonal_cross_entropy(logits, target, ignore_index=19)
    assert result.loss.item() == 0.0
    assert result.stats["diagonal_adaptive_weight_mean"].item() == 0.0
    result.loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_psd_weight_network_initialization_and_manual_uncertainty_objective():
    network = PSDTimeWeightNetwork(32, 64, 0.5)
    s = torch.tensor([0.0, 0.2, 0.9])
    w = network(s)
    torch.testing.assert_close(w, torch.full_like(w, math.log(2.0)))
    torch.testing.assert_close(torch.exp(-w), torch.full_like(w, 0.5))
    losses = torch.tensor([1.0, 3.0, 100.0], requires_grad=True)
    valid = torch.tensor([True, True, False])
    objective, stats = uncertainty_weighted_psd_loss(losses, valid, w)
    expected = torch.tensor((0.5 * 1.0 + math.log(2.0) + 0.5 * 3.0 + math.log(2.0)) / 2)
    torch.testing.assert_close(objective, expected)
    objective.backward()
    assert network.output.bias.grad is not None
    assert losses.grad[0] == pytest.approx(0.25)
    assert losses.grad[2] == 0
    assert stats["psd_effective_multiplier_mean"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "diagonal,psd,expected,conflict",
    [
        ([1.0, 0.0], [1.0, 1.0], [1.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 1.0], [0.0, 1.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], [0.0, 1.0], 0.0),
    ],
)
def test_gradient_projection_toy_cases(diagonal, psd, expected, conflict):
    gd, gp = torch.tensor(diagonal), torch.tensor(psd)
    result = project_conflicting_gradient([gd], [gp])
    torch.testing.assert_close(result.projected[0], torch.tensor(expected))
    assert result.stats["gradient_surgery_conflict"] == conflict
    if conflict:
        assert torch.dot(gd, result.projected[0]).abs() < 1e-6


class _ToyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.endpoint_model = nn.Linear(2, 1, bias=False)
        self.source_model = nn.Linear(1, 1, bias=False)
        self.psd_weight_model = nn.Linear(1, 1, bias=False)


def test_surgery_projects_endpoint_only_and_preserves_other_gradients():
    adapter = _ToyAdapter()
    endpoint = adapter.endpoint_model.weight.flatten()
    source = adapter.source_model.weight.flatten()
    weight = adapter.psd_weight_model.weight.flatten()
    diagonal = endpoint[0]
    psd = -endpoint[0] + endpoint[1] + 3 * source[0] + 4 * weight[0]
    source_objective = 2 * endpoint[0] + 5 * source[0]
    objectives = {
        "diagonal_objective": diagonal,
        "psd_objective": psd,
        "source_objective": source_objective,
        "loss": diagonal + psd + source_objective,
    }
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    stats = apply_global_gradient_surgery(
        adapter=adapter, objectives=objectives, scaler=scaler
    )
    torch.testing.assert_close(adapter.endpoint_model.weight.grad, torch.tensor([[3.0, 1.0]]))
    torch.testing.assert_close(adapter.source_model.weight.grad, torch.tensor([[8.0]]))
    torch.testing.assert_close(adapter.psd_weight_model.weight.grad, torch.tensor([[4.0]]))
    assert stats["gradient_surgery_removed_component_norm"] == pytest.approx(1.0)


def test_zero_schedule_has_zero_weight_network_gradient():
    network = PSDTimeWeightNetwork()
    s = torch.tensor([0.2, 0.7])
    objective, _ = uncertainty_weighted_psd_loss(
        torch.tensor([1.0, 2.0]), torch.tensor([True, True]), network(s)
    )
    (objective * 0.0).backward()
    assert all(
        parameter.grad is None or torch.equal(parameter.grad, torch.zeros_like(parameter))
        for parameter in network.parameters()
    )
