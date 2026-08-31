import copy

import pytest
import torch
import torch.nn as nn

from dfm_stabilization import (
    GradientSurgeryAccumulator,
    apply_global_gradient_surgery,
    project_conflicting_gradient,
)


class _AccumulatorAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.endpoint_model = nn.Linear(2, 1, bias=False)
        self.source_model = nn.Linear(1, 1, bias=False)
        self.psd_weight_model = nn.Linear(1, 1, bias=False)


def _objectives(adapter, diagonal, psd, other, source, weight):
    endpoint_parameter = adapter.endpoint_model.weight.flatten()
    source_parameter = adapter.source_model.weight.flatten()[0]
    weight_parameter = adapter.psd_weight_model.weight.flatten()[0]
    diagonal_objective = (diagonal @ endpoint_parameter).mean()
    psd_objective = (psd @ endpoint_parameter).mean()
    source_objective = (other @ endpoint_parameter).mean()
    loss = (
        diagonal_objective
        + psd_objective
        + source_objective
        + source.mean() * source_parameter
        + weight.mean() * weight_parameter
    )
    return {
        "diagonal_objective": diagonal_objective,
        "psd_objective": psd_objective,
        "source_objective": source_objective,
        "loss": loss,
    }


def _disabled_scaler():
    return torch.amp.GradScaler("cuda", enabled=False)


def _gradient_snapshot(adapter):
    return tuple(
        parameter.grad.detach().clone()
        for parameter in (
            adapter.endpoint_model.weight,
            adapter.source_model.weight,
            adapter.psd_weight_model.weight,
        )
    )


def test_two_microbatches_equal_one_full_batch_for_all_parameter_groups():
    diagonal = torch.tensor([
        [1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 3.0],
        [3.0, 1.0], [1.0, 2.0], [2.0, 2.0], [4.0, 1.0],
    ])
    psd = torch.tensor([
        [-2.0, 1.0], [1.0, -2.0], [-1.0, 0.0], [0.0, -1.0],
        [-3.0, 1.0], [1.0, -3.0], [-2.0, 0.0], [0.0, -2.0],
    ])
    other = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 5.0
    source = torch.arange(1, 9, dtype=torch.float32)
    weight = torch.arange(2, 10, dtype=torch.float32)
    full = _AccumulatorAdapter()
    accumulated = copy.deepcopy(full)

    full_stats = apply_global_gradient_surgery(
        adapter=full,
        objectives=_objectives(full, diagonal, psd, other, source, weight),
        scaler=_disabled_scaler(),
    )
    accumulator = GradientSurgeryAccumulator()
    for indices in (slice(0, 4), slice(4, 8)):
        accumulator.accumulate(
            adapter=accumulated,
            objectives=_objectives(
                accumulated, diagonal[indices], psd[indices], other[indices],
                source[indices], weight[indices],
            ),
            scaler=_disabled_scaler(),
        )
        assert all(parameter.grad is None for parameter in accumulated.parameters())
    stats = accumulator.finalize(adapter=accumulated, scaler=_disabled_scaler())

    for actual, expected in zip(
        _gradient_snapshot(accumulated), _gradient_snapshot(full), strict=True
    ):
        torch.testing.assert_close(actual, expected)
    for key, expected in full_stats.items():
        if key not in {
            "gradient_surgery_accumulated_microbatches",
            "gradient_surgery_accumulation_enabled",
        }:
            torch.testing.assert_close(stats[key], expected)
    full_optimizer = torch.optim.SGD(full.parameters(), lr=0.05)
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.05)
    full_optimizer.step()
    accumulated_optimizer.step()
    for actual, expected in zip(accumulated.parameters(), full.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    assert stats["gradient_surgery_accumulated_microbatches"] == 2
    assert stats["gradient_surgery_accumulation_enabled"] == 1
    assert accumulator.is_empty


def test_projection_happens_after_averaging_not_once_per_microbatch():
    adapter = _AccumulatorAdapter()
    accumulator = GradientSurgeryAccumulator()
    diagonal_vectors = (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]))
    psd_vectors = (torch.tensor([[-2.0, 1.0]]), torch.tensor([[1.0, -2.0]]))
    zeros = torch.zeros(1)
    zero_vectors = torch.zeros(1, 2)
    wrong_microbatch_gradients = []
    for diagonal, psd in zip(diagonal_vectors, psd_vectors, strict=True):
        accumulator.accumulate(
            adapter=adapter,
            objectives=_objectives(
                adapter, diagonal, psd, zero_vectors, zeros, zeros
            ),
            scaler=_disabled_scaler(),
        )
        projected = project_conflicting_gradient([diagonal[0]], [psd[0]]).projected[0]
        wrong_microbatch_gradients.append(diagonal[0] + projected)
    accumulator.finalize(adapter=adapter, scaler=_disabled_scaler())

    torch.testing.assert_close(adapter.endpoint_model.weight.grad, torch.tensor([[0.5, 0.5]]))
    wrong = torch.stack(wrong_microbatch_gradients).mean(0)
    torch.testing.assert_close(wrong, torch.tensor([1.0, 1.0]))
    assert not torch.allclose(adapter.endpoint_model.weight.grad.flatten(), wrong)


def test_partial_window_uses_actual_count_and_reset_prevents_state_leakage():
    adapter = _AccumulatorAdapter()
    accumulator = GradientSurgeryAccumulator()
    diagonal = torch.tensor([[2.0, 0.0]])
    psd = torch.tensor([[0.0, 3.0]])
    zero_vectors = torch.zeros(1, 2)
    zeros = torch.zeros(1)
    for expected_count in (1, 1):
        accumulator.accumulate(
            adapter=adapter,
            objectives=_objectives(
                adapter, diagonal, psd, zero_vectors, zeros, zeros
            ),
            scaler=_disabled_scaler(),
        )
        stats = accumulator.finalize(adapter=adapter, scaler=_disabled_scaler())
        assert stats["gradient_surgery_accumulated_microbatches"] == expected_count
        torch.testing.assert_close(adapter.endpoint_model.weight.grad, torch.tensor([[2.0, 3.0]]))
        assert accumulator.is_empty
        adapter.zero_grad(set_to_none=True)
    with pytest.raises(RuntimeError, match="empty"):
        accumulator.finalize(adapter=adapter, scaler=_disabled_scaler())


def test_one_microbatch_accumulator_matches_backward_compatible_wrapper():
    wrapped = _AccumulatorAdapter()
    direct = copy.deepcopy(wrapped)
    diagonal = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    psd = torch.tensor([[-2.0, 1.0], [1.0, -2.0]])
    other = torch.ones(2, 2)
    source = torch.tensor([1.0, 3.0])
    weight = torch.tensor([2.0, 4.0])
    wrapped_stats = apply_global_gradient_surgery(
        adapter=wrapped,
        objectives=_objectives(wrapped, diagonal, psd, other, source, weight),
        scaler=_disabled_scaler(),
    )
    accumulator = GradientSurgeryAccumulator()
    accumulator.accumulate(
        adapter=direct,
        objectives=_objectives(direct, diagonal, psd, other, source, weight),
        scaler=_disabled_scaler(),
    )
    direct_stats = accumulator.finalize(adapter=direct, scaler=_disabled_scaler())
    for actual, expected in zip(_gradient_snapshot(direct), _gradient_snapshot(wrapped), strict=True):
        torch.testing.assert_close(actual, expected)
    for key in wrapped_stats:
        torch.testing.assert_close(direct_stats[key], wrapped_stats[key])


class _FakeEnabledScaler:
    def __init__(self, scale=8.0):
        self.value = scale

    def is_enabled(self):
        return True

    def scale(self, objective):
        return objective * self.value

    def get_scale(self):
        return self.value


def test_enabled_scaler_keeps_assigned_grads_scaled_but_reports_unscaled_stats():
    unscaled = _AccumulatorAdapter()
    scaled = copy.deepcopy(unscaled)
    diagonal = torch.tensor([[1.0, 0.0]])
    psd = torch.tensor([[-1.0, 1.0]])
    other = torch.zeros(1, 2)
    source = torch.tensor([2.0])
    weight = torch.tensor([3.0])
    reference_stats = apply_global_gradient_surgery(
        adapter=unscaled,
        objectives=_objectives(unscaled, diagonal, psd, other, source, weight),
        scaler=_disabled_scaler(),
    )
    scaled_stats = apply_global_gradient_surgery(
        adapter=scaled,
        objectives=_objectives(scaled, diagonal, psd, other, source, weight),
        scaler=_FakeEnabledScaler(),
    )
    for actual, expected in zip(_gradient_snapshot(scaled), _gradient_snapshot(unscaled), strict=True):
        torch.testing.assert_close(actual, expected * 8.0)
    for key in reference_stats:
        torch.testing.assert_close(scaled_stats[key], reference_stats[key])


def test_scaler_scale_must_not_change_inside_accumulation_window():
    adapter = _AccumulatorAdapter()
    scaler = _FakeEnabledScaler()
    accumulator = GradientSurgeryAccumulator()
    args = (
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        torch.zeros(1, 2),
        torch.zeros(1),
        torch.zeros(1),
    )
    accumulator.accumulate(
        adapter=adapter, objectives=_objectives(adapter, *args), scaler=scaler
    )
    scaler.value = 4.0
    with pytest.raises(RuntimeError, match="scale changed"):
        accumulator.accumulate(
            adapter=adapter, objectives=_objectives(adapter, *args), scaler=scaler
        )
