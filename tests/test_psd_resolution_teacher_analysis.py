from __future__ import annotations

import math

import torch

from psd_resolution_teacher_analysis import (
    QualityAccumulator,
    full_resolution_psd,
    resize_probability,
    semantic_prediction,
)


def _probability(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


def test_probability_resize_renormalizes_and_preserves_graph_roles() -> None:
    teacher = _probability(torch.randn(1, 20, 2, 4)).detach()
    student_logits = torch.randn(1, 20, 2, 4, requires_grad=True)
    student = _probability(student_logits)
    loss, teacher_full, student_full = full_resolution_psd(
        teacher,
        student,
        full_size=(8, 16),
        valid_mask_full=torch.ones(1, 8, 16, dtype=torch.bool),
        probability_eps=1.0e-8,
    )
    assert torch.allclose(teacher_full.sum(dim=1), torch.ones(1, 8, 16), atol=1e-6)
    assert torch.allclose(student_full.sum(dim=1), torch.ones(1, 8, 16), atol=1e-6)
    assert teacher_full.requires_grad is False
    assert student_full.requires_grad is True
    assert loss.requires_grad is True
    student_gradient, = torch.autograd.grad(loss, student_logits)
    assert torch.isfinite(student_gradient).all()
    assert student_gradient.abs().sum() > 0


def test_all_valid_full_psd_matches_manual_cross_entropy() -> None:
    teacher = _probability(torch.randn(1, 20, 2, 2)).detach()
    student = _probability(torch.randn(1, 20, 2, 2, requires_grad=True))
    loss, teacher_full, student_full = full_resolution_psd(
        teacher, student, full_size=(4, 4),
        valid_mask_full=torch.ones(1, 4, 4, dtype=torch.bool),
        probability_eps=1.0e-8,
    )
    manual = -(teacher_full * student_full.clamp_min(1.0e-8).log()).sum(1).mean()
    assert torch.allclose(loss, manual)


def test_void_pixels_are_ignored_by_full_psd() -> None:
    teacher = torch.zeros(1, 20, 1, 2)
    student = torch.zeros(1, 20, 1, 2, requires_grad=True)
    teacher[:, 0] = 1.0
    with torch.no_grad():
        student[:, 0, 0, 0] = 1.0
        student[:, 1, 0, 1] = 1.0
    loss, teacher_full, student_full = full_resolution_psd(
        teacher, student, full_size=(1, 2),
        valid_mask_full=torch.tensor([[[True, False]]]),
        probability_eps=1.0e-8,
    )
    loss_map = -(teacher_full * student_full.clamp_min(1.0e-8).log()).sum(dim=1)
    assert torch.allclose(loss, loss_map[0, 0, 0])
    assert loss_map[0, 0, 1] > 1.0


def test_helper_reuses_input_probability_values() -> None:
    teacher = _probability(torch.randn(1, 20, 2, 2)).detach()
    resized = resize_probability(teacher, (2, 2), 1.0e-8, detach=True)
    expected = teacher.clamp_min(1.0e-8)
    expected = expected / expected.sum(1, keepdim=True)
    assert torch.allclose(resized, expected)


def test_semantic_prediction_excludes_void_class() -> None:
    probability = torch.zeros(1, 20, 1, 1)
    probability[:, 19] = 0.9
    probability[:, 3] = 0.1
    prediction, confidence = semantic_prediction(probability)
    assert prediction.item() == 3
    assert math.isclose(confidence.item(), 0.1, abs_tol=1.0e-6)


def test_teacher_quality_ignores_void_gt_and_counts_high_confidence_wrong() -> None:
    probability = torch.zeros(1, 20, 1, 3)
    probability[0, 0, 0, 0] = 0.95  # correct
    probability[0, 1, 0, 1] = 0.95  # wrong with high confidence
    probability[0, 4, 0, 2] = 1.0   # ignored void GT
    probability += 0.05 / 20.0
    probability /= probability.sum(dim=1, keepdim=True)
    target = torch.tensor([[[0, 2, 19]]])
    accumulator = QualityAccumulator(0.9, 1.0e-8)
    batch, _ = accumulator.update(probability, target)
    result = accumulator.compute()
    assert math.isclose(batch["pixel_accuracy"], 0.5)
    assert math.isclose(result["wrong_pixel_fraction"], 0.5)
    assert math.isclose(result["high_confidence_wrong_valid_fraction"], 0.5)
    assert math.isclose(result["high_confidence_wrong_among_wrong_fraction"], 1.0)
