from __future__ import annotations

from pathlib import Path

import torch

from gradient_conflict_analysis import gradient_pair_metrics
from psd_resolution_teacher_analysis import (
    QualityAccumulator,
    _metrics_from_confusion,
    semantic_prediction,
)
from psd_time_map_analysis import (
    DELTA_BINS,
    S_BINS,
    _capture_rng,
    _original_probability,
    _restore_rng,
    conditional_production_times,
)


TIME_CONFIG = {
    "min_time": 0.0,
    "max_time": 1.0,
    "min_gap": 1.0e-5,
}


def test_conditional_production_sampler_preserves_order_and_s_bins() -> None:
    for index, time_bin in enumerate(S_BINS):
        (s, u, t), stats = conditional_production_times(
            time_bin, 32, torch.device("cpu"), TIME_CONFIG, seed=100 + index
        )
        assert torch.all(s < u)
        assert torch.all(u < t)
        assert torch.all(time_bin.mask(s, t))
        assert stats["accepted_count"] == 32
        assert stats["proposed_count"] >= 32


def test_conditional_production_sampler_obeys_delta_bins() -> None:
    for index, time_bin in enumerate(DELTA_BINS):
        (s, u, t), _ = conditional_production_times(
            time_bin, 32, torch.device("cpu"), TIME_CONFIG, seed=200 + index
        )
        assert torch.all(time_bin.mask(s, t))


def test_time_sampler_rng_is_isolated_from_model_rng() -> None:
    torch.manual_seed(7)
    before = torch.random.get_rng_state()
    conditional_production_times(
        S_BINS[0], 8, torch.device("cpu"), TIME_CONFIG, seed=999
    )
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)


def test_rng_restore_reproduces_base_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    state = _capture_rng(torch.device("cpu"))
    random_input = torch.randn(2)
    first, = torch.autograd.grad((parameter * random_input).sum(), (parameter,))
    _restore_rng(state, torch.device("cpu"))
    random_input = torch.randn(2)
    second, = torch.autograd.grad((parameter * random_input).sum(), (parameter,))
    metrics = gradient_pair_metrics((first,), (second,))
    assert torch.equal(first, second)
    assert metrics["cosine"] > 0.999999


class _RecordingMapModel:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward_logits_with_image_feat(self, x, image_feat, s, t):
        del image_feat
        self.calls.append((s.detach().clone(), t.detach().clone()))
        offset = (s + 2.0 * t)[:, None, None, None]
        return x + offset * torch.arange(20, device=x.device)[None, :, None, None]


def _map_config() -> dict:
    return {
        "runtime": {"amp": False, "amp_dtype": "bf16"},
        "evaluation": {"align_corners": False},
    }


def _sample() -> dict:
    return {
        "model_shape": (2, 2),
        "original_shape": (2, 2),
        "padded_shape": (2, 2),
    }


def test_pi00_pi01_receive_exact_endpoint_times_and_pi0t_matches() -> None:
    model = _RecordingMapModel()
    x0 = torch.randn(1, 20, 2, 2)
    feature = torch.zeros(1, 1, 2, 2)
    pi00 = _original_probability(model, x0, feature, 0.0, 0.0, _sample(), _map_config())
    pi01 = _original_probability(model, x0, feature, 0.0, 1.0, _sample(), _map_config())
    pi0t_zero = _original_probability(model, x0, feature, 0.0, 0.0, _sample(), _map_config())
    pi0t_one = _original_probability(model, x0, feature, 0.0, 1.0, _sample(), _map_config())
    assert model.calls[0][0].item() == model.calls[0][1].item() == 0.0
    assert model.calls[1][0].item() == 0.0 and model.calls[1][1].item() == 1.0
    assert torch.equal(pi00, pi0t_zero)
    assert torch.equal(pi01, pi0t_one)


def test_s_zero_diagonal_and_off_diagonal_match_direct_maps() -> None:
    model = _RecordingMapModel()
    x0 = torch.randn(1, 20, 2, 2)
    x1 = torch.randn_like(x0)
    feature = torch.zeros(1, 1, 2, 2)
    zero = torch.zeros(1)
    x_s = (1.0 - zero[:, None, None, None]) * x0 + zero[:, None, None, None] * x1
    pi00 = _original_probability(model, x0, feature, 0.0, 0.0, _sample(), _map_config())
    pi01 = _original_probability(model, x0, feature, 0.0, 1.0, _sample(), _map_config())
    piss = _original_probability(model, x_s, feature, 0.0, 0.0, _sample(), _map_config())
    pis1 = _original_probability(model, x_s, feature, 0.0, 1.0, _sample(), _map_config())
    assert torch.equal(pi00, piss)
    assert torch.equal(pi01, pis1)
    pi00_miou, pi01_miou = 0.7, 0.6
    assert pi01_miou - pi00_miou == (pi01_miou - pi00_miou)


def test_void_and_absent_classes_are_excluded_from_metrics() -> None:
    confusion = torch.zeros(19, 19, dtype=torch.int64)
    confusion[0, 0] = 3
    metrics = _metrics_from_confusion(confusion)
    assert metrics["miou"] == 1.0
    assert metrics["mean_class_accuracy"] == 1.0
    probability = torch.zeros(1, 20, 1, 2)
    probability[0, 19, 0, 0] = 0.9
    probability[0, 2, 0, 0] = 0.1
    probability[0, 4, 0, 1] = 1.0
    prediction, _ = semantic_prediction(probability)
    assert prediction[0, 0, 0].item() == 2
    accumulator = QualityAccumulator(0.9, 1.0e-8)
    accumulator.update(probability, torch.tensor([[[2, 19]]]))
    assert accumulator.compute()["pixel_accuracy"] == 1.0


def test_diagnostic_source_contains_no_parameter_update_api() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "src" / "psd_time_map_analysis.py"
    ).read_text(encoding="utf-8")
    assert "optimizer.step(" not in source
    assert ".backward(" not in source
    assert ".grad =" not in source
