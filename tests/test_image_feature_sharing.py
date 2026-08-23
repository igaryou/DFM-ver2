from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from torch.func import jvp

import losses
import training_objectives
from discrete_flow_maps import linear_path, sample_prior
from training_objectives import (
    DDPCompatibleTrainingModel,
    compute_model_training_objectives,
)


class TinyUNet(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.projection = nn.Conv2d(classes, classes, 1)
        self.time_scale = nn.Parameter(torch.linspace(-0.15, 0.15, classes))

    def forward(self, fused, s, t):
        time = (0.3 * s + 0.7 * t)[:, None, None, None]
        return (
            self.projection(fused)
            + time * self.time_scale[None, :, None, None]
        )


class TinyEndpoint(nn.Module):
    def __init__(self, classes: int = 4):
        super().__init__()
        self.image_encoder = nn.Conv2d(3, classes, 1)
        self.mask_encoder = nn.Conv2d(classes, classes, 1)
        self.unet = TinyUNet(classes)
        self.feature_grad_flags: list[bool] = []

    def encode_image(self, image):
        return self.image_encoder(image)

    def forward_logits_with_image_feat(self, x, image_feat, s, t):
        self.feature_grad_flags.append(image_feat.requires_grad)
        return self.unet(self.mask_encoder(x) + image_feat, s, t)

    def forward_logits(self, x, image, s, t):
        return self.forward_logits_with_image_feat(
            x, self.encode_image(image), s, t
        )


class TinySource(nn.Module):
    def __init__(self, classes: int = 4):
        super().__init__()
        self.mean = nn.Conv2d(3, classes, 1)
        self.log_variance = nn.Conv2d(3, classes, 1)

    def forward(self, image):
        mean = self.mean(image)
        log_variance = self.log_variance(image).clamp(-2.0, 2.0)
        return mean, mean, log_variance


def _config(
    loss_type: str,
    *,
    jvp_dtype: str | None = None,
    source_prior: str = "gaussian",
) -> dict:
    if jvp_dtype is None and loss_type != "psd":
        jvp_dtype = "bf16"
    return {
        "runtime": {"amp": True, "amp_dtype": "bf16"},
        "dataset": {"num_classes": 4},
        "source": {
            "prior_type": source_prior,
            "prior_noise_std": 1.0,
            "var_weight": 0.2,
            "align_weight": 0.3,
            "align_eps": 1.0e-6,
            "use_loss_align": source_prior == "image_gaussian",
        },
        "flow": {"time_eps": 1.0e-5, "probability_eps": 1.0e-8},
        "time_sampling": {
            "min_time": 0.0,
            "max_time": 1.0,
            "min_gap": 1.0e-4,
        },
        "training": {"label_smoothing": 0.0},
        "loss": {
            "primary": {"weight": 1.2},
            "consistency": {
                "type": loss_type,
                "weight": 0.3,
                "start_epoch": 0,
                "warmup_epochs": 0,
                "max_weight": 1.0,
                "precision": {
                    "jvp_dtype": None if loss_type == "psd" else jvp_dtype,
                    "numerical_dtype": "fp32",
                    "debug_assertions": True,
                },
                "ecld": {
                    "ec_weight": 4.0,
                    "td_weight": 2.0,
                    "time_weighting": "none",
                },
                "adaptive_kl": {
                    "enabled": False,
                    "c": 1.0e-6,
                    "r": 0.5,
                    "normalize_mean": True,
                    "max_weight": 100.0,
                },
                "invalid_teacher": {
                    "strategy": "clamp",
                    "log_eps": 1.0e-6,
                    "skip_batch_threshold": None,
                },
            },
        },
    }


def _inputs(classes: int = 4):
    torch.manual_seed(29)
    image = torch.randn(2, 3, 3, 4)
    target = torch.randint(0, classes, (2, 3, 4))
    one_hot = torch.nn.functional.one_hot(
        target, classes
    ).permute(0, 3, 1, 2).float()
    x_s = torch.softmax(torch.randn_like(one_hot), dim=1)
    s = torch.tensor([0.1, 0.2])
    u = torch.tensor([0.25, 0.35])
    t = torch.tensor([0.45, 0.55])
    return image, target, one_hot, x_s, s, u, t


def _consistency(
    loss_type: str,
    model: nn.Module,
    image: torch.Tensor,
    image_feat: torch.Tensor | None,
    x_s: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
    t: torch.Tensor,
    *,
    jvp_dtype: str = "fp32",
):
    config = _config(loss_type, jvp_dtype=jvp_dtype)
    extra = {"u": u} if loss_type == "psd" else {}
    return losses.compute_consistency_loss(
        loss_type,
        model=model,
        image=image,
        image_feat=image_feat,
        x_s=x_s,
        s=s,
        t=t,
        precision=config["loss"]["consistency"]["precision"],
        config=config,
        **extra,
    )


def _assert_optional_close(
    actual: torch.Tensor | None, expected: torch.Tensor | None
) -> None:
    assert (actual is None) == (expected is None)
    if actual is not None:
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)


@pytest.mark.parametrize(
    ("operation", "loss_type"),
    [
        ("stage1_objectives", "ecld"),
        ("stage2_objectives", "ecld"),
        ("joint_objectives", "ecld"),
        ("joint_objectives", "psd"),
        ("joint_objectives", "csd"),
        ("joint_objectives", "esd"),
    ],
)
def test_bf16_objectives_encode_endpoint_image_exactly_once(operation, loss_type):
    config = _config(loss_type, jvp_dtype="bf16")
    endpoint = TinyEndpoint()
    adapter = DDPCompatibleTrainingModel(endpoint, None, config)
    image, target, one_hot, _, _, _, _ = _inputs()
    forward_count = 0

    def count_forward(_module, _inputs, _output):
        nonlocal forward_count
        forward_count += 1

    handle = endpoint.image_encoder.register_forward_hook(count_forward)
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = compute_model_training_objectives(
                adapter,
                operation=operation,
                image=image,
                target=target,
                epoch_index=0,
                progress_in_epoch=0.5,
            )
        result["loss"].backward()
    finally:
        handle.remove()

    assert forward_count == 1
    assert torch.isfinite(result["loss"])
    image_grad = endpoint.image_encoder.weight.grad
    assert image_grad is not None
    assert torch.isfinite(image_grad).all()
    assert image_grad.norm() > 0


def test_amp_bf16_with_fp32_jvp_uses_one_feature_per_precision():
    config = _config("ecld", jvp_dtype="fp32")
    endpoint = TinyEndpoint()
    adapter = DDPCompatibleTrainingModel(endpoint, None, config)
    image, target, one_hot, _, _, _, _ = _inputs()
    feature_dtypes: list[torch.dtype] = []

    def record_dtype(_module, _inputs, output):
        feature_dtypes.append(output.dtype)

    handle = endpoint.image_encoder.register_forward_hook(record_dtype)
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = compute_model_training_objectives(
                adapter,
                operation="joint_objectives",
                image=image,
                target=target,
                epoch_index=0,
                progress_in_epoch=0.5,
            )
        result["loss"].backward()
    finally:
        handle.remove()

    assert feature_dtypes == [torch.bfloat16, torch.float32]
    assert torch.isfinite(result["loss"])


@pytest.mark.parametrize("loss_type", ["psd", "csd", "ecld", "esd"])
def test_shared_consistency_matches_reference_values_and_gradients(loss_type):
    reference_model = TinyEndpoint()
    shared_model = TinyEndpoint()
    shared_model.load_state_dict(reference_model.state_dict())
    image, _, _, x_s, s, u, t = _inputs()
    reference_count = 0
    shared_count = 0

    def count_reference(_module, _inputs, _output):
        nonlocal reference_count
        reference_count += 1

    def count_shared(_module, _inputs, _output):
        nonlocal shared_count
        shared_count += 1

    reference_handle = reference_model.image_encoder.register_forward_hook(
        count_reference
    )
    shared_handle = shared_model.image_encoder.register_forward_hook(count_shared)
    try:
        reference = _consistency(
            loss_type, reference_model, image, None, x_s, s, u, t
        )
        shared_feat = shared_model.encode_image(image)
        shared = _consistency(
            loss_type, shared_model, image, shared_feat, x_s, s, u, t
        )
        reference.loss.backward()
        shared.loss.backward()
    finally:
        reference_handle.remove()
        shared_handle.remove()

    assert reference_count == (3 if loss_type == "psd" else 2)
    assert shared_count == 1
    torch.testing.assert_close(shared.loss, reference.loss, rtol=1.0e-5, atol=1.0e-6)
    _assert_optional_close(shared.student_prob, reference.student_prob)
    _assert_optional_close(shared.teacher_prob, reference.teacher_prob)
    _assert_optional_close(shared.directional_output, reference.directional_output)
    expected_feature_grad_flags = {
        "psd": [False, False, True],
        "csd": [True, False],
        "ecld": [True, False],
        "esd": [False, True],
    }
    assert shared_model.feature_grad_flags == expected_feature_grad_flags[loss_type]

    reference_parameters = dict(reference_model.named_parameters())
    for name, parameter in shared_model.named_parameters():
        reference_grad = reference_parameters[name].grad
        assert reference_grad is not None
        assert parameter.grad is not None
        torch.testing.assert_close(
            parameter.grad, reference_grad, rtol=1.0e-5, atol=1.0e-6
        )
        assert torch.isfinite(parameter.grad).all()
    image_grad = shared_model.image_encoder.weight.grad
    assert image_grad is not None and image_grad.norm() > 0


def test_bf16_ecld_shared_path_matches_reference_values_and_gradients():
    reference_model = TinyEndpoint()
    shared_model = deepcopy(reference_model)
    image, _, _, x_s, s, u, t = _inputs()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        reference = _consistency(
            "ecld",
            reference_model,
            image,
            None,
            x_s,
            s,
            u,
            t,
            jvp_dtype="bf16",
        )
        image_feat = shared_model.encode_image(image)
        shared = _consistency(
            "ecld",
            shared_model,
            image,
            image_feat,
            x_s,
            s,
            u,
            t,
            jvp_dtype="bf16",
        )

    torch.testing.assert_close(shared.loss, reference.loss, rtol=2.0e-3, atol=2.0e-4)
    _assert_optional_close(shared.student_prob, reference.student_prob)
    _assert_optional_close(shared.teacher_prob, reference.teacher_prob)
    _assert_optional_close(shared.directional_output, reference.directional_output)
    reference.loss.backward()
    shared.loss.backward()
    reference_parameters = dict(reference_model.named_parameters())
    for name, parameter in shared_model.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            reference_parameters[name].grad,
            rtol=2.0e-3,
            atol=2.0e-4,
        )


def test_ecld_shared_feature_preserves_raw_student_logits_and_time_jvp():
    reference_model = TinyEndpoint()
    shared_model = deepcopy(reference_model)
    image, _, _, x_s, s, _, t = _inputs()
    shared_feat = shared_model.encode_image(image)

    reference_logits, reference_derivative = jvp(
        lambda time: reference_model.forward_logits(x_s, image, s, time),
        (t,),
        (torch.ones_like(t),),
    )
    shared_logits, shared_derivative = jvp(
        lambda time: shared_model.forward_logits_with_image_feat(
            x_s, shared_feat, s, time
        ),
        (t,),
        (torch.ones_like(t),),
    )

    torch.testing.assert_close(shared_logits, reference_logits)
    torch.testing.assert_close(shared_derivative, reference_derivative)


def _fixed_consistency_times(_loss_type, batch_size, device, *_args):
    s = torch.full((batch_size,), 0.15, device=device)
    u = torch.full((batch_size,), 0.3, device=device)
    t = torch.full((batch_size,), 0.5, device=device)
    return s, u, t


def _reference_joint_objective(
    adapter: DDPCompatibleTrainingModel,
    image: torch.Tensor,
    one_hot: torch.Tensor,
    target: torch.Tensor,
):
    config = adapter.config
    endpoint = adapter.endpoint_model
    x0, source_stats = sample_prior(
        config, image, one_hot, adapter.source_model, target_full=target
    )
    batch_size = image.shape[0]
    s, u, t = _fixed_consistency_times(
        "ecld", batch_size, image.device
    )
    diagonal_time = torch.full((batch_size,), 0.75, device=image.device)
    diagonal_state = linear_path(x0, one_hot, diagonal_time)
    diagonal_logits = endpoint.forward_logits(
        diagonal_state, image, diagonal_time, diagonal_time
    )
    diagonal_loss = losses.diagonal_cross_entropy(
        diagonal_logits, target, config["training"]["label_smoothing"]
    ).float()
    consistency_state = linear_path(x0, one_hot, s)
    consistency = losses.compute_consistency_loss(
        "ecld",
        model=endpoint,
        image=image,
        x_s=consistency_state,
        s=s,
        u=u,
        t=t,
        precision=config["loss"]["consistency"]["precision"],
        config=config,
    )
    total = (
        config["loss"]["primary"]["weight"] * diagonal_loss
        + config["loss"]["consistency"]["weight"] * consistency.loss
        + source_stats["weighted_var"]
        + source_stats["weighted_align"]
    ).float()
    return total, diagonal_logits, diagonal_loss, consistency, source_stats


def test_joint_ecld_total_and_all_parameter_group_gradients_match_reference(
    monkeypatch,
):
    monkeypatch.setattr(
        training_objectives,
        "sample_consistency_times",
        _fixed_consistency_times,
    )
    monkeypatch.setattr(
        training_objectives,
        "sample_stage1_times",
        lambda batch_size, device, *_args: torch.full(
            (batch_size,), 0.75, device=device
        ),
    )
    config = _config(
        "ecld", jvp_dtype="fp32", source_prior="image_gaussian"
    )
    reference_endpoint = TinyEndpoint()
    shared_endpoint = deepcopy(reference_endpoint)
    reference_source = TinySource()
    shared_source = deepcopy(reference_source)
    reference_adapter = DDPCompatibleTrainingModel(
        reference_endpoint, reference_source, config
    )
    shared_adapter = DDPCompatibleTrainingModel(
        shared_endpoint, shared_source, deepcopy(config)
    )
    image, target, one_hot, _, _, _, _ = _inputs()

    reference = _reference_joint_objective(
        reference_adapter, image, one_hot, target
    )
    shared = compute_model_training_objectives(
        shared_adapter,
        operation="joint_objectives",
        image=image,
        target=target,
        epoch_index=0,
        progress_in_epoch=0.5,
    )
    shared_x0, _shared_mu, _shared_log_variance = shared_source(image)
    diagonal_time = torch.full((image.shape[0],), 0.75)
    shared_diagonal_state = linear_path(shared_x0, one_hot, diagonal_time)
    shared_diagonal_logits = shared_endpoint.forward_logits_with_image_feat(
        shared_diagonal_state,
        shared_endpoint.encode_image(image),
        diagonal_time,
        diagonal_time,
    )

    torch.testing.assert_close(shared["loss"], reference[0])
    torch.testing.assert_close(shared_diagonal_logits, reference[1])
    torch.testing.assert_close(shared["stats"]["loss_diagonal"], reference[2])
    torch.testing.assert_close(
        shared["stats"]["loss_consistency"], reference[3].loss
    )
    torch.testing.assert_close(
        shared["stats"]["loss_source_var"],
        reference[4]["loss_source_var"],
    )
    torch.testing.assert_close(
        shared["stats"]["loss_source_align"],
        reference[4]["loss_source_align"],
    )

    reference[0].backward()
    shared["loss"].backward()
    reference_parameters = dict(reference_adapter.named_parameters())
    for name, parameter in shared_adapter.named_parameters():
        reference_grad = reference_parameters[name].grad
        assert reference_grad is not None
        assert parameter.grad is not None
        torch.testing.assert_close(
            parameter.grad, reference_grad, rtol=1.0e-5, atol=1.0e-6
        )
        assert torch.isfinite(parameter.grad).all()

    for module in (
        shared_endpoint.image_encoder,
        shared_endpoint.mask_encoder,
        shared_endpoint.unet,
        shared_source,
    ):
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_ecld_jvp_with_captured_image_feature_backpropagates_to_encoder():
    model = TinyEndpoint()
    image, _, _, x_s, s, u, t = _inputs()
    image_feat = model.encode_image(image)
    result = _consistency(
        "ecld", model, image, image_feat, x_s, s, u, t, jvp_dtype="fp32"
    )

    assert torch.isfinite(result.loss)
    assert torch.isfinite(result.student_prob).all()
    assert torch.isfinite(result.directional_output).all()
    result.loss.backward()
    gradient = model.image_encoder.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0
