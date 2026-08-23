from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from torch.func import jvp

import losses
from losses import compute_consistency_loss


class TinyFlowModel(nn.Module):
    def __init__(self, classes: int = 4):
        super().__init__()
        self.x_projection = nn.Conv2d(classes, classes, 1)
        self.image_projection = nn.Conv2d(3, classes, 1)
        self.time_scale = nn.Parameter(torch.linspace(-0.2, 0.2, classes))

    def encode_image(self, image):
        return self.image_projection(image)

    def forward_logits_with_image_feat(self, x, image_feat, s, t):
        time = (0.3 * s + 0.7 * t)[:, None, None, None]
        return (
            self.x_projection(x)
            + image_feat
            + time * self.time_scale[None, :, None, None]
        )

    def forward_logits(self, x, image, s, t):
        return self.forward_logits_with_image_feat(
            x, self.encode_image(image), s, t
        )


def _inputs(classes: int = 4):
    torch.manual_seed(17)
    x = torch.softmax(torch.randn(2, classes, 3, 4), dim=1)
    image = torch.randn(2, 3, 3, 4)
    s = torch.tensor([0.1, 0.2])
    u = torch.tensor([0.2, 0.35])
    t = torch.tensor([0.4, 0.55])
    return x, image, s, u, t


def _config(loss_type: str, jvp_dtype):
    return {
        "runtime": {"amp": True, "amp_dtype": "bf16"},
        "flow": {"time_eps": 1.0e-5, "probability_eps": 1.0e-8},
        "loss": {
            "consistency": {
                "type": loss_type,
                "precision": {
                    "jvp_dtype": jvp_dtype,
                    "numerical_dtype": "fp32",
                    "debug_assertions": True,
                },
                "ecld": {
                    "ec_weight": 4.0,
                    "td_weight": 2.0,
                    "time_weighting": "none",
                },
                "adaptive_kl": {
                    "enabled": True,
                    "c": 1.0e-6,
                    "r": 0.5,
                    "normalize_mean": True,
                    "max_weight": 100.0,
                },
                "invalid_teacher": {
                    "strategy": "mask_pixel",
                    "log_eps": 1.0e-6,
                    "skip_batch_threshold": None,
                },
            }
        },
    }


def test_psd_uses_three_times_without_jvp_and_detaches_teacher(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("PSD must not invoke JVP")

    monkeypatch.setattr(losses, "jvp", forbidden)
    model = TinyFlowModel()
    x, image, s, u, t = _inputs()
    result = compute_consistency_loss(
        "psd", model=model, x_s=x, image=image, s=s, u=u, t=t,
        precision=_config("psd", None)["loss"]["consistency"]["precision"],
        config=_config("psd", None),
    )
    assert result.student_prob.shape == x.shape
    assert result.teacher_prob.shape == x.shape
    assert not result.teacher_prob.requires_grad
    assert torch.isfinite(result.loss)
    torch.testing.assert_close(
        result.teacher_prob.sum(1), torch.ones_like(result.teacher_prob[:, 0])
    )
    result.loss.backward()
    assert model.x_projection.weight.grad is not None


@pytest.mark.parametrize("loss_type", ["csd", "ecld", "esd"])
@pytest.mark.parametrize(
    ("jvp_dtype", "expected_dtype"),
    [("fp32", torch.float32), ("bf16", torch.bfloat16)],
)
def test_jvp_losses_use_requested_dtype_then_compute_in_fp32(
    loss_type, jvp_dtype, expected_dtype
):
    model = TinyFlowModel()
    x, image, s, _, t = _inputs()
    config = _config(loss_type, jvp_dtype)
    result = compute_consistency_loss(
        loss_type, model=model, x_s=x, image=image, s=s, t=t,
        precision=config["loss"]["consistency"]["precision"], config=config,
    )
    assert result.dtypes["student_logits_before_cast"] == expected_dtype
    assert result.dtypes["directional_logits_before_cast"] == expected_dtype
    assert result.dtypes["student_logits_after_cast"] == torch.float32
    assert result.dtypes["directional_logits_after_cast"] == torch.float32
    assert result.teacher_prob.dtype == torch.float32
    if result.student_prob is not None:
        assert result.student_prob.dtype == torch.float32
    assert result.loss.dtype == torch.float32
    assert torch.isfinite(result.loss)
    result.loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_ecld_exact_softmax_jvp_matches_autograd_formula():
    model = TinyFlowModel()
    x, image, s, _, t = _inputs()
    config = _config("ecld", "fp32")
    result = compute_consistency_loss(
        "ecld", model=model, x_s=x, image=image, s=s, t=t,
        precision=config["loss"]["consistency"]["precision"], config=config,
    )

    def logits_at(time):
        return model.forward_logits(x, image, s, time)

    logits, derivative = jvp(logits_at, (t,), (torch.ones_like(t),))
    probability = torch.softmax(logits, dim=1)
    expected = probability * (
        derivative - (probability * derivative).sum(dim=1, keepdim=True)
    )
    torch.testing.assert_close(result.directional_output, expected)


@pytest.mark.parametrize("loss_type", ["csd", "ecld", "esd"])
def test_bf16_and_fp32_losses_are_close_and_finite(loss_type):
    model_fp32 = TinyFlowModel()
    model_bf16 = TinyFlowModel()
    model_bf16.load_state_dict(model_fp32.state_dict())
    x, image, s, _, t = _inputs()
    fp32_config = _config(loss_type, "fp32")
    bf16_config = _config(loss_type, "bf16")
    fp32_result = compute_consistency_loss(
        loss_type, model=model_fp32, x_s=x, image=image, s=s, t=t,
        precision=fp32_config["loss"]["consistency"]["precision"],
        config=fp32_config,
    )
    bf16_result = compute_consistency_loss(
        loss_type, model=model_bf16, x_s=x, image=image, s=s, t=t,
        precision=bf16_config["loss"]["consistency"]["precision"],
        config=bf16_config,
    )
    assert torch.isfinite(fp32_result.loss) and torch.isfinite(bf16_result.loss)
    fp32_value = float(fp32_result.loss.detach())
    bf16_value = float(bf16_result.loss.detach())
    difference = abs(fp32_value - bf16_value)
    scale = max(abs(fp32_value), 1.0e-3)
    assert difference / scale < 0.08


def test_psd_rejects_jvp_precision():
    model = TinyFlowModel()
    x, image, s, u, t = _inputs()
    with pytest.raises(ValueError, match="does not use JVP"):
        compute_consistency_loss(
            "psd", model=model, x_s=x, image=image, s=s, u=u, t=t,
            precision={"jvp_dtype": "bf16", "numerical_dtype": "fp32"},
            config=_config("psd", None),
        )


@pytest.mark.parametrize("loss_type", ["psd", "csd", "ecld"])
def test_esd_metadata_does_not_change_other_consistency_losses(loss_type):
    jvp_dtype = None if loss_type == "psd" else "fp32"
    config_without_metadata = _config(loss_type, jvp_dtype)
    config_with_metadata = deepcopy(config_without_metadata)
    config_with_metadata["loss"]["consistency"]["esd"] = {
        "formulation": "stabilized_logit_space",
        "source": "discrete_flow_maps",
        "additional_numerical_safeguards": True,
    }
    x, image, s, u, t = _inputs()
    model_without_metadata = TinyFlowModel()
    model_with_metadata = TinyFlowModel()
    model_with_metadata.load_state_dict(model_without_metadata.state_dict())
    extra = {"u": u} if loss_type == "psd" else {}

    result_without_metadata = compute_consistency_loss(
        loss_type,
        model=model_without_metadata,
        x_s=x,
        image=image,
        s=s,
        t=t,
        precision=config_without_metadata["loss"]["consistency"]["precision"],
        config=config_without_metadata,
        **extra,
    )
    result_with_metadata = compute_consistency_loss(
        loss_type,
        model=model_with_metadata,
        x_s=x,
        image=image,
        s=s,
        t=t,
        precision=config_with_metadata["loss"]["consistency"]["precision"],
        config=config_with_metadata,
        **extra,
    )
    torch.testing.assert_close(
        result_with_metadata.loss, result_without_metadata.loss
    )
