import pytest
import torch
import torch.nn as nn

from config import load_config
from discrete_flow_maps import (
    flow_map,
    linear_path,
    path_coefficient,
    path_derivative,
)
from losses import compute_consistency_loss


LINEAR = {"type": "power", "exponent": 1.0}
POWER2 = {"type": "power", "exponent": 2.0}


def test_power_one_path_and_flow_map_match_legacy_formulas():
    torch.manual_seed(3)
    x0 = torch.randn(2, 3, 2, 2)
    x1 = torch.randn_like(x0)
    time = torch.tensor([0.2, 0.7])
    expected_path = (1 - time[:, None, None, None]) * x0 + time[:, None, None, None] * x1
    torch.testing.assert_close(linear_path(x0, x1, time, LINEAR), expected_path)

    s = torch.tensor([0.1, 0.4])
    t = torch.tensor([0.6, 0.9])
    endpoint = torch.randn_like(x0)
    expected_flow = x0 + (
        ((t - s) / (1 - s))[:, None, None, None] * (endpoint - x0)
    )
    torch.testing.assert_close(
        flow_map(x0, endpoint, s, t, path_config=LINEAR), expected_flow
    )


def test_power_two_coefficients_derivatives_and_endpoints():
    time = torch.tensor([0.0, 0.5, 1.0])
    torch.testing.assert_close(
        path_coefficient(time, POWER2), torch.tensor([0.0, 0.25, 1.0])
    )
    torch.testing.assert_close(
        path_derivative(time, POWER2), torch.tensor([0.0, 1.0, 2.0])
    )
    x0 = torch.randn(3, 2, 1, 1)
    x1 = torch.randn_like(x0)
    path = linear_path(x0, x1, time, POWER2)
    torch.testing.assert_close(path[0], x0[0])
    torch.testing.assert_close(path[-1], x1[-1])


class _TimeOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward_logits(self, x, image, s, t):
        del image, s
        logits = torch.stack((t, -t), dim=1) * self.scale
        return logits[:, :, None, None].expand_as(x)


@pytest.mark.parametrize("path", [LINEAR, POWER2])
def test_psd_teacher_uses_generalized_mixture_coefficient(path):
    model = _TimeOnlyModel()
    x = torch.full((1, 2, 1, 1), 0.5)
    image = torch.zeros(1, 3, 1, 1)
    s, u, t = (torch.tensor([value]) for value in (0.2, 0.5, 0.8))
    config = {
        "flow": {
            "time_eps": 1.0e-5,
            "probability_eps": 1.0e-8,
            "path": path,
        },
        "loss": {
            "consistency": {
                "precision": {"jvp_dtype": None, "numerical_dtype": "fp32"},
                "psd": {"loss_resolution": "state"},
            }
        },
    }
    result = compute_consistency_loss(
        "psd", model=model, x_s=x, image=image, s=s, u=u, t=t,
        config=config,
    )
    alpha_s, alpha_u, alpha_t = (
        path_coefficient(value, path) for value in (s, u, t)
    )
    omega = (
        (1 - alpha_t) * (alpha_u - alpha_s)
        / ((1 - alpha_u) * (alpha_t - alpha_s))
    )
    probability_su = torch.softmax(torch.tensor([u.item(), -u.item()]), dim=0)
    probability_ut = torch.softmax(torch.tensor([t.item(), -t.item()]), dim=0)
    expected = omega * probability_su + (1 - omega) * probability_ut
    torch.testing.assert_close(result.teacher_prob[0, :, 0, 0], expected)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_nonpositive_path_exponent_is_rejected(value):
    with pytest.raises(ValueError, match="flow.path.exponent must be positive"):
        load_config(
            "configs/debug/diagonal/cityscapes.yaml",
            [f"flow.path.exponent={value}"],
        )
