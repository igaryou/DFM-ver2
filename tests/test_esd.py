import torch
import torch.nn as nn

import losses
from losses import esd_loss


class TinyModel(nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        self.x_projection = nn.Conv2d(classes, classes, 1)
        self.image_projection = nn.Conv2d(3, classes, 1)
        self.time_scale = nn.Parameter(torch.linspace(-0.1, 0.1, classes))

    def forward_logits(self, x, image, s, t):
        time = (s + 0.5 * t)[:, None, None, None]
        return (
            self.x_projection(x)
            + self.image_projection(image)
            + time * self.time_scale[None, :, None, None]
        )


class AlwaysInvalidModel(nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(classes))
        self.register_buffer("slopes", torch.tensor([-200.0, -50.0, 50.0, 200.0]))

    def forward_logits(self, x, image, s, t):
        # Primal is uniform, while forward-mode derivative with respect to s is large.
        differentiable_zero = s - s.detach()
        values = self.bias[None] + differentiable_zero[:, None] * self.slopes[None]
        return values[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])


def inputs(classes=4):
    torch.manual_seed(3)
    x = torch.softmax(torch.randn(2, classes, 3, 4), dim=1)
    image = torch.randn(2, 3, 3, 4)
    s = torch.tensor([0.1, 0.2])
    t = torch.tensor([0.2, 0.3])
    return x, image, s, t


def test_joint_jvp_teacher_and_kl_direction_are_correct():
    model = TinyModel()
    x, image, s, t = inputs()
    result = esd_loss(model, x, image, s, t)
    assert result.student_prob.shape == x.shape
    assert result.directional_logits.shape == x.shape
    assert torch.isfinite(result.directional_logits).all()
    assert not result.teacher_prob.requires_grad
    torch.testing.assert_close(
        result.teacher_prob.sum(dim=1), torch.ones_like(s[:, None, None]).expand(-1, 3, 4)
    )
    manual_kl = (
        result.teacher_prob
        * (result.teacher_prob.clamp_min(1e-12).log() - result.student_prob.log())
    ).sum(dim=1)
    torch.testing.assert_close(
        result.loss, manual_kl[result.valid_pixel].mean(), rtol=1e-5, atol=1e-6
    )
    result.loss.backward()
    assert model.x_projection.weight.grad is not None
    assert torch.isfinite(model.x_projection.weight.grad).all()


def test_invalid_teacher_mask_pixel_and_all_invalid_zero_graph():
    model = AlwaysInvalidModel()
    x, image, s, _ = inputs()
    t = torch.full_like(s, 0.9)
    result = esd_loss(model, x, image, s, t, invalid_strategy="mask_pixel")
    assert result.stats["esd_clamp_ratio"] > 0
    assert result.stats["esd_valid_pixel_ratio"] == 0
    assert result.loss.item() == 0.0
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert model.bias.grad is not None
    assert torch.isfinite(model.bias.grad).all()


def test_clamp_ratio_matches_manual_definition_and_skip_batch():
    model = AlwaysInvalidModel()
    x, image, s, _ = inputs()
    t = torch.full_like(s, 0.9)
    result = esd_loss(model, x, image, s, t, invalid_strategy="clamp")
    # Two of four centered slopes make 1-t-(1-s)(t-s)delta non-positive here.
    assert torch.isclose(result.stats["esd_clamp_ratio"], torch.tensor(0.5))
    skipped = esd_loss(
        model, x, image, s, t, invalid_strategy="clamp", skip_batch_threshold=0.4
    )
    assert skipped.loss.item() == 0.0
    assert skipped.stats["esd_skipped_batch"] == 1


def test_adaptive_weight_is_detached_and_optional():
    model = TinyModel()
    x, image, s, t = inputs()
    plain = esd_loss(model, x, image, s, t, adaptive_enabled=False)
    adaptive = esd_loss(
        model, x, image, s, t, adaptive_enabled=True,
        adaptive_c=1e-6, adaptive_r=0.5, adaptive_normalize_mean=True,
    )
    assert not adaptive.adaptive_weight.requires_grad
    assert torch.isfinite(plain.loss) and torch.isfinite(adaptive.loss)
    assert torch.isclose(
        adaptive.adaptive_weight[adaptive.valid_pixel].mean(), torch.tensor(1.0),
        rtol=1e-5, atol=1e-5,
    )


def test_esd_promotes_bf16_autocast_inputs_to_float32_and_backpropagates():
    model = TinyModel()
    x, image, s, t = inputs()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = esd_loss(model, x.to(torch.bfloat16), image.to(torch.bfloat16), s, t)
    assert result.loss.dtype == torch.float32
    assert result.directional_logits.dtype == torch.float32
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

