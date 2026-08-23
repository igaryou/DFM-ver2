import torch
import torch.nn as nn

import losses
from losses import compute_training_losses


class DiagonalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward_logits(self, x, image, s, t):
        return self.conv(x) + 0.0 * image[:, :3] + (s + t)[:, None, None, None]


def test_stage1_never_calls_esd_or_jvp(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("ESD/JVP must not run in Stage 1")

    monkeypatch.setattr(losses, "esd_loss", forbidden)
    monkeypatch.setattr(losses, "jvp", forbidden)
    model = DiagonalModel()
    x = torch.randn(2, 3, 4, 5)
    image = torch.randn(2, 3, 4, 5)
    target = torch.randint(0, 3, (2, 4, 5))
    time = torch.rand(2)
    zero = x.sum() * 0.0
    total, stats, result = compute_training_losses(
        stage="diagonal_pretrain", model=model, x_path=x, image=image,
        target=target, diagonal_time=time, label_smoothing=0.1,
        primary_weight=1.0,
        source_stats={
            "loss_source_var": zero, "loss_source_align": zero,
            "weighted_var": zero, "weighted_align": zero,
        },
    )
    assert result is None
    assert "loss_esd" not in stats
    total.backward()
    assert model.conv.weight.grad is not None

