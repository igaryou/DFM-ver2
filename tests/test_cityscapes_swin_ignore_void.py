from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from config import load_config
from discrete_flow_maps import sample_prior
from losses import compute_consistency_loss, diagonal_cross_entropy
from state_space import prepare_state_targets, state_spatial_size


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "cityscapes" / "psd" / "swin_t_linear_160k.yaml"


class PixelwiseFlowModel(nn.Module):
    def __init__(self, classes: int = 20):
        super().__init__()
        self.state = nn.Conv2d(classes, classes, 1)
        self.image = nn.Conv2d(3, classes, 1)

    def encode_image(self, image):
        return self.image(image)

    def forward_logits_with_image_feat(self, x, image_feat, s, t):
        del s, t
        return self.state(x) + image_feat

    def forward_logits(self, x, image, s, t):
        return self.forward_logits_with_image_feat(x, self.encode_image(image), s, t)


class FixedGaussianSource(nn.Module):
    fixed_std = 1.0

    def __init__(self, mu: torch.Tensor):
        super().__init__()
        self.register_buffer("mu", mu)

    def forward(self, image):
        del image
        logvar = torch.zeros_like(self.mu)
        return self.mu + torch.randn_like(self.mu), self.mu, logvar


def test_new_config_keeps_20_states_and_matches_ade_training_protocol():
    config = load_config(CONFIG)
    expected_dataset = {
        "name": "cityscapes",
        "num_classes": 20,
        "eval_num_classes": 19,
        "void_class_index": 19,
        "image_size": [512, 1024],
    }
    assert {key: config["dataset"][key] for key in expected_dataset} == expected_dataset
    assert config["model"]["num_classes"] == 20
    assert config["model"]["state_downsample_factor"] == 4
    assert config["model"]["fusion_channels"] == 128
    assert config["model"]["image_encoder"] == {
        "type": "swin",
        "variant": "tiny",
        "pretrained": True,
        "freeze": False,
        "input_already_normalized": True,
        "neck": {"type": "ddp_fpn_merge", "channels": 256},
    }
    assert config["loss"]["ignore_index"] == 19
    assert config["loss"]["mask_pixel_losses"] is True
    assert config["source"]["supervision"] == {"type": "align", "weight": 0.2}
    assert config["training"]["max_optimizer_steps"] == 160000
    assert config["training"]["batch_size"] == 16
    assert config["training"]["optimizer"]["parameter_groups"] == {
        "model": {"lr": 1.0e-4}, "source": {"lr": 5.0e-5}
    }
    assert config["loss"]["consistency"]["adaptive_kl"]["enabled"] is False
    assert config["loss"]["consistency"]["invalid_teacher"][
        "skip_batch_threshold"
    ] is None
    assert config["evaluation"]["eval_class_indices"] == [0, 18]
    assert config["evaluation"]["exclude_void_from_prediction"] is True


def test_cityscapes_void_is_state_but_not_supervised_pixel():
    target = torch.tensor([[[0, 5, 19]]])
    targets = prepare_state_targets(
        target,
        num_classes=20,
        state_size=(1, 3),
        ignore_index=19,
        mask_pixel_losses=True,
    )
    assert torch.equal(
        targets.valid_mask_full, torch.tensor([[[True, True, False]]])
    )
    assert torch.equal(targets.valid_mask_state, targets.valid_mask_full)
    assert targets.one_hot_state.shape == (1, 20, 1, 3)
    assert targets.one_hot_state[0, 19, 0, 2] == 1
    assert targets.one_hot_state[:, :, 0, 2].sum() == 1


def test_primary_ce_is_20_way_on_semantic_pixel_and_ignores_void_gt():
    logits = torch.zeros(1, 20, 1, 2)
    logits[:, 19, 0, 0] = 8.0  # Wrong void prediction at semantic GT=5.
    logits[:, 5, 0, 0] = -2.0
    target = torch.tensor([[[5, 19]]])
    loss = diagonal_cross_entropy(logits, target, ignore_index=19)
    assert loss > 9.0
    changed_void_pixel = logits.clone()
    changed_void_pixel[:, :, 0, 1] = torch.linspace(-100, 100, 20)
    torch.testing.assert_close(
        diagonal_cross_entropy(changed_void_pixel, target, ignore_index=19), loss
    )


def test_psd_masks_void_pixel_but_retains_all_20_probability_channels():
    torch.manual_seed(7)
    model = PixelwiseFlowModel()
    x = torch.softmax(torch.randn(1, 20, 1, 2), dim=1)
    image = torch.randn(1, 3, 1, 2)
    valid = torch.tensor([[[True, False]]])
    times = (torch.tensor([0.1]), torch.tensor([0.3]), torch.tensor([0.6]))
    config = load_config(CONFIG)
    precision = config["loss"]["consistency"]["precision"]
    first = compute_consistency_loss(
        "psd", model=model, x_s=x, image=image,
        s=times[0], u=times[1], t=times[2],
        config=config, precision=precision, valid_mask=valid,
    )
    perturbed = image.clone()
    perturbed[:, :, 0, 1] += 1000
    second = compute_consistency_loss(
        "psd", model=model, x_s=x, image=perturbed,
        s=times[0], u=times[1], t=times[2],
        config=config, precision=precision, valid_mask=valid,
    )
    torch.testing.assert_close(first.loss, second.loss)
    assert first.teacher_prob.shape[1] == first.student_prob.shape[1] == 20


def test_source_align_masks_void_gt_and_source_outputs_20_channels():
    config = deepcopy(load_config(CONFIG))
    config["model"]["state_downsample_factor"] = 1
    image = torch.randn(1, 3, 1, 2)
    target = torch.tensor([[[5, 19]]])
    targets = prepare_state_targets(
        target, num_classes=20, state_size=(1, 2),
        ignore_index=19, mask_pixel_losses=True,
    )
    mu = torch.randn(1, 20, 1, 2)
    _, first = sample_prior(
        config, image, targets.one_hot_state, FixedGaussianSource(mu),
        target_full=target, valid_mask_full=targets.valid_mask_full,
    )
    perturbed_mu = mu.clone()
    perturbed_mu[:, :, 0, 1] += torch.linspace(-100, 100, 20)
    _, second = sample_prior(
        config, image, targets.one_hot_state, FixedGaussianSource(perturbed_mu),
        target_full=target, valid_mask_full=targets.valid_mask_full,
    )
    torch.testing.assert_close(first["loss_source_align"], second["loss_source_align"])
    assert mu.shape[1] == 20


def test_cityscapes_swin_and_fusion_spatial_shapes():
    config = load_config(CONFIG)
    assert state_spatial_size((512, 1024), 4) == (128, 256)
    expected_stages = [(128, 256), (64, 128), (32, 64), (16, 32)]
    assert [state_spatial_size((512, 1024), 4 * 2**index)
            for index in range(4)] == expected_stages
    image_feat = torch.empty(1, config["model"]["fusion_channels"], 128, 256)
    mask_feat = torch.empty_like(image_feat)
    assert (image_feat + mask_feat).shape == (1, 128, 128, 256)
