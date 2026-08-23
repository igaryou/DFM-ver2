import pytest
import torch

from discrete_flow_maps import flow_map, sample_prior, sample_sorted_times


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flow_map_identity_shape_and_finite(dtype):
    x = torch.randn(2, 20, 4, 5).to(dtype)
    endpoint = torch.softmax(torch.randn(2, 20, 4, 5), dim=1).to(dtype)
    s = torch.tensor([0.2, 0.9])
    output = flow_map(x, endpoint, s, s, time_eps=1.0e-5)
    assert output.shape == x.shape
    assert torch.equal(output, x)
    assert torch.isfinite(output.float()).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flow_map_reaches_endpoint_at_t_one(dtype):
    x = torch.randn(2, 20, 3, 4).to(dtype)
    endpoint = torch.softmax(torch.randn(2, 20, 3, 4), dim=1).to(dtype)
    s = torch.tensor([0.0, 0.75])
    t = torch.ones_like(s)
    output = flow_map(x, endpoint, s, t, time_eps=1.0e-5)
    torch.testing.assert_close(output, endpoint, rtol=0.02, atol=0.02)


def test_sorted_time_sampling_enforces_gap_and_keeps_endpoint_one_available():
    s, t = sample_sorted_times(
        1024, torch.device("cpu"), min_time=0.0, max_time=1.0, min_gap=1.0e-5
    )
    assert (s >= 0).all() and (t <= 1).all()
    assert ((t - s) >= 0.9999e-5).all()


def test_image_gaussian_prior_reports_actual_detached_cfm_statistics():
    class Source:
        fixed_std = 1.0

        def __call__(self, image):
            mu = torch.linspace(
                -2.0, 2.0, image.shape[0] * 4 * 2 * 3
            ).reshape(image.shape[0], 4, 2, 3)
            logvar = torch.zeros_like(mu)
            x0 = mu + 0.25
            return x0, mu, logvar

    image = torch.randn(2, 3, 2, 3)
    target = torch.nn.functional.one_hot(
        torch.randint(0, 4, (2, 2, 3)), 4
    ).permute(0, 3, 1, 2).float()
    config = {
        "dataset": {"num_classes": 4},
        "source": {
            "prior_type": "image_gaussian",
            "var_weight": 0.0,
            "align_weight": 0.15,
            "use_loss_align": True,
            "align_eps": 1.0e-8,
        },
    }
    x0, stats = sample_prior(config, image, target, Source())
    mu = x0 - 0.25

    assert float(stats["source_mu_abs"]) == pytest.approx(
        float(mu.abs().mean())
    )
    assert float(stats["source_mu_min"]) == pytest.approx(-2.0)
    assert float(stats["source_mu_max"]) == pytest.approx(2.0)
    assert float(stats["source_logvar_mean"]) == 0.0
    assert float(stats["source_sigma_mean"]) == 1.0
    assert float(stats["source_x0_abs"]) == pytest.approx(
        float(x0.abs().mean())
    )
    assert float(stats["target_x1_abs"]) == pytest.approx(
        float(target.abs().mean())
    )
    assert all(not value.requires_grad for value in stats.values())
