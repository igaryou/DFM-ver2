from __future__ import annotations

import torch

from visualize_source_noise_sweep import make_epsilon, select_mean


def test_noise_is_deterministic_and_shared_across_sigma_values():
    mu = torch.tensor([[[[0.2]], [[-0.4]], [[0.1]]]], dtype=torch.float32)
    epsilon = make_epsilon(mu, 42)

    assert torch.equal(epsilon, make_epsilon(mu, 42))
    assert not torch.equal(epsilon, make_epsilon(mu, 43))
    assert torch.equal(mu + 0.0 * epsilon, mu)

    sigma_a = 1.0
    sigma_b = 0.5
    x_a = mu + sigma_a * epsilon
    x_b = mu + sigma_b * epsilon

    torch.testing.assert_close(x_b - mu, 0.5 * (x_a - mu))


def test_effective_mean_applies_bounded_gaussian_transform():
    mu = torch.tensor([[-2.0, 0.0, 2.0]])
    config = {
        "source": {
            "prior_type": "image_bounded_gaussian",
            "bounded_gaussian": {"amplitude": 2.0, "temperature": 4.0},
        }
    }

    torch.testing.assert_close(select_mean(mu, config, "raw"), mu)
    torch.testing.assert_close(
        select_mean(mu, config, "effective"), 2.0 * torch.tanh(mu / 4.0)
    )
    assert torch.equal(
        select_mean(mu, config, "effective").argmax(dim=1),
        mu.argmax(dim=1),
    )
