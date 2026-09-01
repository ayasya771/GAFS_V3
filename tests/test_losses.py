import math

import pytest
import torch

from gafs.models.critic import Critic
from gafs.training.augmentations import augment, jitter, scaling, time_warp
from gafs.training.losses import (
    critic_wgan_loss,
    generator_wgan_loss,
    gradient_penalty,
    nt_xent,
    rbf_mmd,
    uniformity,
)

B, K, H, F, A, C = 6, 20, 12, 5, 3, 2


def _critic():
    torch.manual_seed(0)
    return Critic(
        n_assets=A, n_features=F, cond_dim=C, horizon=H,
        hidden=16, arch="resnet", channels=(8, 16), ctx_channels=4, proj_dim=8,
        heads=4, dropout=0.0,
    )


def test_wgan_loss_signs():
    d_real = torch.tensor([2.0, 2.0])
    d_fake = torch.tensor([-1.0, -1.0])
    assert critic_wgan_loss(d_real, d_fake).item() == pytest.approx(-3.0)
    assert generator_wgan_loss(d_fake).item() == pytest.approx(1.0)


def test_gradient_penalty_positive_finite_and_differentiable():
    critic = _critic()
    y_real = torch.randn(B, H, A)
    y_fake = torch.randn(B, H, A)
    x = torch.randn(B, K, F)
    c = torch.randn(B, C)
    gp = gradient_penalty(critic, y_real, y_fake, x, c)
    assert gp.item() >= 0.0 and math.isfinite(gp.item())
    gp.backward()
    grads = [p.grad for p in critic.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_nt_xent_prefers_aligned_pairs():
    torch.manual_seed(1)
    z = torch.randn(32, 8)
    aligned = nt_xent(z, z + 0.01 * torch.randn_like(z), tau=0.2)
    shuffled = nt_xent(z, z[torch.randperm(32)], tau=0.2)
    assert aligned.item() < shuffled.item()
    assert aligned.item() >= 0.0


def test_uniformity_orders_spread():
    torch.manual_seed(2)
    collapsed = torch.ones(64, 8) + 0.001 * torch.randn(64, 8)
    spread = torch.randn(64, 8)
    assert uniformity(spread).item() < uniformity(collapsed).item()


def test_mmd_small_for_same_distribution_positive_for_different():
    torch.manual_seed(3)
    x = torch.randn(128, 8)
    y_same = torch.randn(128, 8)
    same = rbf_mmd(x, y_same)
    far = rbf_mmd(x, y_same + 3.0)
    assert abs(same.item()) < 0.10
    assert far.item() > 0.2
    assert far.item() > same.item()


def test_augmentations_preserve_shape_and_change_values():
    torch.manual_seed(4)
    x = torch.randn(B, H, A)
    for fn in (jitter, scaling, time_warp, augment):
        out = fn(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        assert not torch.equal(out, x)


def test_time_warp_keeps_range_reasonable():
    torch.manual_seed(5)
    x = torch.randn(B, 30, A).cumsum(dim=1)
    out = time_warp(x, sigma=0.2)
    assert out.min() >= x.min() - 1e-4
    assert out.max() <= x.max() + 1e-4
