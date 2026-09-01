"""Losses: WGAN-GP adversarial terms and the contrastive regularisers.

Critic:      L_D = E[D(x_g)] - E[D(x)] + lambda * E[(||grad D(x_hat)||_2 - 1)^2]
Generator:   L_G = -E[D(x_g)]  (+ latent coverage terms below)
NT-Xent:     normalized temperature-scaled cross entropy over positive pairs
Uniformity:  log E exp(-t ||z_i - z_j||^2) on the unit hypersphere; minimising
             it spreads generated embeddings, countering mode collapse
MMD:         multi-bandwidth RBF discrepancy aligning generated and real
             latent distributions
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def critic_wgan_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    return d_fake.mean() - d_real.mean()


def generator_wgan_loss(d_fake: torch.Tensor) -> torch.Tensor:
    return -d_fake.mean()


def gradient_penalty(
    critic,
    y_real: torch.Tensor,
    y_fake: torch.Tensor,
    x_hist: torch.Tensor,
    cond: torch.Tensor | None,
) -> torch.Tensor:
    """One-Lipschitz enforcement on interpolates x_hat = eps*x + (1-eps)*x_g."""
    B = y_real.shape[0]
    eps = torch.rand(B, 1, 1, device=y_real.device, dtype=y_real.dtype)
    y_hat = (eps * y_real + (1.0 - eps) * y_fake).requires_grad_(True)
    scores = critic(y_hat, x_hist, cond)
    grads = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=y_hat,
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_norm = grads.flatten(start_dim=1).norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """NT-Xent over a batch of positive pairs (z1_i, z2_i).

    Penalises far-apart views of the same window and too-close embeddings of
    different windows.
    """
    if z1.shape[0] < 2:
        return z1.new_zeros(())
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    n = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.t()) / tau
    sim.fill_diagonal_(-1e9)
    targets = torch.arange(2 * n, device=z.device)
    targets = (targets + n) % (2 * n)
    return F.cross_entropy(sim, targets)


def uniformity(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """Wang & Isola uniformity: lower = better spread over the hypersphere."""
    if z.shape[0] < 2:
        return z.new_zeros(())
    z = F.normalize(z, dim=1)
    sq_dists = torch.pdist(z, p=2).pow(2)
    return torch.logsumexp(-t * sq_dists, dim=0) - torch.log(
        torch.tensor(float(sq_dists.numel()), device=z.device)
    )


def rbf_mmd(x: torch.Tensor, y: torch.Tensor, scales: tuple[float, ...] = (0.5, 1.0, 2.0)) -> torch.Tensor:
    """Multi-bandwidth RBF MMD^2 with a median-heuristic base bandwidth."""
    if x.shape[0] < 2 or y.shape[0] < 2:
        return x.new_zeros(())
    xy = torch.cat([x, y], dim=0)
    d2 = torch.cdist(xy, xy).pow(2)
    with torch.no_grad():
        pos = d2[d2 > 0]
        med = pos.median() if pos.numel() else torch.tensor(1.0, device=x.device)
        med = med.clamp(min=1e-6)
    n, m = x.shape[0], y.shape[0]
    k = torch.zeros_like(d2)
    for s in scales:
        k = k + torch.exp(-d2 / (s * med))
    k_xx = (k[:n, :n].sum() - k[:n, :n].diagonal().sum()) / (n * (n - 1))
    k_yy = (k[n:, n:].sum() - k[n:, n:].diagonal().sum()) / (m * (m - 1))
    k_xy = k[:n, n:].mean()
    return k_xx + k_yy - 2.0 * k_xy
