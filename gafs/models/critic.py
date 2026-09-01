"""WGAN-GP critic with a SimCLR projection head.

Scores whether a future path Y belongs to the real market distribution P_r or
the generated distribution P_g, conditioned on the same history and macro
state the generator saw. Outputs an unbounded scalar D(x) (Wasserstein
critic), never a sigmoid probability.

Two interchangeable trunks:
  * 1D ResNet with GroupNorm (per-sample normalisation only; batch norm would
    invalidate the per-sample gradient penalty),
  * Transformer encoder over time steps.

The `project` head maps pooled features to the contrastive latent space z
used by the NT-Xent loss and the generator's latent coverage terms.
"""

from __future__ import annotations

import torch
from torch import nn


class _CondEncoder(nn.Module):
    """History + macro conditioning -> context vector [B, H]."""

    def __init__(self, n_features: int, cond_dim: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        cond_in = max(cond_dim, 1)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, hidden)
        )
        self.fuse = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.LeakyReLU(0.2))

    def forward(self, x_hist: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        _, h_n = self.gru(x_hist)
        h = h_n[-1]
        if cond is None or cond.shape[-1] == 0:
            cond = torch.zeros(x_hist.shape[0], 1, device=x_hist.device, dtype=x_hist.dtype)
        c = self.cond_mlp(cond)
        return self.fuse(torch.cat([h, c], dim=-1))


class _ResBlock1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=pad)
        self.n1 = nn.GroupNorm(1, c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad)
        self.n2 = nn.GroupNorm(1, c_out)
        self.act = nn.LeakyReLU(0.2)
        self.skip = (
            nn.Conv1d(c_in, c_out, 1, stride=stride)
            if (c_in != c_out or stride != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.n1(self.conv1(x)))
        h = self.n2(self.conv2(h))
        return self.act(h + self.skip(x))


class Critic(nn.Module):
    """Common interface: forward(y, x_hist, cond) -> score [B] (+ features)."""

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        cond_dim: int,
        horizon: int,
        hidden: int = 64,
        arch: str = "resnet",
        channels: tuple[int, ...] = (64, 128, 128),
        ctx_channels: int = 16,
        proj_dim: int = 32,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if arch not in ("resnet", "transformer"):
            raise ValueError("critic arch must be 'resnet' or 'transformer'")
        self.arch = arch
        self.horizon = horizon
        self.cond_encoder = _CondEncoder(n_features, cond_dim, hidden)
        self.ctx_to_channels = nn.Linear(hidden, ctx_channels)

        in_ch = n_assets + ctx_channels
        if arch == "resnet":
            blocks: list[nn.Module] = []
            prev = in_ch
            for i, ch in enumerate(channels):
                blocks.append(_ResBlock1d(prev, ch, stride=2 if i > 0 else 1))
                prev = ch
            self.trunk = nn.Sequential(*blocks)
            self.feature_dim = prev * 2
        else:
            self.in_proj = nn.Linear(in_ch, hidden)
            self.pos = nn.Parameter(torch.randn(1, horizon, hidden) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=hidden * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.trunk = nn.TransformerEncoder(
                layer, num_layers=2, enable_nested_tensor=False
            )
            self.feature_dim = hidden * 2

        self.score_head = nn.Sequential(
            nn.Linear(self.feature_dim + hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, proj_dim),
        )

    def _features(self, y: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, h, A = y.shape
        ctx_ch = self.ctx_to_channels(ctx)
        if self.arch == "resnet":
            ctx_map = ctx_ch.unsqueeze(-1).expand(-1, -1, h)
            x = torch.cat([y.transpose(1, 2), ctx_map], dim=1)
            feat_map = self.trunk(x)
            feat = torch.cat([feat_map.mean(dim=-1), feat_map.amax(dim=-1)], dim=-1)
        else:
            ctx_map = ctx_ch.unsqueeze(1).expand(-1, h, -1)
            x = self.in_proj(torch.cat([y, ctx_map], dim=-1)) + self.pos[:, :h]
            enc = self.trunk(x)
            feat = torch.cat([enc.mean(dim=1), enc.amax(dim=1)], dim=-1)
        return feat

    def forward(
        self,
        y: torch.Tensor,
        x_hist: torch.Tensor,
        cond: torch.Tensor | None = None,
        return_features: bool = False,
    ):
        ctx = self.cond_encoder(x_hist, cond)
        feat = self._features(y, ctx)
        score = self.score_head(torch.cat([feat, ctx], dim=-1)).squeeze(-1)
        if return_features:
            return score, feat
        return score

    def project(self, features: torch.Tensor) -> torch.Tensor:
        """Contrastive latent z (unnormalised; NT-Xent normalises internally)."""
        return self.proj_head(features)


def build_critic(
    n_assets: int,
    n_features: int,
    cond_dim: int,
    horizon: int,
    hidden: int,
    arch: str,
    channels: list[int],
    ctx_channels: int,
    proj_dim: int,
    heads: int,
    dropout: float,
) -> Critic:
    return Critic(
        n_assets=n_assets,
        n_features=n_features,
        cond_dim=cond_dim,
        horizon=horizon,
        hidden=hidden,
        arch=arch,
        channels=tuple(channels),
        ctx_channels=ctx_channels,
        proj_dim=proj_dim,
        heads=heads,
        dropout=dropout,
    )
