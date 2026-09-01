"""Time-series augmentations for SimCLR positive pairs.

Each transform takes and returns a tensor [B, T, F]. Two independently
augmented views of the same real window form a positive pair; views of
different windows are negatives.
"""

from __future__ import annotations

import torch


def jitter(x: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
    """Add minor Gaussian noise."""
    return x + sigma * torch.randn_like(x)


def scaling(x: torch.Tensor, sigma: float = 0.10) -> torch.Tensor:
    """Multiply magnitude by a random scalar per sample and feature."""
    factor = 1.0 + sigma * torch.randn(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
    return x * factor


def time_warp(x: torch.Tensor, sigma: float = 0.20, knots: int = 4) -> torch.Tensor:
    """Smoothly stretch/compress the sequence in time.

    A random positive speed profile is built from `knots + 2` anchors,
    integrated into a monotone warp of [0, T-1], and the series is linearly
    re-interpolated at the warped positions.
    """
    B, T, F = x.shape
    if T < 3:
        return x
    device, dtype = x.device, x.dtype

    anchors = torch.clamp(
        1.0 + sigma * torch.randn(B, knots + 2, device=device, dtype=dtype), min=0.2
    )
    grid = torch.linspace(0, knots + 1, T, device=device, dtype=dtype)
    lo = grid.floor().long().clamp(max=knots)
    frac = (grid - lo.to(dtype)).unsqueeze(0)
    speed = torch.gather(anchors, 1, lo.unsqueeze(0).expand(B, -1)) * (1 - frac) + \
        torch.gather(anchors, 1, (lo + 1).unsqueeze(0).expand(B, -1)) * frac

    cum = torch.cumsum(speed, dim=1)
    cum = cum - cum[:, :1]
    denom = cum[:, -1:].clamp(min=1e-8)
    src = cum / denom * (T - 1)

    lo_idx = src.floor().long().clamp(0, T - 2)
    w = (src - lo_idx.to(dtype)).unsqueeze(-1)
    lo_gather = lo_idx.unsqueeze(-1).expand(-1, -1, F)
    x_lo = torch.gather(x, 1, lo_gather)
    x_hi = torch.gather(x, 1, lo_gather + 1)
    return x_lo * (1 - w) + x_hi * w


def augment(
    x: torch.Tensor,
    jitter_sigma: float = 0.03,
    scale_sigma: float = 0.10,
    warp_sigma: float = 0.20,
    p: float = 0.5,
) -> torch.Tensor:
    """Random composition of the three transforms; at least one is applied."""
    applied = False
    out = x
    if torch.rand(()) < p:
        out = scaling(out, scale_sigma)
        applied = True
    if torch.rand(()) < p:
        out = time_warp(out, warp_sigma)
        applied = True
    if torch.rand(()) < p or not applied:
        out = jitter(out, jitter_sigma)
    return out
