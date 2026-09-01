"""Batch sampling of generated paths across many historical contexts.

Used by evaluation: draws `samples_per_anchor` stochastic paths at up to
`n_anchors` distinct test-set forecast origins, then unscales everything to
log-return space with each origin's own volatility so the comparison against
real returns is made in the units that matter.
"""

from __future__ import annotations

import numpy as np
import torch

from ..data.preprocess import unscale_asset_paths
from ..data.windows import WindowArrays


@torch.no_grad()
def sample_paths_over_anchors(
    generator,
    arrays: WindowArrays,
    indices: np.ndarray,
    n_anchors: int = 100,
    samples_per_anchor: int = 8,
    device: torch.device | str = "cpu",
    meta: dict | None = None,
    batch_windows: int = 32,
    seed: int | None = None,
) -> dict:
    """Return generated paths in scaled and (optionally) return space.

    Output dict:
      fake_scaled  [M, h, A]  generator output in feature space
      fake_returns [M, h, A]  log returns (only when meta permits inversion)
      anchor_t0    [M]        forecast-origin timestamps, repeated per sample
    """
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device(device)
    generator = generator.to(device).eval()

    indices = np.asarray(indices)
    if len(indices) == 0:
        raise ValueError("No anchor indices supplied.")
    pick = np.unique(np.linspace(0, len(indices) - 1, min(n_anchors, len(indices)), dtype=int))
    anchors = indices[pick]

    scaled_chunks: list[np.ndarray] = []
    vol_chunks: list[np.ndarray] = []
    t0_chunks: list[np.ndarray] = []
    for lo in range(0, len(anchors), batch_windows):
        chunk = anchors[lo : lo + batch_windows]
        x = torch.from_numpy(arrays.x_hist[chunk]).to(device)
        c = torch.from_numpy(arrays.cond[chunk]).to(device)
        x = x.repeat_interleave(samples_per_anchor, dim=0)
        c = c.repeat_interleave(samples_per_anchor, dim=0) if c.shape[-1] else None
        y = generator(x, c)
        scaled_chunks.append(y.cpu().numpy())
        vol_chunks.append(np.repeat(arrays.last_vol[chunk], samples_per_anchor, axis=0))
        t0_chunks.append(np.repeat(arrays.t0[chunk], samples_per_anchor, axis=0))

    fake_scaled = np.concatenate(scaled_chunks, axis=0)
    last_vol = np.concatenate(vol_chunks, axis=0)
    out = {
        "fake_scaled": fake_scaled,
        "anchor_t0": np.concatenate(t0_chunks, axis=0),
        "n_anchors": int(len(anchors)),
        "samples_per_anchor": int(samples_per_anchor),
    }
    if meta is not None and meta.get("stationarity") == "log_returns":
        out["fake_returns"] = unscale_asset_paths(fake_scaled, meta, last_vol)
    return out
