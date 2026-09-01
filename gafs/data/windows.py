"""Windowing into overlapping 3D tensors.

Each sample provides:
  x_hist    [lookback, F_all]  full feature history X_{t-k:t}
  cond      [F_macro]          macro conditioning vector C_t at the origin
  y         [horizon, A]       future asset paths (feature space) to learn
  last_vol  [A]                return volatility at the origin (reconstruction)
  last_close[A]                close prices at the origin (reconstruction)

Splits are chronological with a purge gap of (lookback + horizon) windows
between train/val/test so overlapping windows never leak across splits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .preprocess import ProcessedData


@dataclass
class WindowArrays:
    x_hist: np.ndarray
    cond: np.ndarray
    y: np.ndarray
    last_vol: np.ndarray
    last_close: np.ndarray
    t0: np.ndarray
    feature_cols: list[str]
    asset_cols: list[str]
    macro_cols: list[str]
    lookback: int
    horizon: int

    def __len__(self) -> int:
        return self.x_hist.shape[0]


def build_windows(
    processed: ProcessedData,
    lookback: int = 90,
    horizon: int = 30,
    stride: int = 1,
) -> WindowArrays:
    feats = processed.features
    aux = processed.aux
    asset_cols = processed.asset_cols
    macro_cols = processed.macro_cols
    feature_cols = list(feats.columns)

    F = feats.to_numpy(dtype=np.float32)
    T, n_feat = F.shape
    if T < lookback + horizon + 1:
        raise ValueError(
            f"Need at least lookback + horizon + 1 = {lookback + horizon + 1} rows, got {T}."
        )

    asset_pos = [feature_cols.index(a) for a in asset_cols]
    macro_pos = [feature_cols.index(m) for m in macro_cols]

    starts = np.arange(0, T - lookback - horizon + 1, stride)
    hist = sliding_window_view(F, lookback, axis=0)
    hist = np.transpose(hist, (0, 2, 1))
    x_hist = np.ascontiguousarray(hist[starts])

    A = F[:, asset_pos]
    fut = sliding_window_view(A, horizon, axis=0)
    fut = np.transpose(fut, (0, 2, 1))
    y = np.ascontiguousarray(fut[starts + lookback])

    origin = starts + lookback - 1
    cond = (
        F[origin][:, macro_pos]
        if macro_pos
        else np.zeros((len(starts), 0), dtype=np.float32)
    )

    vol_cols = [f"{a}__vol" for a in asset_cols]
    close_cols = [f"{a}__close" for a in asset_cols]
    last_vol = aux[vol_cols].to_numpy(dtype=np.float32)[origin]
    last_close = aux[close_cols].to_numpy(dtype=np.float32)[origin]
    t0 = feats.index.to_numpy()[origin]

    return WindowArrays(
        x_hist=x_hist,
        cond=np.ascontiguousarray(cond, dtype=np.float32),
        y=y,
        last_vol=last_vol,
        last_close=last_close,
        t0=t0,
        feature_cols=feature_cols,
        asset_cols=asset_cols,
        macro_cols=macro_cols,
        lookback=lookback,
        horizon=horizon,
    )


def chronological_split(
    n_windows: int,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    purge: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological train/val/test window indices with purge gaps.

    `purge` should be at least lookback + horizon - 1 when stride is 1 so no
    timestamp appears in two splits.
    """
    n_val = int(round(n_windows * val_frac))
    n_test = int(round(n_windows * test_frac))
    n_train = n_windows - n_val - n_test - 2 * purge
    if n_train < 1:
        raise ValueError(
            f"Split leaves no training windows (n={n_windows}, purge={purge}); "
            "reduce val/test fractions or the purge gap."
        )
    train = np.arange(0, n_train)
    val_start = n_train + purge
    val = np.arange(val_start, val_start + n_val)
    test_start = val_start + n_val + purge
    test = np.arange(test_start, n_windows)
    return train, val, test


def split_windows(
    arrays: WindowArrays,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict[str, np.ndarray]:
    purge = arrays.lookback + arrays.horizon - 1
    train, val, test = chronological_split(len(arrays), val_frac, test_frac, purge)
    return {"train": train, "val": val, "test": test}


def to_frame_returns(processed: ProcessedData) -> pd.DataFrame:
    """Real per-date log returns panel (for stylized-fact baselines)."""
    cols = [f"{a}__ret" for a in processed.asset_cols]
    out = processed.aux[cols].dropna()
    out.columns = processed.asset_cols
    return out
