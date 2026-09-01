"""Fixed-width fractional differencing (FFD).

For d in (0, 1) the fractionally differenced series is stationary while
preserving long memory that plain log returns (d = 1) destroy. Weights follow
the binomial expansion

    w_0 = 1,   w_k = -w_{k-1} * (d - k + 1) / k

truncated where |w_k| drops below `threshold` (Lopez de Prado, Advances in
Financial Machine Learning, ch. 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def ffd_weights(d: float, threshold: float = 1e-4, max_width: int = 10_000) -> np.ndarray:
    """Return FFD weights ordered oldest -> newest (ready for a dot product)."""
    if not 0.0 <= d <= 1.0:
        raise ValueError(f"d must lie in [0, 1], got {d}")
    weights = [1.0]
    k = 1
    while k < max_width:
        w = -weights[-1] * (d - k + 1.0) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    return np.asarray(weights[::-1], dtype=float)


def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    """Apply fixed-width fractional differencing to one series.

    The first (width - 1) observations, and any window containing NaN, come
    back as NaN so downstream code can drop the warm-up explicitly.
    """
    w = ffd_weights(d, threshold)
    width = len(w)
    values = series.to_numpy(dtype=float)
    out = np.full(values.shape[0], np.nan)
    if values.shape[0] >= width:
        windows = sliding_window_view(values, width)
        res = windows @ w
        res[np.isnan(windows).any(axis=1)] = np.nan
        out[width - 1:] = res
    return pd.Series(out, index=series.index, name=series.name)


def frac_diff_frame(df: pd.DataFrame, d: float, threshold: float = 1e-4) -> pd.DataFrame:
    """Column-wise FFD with a shared weight vector."""
    return pd.concat(
        {c: frac_diff_ffd(df[c], d, threshold) for c in df.columns}, axis=1
    )[df.columns]
