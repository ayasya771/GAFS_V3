"""Stylized-facts battery: does the synthetic data behave like markets?

Checks the properties the synthetic data must preserve to be usable, with the SAME estimator applied to real and generated windows so
the comparison is apples to apples:

* fat tails             excess kurtosis, Hill tail exponent (lower = fatter)
* no linear memory      ACF of returns ~ 0 at all lags
* volatility clustering slowly decaying ACF of |returns|
* leverage effect       corr(r_t, |r_{t+l}|) < 0 for l >= 1 (equities)
* cross-asset structure correlation matrix distance
* distributional match  1D Wasserstein distance per asset
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats


def series_moments(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    return {
        "mean_daily": float(np.mean(x)),
        "ann_vol": float(np.std(x, ddof=1) * np.sqrt(252.0)),
        "skew": float(stats.skew(x)),
        "excess_kurtosis": float(stats.kurtosis(x, fisher=True)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def window_acf(windows: np.ndarray, max_lag: int = 15, absolute: bool = False) -> np.ndarray:
    """Average sample ACF across windows. windows: [N, T]."""
    w = np.abs(windows) if absolute else windows
    w = w - w.mean(axis=1, keepdims=True)
    T = w.shape[1]
    max_lag = min(max_lag, T - 2)
    denom = (w * w).sum(axis=1)
    denom = np.where(denom < 1e-12, np.nan, denom)
    out = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        num = (w[:, :-lag] * w[:, lag:]).sum(axis=1)
        out[lag - 1] = np.nanmean(num / denom)
    return out


def leverage_correlation(windows: np.ndarray, lags: tuple[int, ...] = (1, 2, 3, 4, 5)) -> float:
    """Mean corr(r_t, |r_{t+l}|) pooled across windows; negative = leverage."""
    vals = []
    for lag in lags:
        if windows.shape[1] <= lag + 1:
            continue
        a = windows[:, :-lag].ravel()
        b = np.abs(windows[:, lag:]).ravel()
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        vals.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(vals)) if vals else float("nan")


def hill_alpha(x: np.ndarray, tail_frac: float = 0.05) -> float:
    """Hill estimator of the tail exponent on |x| (alpha ~ 3-5 for equities;
    smaller alpha means fatter tails)."""
    a = np.abs(np.asarray(x, dtype=float).ravel())
    a = a[np.isfinite(a) & (a > 0)]
    if a.size < 50:
        return float("nan")
    a = np.sort(a)[::-1]
    k = max(10, int(tail_frac * a.size))
    k = min(k, a.size - 1)
    tail = a[:k]
    return float(k / np.sum(np.log(tail / a[k])))


def real_to_windows(series: np.ndarray, horizon: int, stride: int | None = None) -> np.ndarray:
    """Slice one real return series [T] into [M, horizon] windows."""
    stride = stride or max(1, horizon // 2)
    x = np.asarray(series, dtype=float)
    if x.size < horizon:
        raise ValueError(f"Series length {x.size} shorter than horizon {horizon}.")
    sw = sliding_window_view(x, horizon)
    return np.ascontiguousarray(sw[::stride])


def evaluate_stylized_facts(
    real_returns: pd.DataFrame,
    fake_returns: np.ndarray,
    asset_names: list[str] | None = None,
    max_lag: int = 15,
) -> dict:
    """Compare a real return panel [T, A] against generated windows [N, h, A]."""
    asset_names = asset_names or list(real_returns.columns)
    N, h, A = fake_returns.shape
    if A != len(asset_names):
        raise ValueError(f"fake has {A} assets, names give {len(asset_names)}")

    results: dict = {"horizon": h, "n_fake_windows": int(N), "assets": {}}
    for i, name in enumerate(asset_names):
        real_series = real_returns[name].dropna().to_numpy()
        rw = real_to_windows(real_series, h)
        fw = fake_returns[:, :, i]

        results["assets"][name] = {
            "real": {
                **series_moments(real_series),
                "acf_ret": window_acf(rw, max_lag).tolist(),
                "acf_abs": window_acf(rw, max_lag, absolute=True).tolist(),
                "leverage": leverage_correlation(rw),
                "hill_alpha": hill_alpha(real_series),
            },
            "fake": {
                **series_moments(fw),
                "acf_ret": window_acf(fw, max_lag).tolist(),
                "acf_abs": window_acf(fw, max_lag, absolute=True).tolist(),
                "leverage": leverage_correlation(fw),
                "hill_alpha": hill_alpha(fw),
            },
            "wasserstein": float(
                stats.wasserstein_distance(real_series, fw.ravel())
            ),
        }

    real_corr = np.corrcoef(real_returns[asset_names].dropna().to_numpy().T)
    fake_corr = np.corrcoef(fake_returns.reshape(N * h, A).T)
    results["correlation"] = {
        "real": real_corr.tolist(),
        "fake": fake_corr.tolist(),
        "frobenius": float(np.linalg.norm(real_corr - fake_corr)),
        "mean_abs_diff": float(np.mean(np.abs(real_corr - fake_corr))),
    }
    return results


def write_markdown_report(results: dict, path, title: str = "Stylized-facts report") -> None:
    from pathlib import Path

    lines = [f"# {title}", ""]
    lines.append(
        f"Generated windows: {results['n_fake_windows']} of length {results['horizon']} steps."
    )
    lines.append("")
    header = (
        "| Asset | Metric | Real | Generated |\n"
        "|---|---|---:|---:|"
    )
    rows: list[str] = [header]
    metric_names = [
        ("ann_vol", "Annualised volatility"),
        ("skew", "Skewness"),
        ("excess_kurtosis", "Excess kurtosis (fat tails)"),
        ("hill_alpha", "Hill tail exponent"),
        ("leverage", "Leverage corr(r_t, |r_t+l|)"),
    ]
    for name, res in results["assets"].items():
        for key, label in metric_names:
            rows.append(
                f"| {name} | {label} | {res['real'][key]:.4f} | {res['fake'][key]:.4f} |"
            )
        acf_r = np.mean(np.abs(res["real"]["acf_ret"]))
        acf_f = np.mean(np.abs(res["fake"]["acf_ret"]))
        rows.append(f"| {name} | Mean abs ACF of returns | {acf_r:.4f} | {acf_f:.4f} |")
        abs_r = np.mean(res["real"]["acf_abs"][:5])
        abs_f = np.mean(res["fake"]["acf_abs"][:5])
        rows.append(f"| {name} | ACF of abs returns (lags 1-5) | {abs_r:.4f} | {abs_f:.4f} |")
        rows.append(f"| {name} | Wasserstein distance | | {res['wasserstein']:.6f} |")
    corr = results["correlation"]
    lines.extend(rows)
    lines.append("")
    lines.append(
        f"Correlation structure: Frobenius distance {corr['frobenius']:.4f}, "
        f"mean absolute entry difference {corr['mean_abs_diff']:.4f}."
    )
    lines.append("")
    lines.append(
        "Reading guide: generated values should sit near the real column. "
        "Excess kurtosis well above 0 and a slowly decaying ACF of absolute "
        "returns indicate fat tails and volatility clustering; a negative "
        "leverage correlation reproduces the equity leverage effect; the mean "
        "absolute ACF of raw returns should stay near zero (no free lunch)."
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))
