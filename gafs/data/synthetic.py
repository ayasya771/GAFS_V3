"""Calibrated synthetic multi-asset market for offline development and tests.

The generator below is NOT the GAN. It is a classical econometric simulator
(correlated GJR-GARCH with common and idiosyncratic jumps plus a 2-state
volatility regime chain) used to produce training data with authentic stylized
facts when market-data endpoints are unreachable:

* fat tails            (jumps + GARCH mixture)
* volatility clustering (GARCH persistence alpha + beta near 1)
* leverage effect       (GJR asymmetry: negative returns raise variance more)
* cross-asset correlation with crisis co-crashes (common jump factor)

Macro conditioning proxies are derived the way their real counterparts behave:
a VIX-like index tracking short-horizon realized volatility, a mean-reverting
10y rate, and a credit spread that widens with market stress.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_ASSETS = ("EQ_LARGE", "EQ_TECH", "CMD_GOLD")


def generate_synthetic_market(
    n_days: int = 6000,
    assets: tuple[str, ...] = DEFAULT_ASSETS,
    seed: int = 7,
    start: str = "2001-01-01",
) -> pd.DataFrame:
    """Return a business-day DataFrame with `{asset}_close` price columns and
    macro columns VIX_PROXY, RATE_10Y, CREDIT_SPREAD (UTC index)."""
    rng = np.random.default_rng(seed)
    n_assets = len(assets)

    corr = np.full((n_assets, n_assets), 0.35)
    np.fill_diagonal(corr, 1.0)
    if n_assets >= 2:
        corr[0, 1] = corr[1, 0] = 0.80
    if n_assets >= 3:
        corr[0, 2] = corr[2, 0] = 0.10
        corr[1, 2] = corr[2, 1] = 0.05
    chol = np.linalg.cholesky(corr)

    target_ann = np.linspace(0.16, 0.24, n_assets)
    daily_var = (target_ann / np.sqrt(252.0)) ** 2
    alpha, beta, gamma = 0.05, 0.90, 0.06
    omega = daily_var * (1.0 - alpha - beta - gamma / 2.0)
    mu = 0.0003

    p_calm_stay, p_stress_stay = 0.995, 0.980
    regime_mult = np.array([1.0, 2.1])

    p_common_jump, p_idio_jump = 0.0045, 0.0020
    common_beta = np.linspace(1.2, 0.5, n_assets)

    sigma2 = daily_var.copy()
    state = 0
    rets = np.zeros((n_days, n_assets))
    regimes = np.zeros(n_days, dtype=int)

    sigma2_cap = (0.15) ** 2
    for t in range(n_days):
        stay = p_calm_stay if state == 0 else p_stress_stay
        if rng.random() > stay:
            state = 1 - state
        regimes[t] = state

        eps = chol @ rng.standard_normal(n_assets)
        base_innov = np.sqrt(sigma2) * eps
        r = mu + regime_mult[state] * base_innov

        if rng.random() < p_common_jump:
            size = rng.normal(-0.05, 0.02)
            r = r + common_beta * size
            state = 1
        idio = rng.random(n_assets) < p_idio_jump
        if idio.any():
            r[idio] += rng.normal(-0.02, 0.025, idio.sum())

        sigma2 = (
            omega
            + (alpha + gamma * (base_innov < 0)) * base_innov**2
            + beta * sigma2
        )
        sigma2 = np.clip(sigma2, 1e-10, sigma2_cap)
        rets[t] = r

    index = pd.bdate_range(start=start, periods=n_days, tz="UTC")
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    df = pd.DataFrame(prices, index=index, columns=[f"{a}_close" for a in assets])

    mkt_ret = rets[:, 0]
    rv = pd.Series(mkt_ret, index=index).ewm(span=10).std().bfill()
    vix = (100.0 * np.sqrt(252.0) * rv * regime_mult[regimes] ** 0.5
           + rng.normal(0.0, 0.8, n_days))
    df["VIX_PROXY"] = np.clip(vix, 8.0, None)

    rate = np.zeros(n_days)
    rate[0] = 3.0
    for t in range(1, n_days):
        rate[t] = rate[t - 1] + 0.002 * (3.0 - rate[t - 1]) + rng.normal(0.0, 0.03)
    df["RATE_10Y"] = np.clip(rate, 0.1, 9.0)

    spread = 0.9 + 0.045 * (df["VIX_PROXY"].to_numpy() - 15.0) + rng.normal(0.0, 0.05, n_days)
    df["CREDIT_SPREAD"] = np.clip(spread, 0.3, None)

    return df


def macro_columns(df: pd.DataFrame) -> list[str]:
    """Columns that act as conditioning vectors (everything not *_close)."""
    return [c for c in df.columns if not c.endswith("_close")]


def price_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_close")]
