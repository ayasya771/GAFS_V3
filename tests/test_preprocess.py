import numpy as np
import pandas as pd
import pytest

from gafs.config import PreprocessConfig
from gafs.data.preprocess import (
    align_join,
    ensure_utc,
    mad_clean_prices,
    preprocess_market,
    to_log_returns,
)
from gafs.data.synthetic import generate_synthetic_market, macro_columns, price_columns


def _dates(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def test_ensure_utc_localizes_and_sorts():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame({"a": range(5)}, index=idx[::-1])
    out = ensure_utc(df)
    assert str(out.index.tz) == "UTC"
    assert out.index.is_monotonic_increasing


def test_align_join_ffill_limit_three():
    idx = _dates(10)
    a = pd.DataFrame({"A_close": np.arange(10.0) + 1}, index=idx)
    b = pd.DataFrame({"B_close": np.arange(10.0) + 1}, index=idx)
    b.iloc[3:5] = np.nan
    b.iloc[6:10] = np.nan
    joined = align_join([a, b], ffill_limit=3)
    assert idx[3] in joined.index and idx[4] in joined.index
    assert joined.loc[idx[4], "B_close"] == 3.0
    assert idx[9] not in joined.index
    assert len(joined) == 9


def test_mad_removes_spike_keeps_crash():
    n = 400
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        {"X_close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))},
        index=_dates(n),
    )
    spiked = prices.copy()
    spiked.iloc[200, 0] *= 1.60
    crashed = spiked.copy()
    crashed.iloc[300:, 0] *= 0.75

    clean, flagged = mad_clean_prices(crashed, window=51, n_sigmas=8.0)
    assert flagged["X_close"] >= 1
    assert abs(np.log(clean.iloc[200, 0] / crashed.iloc[199, 0])) < 0.10
    assert np.allclose(clean.iloc[300:, 0], crashed.iloc[300:, 0])


def test_log_returns_definition():
    prices = pd.DataFrame({"A_close": [100.0, 110.0, 99.0]}, index=_dates(3))
    rets = to_log_returns(prices)
    assert list(rets.columns) == ["A"]
    assert rets.iloc[0, 0] == pytest.approx(np.log(1.10))


def test_preprocess_market_end_to_end_no_lookahead():
    panel = generate_synthetic_market(n_days=900, seed=3)
    prices, macro = panel[price_columns(panel)], panel[macro_columns(panel)]
    cfg = PreprocessConfig(vol_window=30, train_frac=0.7)
    full = preprocess_market(prices, macro, cfg)

    assert full.meta["n_rows"] == len(full.features)
    assert not full.features.isna().any().any()
    assert set(full.meta["asset_cols"]) | set(full.meta["macro_cols"]) == set(
        full.features.columns
    )

    cut = len(panel) - 150
    truncated = preprocess_market(prices.iloc[:cut], macro.iloc[:cut], cfg)
    common = truncated.features.index[:-30]
    for a in full.meta["asset_cols"]:
        pd.testing.assert_series_equal(
            full.features.loc[common, a],
            truncated.features.loc[common, a],
            rtol=1e-10,
        )


def test_preprocess_zscore_and_fracdiff_modes():
    panel = generate_synthetic_market(n_days=700, seed=5)
    prices, macro = panel[price_columns(panel)], panel[macro_columns(panel)]
    z = preprocess_market(prices, macro, PreprocessConfig(scaling="zscore"))
    assert z.meta["asset_stats"]
    f = preprocess_market(
        prices, macro,
        PreprocessConfig(stationarity="fracdiff", fracdiff_d=0.4),
    )
    assert f.meta["stationarity"] == "fracdiff"
    assert len(f.features) > 100
