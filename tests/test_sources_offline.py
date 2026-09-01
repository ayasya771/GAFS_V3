"""Offline-safe tests for the data source adapters (no network calls)."""

import numpy as np
import pandas as pd
import pytest

from gafs.data.sources import DataSourceError, load_dukascopy_csv
from gafs.data.sources.base import to_utc_index


def test_to_utc_index_sorts_dedups_localizes():
    idx = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-01"])
    df = pd.DataFrame({"x": [3.0, 1.0, 2.0]}, index=idx)
    out = to_utc_index(df)
    assert str(out.index.tz) == "UTC"
    assert list(out["x"]) == [2.0, 3.0]


def test_dukascopy_bar_export(tmp_path):
    path = tmp_path / "eurusd-m1.csv"
    ts = pd.date_range("2021-01-04", periods=5, freq="min").astype("int64") // 10**6
    pd.DataFrame(
        {"timestamp": ts, "open": 1.1, "high": 1.2, "low": 1.0,
         "close": np.linspace(1.10, 1.14, 5), "volume": 10}
    ).to_csv(path, index=False)
    out = load_dukascopy_csv(path)
    assert list(out.columns) == ["EURUSD_close"]
    assert len(out) == 5
    assert str(out.index.tz) == "UTC"
    assert out["EURUSD_close"].iloc[-1] == pytest.approx(1.14)


def test_dukascopy_tick_export_uses_mid(tmp_path):
    path = tmp_path / "usa500idxusd-tick.csv"
    ts = pd.date_range("2021-01-04", periods=3, freq="s").astype("int64") // 10**6
    pd.DataFrame(
        {"timestamp": ts, "askPrice": [4001.0, 4002.0, 4003.0],
         "bidPrice": [4000.0, 4001.0, 4002.0]}
    ).to_csv(path, index=False)
    out = load_dukascopy_csv(path, name="US500")
    assert list(out.columns) == ["US500_close"]
    assert out["US500_close"].iloc[0] == pytest.approx(4000.5)


def test_dukascopy_missing_file_message(tmp_path):
    with pytest.raises(DataSourceError, match="dukascopy-node"):
        load_dukascopy_csv(tmp_path / "nope.csv")


def test_dukascopy_rejects_unusable_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": [1, 2], "weird": [1.0, 2.0]}).to_csv(path, index=False)
    with pytest.raises(DataSourceError, match="close or bid/ask"):
        load_dukascopy_csv(path)
