"""Binance historical archive: extreme-regime crypto proxy.

Downloads monthly kline zips from data.binance.vision (no API key). Crypto
tick/minute data exhibits hyper-fat tails and severe flash crashes, which
teaches the model extreme non-linear crash dynamics.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
import zipfile

import pandas as pd

from .base import DataSourceError, to_utc_index

_ARCHIVE = (
    "https://data.binance.vision/data/spot/monthly/klines/"
    "{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
)
_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "n_trades", "taker_base", "taker_quote", "ignore",
]
_TIMEOUT = 60


def _fetch_month(symbol: str, interval: str, ym: str) -> pd.DataFrame | None:
    url = _ARCHIVE.format(symbol=symbol, interval=interval, ym=ym)
    req = urllib.request.Request(url, headers={"User-Agent": "gafs/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise DataSourceError(f"Binance archive HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DataSourceError(f"Binance archive fetch failed: {exc}") from exc

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            df = pd.read_csv(f, header=None, names=_KLINE_COLS)
    if df.iloc[0]["open_time"] == "open_time":
        df = df.iloc[1:]
    ts = pd.to_numeric(df["open_time"], errors="coerce")
    unit = "us" if ts.iloc[0] > 1e14 else "ms"
    out = pd.DataFrame(
        {
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
        },
        index=pd.to_datetime(ts, unit=unit, utc=True),
    )
    return out.dropna()


def fetch_binance_klines(
    symbols: list[str],
    start: str,
    end: str,
    interval: str = "1m",
) -> pd.DataFrame:
    """Return `{symbol}_close` (and `{symbol}_volume`) columns for the range."""
    months = pd.period_range(start=start, end=end, freq="M").strftime("%Y-%m")
    frames = []
    for symbol in symbols:
        parts = [m for ym in months if (m := _fetch_month(symbol, interval, ym)) is not None]
        if not parts:
            raise DataSourceError(
                f"No Binance archive months found for {symbol} {interval} in range."
            )
        sym = pd.concat(parts).sort_index()
        sym = sym.rename(columns={"close": f"{symbol}_close", "volume": f"{symbol}_volume"})
        frames.append(sym)
    return to_utc_index(pd.concat(frames, axis=1))
