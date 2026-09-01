"""Alpaca Market Data: free-tier US equity minute bars.

Requires the `alpaca-py` package and free API keys in the environment:
ALPACA_API_KEY / ALPACA_SECRET_KEY (paper account keys work).
"""

from __future__ import annotations

import os

import pandas as pd

from .base import DataSourceError, to_utc_index


def fetch_alpaca_bars(
    tickers: list[str],
    start: str,
    end: str,
    timeframe: str = "1Min",
) -> pd.DataFrame:
    """Return `{ticker}_close` minute/daily bars from Alpaca's IEX feed."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError as exc:
        raise DataSourceError(
            "alpaca-py is not installed. Run: pip install alpaca-py"
        ) from exc

    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise DataSourceError(
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables "
            "(free keys: https://alpaca.markets)."
        )

    unit_map = {"Min": TimeFrameUnit.Minute, "Hour": TimeFrameUnit.Hour, "Day": TimeFrameUnit.Day}
    amount = int("".join(ch for ch in timeframe if ch.isdigit()) or 1)
    unit_name = "".join(ch for ch in timeframe if ch.isalpha())
    if unit_name not in unit_map:
        raise DataSourceError(f"Unsupported Alpaca timeframe: {timeframe}")

    client = StockHistoricalDataClient(key, secret)
    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame(amount, unit_map[unit_name]),
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
    )
    try:
        bars = client.get_stock_bars(request).df
    except Exception as exc:
        raise DataSourceError(f"Alpaca request failed: {exc}") from exc
    if bars.empty:
        raise DataSourceError("Alpaca returned no bars (check keys, tickers, dates).")

    close = bars["close"].unstack(level=0)
    close = close.rename(columns={t: f"{t}_close" for t in close.columns})
    return to_utc_index(close)
