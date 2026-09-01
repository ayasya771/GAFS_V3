"""Yahoo Finance daily/intraday bars via yfinance.

Long-horizon daily data with adjusted closes, so splits and dividends do not
masquerade as crashes across 1987/2000/2008-style histories.
"""

from __future__ import annotations

import pandas as pd

from .base import DataSourceError, to_utc_index


def fetch_yahoo(
    tickers: list[str],
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices, columns `{ticker}_close`."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataSourceError(
            "yfinance is not installed. Run: pip install yfinance"
        ) from exc

    try:
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception as exc:
        raise DataSourceError(f"Yahoo Finance download failed: {exc}") from exc

    if raw is None or len(raw) == 0:
        raise DataSourceError("Yahoo Finance returned no rows (check tickers/dates or connectivity).")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(close, pd.DataFrame):
        close = close.to_frame()
    if len(tickers) == 1 and close.shape[1] == 1:
        close.columns = [tickers[0]]

    close = close.rename(columns={t: f"{t}_close" for t in close.columns})
    close = close.dropna(how="all")
    if close.empty:
        raise DataSourceError("Yahoo Finance returned only empty columns.")
    return to_utc_index(close)
