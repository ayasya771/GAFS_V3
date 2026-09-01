"""Shared plumbing for data source adapters."""

from __future__ import annotations

import pandas as pd


class DataSourceError(RuntimeError):
    """Raised when a source is unreachable or returns unusable data."""


def to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-duplicate and convert the index to tz-aware UTC."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df
