"""FRED macro conditioning series, no API key required.

Uses the public fredgraph.csv endpoint for series such as DGS10 (10y treasury),
VIXCLS (VIX close) and BAMLC0A0CM (IG credit spread).
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request

import pandas as pd

from .base import DataSourceError, to_utc_index

_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}&coed={end}"
_TIMEOUT = 30


def _fetch_one(series_id: str, start: str, end: str) -> pd.Series:
    url = _FRED_CSV.format(sid=series_id, start=start, end=end)
    req = urllib.request.Request(url, headers={"User-Agent": "gafs/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            payload = resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DataSourceError(f"FRED fetch failed for {series_id}: {exc}") from exc

    df = pd.read_csv(io.StringIO(payload), na_values=["."])
    date_col = df.columns[0]
    if series_id not in df.columns:
        raise DataSourceError(f"FRED response for {series_id} missing its value column: {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col])
    series = df.set_index(date_col)[series_id].astype(float)
    series.name = series_id
    return series


def fetch_fred(series_ids: list[str], start: str, end: str) -> pd.DataFrame:
    """Return one column per series id, outer-joined on date, UTC index."""
    if not series_ids:
        raise DataSourceError("No FRED series ids given.")
    columns = [_fetch_one(sid, start, end) for sid in series_ids]
    df = pd.concat(columns, axis=1).sort_index()
    return to_utc_index(df)
