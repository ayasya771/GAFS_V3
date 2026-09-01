"""Dukascopy tick/minute data: FX, indices, commodities.

Dukascopy serves raw .bi5 tick files; the practical free workflow is the
open-source `dukascopy-node` CLI, which handles download, decoding and
aggregation. Bulk-download example (minute bars for EUR/USD and the S&P CFD):

    npx dukascopy-node -i eurusd -from 2015-01-01 -to 2025-01-01 \
        -t m1 -f csv -dir data/raw/dukascopy
    npx dukascopy-node -i usa500idxusd -from 2015-01-01 -to 2025-01-01 \
        -t m1 -f csv -dir data/raw/dukascopy

This module then loads those CSV exports into the GAFS pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import DataSourceError, to_utc_index


def load_dukascopy_csv(path: str | Path, name: str | None = None) -> pd.DataFrame:
    """Load a dukascopy-node CSV export; returns `{name}_close` (UTC index).

    Handles both tick exports (bidPrice/askPrice: mid is used) and aggregated
    bar exports (close). Timestamps are epoch milliseconds or ISO strings.
    """
    path = Path(path)
    if not path.exists():
        raise DataSourceError(
            f"Dukascopy CSV not found: {path}. Download it first with "
            "dukascopy-node (see this module's docstring)."
        )
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    ts_col = next((cols[k] for k in ("timestamp", "time", "date") if k in cols), None)
    if ts_col is None:
        raise DataSourceError(f"No timestamp column in {path}; columns: {list(df.columns)}")
    ts = df[ts_col]
    if pd.api.types.is_numeric_dtype(ts):
        idx = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        idx = pd.to_datetime(ts, utc=True)

    if "close" in cols:
        px = pd.to_numeric(df[cols["close"]], errors="coerce")
    elif "bidprice" in cols and "askprice" in cols:
        bid = pd.to_numeric(df[cols["bidprice"]], errors="coerce")
        ask = pd.to_numeric(df[cols["askprice"]], errors="coerce")
        px = (bid + ask) / 2.0
    else:
        raise DataSourceError(f"No close or bid/ask columns in {path}.")

    instrument = name or path.stem.split("-")[0].upper()
    out = pd.DataFrame({f"{instrument}_close": px.to_numpy()}, index=idx).dropna()
    if out.empty:
        raise DataSourceError(f"{path} parsed to zero usable rows.")
    return to_utc_index(out)
