"""Free-tier market data source adapters.

Every adapter returns a pandas DataFrame with a tz-aware UTC DatetimeIndex.
Network failures raise DataSourceError with an actionable message so the CLI
can degrade gracefully source by source.
"""

from .base import DataSourceError
from .yahoo import fetch_yahoo
from .fred import fetch_fred
from .binance import fetch_binance_klines
from .alpaca import fetch_alpaca_bars
from .dukascopy import load_dukascopy_csv

__all__ = [
    "DataSourceError",
    "fetch_yahoo",
    "fetch_fred",
    "fetch_binance_klines",
    "fetch_alpaca_bars",
    "load_dukascopy_csv",
]
