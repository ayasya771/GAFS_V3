"""Data acquisition, synthesis, preprocessing and windowing."""

from .synthetic import generate_synthetic_market, macro_columns, price_columns
from .fracdiff import ffd_weights, frac_diff_ffd
from .preprocess import (
    ProcessedData,
    align_join,
    ensure_utc,
    mad_clean_prices,
    preprocess_market,
    to_log_returns,
)
from .windows import WindowArrays, build_windows, chronological_split

__all__ = [
    "generate_synthetic_market",
    "macro_columns",
    "price_columns",
    "ffd_weights",
    "frac_diff_ffd",
    "ProcessedData",
    "align_join",
    "ensure_utc",
    "mad_clean_prices",
    "preprocess_market",
    "to_log_returns",
    "WindowArrays",
    "build_windows",
    "chronological_split",
]
