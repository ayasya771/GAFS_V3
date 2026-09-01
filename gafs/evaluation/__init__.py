"""Stylized-facts evaluation and diagnostic plotting."""

from .stylized_facts import (
    evaluate_stylized_facts,
    hill_alpha,
    leverage_correlation,
    real_to_windows,
    series_moments,
    window_acf,
    write_markdown_report,
)

__all__ = [
    "evaluate_stylized_facts",
    "hill_alpha",
    "leverage_correlation",
    "real_to_windows",
    "series_moments",
    "window_acf",
    "write_markdown_report",
]
