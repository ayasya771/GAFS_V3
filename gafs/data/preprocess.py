"""Preprocessing pipeline, in the order the transforms must be applied:

1. UTC normalisation and exact timestamp alignment, forward-filling at most
   `ffill_limit` periods and dropping rows that stay incomplete.
2. MAD-based bad-tick removal that preserves genuine flash crashes: a point is
   removed only when it is a robust outlier AND the next observation reverts
   toward the local median (isolated spike). Persistent moves survive.
3. Stationarity: log returns, or fixed-width fractional differencing.
4. Standardisation: division by the 30-day rolling realized volatility
   (lagged one step, so it is strictly causal) or train-fitted z-scores.
5. Windowing into [batch, sequence, features] tensors lives in windows.py.

All scaler statistics that require fitting are computed on the training prefix
only; nothing in this module looks ahead of time t when transforming time t.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ..config import PreprocessConfig
from .fracdiff import frac_diff_frame

PRICE_SUFFIX = "_close"


def ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-duplicate and convert the index to tz-aware UTC."""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out = out.sort_index()
    return out[~out.index.duplicated(keep="last")]


def align_join(frames: Iterable[pd.DataFrame], ffill_limit: int = 3) -> pd.DataFrame:
    """Exact timestamp join across sources with bounded forward-filling.

    Union-join on UTC timestamps, forward-fill gaps of at most `ffill_limit`
    periods, then drop any row that is still incomplete. The result is an
    exactly aligned panel (equivalent to an inner join after limited ffill).
    """
    frames = [ensure_utc(f) for f in frames]
    joined = pd.concat(frames, axis=1, join="outer", sort=True)
    dup = joined.columns[joined.columns.duplicated()]
    if len(dup):
        raise ValueError(f"Duplicate columns across sources: {sorted(set(dup))}")
    joined = joined.sort_index()
    joined = joined.ffill(limit=ffill_limit) if ffill_limit > 0 else joined
    return joined.dropna(how="any")


def mad_clean_prices(
    prices: pd.DataFrame,
    window: int = 51,
    n_sigmas: float = 10.0,
    reversal_tol: float = 0.40,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove erroneous spikes while preserving real crashes.

    Standard-deviation filters are skewed by the very crashes we want to learn,
    so a centered rolling median/MAD flags candidates in log-price space. A
    candidate is treated as a bad tick only if the NEXT observation reverts to
    within `reversal_tol` of the deviation (an isolated spike); it is then
    replaced by the local median. A crash shifts the level persistently, fails
    the reversal test, and is kept untouched.
    """
    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive for log-space cleaning.")
    if window % 2 == 0:
        window += 1

    clean = prices.copy()
    flagged: dict[str, int] = {}
    min_p = max(5, window // 4)
    for col in prices.columns:
        p = np.log(prices[col].astype(float))
        med = p.rolling(window, center=True, min_periods=min_p).median()
        dev = p - med
        mad = dev.abs().rolling(window, center=True, min_periods=min_p).median()
        sigma = 1.4826 * mad
        z = dev.abs() / sigma.replace(0.0, np.nan)
        candidates = (z > n_sigmas).fillna(False)

        next_dev = (p.shift(-1) - med).abs()
        reverts = (next_dev <= reversal_tol * dev.abs()).fillna(False)
        bad = candidates & reverts
        flagged[col] = int(bad.sum())
        if bad.any():
            fixed = p.where(~bad, med)
            clean[col] = np.exp(fixed)
    return clean, flagged


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns r_t = ln(P_t / P_{t-1}); columns lose the '_close' suffix."""
    rets = np.log(prices.astype(float)).diff().iloc[1:]
    rets.columns = [c.removesuffix(PRICE_SUFFIX) for c in rets.columns]
    return rets


def _stationarize(prices: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    if cfg.stationarity == "log_returns":
        return to_log_returns(prices)
    log_p = np.log(prices.astype(float))
    log_p.columns = [c.removesuffix(PRICE_SUFFIX) for c in log_p.columns]
    return frac_diff_frame(log_p, cfg.fracdiff_d, cfg.fracdiff_threshold)


@dataclass
class ProcessedData:
    """Model-ready panel plus everything needed to invert the transforms."""

    features: pd.DataFrame
    aux: pd.DataFrame
    meta: dict = field(default_factory=dict)

    @property
    def asset_cols(self) -> list[str]:
        return list(self.meta["asset_cols"])

    @property
    def macro_cols(self) -> list[str]:
        return list(self.meta["macro_cols"])

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.features.to_parquet(out / "features.parquet")
        self.aux.to_parquet(out / "aux.parquet")
        with open(out / "meta.json", "w") as f:
            json.dump(self.meta, f, indent=2, default=str)
        return out

    @classmethod
    def load(cls, in_dir: str | Path) -> "ProcessedData":
        p = Path(in_dir)
        features = pd.read_parquet(p / "features.parquet")
        aux = pd.read_parquet(p / "aux.parquet")
        with open(p / "meta.json") as f:
            meta = json.load(f)
        return cls(features=features, aux=aux, meta=meta)


def preprocess_market(
    prices: pd.DataFrame,
    macro: pd.DataFrame | None,
    cfg: PreprocessConfig | None = None,
) -> ProcessedData:
    """Run the full preprocessing pipeline on a raw price panel + macro panel.

    `prices` columns must end in '_close'. `macro` may be None (unconditional
    model); its columns become the conditioning vector C_t after train-fitted
    z-scoring.
    """
    cfg = cfg or PreprocessConfig()
    prices = ensure_utc(prices)
    price_cols = [c for c in prices.columns if c.endswith(PRICE_SUFFIX)]
    if not price_cols:
        raise ValueError("No '*_close' price columns found.")
    prices = prices[price_cols]

    frames = [prices]
    macro_cols: list[str] = []
    if macro is not None and len(macro.columns):
        macro = ensure_utc(macro)
        macro_cols = list(macro.columns)
        frames.append(macro)

    panel = align_join(frames, cfg.ffill_limit)
    prices_al = panel[price_cols]
    macro_al = panel[macro_cols] if macro_cols else None

    prices_clean, flagged = mad_clean_prices(
        prices_al, cfg.mad_window, cfg.mad_threshold, cfg.mad_reversal_tol
    )

    stat = _stationarize(prices_clean, cfg)
    asset_cols = list(stat.columns)
    rets = to_log_returns(prices_clean)
    ret_vol = rets.rolling(cfg.vol_window).std()

    if cfg.scaling == "vol":
        divisor = stat.rolling(cfg.vol_window).std().shift(1)
        scaled_assets = stat / (divisor + cfg.eps)
        asset_stats: dict[str, dict[str, float]] = {}
    else:
        scaled_assets = None
        asset_stats = {}

    parts = [stat if cfg.scaling == "zscore" else scaled_assets]
    if macro_al is not None:
        parts.append(macro_al)
    candidate = pd.concat(parts, axis=1, sort=False).dropna().index
    if len(candidate) < 10:
        raise ValueError("Too few aligned observations after preprocessing.")
    train_end_pos = max(1, int(cfg.train_frac * len(candidate)) - 1)
    train_end_ts = candidate[train_end_pos]

    if cfg.scaling == "zscore":
        train_slice = stat.loc[:train_end_ts]
        mean, std = train_slice.mean(), train_slice.std().replace(0.0, 1.0)
        scaled_assets = (stat - mean) / (std + cfg.eps)
        asset_stats = {
            c: {"mean": float(mean[c]), "std": float(std[c])} for c in asset_cols
        }

    macro_stats: dict[str, dict[str, float]] = {}
    if macro_al is not None:
        m_train = macro_al.loc[:train_end_ts]
        m_mean, m_std = m_train.mean(), m_train.std().replace(0.0, 1.0)
        macro_scaled = (macro_al - m_mean) / (m_std + cfg.eps)
        macro_stats = {
            c: {"mean": float(m_mean[c]), "std": float(m_std[c])} for c in macro_cols
        }
        features = pd.concat([scaled_assets, macro_scaled], axis=1, sort=False).dropna()
    else:
        features = scaled_assets.dropna()

    aux = pd.concat(
        {
            **{f"{a}__ret": rets[a] for a in asset_cols},
            **{f"{a}__vol": ret_vol[a] for a in asset_cols},
            **{f"{a}__close": prices_clean[f"{a}{PRICE_SUFFIX}"] for a in asset_cols},
        },
        axis=1,
        sort=False,
    )
    aux.columns = [c[0] if isinstance(c, tuple) else c for c in aux.columns]
    aux = aux.reindex(features.index)

    meta = {
        "asset_cols": asset_cols,
        "macro_cols": macro_cols,
        "feature_cols": list(features.columns),
        "stationarity": cfg.stationarity,
        "fracdiff_d": cfg.fracdiff_d if cfg.stationarity == "fracdiff" else None,
        "scaling": cfg.scaling,
        "vol_window": cfg.vol_window,
        "eps": cfg.eps,
        "asset_stats": asset_stats,
        "macro_stats": macro_stats,
        "train_end_ts": str(train_end_ts),
        "mad_flagged": flagged,
        "preprocess_config": dataclasses.asdict(cfg),
        "n_rows": int(len(features)),
    }
    return ProcessedData(features=features, aux=aux, meta=meta)


def unscale_asset_paths(
    scaled: np.ndarray,
    meta: Mapping,
    last_vol: np.ndarray,
    eps: float | None = None,
) -> np.ndarray:
    """Invert the asset scaling for generated paths.

    scaled:   [N, h, A] generator output in feature space
    last_vol: [N, A] rolling return volatility at each path's forecast origin
    Returns log returns [N, h, A]. Only valid when stationarity was
    'log_returns'; fracdiff paths stay in fracdiff space by design.
    """
    if meta["stationarity"] != "log_returns":
        raise ValueError(
            "Price reconstruction requires stationarity='log_returns'; "
            "fractionally differenced paths are analysed in feature space."
        )
    eps = float(meta.get("eps", 1e-8)) if eps is None else eps
    if meta["scaling"] == "vol":
        return scaled * (last_vol[:, None, :] + eps)
    stats = meta["asset_stats"]
    mean = np.array([stats[a]["mean"] for a in meta["asset_cols"]], dtype=float)
    std = np.array([stats[a]["std"] for a in meta["asset_cols"]], dtype=float)
    return scaled * (std[None, None, :] + eps) + mean[None, None, :]
