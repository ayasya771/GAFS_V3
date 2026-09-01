"""Conditioned scenario generation and macro stress testing (the payoff).

Given a trained generator, a preprocessed panel and a forecast anchor, draws
N stochastic future paths Y = G(X_{t-k:t}, C_t, Z). The macro conditioning
vector C_t is the stress lever: shocks are specified in RAW units (for
example VIX_PROXY +20 points) and mapped into the scaled space via the stored
train-set statistics, so "what if volatility spiked and spreads blew out"
becomes a direct query on the model.

Reconstruction: generated paths live in feature space; they are unscaled
with the anchor's rolling volatility (or z-score stats) into log returns and
compounded from the anchor close into price paths. VaR/ES and drawdown
summaries are computed on a weighted portfolio of the simulated paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.preprocess import ProcessedData, unscale_asset_paths


@dataclass
class ScenarioSet:
    returns: np.ndarray
    prices: np.ndarray
    asset_names: list[str]
    anchor_ts: str
    macro_shock: dict[str, float] | None = None
    macro_raw_at_anchor: dict[str, float] = field(default_factory=dict)

    @property
    def n_scenarios(self) -> int:
        return self.returns.shape[0]

    @property
    def horizon(self) -> int:
        return self.returns.shape[1]

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "scenarios.npz",
            returns=self.returns,
            prices=self.prices,
            asset_names=np.array(self.asset_names),
        )
        with open(out / "scenario_meta.json", "w") as f:
            json.dump(
                {
                    "anchor_ts": self.anchor_ts,
                    "n_scenarios": int(self.n_scenarios),
                    "horizon": int(self.horizon),
                    "macro_shock": self.macro_shock,
                    "macro_raw_at_anchor": self.macro_raw_at_anchor,
                },
                f,
                indent=2,
            )
        return out


def build_context(
    processed: ProcessedData,
    lookback: int,
    anchor: str | pd.Timestamp | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Timestamp]:
    """Assemble (x_hist [1,k,F], cond [1,C], last_vol [1,A], last_close [1,A])."""
    feats = processed.features
    if anchor is None:
        pos = len(feats) - 1
    else:
        ts = pd.Timestamp(anchor)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        pos = int(feats.index.get_indexer([ts], method="pad")[0])
        if pos < 0:
            raise ValueError(f"Anchor {anchor} precedes the processed data range.")
    if pos < lookback - 1:
        raise ValueError(
            f"Anchor at row {pos} leaves fewer than lookback={lookback} rows of history."
        )

    x = np.array(
        feats.to_numpy(dtype=np.float32)[pos - lookback + 1 : pos + 1][None, ...]
    )
    macro_pos = [list(feats.columns).index(m) for m in processed.macro_cols]
    cond = (
        np.array(x[:, -1, macro_pos])
        if macro_pos
        else np.zeros((1, 0), dtype=np.float32)
    )
    aux_row = processed.aux.iloc[pos]
    last_vol = np.array(
        [[aux_row[f"{a}__vol"] for a in processed.asset_cols]], dtype=np.float32
    )
    last_close = np.array(
        [[aux_row[f"{a}__close"] for a in processed.asset_cols]], dtype=np.float32
    )
    if not np.isfinite(last_vol).all() or not np.isfinite(last_close).all():
        raise ValueError("Anchor row has undefined volatility or close; pick a later anchor.")
    return x, cond.astype(np.float32), last_vol, last_close, feats.index[pos]


def apply_macro_shock(
    cond: np.ndarray,
    meta: dict,
    shock: dict[str, float],
) -> np.ndarray:
    """Shift the scaled conditioning vector by raw-unit shocks.

    cond is z-scored with train stats, so a raw shift of Delta maps to
    Delta / std in scaled space.
    """
    macro_cols = list(meta["macro_cols"])
    stats = meta["macro_stats"]
    out = cond.copy()
    for name, delta in shock.items():
        if name not in macro_cols:
            raise KeyError(f"Unknown macro column {name!r}; available: {macro_cols}")
        std = float(stats[name]["std"]) or 1.0
        out[:, macro_cols.index(name)] += float(delta) / std
    return out


def generate_scenarios(
    generator,
    processed: ProcessedData,
    n_scenarios: int = 1000,
    device: torch.device | str = "cpu",
    macro_shock: dict[str, float] | None = None,
    anchor: str | pd.Timestamp | None = None,
    batch_size: int = 256,
    seed: int | None = None,
) -> ScenarioSet:
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device(device)
    generator = generator.to(device).eval()
    lookback = generator.lookback

    x, cond, last_vol, last_close, anchor_ts = build_context(processed, lookback, anchor)
    macro_raw = {}
    if processed.macro_cols:
        stats = processed.meta["macro_stats"]
        for j, m in enumerate(processed.macro_cols):
            macro_raw[m] = float(cond[0, j] * stats[m]["std"] + stats[m]["mean"])
    if macro_shock:
        cond = apply_macro_shock(cond, processed.meta, macro_shock)

    x_t = torch.from_numpy(x).to(device)
    c_t = torch.from_numpy(cond).to(device) if cond.shape[-1] else None

    with torch.no_grad():
        y_scaled = generator.sample(x_t, c_t, n_scenarios, batch_size=batch_size)
    y_scaled = y_scaled.cpu().numpy()

    last_vol_n = np.repeat(last_vol, n_scenarios, axis=0)
    rets = unscale_asset_paths(y_scaled, processed.meta, last_vol_n)

    cum = np.cumsum(rets, axis=1)
    prices = last_close[0][None, None, :] * np.exp(cum)
    anchor_prices = np.repeat(last_close, n_scenarios, axis=0)[:, None, :]
    prices = np.concatenate([anchor_prices, prices], axis=1)

    return ScenarioSet(
        returns=rets,
        prices=prices,
        asset_names=list(processed.asset_cols),
        anchor_ts=str(anchor_ts),
        macro_shock=dict(macro_shock) if macro_shock else None,
        macro_raw_at_anchor=macro_raw,
    )


def scenario_summary(
    scn: ScenarioSet,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Percentiles, VaR/ES and drawdowns per asset and for a weighted portfolio."""
    N, h, A = scn.returns.shape
    if weights is None:
        weights = np.full(A, 1.0 / A)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (A,) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must have one entry per asset and sum to 1.")

    norm_paths = scn.prices / scn.prices[:, :1, :]
    port_paths = (norm_paths * weights[None, None, :]).sum(axis=2)

    rows = []
    for i, name in enumerate(scn.asset_names + ["PORTFOLIO"]):
        paths = norm_paths[:, :, i] if i < A else port_paths
        horizon_ret = paths[:, -1] - 1.0
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdown = (paths / running_max - 1.0).min(axis=1)
        q = lambda p: float(np.percentile(horizon_ret, p))
        var95, var99 = -q(5), -q(1)
        es95 = -float(horizon_ret[horizon_ret <= q(5)].mean())
        es99 = -float(horizon_ret[horizon_ret <= q(1)].mean())
        rows.append(
            {
                "name": name,
                "mean_ret": float(horizon_ret.mean()),
                "p05": q(5),
                "p50": q(50),
                "p95": q(95),
                "VaR95": var95,
                "VaR99": var99,
                "ES95": es95,
                "ES99": es99,
                "median_max_drawdown": float(np.median(drawdown)),
                "p95_max_drawdown": float(np.percentile(drawdown, 5)),
            }
        )
    return pd.DataFrame(rows).set_index("name")
