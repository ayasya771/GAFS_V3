#!/usr/bin/env python3
"""Run the preprocessing pipeline and persist a model-ready panel.

    python scripts/preprocess_data.py                       # bundled panel
    python scripts/preprocess_data.py --raw data/raw        # downloaded data
    python scripts/preprocess_data.py --panel data/market/holdout_panel.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from gafs.config import load_config
from gafs.data.preprocess import preprocess_market
from gafs.data.synthetic import generate_synthetic_market, macro_columns, price_columns


def load_panel_csv(path: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load a single combined CSV panel: date index, `*_close` prices, macro."""
    if not path.exists():
        raise FileNotFoundError(
            f"Panel not found: {path}. Regenerate the bundled panels with "
            "python scripts/make_reference_data.py"
        )
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    price_cols = [c for c in df.columns if c.endswith("_close")]
    if not price_cols:
        raise ValueError(f"{path} has no '*_close' price columns.")
    macro_cols = [c for c in df.columns if c not in price_cols]
    return df[price_cols], (df[macro_cols] if macro_cols else None)


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    prices_parts = []
    for name in ("equities.parquet", "crypto.parquet"):
        p = raw_dir / name
        if p.exists():
            df = pd.read_parquet(p)
            prices_parts.append(df[[c for c in df.columns if c.endswith("_close")]])
    if not prices_parts:
        raise FileNotFoundError(
            f"No equities.parquet or crypto.parquet under {raw_dir}. "
            "Run scripts/download_data.py first, or pass --synthetic."
        )
    prices = pd.concat(prices_parts, axis=1)

    macro = None
    macro_path = raw_dir / "macro_fred.parquet"
    if macro_path.exists():
        macro = pd.read_parquet(macro_path)
    return prices, macro


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--raw", default=None,
                        help="directory of downloaded parquet files")
    parser.add_argument("--panel", default=None,
                        help="single combined CSV panel (default: the bundled "
                             "data/market/reference_panel.csv)")
    parser.add_argument("--out", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--synthetic", action="store_true",
                        help="generate a fresh panel instead of reading a file")
    parser.add_argument("--days", type=int, default=6000, help="generated length")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.synthetic:
        panel = generate_synthetic_market(n_days=args.days, seed=args.seed)
        prices = panel[price_columns(panel)]
        macro = panel[macro_columns(panel)]
        print(f"Generated panel: {len(panel)} days, assets={price_columns(panel)}")
    elif args.raw:
        prices, macro = load_raw(Path(args.raw))
        print(f"Downloaded panel: prices {prices.shape}, macro "
              f"{None if macro is None else macro.shape}")
    else:
        panel_path = Path(args.panel or (ROOT / "data" / "market" / "reference_panel.csv"))
        prices, macro = load_panel_csv(panel_path)
        print(f"Panel {panel_path.name}: prices {prices.shape}, macro "
              f"{None if macro is None else macro.shape}")

    processed = preprocess_market(prices, macro, cfg.preprocess)
    out = processed.save(args.out)
    m = processed.meta
    print(f"Processed rows: {m['n_rows']}  features: {len(m['feature_cols'])} "
          f"({len(m['asset_cols'])} assets + {len(m['macro_cols'])} macro)")
    print(f"MAD bad ticks removed: {m['mad_flagged']}")
    print(f"Scaler train cutoff: {m['train_end_ts']}")
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
