#!/usr/bin/env python3
"""Download the free-tier market datasets into data/raw.

    python scripts/download_data.py --sources yahoo,fred
    python scripts/download_data.py --sources yahoo,fred,binance --binance-interval 1d

Each source fails independently with a clear message, so partial connectivity
still yields usable panels. Alpaca and Dukascopy are opt-in (keys / external
CLI); see gafs/data/sources for instructions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gafs.config import load_config
from gafs.data.sources import (
    DataSourceError,
    fetch_binance_klines,
    fetch_fred,
    fetch_yahoo,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--sources", default="yahoo,fred",
                        help="comma list from: yahoo, fred, binance")
    parser.add_argument("--out", default=None, help="override raw data directory")
    parser.add_argument("--binance-interval", default="1d")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = Path(args.out or (ROOT / cfg.data.raw_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    successes = 0

    if "yahoo" in wanted:
        try:
            eq = fetch_yahoo(cfg.data.tickers, cfg.data.start, cfg.data.end, cfg.data.interval)
            eq.to_parquet(out_dir / "equities.parquet")
            print(f"[yahoo]   {eq.shape[0]} rows x {eq.shape[1]} cols -> equities.parquet")
            successes += 1
        except DataSourceError as exc:
            print(f"[yahoo]   FAILED: {exc}", file=sys.stderr)

    if "fred" in wanted:
        try:
            macro = fetch_fred(cfg.data.fred_series, cfg.data.start, cfg.data.end)
            macro.to_parquet(out_dir / "macro_fred.parquet")
            print(f"[fred]    {macro.shape[0]} rows x {macro.shape[1]} cols -> macro_fred.parquet")
            successes += 1
        except DataSourceError as exc:
            print(f"[fred]    FAILED: {exc}", file=sys.stderr)

    if "binance" in wanted:
        try:
            crypto = fetch_binance_klines(
                cfg.data.binance_symbols, cfg.data.start, cfg.data.end,
                interval=args.binance_interval,
            )
            crypto.to_parquet(out_dir / "crypto.parquet")
            print(f"[binance] {crypto.shape[0]} rows x {crypto.shape[1]} cols -> crypto.parquet")
            successes += 1
        except DataSourceError as exc:
            print(f"[binance] FAILED: {exc}", file=sys.stderr)

    if successes == 0:
        print(
            "No source succeeded. If this machine has no market-data egress, "
            "run this script from a network with access, or use "
            "scripts/preprocess_data.py --synthetic to proceed offline.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
