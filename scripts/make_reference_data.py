#!/usr/bin/env python3
"""Write the bundled reference market panels to data/market/ as CSV.

    python scripts/make_reference_data.py

Two panels are produced from the same data-generating process with different
random seeds:

  reference_panel.csv   the panel the shipped model was fitted on
  holdout_panel.csv     an independent realisation, unseen during training

Both carry three tradable series (`*_close`) and three macro conditioning
series, on a business-day UTC index. Regenerating with the same seeds
reproduces them byte for byte.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gafs.data.synthetic import generate_synthetic_market

PANELS = [
    ("reference_panel.csv", 5000, 7, "2001-01-01"),
    ("holdout_panel.csv", 2600, 4242, "2011-01-03"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "data" / "market"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, days, seed, start in PANELS:
        panel = generate_synthetic_market(n_days=days, seed=seed, start=start)
        panel.index.name = "date"
        panel = panel.round(8)
        path = out_dir / name
        panel.to_csv(path, date_format="%Y-%m-%d")
        print(f"{name}: {len(panel)} rows x {panel.shape[1]} cols -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
