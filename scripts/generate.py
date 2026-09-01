#!/usr/bin/env python3
"""Generate stress-testable scenarios from a trained model.

    python scripts/generate.py --ckpt outputs/run/ckpt_final.pt --n 1000
    python scripts/generate.py --ckpt outputs/run/ckpt_final.pt \
        --shock VIX_PROXY=+20 --shock CREDIT_SPREAD=+1.5 --n 2000

Shocks are raw-unit shifts of the macro conditioning vector at the anchor
(for example VIX up 20 points), mapped into model space via stored
train-set statistics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from gafs.config import load_config
from gafs.data.preprocess import ProcessedData
from gafs.evaluation.plots import plot_fan_chart
from gafs.simulation.scenarios import generate_scenarios, scenario_summary
from gafs.training.trainer import load_generator
from gafs.utils import resolve_device, set_seed


def parse_shocks(items: list[str]) -> dict[str, float] | None:
    if not items:
        return None
    shocks: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Bad --shock {item!r}; expected NAME=+DELTA")
        name, value = item.split("=", 1)
        shocks[name.strip()] = float(value)
    return shocks


def parse_weights(text: str | None, n_assets: int) -> np.ndarray | None:
    if not text:
        return None
    w = np.array([float(x) for x in text.split(",")], dtype=float)
    if len(w) != n_assets:
        raise SystemExit(f"--weights needs {n_assets} entries")
    return w / w.sum()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "scenarios"))
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--anchor", default=None, help="forecast origin date (default: last)")
    parser.add_argument("--shock", action="append", default=[],
                        help="macro shock NAME=+DELTA in raw units; repeatable")
    parser.add_argument("--weights", default=None, help="portfolio weights, comma list")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = resolve_device(cfg.train.device)

    generator, meta = load_generator(args.ckpt, map_location=str(device))
    processed = ProcessedData.load(args.data)
    shocks = parse_shocks(args.shock)

    scn = generate_scenarios(
        generator,
        processed,
        n_scenarios=args.n,
        device=device,
        macro_shock=shocks,
        anchor=args.anchor,
        seed=args.seed,
    )
    out_dir = Path(args.out)
    scn.save(out_dir)

    weights = parse_weights(args.weights, len(scn.asset_names))
    summary = scenario_summary(scn, weights)
    summary.to_csv(out_dir / "summary.csv")

    label = "stressed" if shocks else "baseline"
    plot_fan_chart(
        scn.prices, scn.asset_names, out_dir / f"fan_{label}.png",
        title=f"Scenario fan ({label}), anchor {scn.anchor_ts[:10]}"
              + (f", shocks {shocks}" if shocks else ""),
    )

    print(f"Anchor: {scn.anchor_ts}  scenarios: {scn.n_scenarios}  horizon: {scn.horizon}")
    if scn.macro_raw_at_anchor:
        print("Macro at anchor (raw): "
              + ", ".join(f"{k}={v:.2f}" for k, v in scn.macro_raw_at_anchor.items()))
    if shocks:
        print(f"Applied shocks: {shocks}")
    print(summary.round(4).to_string())
    print(f"Saved -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
