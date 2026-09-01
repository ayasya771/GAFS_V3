#!/usr/bin/env python3
"""End-to-end demo on the calibrated synthetic market (no network needed).

    python scripts/quickstart_demo.py                    # ~10-15 min on CPU
    python scripts/quickstart_demo.py --steps 300 --days 3000   # faster smoke

Pipeline: synthetic data -> preprocessing -> WGAN-GP + contrastive training
-> stylized-facts evaluation -> baseline and VIX-shock stress scenarios.
All outputs land under --out (default outputs/demo).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--days", type=int, default=5000)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "demo"))
    parser.add_argument("--scenarios", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    out = Path(args.out)

    from scripts.preprocess_data import main as preprocess_main
    from scripts.train import main as train_main
    from scripts.evaluate import main as evaluate_main
    from scripts.generate import main as generate_main

    processed_dir = out / "processed"
    run_dir = out / "run"
    print("=== 1/4 preprocess (synthetic market) ===", flush=True)
    rc = preprocess_main(["--synthetic", "--days", str(args.days), "--out", str(processed_dir)])
    if rc:
        return rc

    print("=== 2/4 train ===", flush=True)
    train_args = [
        "--data", str(processed_dir), "--out", str(run_dir),
        "--steps", str(args.steps), "--batch-size", str(args.batch_size),
    ]
    rc = train_main(train_args)
    if rc:
        return rc
    ckpt = run_dir / "ckpt_final.pt"

    print("=== 3/4 evaluate stylized facts ===", flush=True)
    rc = evaluate_main([
        "--ckpt", str(ckpt), "--data", str(processed_dir),
        "--out", str(out / "evaluation"),
        "--n-anchors", "60", "--samples-per-anchor", "8",
    ])
    if rc:
        return rc

    print("=== 4/4 scenarios: baseline + VIX stress ===", flush=True)
    rc = generate_main([
        "--ckpt", str(ckpt), "--data", str(processed_dir),
        "--out", str(out / "scenarios_baseline"), "--n", str(args.scenarios),
    ])
    if rc:
        return rc
    rc = generate_main([
        "--ckpt", str(ckpt), "--data", str(processed_dir),
        "--out", str(out / "scenarios_stressed"), "--n", str(args.scenarios),
        "--shock", "VIX_PROXY=+20", "--shock", "CREDIT_SPREAD=+1.5",
    ])
    if rc:
        return rc

    print("\nDemo complete. Key outputs:")
    for p in (
        out / "evaluation" / "stylized_facts.md",
        out / "evaluation" / "distributions.png",
        out / "evaluation" / "acf_abs.png",
        out / "evaluation" / "correlations.png",
        out / "evaluation" / "fan_chart.png",
        out / "scenarios_baseline" / "summary.csv",
        out / "scenarios_stressed" / "summary.csv",
        run_dir / "training_history.png",
    ):
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
