#!/usr/bin/env python3
"""Stylized-facts evaluation of a trained generator on the held-out test set.

    python scripts/evaluate.py --ckpt outputs/run/ckpt_final.pt --data data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gafs.config import load_config
from gafs.data.preprocess import ProcessedData
from gafs.data.windows import build_windows, split_windows, to_frame_returns
from gafs.evaluation.plots import (
    plot_acf_comparison,
    plot_correlation_heatmaps,
    plot_fan_chart,
    plot_return_distributions,
)
from gafs.evaluation.sampling import sample_paths_over_anchors
from gafs.evaluation.stylized_facts import evaluate_stylized_facts, write_markdown_report
from gafs.simulation.scenarios import generate_scenarios
from gafs.training.trainer import load_generator
from gafs.utils import resolve_device, set_seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "evaluation"))
    parser.add_argument("--n-anchors", type=int, default=100)
    parser.add_argument("--samples-per-anchor", type=int, default=8)
    parser.add_argument("--fan-scenarios", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = resolve_device(cfg.train.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    generator, meta = load_generator(args.ckpt, map_location=str(device))
    processed = ProcessedData.load(args.data)
    if processed.meta["stationarity"] != "log_returns":
        print("Note: fracdiff mode evaluates in feature space only.", file=sys.stderr)

    arrays = build_windows(
        processed, meta["lookback"], meta["horizon"], cfg.window.stride
    )
    splits = split_windows(arrays, cfg.window.val_frac, cfg.window.test_frac)
    test_idx = splits["test"]
    if len(test_idx) == 0:
        raise SystemExit("Test split is empty; retrain with more data.")

    sampled = sample_paths_over_anchors(
        generator,
        arrays,
        test_idx,
        n_anchors=args.n_anchors,
        samples_per_anchor=args.samples_per_anchor,
        device=device,
        meta=processed.meta,
        seed=args.seed,
    )
    if "fake_returns" not in sampled:
        raise SystemExit(
            "Evaluation in return space needs stationarity='log_returns'."
        )
    fake = sampled["fake_returns"]

    real = to_frame_returns(processed)
    first_test_t0 = arrays.t0[test_idx[0]]
    real_test = real[real.index >= first_test_t0]
    print(f"Real test rows: {len(real_test)}  generated windows: {fake.shape[0]}")

    results = evaluate_stylized_facts(real_test, fake, list(arrays.asset_cols))
    with open(out_dir / "stylized_facts.json", "w") as f:
        json.dump(results, f, indent=2)
    write_markdown_report(results, out_dir / "stylized_facts.md",
                          title="GAFS stylized-facts report (test split)")

    plot_return_distributions(real_test, fake, arrays.asset_cols, out_dir / "distributions.png")
    plot_acf_comparison(results, out_dir / "acf_abs.png")
    plot_correlation_heatmaps(results, out_dir / "correlations.png")

    anchor_pos = test_idx[len(test_idx) // 2]
    anchor_ts = arrays.t0[anchor_pos]
    scn = generate_scenarios(
        generator, processed, n_scenarios=args.fan_scenarios,
        device=device, anchor=str(anchor_ts), seed=args.seed,
    )
    close_cols = [f"{a}__close" for a in processed.asset_cols]
    future = processed.aux.loc[processed.aux.index >= anchor_ts, close_cols]
    real_tail = future.iloc[: meta["horizon"] + 1].to_numpy()
    plot_fan_chart(
        scn.prices, arrays.asset_cols, out_dir / "fan_chart.png",
        real_tail=real_tail if len(real_tail) > 1 else None,
        title=f"Generated fan vs realised path, anchor {str(anchor_ts)[:10]}",
    )

    print(f"Report -> {out_dir / 'stylized_facts.md'}")
    for name in ("distributions.png", "acf_abs.png", "correlations.png", "fan_chart.png"):
        print(f"Figure -> {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
