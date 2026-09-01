#!/usr/bin/env python3
"""Build the browser demo bundle in docs/ from a trained checkpoint.

    python scripts/build_site.py --ckpt outputs/demo/run/ckpt_final.pt

Writes:

    docs/model/generator.bin    float32 weights, tensors concatenated
    docs/model/generator.json   manifest: config, tensor offsets, scaler stats
    docs/model/parity.json      a reference input/noise/output triple so the
                                browser can verify its own arithmetic against
                                this PyTorch build at load time
    docs/data/*.csv             the market panels the page fetches

The docs/ tree is what GitHub Pages publishes; everything in it is a build
artifact of this script plus the checked-in page sources.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gafs.config import load_config
from gafs.data.preprocess import ProcessedData
from gafs.training.trainer import load_generator
from gafs.utils import count_parameters, set_seed


def export_weights(generator, out_dir: Path) -> tuple[dict, int]:
    """Flatten the state dict into one float32 blob plus an offset table."""
    tensors, offset, blob = {}, 0, []
    for name, tensor in generator.state_dict().items():
        arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        flat = np.ascontiguousarray(arr).reshape(-1)
        tensors[name] = {"shape": list(arr.shape), "offset": offset, "size": int(flat.size)}
        blob.append(flat)
        offset += int(flat.size)
    payload = np.concatenate(blob) if blob else np.zeros(0, dtype=np.float32)
    (out_dir / "generator.bin").write_bytes(payload.tobytes())
    return tensors, offset


def parity_sample(generator, meta: dict) -> dict:
    """A fixed (x_hist, cond, z) -> y triple for the browser to reproduce."""
    set_seed(1234)
    k, f = meta["lookback"], meta["n_features"]
    x = torch.randn(1, k, f)
    cond = torch.randn(1, meta["cond_dim"]) if meta["cond_dim"] else None
    z = torch.randn(1, meta["horizon"], meta["z_dim"])
    with torch.no_grad():
        y = generator(x, cond, z)
    return {
        "x_hist": x.reshape(-1).tolist(),
        "cond": (cond.reshape(-1).tolist() if cond is not None else []),
        "z": z.reshape(-1).tolist(),
        "y": y.reshape(-1).tolist(),
        "shapes": {
            "x_hist": [k, f],
            "cond": [meta["cond_dim"]],
            "z": [meta["horizon"], meta["z_dim"]],
            "y": [meta["horizon"], meta["n_assets"]],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=str(ROOT / "outputs" / "demo" / "run" / "ckpt_final.pt"))
    parser.add_argument("--processed", default=str(ROOT / "outputs" / "demo" / "processed"),
                        help="processed panel whose scaler statistics are exported")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--docs", default=str(ROOT / "docs"))
    parser.add_argument("--panels", default=str(ROOT / "data" / "market"))
    args = parser.parse_args(argv)

    docs = Path(args.docs)
    model_dir = docs / "model"
    data_dir = docs / "data"
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    generator, meta = load_generator(args.ckpt, map_location="cpu")
    generator.eval()
    processed = ProcessedData.load(args.processed)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    tensors, n_params = export_weights(generator, model_dir)

    manifest = {
        "format": 1,
        "weights": "generator.bin",
        "dtype": "float32",
        "param_count": int(n_params),
        "trainable_params": int(count_parameters(generator)),
        "trained_steps": int(ckpt.get("step", 0)),
        "arch": {
            "n_features": meta["n_features"],
            "n_assets": meta["n_assets"],
            "cond_dim": meta["cond_dim"],
            "lookback": meta["lookback"],
            "horizon": meta["horizon"],
            "hidden": meta["hidden"],
            "heads": meta["heads"],
            "z_dim": meta["z_dim"],
            "lstm_layers": meta["lstm_layers"],
        },
        "columns": {
            "features": list(meta["feature_cols"]),
            "assets": list(meta["asset_cols"]),
            "macro": list(meta["macro_cols"]),
        },
        "preprocess": {
            "stationarity": processed.meta["stationarity"],
            "scaling": processed.meta["scaling"],
            "vol_window": processed.meta["vol_window"],
            "eps": processed.meta["eps"],
            "ffill_limit": processed.meta["preprocess_config"]["ffill_limit"],
            "mad_window": processed.meta["preprocess_config"]["mad_window"],
            "mad_threshold": processed.meta["preprocess_config"]["mad_threshold"],
            "mad_reversal_tol": processed.meta["preprocess_config"]["mad_reversal_tol"],
            "train_frac": processed.meta["preprocess_config"]["train_frac"],
            "fracdiff_d": processed.meta["preprocess_config"]["fracdiff_d"],
            "fracdiff_threshold": processed.meta["preprocess_config"]["fracdiff_threshold"],
            "macro_stats": processed.meta["macro_stats"],
            "asset_stats": processed.meta["asset_stats"],
        },
        "window": {
            "lookback": cfg.window.lookback,
            "horizon": cfg.window.horizon,
            "val_frac": cfg.window.val_frac,
            "test_frac": cfg.window.test_frac,
        },
        "training": {
            "n_critic": cfg.train.n_critic,
            "lambda_gp": cfg.train.lambda_gp,
            "tau": cfg.train.tau,
            "batch_size": cfg.train.batch_size,
            "lr_g": cfg.train.lr_g,
            "lr_d": cfg.train.lr_d,
            "w_contrastive": cfg.train.w_contrastive,
            "w_uniform": cfg.train.w_uniform,
            "w_mmd": cfg.train.w_mmd,
            "ema_decay": cfg.train.ema_decay,
        },
        "tensors": tensors,
    }
    (model_dir / "generator.json").write_text(json.dumps(manifest, indent=1))
    (model_dir / "parity.json").write_text(json.dumps(parity_sample(generator, meta)))

    panels = sorted(Path(args.panels).glob("*.csv"))
    if not panels:
        raise SystemExit(
            f"No panels in {args.panels}; run scripts/make_reference_data.py first."
        )
    index = []
    for p in panels:
        shutil.copy2(p, data_dir / p.name)
        head = p.read_text().splitlines()
        cols = head[0].split(",")[1:]
        index.append({
            "file": p.name,
            "label": p.stem.replace("_", " ").title(),
            "rows": len(head) - 1,
            "start": head[1].split(",")[0],
            "end": head[-1].split(",")[0],
            "assets": [c for c in cols if c.endswith("_close")],
            "macro": [c for c in cols if not c.endswith("_close")],
            "trained_on": p.name == "reference_panel.csv",
        })
    (data_dir / "index.json").write_text(json.dumps(index, indent=1))

    size_kb = (model_dir / "generator.bin").stat().st_size / 1024
    print(f"weights   {n_params:,} float32 ({size_kb:.0f} KB) -> {model_dir/'generator.bin'}")
    print(f"manifest  {len(tensors)} tensors -> {model_dir/'generator.json'}")
    print(f"parity    reference triple -> {model_dir/'parity.json'}")
    for entry in index:
        print(f"data      {entry['file']}: {entry['rows']} rows "
              f"{entry['start']}..{entry['end']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
