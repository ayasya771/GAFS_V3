#!/usr/bin/env python3
"""Train the WGAN-GP + contrastive model on a processed panel.

    python scripts/train.py --data data/processed --out outputs/run1
    python scripts/train.py --data data/processed --steps 500   # quick pass
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gafs.config import load_config, save_config
from gafs.data.dataset import WindowDataset
from gafs.data.preprocess import ProcessedData
from gafs.data.windows import build_windows, split_windows
from gafs.evaluation.plots import plot_training_history
from gafs.training.trainer import GANTrainer, build_models
from gafs.utils import count_parameters, resolve_device, set_seed


def make_model_meta(cfg, arrays) -> dict:
    return {
        "n_features": arrays.x_hist.shape[2],
        "n_assets": arrays.y.shape[2],
        "cond_dim": arrays.cond.shape[1],
        "lookback": arrays.lookback,
        "horizon": arrays.horizon,
        "hidden": cfg.model.hidden,
        "heads": cfg.model.heads,
        "z_dim": cfg.model.z_dim,
        "dropout": cfg.model.dropout,
        "lstm_layers": cfg.model.lstm_layers,
        "critic_arch": cfg.model.critic_arch,
        "critic_channels": list(cfg.model.critic_channels),
        "critic_ctx_channels": cfg.model.critic_ctx_channels,
        "proj_dim": cfg.model.proj_dim,
        "asset_cols": list(arrays.asset_cols),
        "macro_cols": list(arrays.macro_cols),
        "feature_cols": list(arrays.feature_cols),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "run"))
    parser.add_argument("--steps", type=int, default=None,
                        help="generator updates to run (this session)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None,
                        help="checkpoint to continue training from")
    args = parser.parse_args(argv)

    overrides: dict = {"train": {}}
    if args.steps is not None:
        overrides["train"]["steps"] = args.steps
    if args.batch_size is not None:
        overrides["train"]["batch_size"] = args.batch_size
    if args.device is not None:
        overrides["train"]["device"] = args.device
    cfg = load_config(args.config, overrides)

    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    print(f"Device: {device}")

    processed = ProcessedData.load(args.data)
    arrays = build_windows(
        processed, cfg.window.lookback, cfg.window.horizon, cfg.window.stride
    )
    splits = split_windows(arrays, cfg.window.val_frac, cfg.window.test_frac)
    print(
        f"Windows: {len(arrays)} total  "
        f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
    )

    train_ds = WindowDataset(arrays, splits["train"])
    if args.resume:
        import torch

        meta = torch.load(args.resume, map_location="cpu", weights_only=False)["model_meta"]
    else:
        meta = make_model_meta(cfg, arrays)
    generator, critic = build_models(meta)
    print(f"Generator params: {count_parameters(generator):,}  "
          f"Critic params: {count_parameters(critic):,}")

    trainer = GANTrainer(generator, critic, cfg.train, meta, device, out_dir=args.out)
    if args.resume:
        step = trainer.load_state(args.resume)
        trainer.logger.path = Path(args.out) / f"history_from_{step}.csv"
        print(f"Resumed from {args.resume} at step {step}")
    save_config(cfg, Path(args.out) / "config_used.yaml")
    history = trainer.fit(train_ds)
    if history:
        plot_training_history(history, Path(args.out) / "training_history.png")
    print(f"Final checkpoint: {Path(args.out) / 'ckpt_final.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
