"""Typed configuration for the whole pipeline, loadable from YAML.

Every tunable parameter lives here with its default:
ffill limit 3, MAD outlier filtering, fractional differencing d in (0, 1),
30-day volatility scaling, lookback k = 90, horizon h = 30, WGAN-GP
lambda = 10, n_critic = 5, NT-Xent temperature, and the SimCLR integration
weights for the generator.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class DataConfig:
    tickers: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "GLD", "TLT"])
    fred_series: list[str] = field(default_factory=lambda: ["DGS10", "VIXCLS", "BAMLC0A0CM"])
    binance_symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    start: str = "2000-01-01"
    end: str = "2025-12-31"
    interval: str = "1d"
    raw_dir: str = "data/raw"


@dataclass
class PreprocessConfig:
    ffill_limit: int = 3
    mad_window: int = 51
    mad_threshold: float = 10.0
    mad_reversal_tol: float = 0.40
    stationarity: str = "log_returns"
    fracdiff_d: float = 0.4
    fracdiff_threshold: float = 1e-4
    scaling: str = "vol"
    vol_window: int = 30
    train_frac: float = 0.70
    eps: float = 1e-8


@dataclass
class WindowConfig:
    lookback: int = 90
    horizon: int = 30
    stride: int = 1
    val_frac: float = 0.15
    test_frac: float = 0.15


@dataclass
class ModelConfig:
    hidden: int = 64
    heads: int = 4
    z_dim: int = 16
    dropout: float = 0.10
    lstm_layers: int = 1
    critic_arch: str = "resnet"
    critic_channels: list[int] = field(default_factory=lambda: [64, 128, 128])
    critic_ctx_channels: int = 16
    proj_dim: int = 32


@dataclass
class TrainConfig:
    batch_size: int = 64
    steps: int = 4000
    n_critic: int = 5
    lambda_gp: float = 10.0
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    beta1: float = 0.0
    beta2: float = 0.9
    tau: float = 0.2
    w_contrastive: float = 1.0
    w_uniform: float = 0.5
    w_mmd: float = 1.0
    ema_decay: float = 0.999
    grad_clip: float = 0.0
    log_every: int = 50
    ckpt_every: int = 1000
    seed: int = 7
    device: str = "auto"
    num_workers: int = 0


@dataclass
class PathsConfig:
    out_dir: str = "outputs"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _build(cls, payload: Mapping[str, Any]):
    """Instantiate a dataclass from a mapping, ignoring unknown keys."""
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(payload) - names
    if unknown:
        raise KeyError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**{k: v for k, v in payload.items() if k in names})


def load_config(path: str | Path | None = None, overrides: Mapping[str, Mapping[str, Any]] | None = None) -> Config:
    """Load YAML config; missing sections and keys fall back to defaults.

    `overrides` is a nested mapping like {'train': {'steps': 200}} applied on
    top of the file values.
    """
    payload: dict[str, Any] = {}
    if path is not None:
        with open(path) as f:
            payload = yaml.safe_load(f) or {}
    if overrides:
        for section, values in overrides.items():
            payload.setdefault(section, {})
            payload[section] = {**payload[section], **dict(values)}

    sections = {
        "data": DataConfig,
        "preprocess": PreprocessConfig,
        "window": WindowConfig,
        "model": ModelConfig,
        "train": TrainConfig,
        "paths": PathsConfig,
    }
    kwargs = {}
    for name, cls in sections.items():
        kwargs[name] = _build(cls, payload.get(name, {}) or {})
    cfg = Config(**kwargs)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    p, w, t = cfg.preprocess, cfg.window, cfg.train
    if p.stationarity not in ("log_returns", "fracdiff"):
        raise ValueError("preprocess.stationarity must be 'log_returns' or 'fracdiff'")
    if p.stationarity == "fracdiff" and not (0.0 < p.fracdiff_d < 1.0):
        raise ValueError("preprocess.fracdiff_d must lie in (0, 1)")
    if p.scaling not in ("vol", "zscore"):
        raise ValueError("preprocess.scaling must be 'vol' or 'zscore'")
    if p.ffill_limit < 0:
        raise ValueError("preprocess.ffill_limit must be >= 0")
    if w.lookback < 2 or w.horizon < 1:
        raise ValueError("window.lookback must be >= 2 and window.horizon >= 1")
    if not (0 < w.val_frac < 1 and 0 < w.test_frac < 1 and w.val_frac + w.test_frac < 1):
        raise ValueError("window val_frac/test_frac must be in (0, 1) and sum below 1")
    if t.n_critic < 1:
        raise ValueError("train.n_critic must be >= 1")
    if cfg.model.hidden % cfg.model.heads != 0:
        raise ValueError("model.hidden must be divisible by model.heads")


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
