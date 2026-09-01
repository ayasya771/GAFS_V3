"""Tiny end-to-end training + scenario smoke tests (CPU, under a minute)."""

import numpy as np
import pytest
import torch

from gafs.config import PreprocessConfig, TrainConfig
from gafs.data.dataset import WindowDataset
from gafs.data.preprocess import preprocess_market
from gafs.data.synthetic import generate_synthetic_market, macro_columns, price_columns
from gafs.data.windows import build_windows, split_windows
from gafs.evaluation.sampling import sample_paths_over_anchors
from gafs.evaluation.stylized_facts import evaluate_stylized_facts
from gafs.simulation.scenarios import (
    apply_macro_shock,
    generate_scenarios,
    scenario_summary,
)
from gafs.training.trainer import GANTrainer, build_models, load_generator
from gafs.utils import set_seed


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    set_seed(0)
    panel = generate_synthetic_market(n_days=700, seed=21)
    processed = preprocess_market(
        panel[price_columns(panel)], panel[macro_columns(panel)],
        PreprocessConfig(vol_window=20),
    )
    arrays = build_windows(processed, lookback=30, horizon=10)
    splits = split_windows(arrays, 0.15, 0.15)
    meta = {
        "n_features": arrays.x_hist.shape[2],
        "n_assets": arrays.y.shape[2],
        "cond_dim": arrays.cond.shape[1],
        "lookback": arrays.lookback,
        "horizon": arrays.horizon,
        "hidden": 16,
        "heads": 2,
        "z_dim": 4,
        "dropout": 0.0,
        "lstm_layers": 1,
        "critic_arch": "resnet",
        "critic_channels": [8, 16],
        "critic_ctx_channels": 4,
        "proj_dim": 8,
        "asset_cols": list(arrays.asset_cols),
        "macro_cols": list(arrays.macro_cols),
        "feature_cols": list(arrays.feature_cols),
    }
    cfg = TrainConfig(
        batch_size=16, steps=6, n_critic=2, log_every=2, ckpt_every=0,
        w_contrastive=1.0, w_uniform=0.5, w_mmd=1.0, seed=0, device="cpu",
    )
    out_dir = tmp_path_factory.mktemp("run")
    generator, critic = build_models(meta)
    trainer = GANTrainer(generator, critic, cfg, meta, torch.device("cpu"), out_dir)
    train_ds = WindowDataset(arrays, splits["train"])
    history = trainer.fit(train_ds, verbose=False)
    return processed, arrays, splits, trainer, history, out_dir


def test_training_runs_and_logs(setup):
    _, _, _, trainer, history, out_dir = setup
    assert trainer.step == 6
    assert len(history) >= 2
    for row in history:
        for key in ("d_adv", "d_gp", "d_con", "g_adv", "g_unif", "g_mmd"):
            assert np.isfinite(row[key])
    assert (out_dir / "history.csv").exists()
    assert (out_dir / "ckpt_final.pt").exists()


def test_checkpoint_roundtrip_and_ema(setup):
    *_, out_dir = setup
    generator, meta = load_generator(out_dir / "ckpt_final.pt")
    assert generator.lookback == 30 and generator.horizon == 10
    assert meta["asset_cols"]
    x = torch.randn(2, 30, meta["n_features"])
    c = torch.randn(2, meta["cond_dim"])
    y = generator(x, c)
    assert y.shape == (2, 10, meta["n_assets"])


def test_scenarios_and_stress_shift_distribution(setup):
    processed, *_ , out_dir = setup
    generator, _ = load_generator(out_dir / "ckpt_final.pt")
    base = generate_scenarios(generator, processed, n_scenarios=64,
                              device="cpu", seed=1)
    assert base.returns.shape == (64, 10, 3)
    assert base.prices.shape == (64, 11, 3)
    assert np.isfinite(base.prices).all() and (base.prices > 0).all()
    assert np.allclose(base.prices[:, 0, :], base.prices[0, 0, :])

    stressed = generate_scenarios(
        generator, processed, n_scenarios=64, device="cpu", seed=1,
        macro_shock={"VIX_PROXY": 20.0},
    )
    assert not np.allclose(base.returns, stressed.returns)

    summary = scenario_summary(base)
    assert "PORTFOLIO" in summary.index
    assert (summary["ES95"] >= summary["VaR95"] - 1e-9).all()


def test_apply_macro_shock_units(setup):
    processed, *_ = setup
    cond = np.zeros((1, len(processed.macro_cols)), dtype=np.float32)
    shocked = apply_macro_shock(cond, processed.meta, {"VIX_PROXY": 10.0})
    j = processed.macro_cols.index("VIX_PROXY")
    expected = 10.0 / processed.meta["macro_stats"]["VIX_PROXY"]["std"]
    assert shocked[0, j] == pytest.approx(expected, rel=1e-5)
    with pytest.raises(KeyError):
        apply_macro_shock(cond, processed.meta, {"NOPE": 1.0})


def test_sampling_and_stylized_facts_pipeline(setup):
    processed, arrays, splits, trainer, *_ = setup
    sampled = sample_paths_over_anchors(
        trainer.g_ema, arrays, splits["test"], n_anchors=10,
        samples_per_anchor=3, device="cpu", meta=processed.meta, seed=0,
    )
    fake = sampled["fake_returns"]
    assert fake.shape[1:] == (10, 3)
    from gafs.data.windows import to_frame_returns

    real = to_frame_returns(processed)
    results = evaluate_stylized_facts(real, fake, list(arrays.asset_cols), max_lag=5)
    for asset in results["assets"].values():
        assert np.isfinite(asset["real"]["excess_kurtosis"])
        assert np.isfinite(asset["fake"]["ann_vol"])
        assert asset["wasserstein"] >= 0.0
    assert len(results["correlation"]["real"]) == 3
