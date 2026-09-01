import numpy as np
import pytest

from gafs.config import PreprocessConfig
from gafs.data.preprocess import preprocess_market
from gafs.data.synthetic import generate_synthetic_market, macro_columns, price_columns
from gafs.data.windows import build_windows, chronological_split, split_windows


@pytest.fixture(scope="module")
def processed():
    panel = generate_synthetic_market(n_days=800, seed=11)
    return preprocess_market(
        panel[price_columns(panel)], panel[macro_columns(panel)], PreprocessConfig()
    )


def test_window_shapes_and_alignment(processed):
    k, h = 60, 20
    arrays = build_windows(processed, lookback=k, horizon=h, stride=1)
    T = len(processed.features)
    assert arrays.x_hist.shape == (T - k - h + 1, k, len(arrays.feature_cols))
    assert arrays.y.shape == (T - k - h + 1, h, len(arrays.asset_cols))
    assert arrays.cond.shape == (T - k - h + 1, len(arrays.macro_cols))
    assert np.isfinite(arrays.x_hist).all() and np.isfinite(arrays.y).all()

    F = processed.features.to_numpy(dtype=np.float32)
    asset_pos = [arrays.feature_cols.index(a) for a in arrays.asset_cols]
    assert np.allclose(arrays.x_hist[0], F[:k])
    assert np.allclose(arrays.y[0], F[k : k + h][:, asset_pos])
    macro_pos = [arrays.feature_cols.index(m) for m in arrays.macro_cols]
    assert np.allclose(arrays.cond[0], F[k - 1, macro_pos])
    assert np.allclose(arrays.x_hist[0, -1, macro_pos], arrays.cond[0])


def test_chronological_split_no_overlap_with_purge():
    train, val, test = chronological_split(1000, 0.15, 0.15, purge=50)
    assert train.max() < val.min() - 49
    assert val.max() < test.min() - 49
    assert len(np.intersect1d(train, val)) == 0
    assert len(np.intersect1d(val, test)) == 0
    assert test.max() == 999


def test_split_windows_timestamps_disjoint(processed):
    arrays = build_windows(processed, lookback=40, horizon=10)
    splits = split_windows(arrays, 0.15, 0.15)
    span = arrays.lookback + arrays.horizon
    last_train_row = splits["train"].max() + span - 1
    first_val_row = splits["val"].min()
    assert last_train_row < first_val_row


def test_split_raises_when_too_small():
    with pytest.raises(ValueError):
        chronological_split(60, 0.15, 0.15, purge=40)
