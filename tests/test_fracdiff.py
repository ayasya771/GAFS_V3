import numpy as np
import pandas as pd
import pytest

from gafs.data.fracdiff import ffd_weights, frac_diff_ffd


def test_weights_d1_is_first_difference():
    w = ffd_weights(1.0, threshold=1e-8)
    assert np.allclose(w, [-1.0, 1.0])


def test_weights_d0_is_identity():
    w = ffd_weights(0.0, threshold=1e-8)
    assert np.allclose(w, [1.0])


def test_weights_decay_and_signs():
    w = ffd_weights(0.4, threshold=1e-5)[::-1]
    assert w[0] == 1.0
    assert w[1] == pytest.approx(-0.4)
    assert np.all(np.abs(w[1:]) < 1.0)
    assert np.all(np.diff(np.abs(w[1:])) <= 1e-12)


def test_ffd_reduces_random_walk_nonstationarity():
    rng = np.random.default_rng(0)
    walk = pd.Series(np.cumsum(rng.standard_normal(4000)))
    ffd = frac_diff_ffd(walk, d=0.5, threshold=1e-4).dropna()
    first, second = ffd.iloc[: len(ffd) // 2], ffd.iloc[len(ffd) // 2 :]
    ratio = second.var() / first.var()
    assert 0.5 < ratio < 2.0
    assert ffd.autocorr(1) > 0.2


def test_ffd_matches_manual_dot_product():
    s = pd.Series(np.arange(200, dtype=float) ** 1.3)
    d, thr = 0.35, 1e-2
    w = ffd_weights(d, thr)
    out = frac_diff_ffd(s, d, thr)
    width = len(w)
    i = width + 5
    manual = float(np.dot(w, s.to_numpy()[i - width + 1 : i + 1]))
    assert out.iloc[i] == pytest.approx(manual)
    assert out.iloc[: width - 1].isna().all()
    assert out.iloc[width - 1 :].notna().all()
