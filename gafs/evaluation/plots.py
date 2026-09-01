"""Diagnostic figures for real-vs-generated comparison.

Colors follow a validated colorblind-safe palette: real data is blue
(#2a78d6), generated data is orange (#eb6834) - a validated adjacent pair;
fan-chart bands use a single-hue sequential blue ramp; the correlation
heatmap uses a blue-red diverging map with a neutral midpoint. Chrome stays
recessive (hairline grid, muted labels) so the data carries the figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REAL = "#2a78d6"
FAKE = "#eb6834"
SEQ = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#104281"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"

DIVERGING = LinearSegmentedColormap.from_list(
    "gafs_div", ["#104281", "#3987e5", "#f0efec", "#ec835a", "#c22f2e"]
)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


def _new_fig(n_axes: int, width: float = 4.2, height: float = 3.2):
    fig, axes = plt.subplots(
        1, n_axes, figsize=(width * n_axes, height), facecolor=SURFACE, squeeze=False
    )
    return fig, axes[0]


def plot_return_distributions(
    real_returns, fake_returns: np.ndarray, asset_names: list[str], out_path: str | Path
) -> Path:
    """Log-scale step histograms: fat tails show up as slow lateral decay."""
    fig, axes = _new_fig(len(asset_names))
    for i, (ax, name) in enumerate(zip(axes, asset_names)):
        r = real_returns[name].dropna().to_numpy()
        f = fake_returns[:, :, i].ravel()
        lo, hi = np.quantile(np.concatenate([r, f]), [0.001, 0.999])
        bins = np.linspace(lo, hi, 60)
        ax.hist(r, bins=bins, density=True, histtype="step", lw=2.0, color=REAL, label="Real")
        ax.hist(f, bins=bins, density=True, histtype="step", lw=2.0, color=FAKE, label="Generated")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("log return")
        if i == 0:
            ax.set_ylabel("density (log scale)")
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        _style(ax)
    fig.suptitle("Return distributions, log density", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out_path)


def plot_acf_comparison(results: dict, out_path: str | Path) -> Path:
    """ACF of absolute returns (volatility clustering) per asset."""
    assets = list(results["assets"].keys())
    fig, axes = _new_fig(len(assets))
    for i, (ax, name) in enumerate(zip(axes, assets)):
        res = results["assets"][name]
        lags = np.arange(1, len(res["real"]["acf_abs"]) + 1)
        ax.plot(lags, res["real"]["acf_abs"], color=REAL, lw=2, marker="o", ms=4, label="Real")
        ax.plot(lags, res["fake"]["acf_abs"], color=FAKE, lw=2, marker="o", ms=4, label="Generated")
        ax.axhline(0.0, color=BASE, lw=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("lag (days)")
        if i == 0:
            ax.set_ylabel("ACF of |returns|")
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        _style(ax)
    fig.suptitle("Volatility clustering: ACF of absolute returns", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out_path)


def plot_fan_chart(
    price_paths: np.ndarray,
    asset_names: list[str],
    out_path: str | Path,
    real_tail: np.ndarray | None = None,
    title: str = "Generated scenario fan",
) -> Path:
    """Percentile fan of generated price paths [N, h+1, A], indexed to 100."""
    n, steps, A = price_paths.shape
    x = np.arange(steps)
    fig, axes = _new_fig(len(asset_names))
    bands = [(5, 95), (10, 90), (25, 75)]
    for i, (ax, name) in enumerate(zip(axes, asset_names)):
        paths = price_paths[:, :, i] / price_paths[:, :1, i] * 100.0
        for j, (lo, hi) in enumerate(bands):
            ax.fill_between(
                x,
                np.percentile(paths, lo, axis=0),
                np.percentile(paths, hi, axis=0),
                color=SEQ[j],
                lw=0,
                label=f"{lo}-{hi}%" if i == 0 else None,
            )
        ax.plot(x, np.median(paths, axis=0), color=SEQ[4], lw=2, label="Median" if i == 0 else None)
        if real_tail is not None:
            rt = real_tail[:, i] / real_tail[0, i] * 100.0
            ax.plot(np.arange(len(rt)), rt, color=INK, lw=1.4, ls="--",
                    label="Realised" if i == 0 else None)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("days ahead")
        if i == 0:
            ax.set_ylabel("price (origin = 100)")
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        _style(ax)
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out_path)


def plot_correlation_heatmaps(results: dict, out_path: str | Path) -> Path:
    assets = list(results["assets"].keys())
    real = np.array(results["correlation"]["real"])
    fake = np.array(results["correlation"]["fake"])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), facecolor=SURFACE)
    for ax, mat, label in ((axes[0], real, "Real"), (axes[1], fake, "Generated")):
        im = ax.imshow(mat, cmap=DIVERGING, vmin=-1, vmax=1)
        ax.set_xticks(range(len(assets)), assets, rotation=45, ha="right", fontsize=8, color=MUTED)
        ax.set_yticks(range(len(assets)), assets, fontsize=8, color=MUTED)
        ax.set_title(f"{label} correlation", fontsize=10, color=INK)
        for a in range(len(assets)):
            for b in range(len(assets)):
                ax.text(b, a, f"{mat[a, b]:.2f}", ha="center", va="center",
                        fontsize=8, color=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_visible(False)
    fig.suptitle("Cross-asset correlation structure", color=INK, fontsize=11)
    return _save(fig, out_path)


def plot_training_history(history: list[dict], out_path: str | Path) -> Path:
    steps = [h["step"] for h in history]
    fig, axes = _new_fig(2, width=4.6)
    axes[0].plot(steps, [h["wasserstein"] for h in history], color=REAL, lw=2)
    axes[0].set_title("Critic Wasserstein estimate", fontsize=10)
    axes[0].set_xlabel("generator step")
    axes[1].plot(steps, [h["d_gp"] for h in history], color=FAKE, lw=2, label="Gradient penalty")
    axes[1].plot(steps, [h["d_con"] for h in history], color=SEQ[3], lw=2, label="NT-Xent")
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK)
    axes[1].set_title("Critic regularisers", fontsize=10)
    axes[1].set_xlabel("generator step")
    for ax in axes:
        _style(ax)
    fig.suptitle("Training diagnostics", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, out_path)


def _save(fig, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out
