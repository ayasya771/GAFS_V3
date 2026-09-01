# Reference market panels

Two multi-asset panels ship with the repository so the pipeline, the tests
and the browser demo all run without network access.

| File | Rows | Span | Purpose |
|---|---|---|---|
| `reference_panel.csv` | 5000 business days | 2001-01-01 onward | the panel the released model was fitted on |
| `holdout_panel.csv` | 2600 business days | 2011-01-03 onward | an independent realisation, never seen in training |

## Provenance

These are **simulated** panels, not recorded market history. They come from
`gafs.data.synthetic.generate_synthetic_market`, a classical econometric
data-generating process, and are regenerated exactly by

    python scripts/make_reference_data.py

The process is a correlated GJR-GARCH(1,1) system with:

* a two-state volatility regime chain (calm and stressed) with persistent
  transition probabilities,
* a rare common jump factor with asset-specific loadings, so drawdowns hit
  the equity-like series harder than the commodity-like one, and crashes
  arrive simultaneously across assets,
* idiosyncratic jumps per asset,
* GJR asymmetry, so negative innovations raise conditional variance more
  than positive ones (the leverage effect),
* a Cholesky-factored correlation structure: the two equity-like series are
  tightly coupled (0.80), the commodity-like series is nearly independent.

The resulting series carry the stylized facts the model has to learn: fat
tails, volatility clustering, leverage, near-zero return autocorrelation and
a realistic cross-asset correlation matrix.

Macro conditioning series are derived the way their real counterparts
behave: `VIX_PROXY` tracks short-horizon realised volatility scaled by the
current regime, `RATE_10Y` is a mean-reverting rate process, and
`CREDIT_SPREAD` widens with market stress.

## Why simulated data ships instead of market history

Vendor terms for free market data generally permit downloading for personal
research but not redistribution, so committing recorded price history to a
public repository is not appropriate. A simulated panel with the right
statistical structure keeps the repository self-contained and reproducible
while sidestepping that entirely. It also carries a ground-truth regime
label, which recorded data does not.

For real data, run

    python scripts/download_data.py --sources yahoo,fred
    python scripts/preprocess_data.py --raw data/raw

and the rest of the pipeline is unchanged.

## Schema

    date              ISO date, business-day frequency, UTC
    EQ_LARGE_close    broad equity-like series
    EQ_TECH_close     high-beta equity-like series
    CMD_GOLD_close    commodity-like series, weakly correlated
    VIX_PROXY         implied-volatility-like index, points
    RATE_10Y          long rate, percent
    CREDIT_SPREAD     investment-grade spread, percent

Columns ending in `_close` are treated as tradable prices; every other
column is treated as a macro conditioning variable. Any CSV following this
convention can be dropped in and used with `--panel`.
