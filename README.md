# Generative Adversarial Financial Simulation (GAFS)

Synthetic multi-asset market scenarios from a Wasserstein GAN. The generator
is a Conditional Temporal Fusion Transformer driven by per-step Gaussian
noise; the critic is a WGAN-GP discriminator carrying a SimCLR-style
contrastive head that keeps the latent space organised, so the model keeps
producing rare tail events instead of collapsing onto average days. The
output is a scenario engine: thousands of statistically faithful price paths,
conditioned on the current macro state, that you can shock (VIX up 20 points,
credit spreads out 150 bp) for portfolio stress testing.

Synthetic paths are validated against the stylized facts that matter: fat
tails, volatility clustering, the leverage effect, near-zero return
autocorrelation, and the cross-asset correlation structure.

There is an interactive demo in `docs/`: the trained generator, the full
preprocessing pipeline and the whole validation suite ported to JavaScript
and run client-side, with no server and no dependencies. See
[Browser](#browser-demo).

## How it works

Preprocessing (strict order): timestamps are
normalised to UTC and joined exactly, with forward-filling capped at 3
periods; bad ticks are removed with a rolling median/MAD filter that keeps
genuine crashes (an outlier is only deleted when the next print reverts to
the local median); prices become log returns, or fractionally differenced
log prices (fixed-width FFD, d in (0,1)) when long memory should survive;
series are standardised by their lagged 30-day realized volatility (or
train-fitted z-scores); the panel is then windowed into
[batch, sequence, features] tensors with lookback k = 90 and horizon h = 30.

Generator, a conditional TFT: a Variable Selection Network weighs the input
variables at every time step, Gated Residual Networks let simple mappings
bypass nonlinear processing, an LSTM encoder-decoder pair is initialised
from the macro conditioning vector C_t, and causal multi-head attention maps
history X_{t-k:t} onto the projection horizon. A noise sequence
Z_t ~ N(0, I) drives the decoder and is concatenated again at the output
head, giving Y_hat = G(X_{t-k:t}, C_t, Z_t).

Critic, WGAN-GP: a 1D ResNet (or transformer encoder) over candidate future
paths, conditioned on the same history and macro state, returning an
unbounded scalar score. Training minimises

    L_D = E[D(x_g)] - E[D(x)] + lambda * E[(||grad D(x_hat)||_2 - 1)^2]

with lambda = 10 and interpolates x_hat = eps x + (1 - eps) x_g,
eps ~ U[0, 1]. The critic trains n_critic = 5 times per generator step.

Anti-mode-collapse, SimCLR adapted to time series: two augmented views of
each real window (jittering, scaling, time warping) pass through the
critic's projection head and are pulled together by the NT-Xent loss while
different windows are pushed apart. The generator is then held to that
organised latent space through a uniformity term (spread your samples over
the hypersphere) and an RBF-MMD term (match the real windows' embedding
distribution), on top of its adversarial loss.

## Repository layout

    config/default.yaml        every tunable parameter and its default
    gafs/config.py             typed config loading and validation
    gafs/data/sources/         Yahoo, FRED, Binance archive, Alpaca, Dukascopy
    gafs/data/synthetic.py     calibrated GJR-GARCH + jumps offline market
    gafs/data/preprocess.py    UTC join, MAD filter, stationarity, scaling
    gafs/data/fracdiff.py      fixed-width fractional differencing
    gafs/data/windows.py       tensor windowing, purged chronological splits
    gafs/models/layers.py      GLU, GRN, VSN
    gafs/models/generator_tft.py
    gafs/models/critic.py      ResNet / transformer critic + projection head
    gafs/training/             augmentations, losses, WGAN-GP trainer
    gafs/evaluation/           stylized facts, diagnostics, batch sampling
    gafs/simulation/           conditioned scenarios, macro shocks, VaR/ES
    scripts/                   download, preprocess, train, evaluate, generate
    data/market/               bundled reference panels (see its README)
    docs/                      the browser demo, published by GitHub Pages
    reports/                   figures and metric tables from the released run
    tests/                     unit, smoke and web-bundle tests

## Quickstart

    pip install -r requirements.txt
    python scripts/quickstart_demo.py

No network required: the run preprocesses the bundled reference panel, trains
the full WGAN-GP + contrastive stack, writes a stylized-facts report with
figures, and produces baseline and VIX-stress scenario sets under
`outputs/demo/`. Use `--steps 300 --days 3000` for a fast smoke run; the
default 1500 steps takes roughly 10 to 15 minutes on a laptop CPU.

## Browser demo

`docs/` is a self-contained page that loads the trained weights and runs the
real model in the browser. It is not a set of exported screenshots: the
generator forward pass, the preprocessing pipeline and every statistic on the
page are computed live in JavaScript from the committed weights and panels.

To view it locally, serve the folder over HTTP (fetch will not read model
files from `file://`):

    python -m http.server -d docs 8000    # then open http://localhost:8000

To publish it, push the repository and set GitHub Pages to deploy from the
`main` branch, `/docs` folder. Nothing else is required: there is no build
step, no bundler and no external dependency, and the page makes no network
request after the initial load.

What the page does:

* **Data** loads a bundled panel or a CSV you drop in, and reports its return
  characteristics with the same estimators the Python evaluation uses.
* **Pipeline** runs each preprocessing stage live. It can inject bad ticks and
  a crash into the raw series so you can watch the MAD filter delete the
  former and leave the latter alone, which is the property that matters.
* **Scenarios** draws conditional paths from a chosen origin, with the macro
  state shockable in raw units, and reports VaR, expected shortfall and
  drawdowns from the simulated paths.
* **Validation** re-runs the stylized-facts battery in the browser at held-out
  origins and shows real against generated side by side.
* **Model** describes the architecture, shows live variable-selection weights,
  and states what the demo does not claim.

The page verifies its own arithmetic. `scripts/build_site.py` exports a
reference input, noise draw and output from PyTorch alongside the weights; on
load the page reproduces that output with its own forward pass and displays
the maximum deviation in the header. It is a float32 rounding difference
(order 1e-7), and it is measured on every page load rather than asserted.

Rebuild the bundle after training:

    python scripts/build_site.py --ckpt outputs/demo/run/ckpt_final.pt

## Data

Two reference panels ship with the repository so everything runs offline;
`data/market/README.md` documents how they are generated and why simulated
data is used rather than redistributed market history. Regenerate them with
`python scripts/make_reference_data.py`.

For real data:

    python scripts/download_data.py --sources yahoo,fred
    python scripts/preprocess_data.py --raw data/raw
    python scripts/train.py --data data/processed --out outputs/run1
    python scripts/evaluate.py --ckpt outputs/run1/ckpt_final.pt
    python scripts/generate.py --ckpt outputs/run1/ckpt_final.pt \
        --n 2000 --shock VIXCLS=+20 --shock BAMLC0A0CM=+1.5

Tickers, FRED series and date ranges live in `config/default.yaml`. Binance
minute data adds crypto crash dynamics (`--sources yahoo,fred,binance`).
The Alpaca adapter needs free API keys in `ALPACA_API_KEY` and
`ALPACA_SECRET_KEY`; for Dukascopy tick data, bulk-download CSVs with
dukascopy-node as described in `gafs/data/sources/dukascopy.py`.

Shocks are raw-unit shifts of the macro conditioning vector at the forecast
anchor, translated into model space with stored train-set statistics.
`scripts/generate.py` prints per-asset and portfolio percentiles, VaR and
expected shortfall at 95/99, and drawdown statistics, and saves the full
path array (`scenarios.npz`) plus a fan chart.

## Evaluation

`scripts/evaluate.py` samples paths at up to 100 held-out test anchors,
converts them back to return space with each anchor's own volatility, and
compares them with real test-period returns using the same estimators on
both sides: moments and excess kurtosis, Hill tail exponents, ACFs of raw
and absolute returns, leverage correlations corr(r_t, |r_{t+l}|), per-asset
Wasserstein distances, and correlation matrix distance. Results land in
`stylized_facts.md` / `.json` with four diagnostic figures.

The output of the released run is committed under `reports/`, so the metric
tables and figures can be read without rerunning anything.

## Testing

    pytest

The suite covers the FFD weights against closed forms, ffill limits, the
spike-versus-crash behaviour of the MAD filter, causality of the scaling
(recomputing on truncated data reproduces the prefix), window alignment and
purged splits, model shapes, conditioning and the causal attention mask,
loss properties (gradient penalty, NT-Xent ordering, uniformity, MMD), a
six-step training smoke run with checkpoint round-trip, and scenario
generation including stress-shock unit mapping.

`tests/test_web_bundle.py` additionally guards the browser demo: the manifest
must account for every float in the weight blob, the exported scalers must
match the processed panel, and - where node is installed - the JavaScript
generator and preprocessing pipeline must reproduce their PyTorch and pandas
counterparts to within float32 precision. A change to either side that breaks
the port fails the test suite rather than the page.

## Design notes

The critic uses GroupNorm only; batch normalisation would couple samples and
invalidate the per-sample gradient penalty. Feature scaling is strictly
causal (lagged rolling volatility), and z-score statistics come from the
training prefix alone; the one deliberately non-causal step is the MAD
despiking filter, which inspects a centered neighbourhood to tell isolated
bad ticks from real moves, standard for offline cleaning and never a source
of value leakage into the features. Chronological splits are purged by lookback + horizon windows
so no timestamp leaks across splits. Generator weights are tracked with an
EMA copy, which is what checkpoint loading returns by default. When
fractional differencing is selected the model works in fracdiff space and
price-path reconstruction is deliberately disabled rather than silently
wrong; log-returns mode is the default and supports full price
reconstruction.

## References

Gulrajani et al., Improved Training of Wasserstein GANs, 2017.
Lim et al., Temporal Fusion Transformers for Interpretable Multi-horizon
Time Series Forecasting, 2021. Chen et al., A Simple Framework for
Contrastive Learning of Visual Representations, 2020. Wang and Isola,
Understanding Contrastive Representation Learning, 2020. Lopez de Prado,
Advances in Financial Machine Learning, 2018 (fractional differencing).
Cont, Empirical Properties of Asset Returns: Stylized Facts and Statistical
Issues, 2001.

MIT licensed.
