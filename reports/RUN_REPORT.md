# Validation run report

PyTorch 2.13, Python 3.11, 8 CPU cores, no GPU. This run validates the full
stack on the bundled reference market panel (`data/market/`), a simulated
multi-asset panel with realistic dynamics. The identical pipeline runs on
real market data once `scripts/download_data.py` has been executed from a
machine with access to the data providers.

## What ran

Panel: 5000 business days, 3 assets, 3 macro conditioning series.
Preprocessing: UTC alignment, MAD despiking (0 bad ticks in the clean
reference panel, as expected), log returns, causal 30-day volatility
scaling, macro z-scores fit on the training prefix. Windows: 4850 total
(lookback 90, horizon 30), split 3156 train / 728 validation / 728 test with
purged boundaries.

Training: WGAN-GP with n_critic 5 and lambda 10, an NT-Xent contrastive head
on the critic, and uniformity + MMD latent coverage terms on the generator.
Batch 64, Adam 1e-4, EMA weights used for sampling.

Evaluation: generated windows drawn at 60 held-out test anchors and compared
with the real test returns using identical estimators on both sides.

## Reading the metrics

`stylized_facts.md` holds the full table. The generator should land close to
the real column on annualised volatility, excess kurtosis and the Hill tail
exponent, keep the mean absolute autocorrelation of raw returns near zero,
and reproduce the cross-asset correlation structure. Wasserstein distances
in the low thousandths indicate the marginal return distributions match.

The scenario engine is exercised at the final anchor with a baseline set and
a stressed set (VIX +20 points, credit spreads +150 bp). Per-asset and
portfolio percentiles, VaR and expected shortfall at 95/99 and drawdown
statistics land in `scenarios_*/summary.csv`, with the full path arrays in
`scenarios.npz`.

## Artifacts

    evaluation/stylized_facts.md, .json   metric tables
    evaluation/distributions.png          real vs generated log densities
    evaluation/acf_abs.png                volatility clustering diagnostic
    evaluation/correlations.png           correlation heatmaps
    evaluation/fan_chart.png              generated fan vs realised path
    run/ckpt_final.pt                     trained weights (EMA included)
    run/history*.csv, training_history.png
    scenarios_baseline/, scenarios_stressed/
