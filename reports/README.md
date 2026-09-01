# Run artifacts

Output from the released training run, committed so the repository shows its
own results without requiring a rerun.

    RUN_REPORT.md                  what was run and how to read it
    stylized_facts.md              real vs generated metric table (test split)
    distributions.png              return densities, log scale
    acf_abs.png                    volatility clustering diagnostic
    correlations.png               cross-asset correlation, real vs generated
    fan_chart.png                  generated fan against the realised path
    training_history.png           critic Wasserstein estimate and regularisers
    scenario_summary_*.csv         baseline and stressed tail-risk summaries

Checkpoints and the full path arrays are not committed; they are regenerated
by `python scripts/quickstart_demo.py`. The trained generator weights the
browser demo uses are committed separately under `docs/model/`.
