# Notebook Preview Branch

This branch contains executed notebook previews for visual inspection.

The `main` branch keeps notebooks output-stripped for reproducibility and package size. Outputs on
this branch are generated from local artefacts and may not be reproducible without the same local
data, checkpoints, summaries, or generated figures.

Do not merge this branch into `main`.

## First Preview Set

Executed with `poetry run jupyter nbconvert --execute --inplace` on 2026-06-20.

| Notebook | Status | Notes |
| --- | --- | --- |
| `notebooks/benchmarks/hawkes_jump_dataset.ipynb` | Executed | Public Hawkes/SVMHJD dataset-analysis preview. |
| `notebooks/report/hawkes_jump_model_comparison.ipynb` | Executed | Uses local report-comparison artefacts where available. |
| `notebooks/report/sp500_vix_report_figures.ipynb` | Executed | Uses local S&P500/VIX report artefacts where available. |
| `notebooks/report/sample_geometry_diagnostics.ipynb` | Executed | Uses local sample-geometry artefacts where available. |
| `notebooks/continuous/sp500_vix.ipynb` | Executed | Guarded continuous S&P500/VIX demo path. |
| `notebooks/discrete/sp500_vix.ipynb` | Executed | Guarded discrete S&P500/VIX demo path. |

No multidimensional notebooks were executed in this pass.
