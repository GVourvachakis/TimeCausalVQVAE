# Notebook Preview Branch

This branch contains executed notebook previews for visual inspection.

The `main` branch keeps notebooks output-stripped for reproducibility and package size. Outputs on
this branch are generated from local artefacts and may not be reproducible without the same local
outputs, checkpoints, summaries, generated figures, or processed data conventions.

Do not merge this branch into `main`. Do not commit raw or processed downloaded data,
checkpoints, W&B artefacts, generated tensors, or large generated outputs outside notebooks.
Large embedded notebook outputs should be avoided so this preview branch remains inspectable.

## Closure Status

This branch is closed as an executed-notebook preview branch. It is intended only for public visual
inspection of rendered notebook outputs and should not be merged into `main`.

Preview outputs depend on local artefacts, checkpoints, processed-data conventions, and cached
evaluation summaries that may not exist on another machine. The notebooks are therefore useful for
inspection, but they are not a reproducibility contract for the package release.

The initial preview passes intentionally skipped multidimensional notebooks. A later
multidimensional preview pass executed the new profile-aware notebooks for visual inspection only,
with training, downloads, checkpoint evaluation, and expensive metrics disabled. Expensive metrics
were run only as capped diagnostics in the earlier report notebooks. The retained expensive
diagnostic is Gaussian path MMD on small cached batches; those values are preview smoke checks, not
model-selection evidence. Signature MMD was attempted in the report notebooks and skipped because
the optional `signatory` dependency is unavailable. Adapted Wasserstein remained disabled.

## First Preview Set

Executed with `poetry run python scripts/run_notebook_previews.py --manifest
notebooks/preview_notebook_manifest.yaml --continue-on-error` on 2026-06-20.

| Notebook | Status | Notes |
| --- | --- | --- |
| `notebooks/benchmarks/hawkes_jump_dataset.ipynb` | Executed | Public Hawkes/SVMHJD dataset-analysis preview. |
| `notebooks/report/hawkes_jump_model_comparison.ipynb` | Executed | Uses local report-comparison artefacts where available. |
| `notebooks/report/sp500_vix_report_figures.ipynb` | Executed | Uses local S&P500/VIX report artefacts where available. |
| `notebooks/report/sample_geometry_diagnostics.ipynb` | Executed | Uses local sample-geometry artefacts where available. |
| `notebooks/continuous/sp500_vix.ipynb` | Executed | Guarded continuous S&P500/VIX demo path. |
| `notebooks/discrete/sp500_vix.ipynb` | Executed | Guarded discrete S&P500/VIX demo path. |
| `notebooks/continuous/hawkes_jump.ipynb` | Executed | Guarded continuous Hawkes/SVMHJD research-candidate demo path. |
| `notebooks/discrete/hawkes_jump.ipynb` | Executed | Guarded discrete Hawkes/SVMHJD research-candidate demo path. |

No multidimensional notebooks were executed in this pass.

## Multidimensional Preview Set

Executed with `poetry run python scripts/run_notebook_previews.py --manifest
/tmp/multidim_preview_notebook_manifest.yaml --parameter-mode preview --continue-on-error` on
2026-06-20. The temporary manifest contained only the three multidimensional entries that were
added to `notebooks/preview_notebook_manifest.yaml`.

This pass kept `RUN_TRAINING=False`, `RUN_EVALUATION=False`, `RUN_DOWNLOAD=False`,
`RUN_EXPENSIVE_METRICS=False`, and `ALLOW_MISSING_OUTPUTS=True`. No full training, downloads, or
checkpoint evaluation were run. The S&P500 50-stock panel notebook was executed with
`SP50050_PANEL_PROCESSED_DIR` pointed at a missing temporary directory, so the rendered notebook
shows the clean-checkout missing-data skip path. The multidimensional comparison notebook keeps
local-output summary, tensor-batch, and token-diagnostic discovery disabled in this preview pass.

| Notebook | Status | Notes |
| --- | --- | --- |
| `notebooks/benchmarks/multifactor_market_50d.ipynb` | Executed | Shows lightweight in-memory synthetic multifactor EDA and experimental profile metadata. |
| `notebooks/benchmarks/sp500_50_panel.ipynb` | Executed | Shows the static universe, guarded download command, missing-data skip path, and experimental profile metadata. |
| `notebooks/report/multidim_experimental_comparison.ipynb` | Executed | Shows multidimensional profile metadata, representative metric tables, disabled local-output discovery, and printed commands only. |

## Full-Output Preview Set

Executed with `poetry run python scripts/run_notebook_previews.py --manifest
notebooks/full_preview_notebook_manifest.yaml --parameter-mode full-preview
--max-total-runtime-hours 6 --continue-on-error` on 2026-06-20.

This pass covers all non-multidimensional notebooks currently under `notebooks/`. The runner
overrides existing preview parameters to enable full/heavy preview cells while keeping
`RUN_TRAINING=False`, `RUN_EVALUATION=False`, `RUN_EXPENSIVE_METRICS=False`,
`RUN_SIGNATURE_KERNEL=False`, `RUN_ADAPTED_WASSERSTEIN=False`, and
`ALLOW_MISSING_OUTPUTS=True`.

All notebooks in `notebooks/full_preview_notebook_manifest.yaml` passed. The initial pass exposed
a tensor truth-value issue in `notebooks/continuous/hawkes_jump.ipynb` when a local
`evaluation_batch.pt` was present; the notebook now selects optional tensors without boolean
coercion and passed when rerun alone. Expensive signature-kernel and adapted-Wasserstein metrics
remain disabled and documented in `NOTEBOOK_EXPENSIVE_METRIC_RECOMMENDATIONS.md`.
