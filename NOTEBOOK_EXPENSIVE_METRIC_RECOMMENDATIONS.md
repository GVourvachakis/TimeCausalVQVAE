# Notebook Expensive Metric Recommendations

This note records optional expensive diagnostics for future notebook work. The full-preview pass
keeps `RUN_EXPENSIVE_METRICS=False`, `RUN_SIGNATURE_KERNEL=False`, and
`RUN_ADAPTED_WASSERSTEIN=False`.

The executed Gaussian path MMD values on this branch are preview and smoke diagnostics only. They
are not model-selection evidence. Expensive path-space diagnostics should remain concentrated in
the report notebooks, where cached local artefacts and small deterministic caps can be documented,
rather than being added to every per-experiment walkthrough notebook.

## Signature MMD

The repository already exposes optional signature-style metric infrastructure through
`time_causal_vae.evaluation.metrics.SignatureMMD`. None of the preview notebooks should run it by
default. sigMMD should remain optional and dependency-gated because the executed small-cap report
pass skipped it when the optional `signatory` dependency was unavailable.

Recommended optional cells:

| Notebook group | Recommendation | Suggested cap |
| --- | --- | --- |
| `notebooks/report/hawkes_jump_model_comparison.ipynb` | Good candidate for sigMMD because clustered jump timing is path-order sensitive. Keep it as an optional comparison cell. | `n_sample <= 128`, `n_timestep <= 60`, CPU only unless explicitly profiled. |
| `notebooks/report/sp500_vix_report_figures.ipynb` | Useful only as a supplemental path-shape diagnostic for existing local generated batches. | `n_sample <= 128`; subsample generated and real paths deterministically. |
| `notebooks/report/sample_geometry_diagnostics.ipynb` | Useful as an optional scalar column for the same subsampled models used in KDE/t-SNE figures. | `n_sample <= 96`; reuse the notebook's lightweight sample cap. |
| Continuous/discrete one-dimensional demo notebooks | Usually not necessary in demos; prefer printing commands and registry summaries. | `n_sample <= 64` if added for smoke inspection. |

## Adapted Or Causal Wasserstein

The evaluation package retains adapted-Wasserstein-style optional behaviour through
`time_causal_vae.evaluation.metrics`. Current notebooks already display ordinary one-dimensional
Wasserstein summaries when local evaluation artefacts contain them.

Adapted Wasserstein should remain disabled by default. If it is enabled later, it should use cached
data only, deterministic subsampling, and small sample sizes.

Recommended optional cells:

| Notebook group | Recommendation | Suggested cap |
| --- | --- | --- |
| `notebooks/report/hawkes_jump_model_comparison.ipynb` | Strong candidate because jump timing and lower-tail path order matter. Keep as optional and cached-output-only. | `n_sample <= 64`, `n_timestep <= 60`, fixed seed and deterministic subsampling. |
| `notebooks/report/sp500_vix_report_figures.ipynb` | Useful for a small VIX-conditioned subset if local generated paths are already available. | `n_sample <= 64` per VIX bucket, or `n_sample <= 128` globally. |
| `notebooks/report/sample_geometry_diagnostics.ipynb` | Optional table-only diagnostic; avoid combining with t-SNE recomputation in the same uncached cell. | `n_sample <= 64` per model pair. |
| Benchmark and simple continuous/discrete demos | Do not add by default. These notebooks should remain command/registry previews. | Only smoke caps of `n_sample <= 32` if added. |

All expensive diagnostics should remain behind explicit guards, avoid full checkpoint evaluation,
prefer existing local output batches, and print a clear skipped message when required artefacts are
missing.

## Executed Small-Cap Report Pass

Executed on 2026-06-20 for the three report notebooks only with
`RUN_EXPENSIVE_METRICS=True`, `RUN_SIGNATURE_KERNEL=True`,
`RUN_ADAPTED_WASSERSTEIN=False`, `MAX_EXPENSIVE_METRIC_PATHS=128`, and
`MAX_AWD_PATHS=32`.

| Notebook | Outcome | Recommendation |
| --- | --- | --- |
| `notebooks/report/hawkes_jump_model_comparison.ipynb` | Gaussian path MMD computed for three cached Hawkes batches at 128 paths and 60 time steps. Signature MMD was attempted and skipped because the optional `signatory` dependency is unavailable. Adapted Wasserstein remained disabled. | Keep the Gaussian path MMD preview table. Add sigMMD only when `signatory` is installed, and keep adapted Wasserstein optional with `n_sample <= 32` unless a faster cached implementation is available. |
| `notebooks/report/sp500_vix_report_figures.ipynb` | Gaussian path MMD computed for five cached S&P 500/VIX batches at 128 paths and 60 time steps. Signature MMD was skipped by the same dependency gate. | Keep the bounded Gaussian diagnostic as a report preview. Treat sigMMD as an optional dependency-backed cell, not a default branch requirement. |
| `notebooks/report/sample_geometry_diagnostics.ipynb` | Gaussian path MMD computed for four cached geometry panels at 128 paths and 60 time steps. Signature MMD skipped; adapted Wasserstein disabled. | Keep the table as a small path-space smoke diagnostic. Do not interpret the values as model-selection evidence without a larger, explicitly profiled evaluation run. |

The executed outputs are informative enough to retain on the preview branch. They should remain
small, cached-output-only diagnostics and should not trigger training or full checkpoint evaluation.
