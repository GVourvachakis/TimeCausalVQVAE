# Notebook Expensive Metric Recommendations

This note records optional expensive diagnostics for future notebook work. The full-preview pass
keeps `RUN_EXPENSIVE_METRICS=False`, `RUN_SIGNATURE_KERNEL=False`, and
`RUN_ADAPTED_WASSERSTEIN=False`.

## Signature MMD

The repository already exposes optional signature-style metric infrastructure through
`time_causal_vae.evaluation.metrics.SignatureMMD`. None of the preview notebooks should run it by
default.

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
