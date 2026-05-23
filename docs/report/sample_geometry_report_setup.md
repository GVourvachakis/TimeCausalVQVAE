# Sample Geometry Report Setup Audit

This audit records the inputs needed for final report t-SNE and KDE/ECDF figures.
It does not train models, add model code, modify the model registry, or commit
generated outputs.

## Intended Report Notebook

- Intended notebook path: `notebooks/report/final_sample_geometry_report.ipynb`.
- Current `notebooks/report/` contents:
  - `notebooks/report/sp500_vix_report_figures.ipynb`
  - `notebooks/report/hawkes_jump_model_comparison.ipynb`
- The intended final sample-geometry notebook is not present yet.

## Sample Geometry Tooling

The current checkout and local `main` do not contain the shared sample-geometry
tooling:

- Missing on current checkout and `main`:
  - `src/time_causal_vae/evaluation/sample_geometry.py`
  - `scripts/plot_sample_geometry.py`
  - `docs/verification/sample_geometry_diagnostics.md`

The source branch `research/continuous-lsgm-prior` contains all three files. They
were inspected with `git show` only; they were not copied into this branch.

- `src/time_causal_vae/evaluation/sample_geometry.py` builds per-path financial
  feature matrices, fits qualitative t-SNE projections with a deterministic PCA
  fallback, and produces ECDF/KDE feature summaries.
- `scripts/plot_sample_geometry.py` loads real and generated tensor payloads,
  computes feature matrices, and writes local diagnostic artefacts below
  `outputs/`.
- `docs/verification/sample_geometry_diagnostics.md` documents the diagnostic
  scope, feature set, CLI use, report layout, and verification commands.

For the final notebook, the missing tooling should be copied or otherwise
ported from `research/continuous-lsgm-prior` before relying on a shared
implementation.

## Notebook And Registry Structure

`notebooks/README.md` describes registry-driven continuous and discrete demos
for all core experiments, plus report notebooks that read local `outputs/`
paths and do not train models by default.

Available notebook groups:

- `notebooks/continuous/`: `black_scholes.ipynb`, `heston.ipynb`,
  `pdv.ipynb`, `sp500_vix.ipynb`, `hawkes_jump.ipynb`.
- `notebooks/discrete/`: `black_scholes.ipynb`, `heston.ipynb`,
  `pdv.ipynb`, `sp500_vix.ipynb`, `hawkes_jump.ipynb`,
  `discrete_latent_geometry.ipynb`.
- `notebooks/report/`: S&P500/VIX and Hawkes/SVMHJD report notebooks only.

The registry file `trained_models/model_registry.yaml` includes all requested
experiments:

| Experiment | Continuous selection | Discrete selection | Notes |
| --- | --- | --- | --- |
| `black_scholes` | `beta_cvae` | `hidden128_conv_transformer_k3` | Public workflow entry. |
| `heston` | `info_cvae` | `standard_vq_additive_ar` | Public workflow entry. |
| `pdv` | `info_cvae` | `conditional_standard_vq_additive_ar` | Public workflow entry. |
| `sp500_vix` | `beta_cvae` | `conditional_standard_vq_additive_ar` | Also has optional `conditional_hidden128_conv_transformer_k3`. |
| `hawkes_jump` | `beta_cvae_logreturn_identity` | `hidden128_logreturn_cb64_conv_transformer_k3` | Present as a `research_candidate`; public default is false. |

`hawkes_jump` is present in this checkout, so the final notebook does not need
to degrade because of a missing registry entry here. It should still degrade
gracefully if the registry entry or local Hawkes outputs are absent in another
checkout.

## Local Output Batches

The requested scan,
`find outputs -path '*evaluation_batch.pt' -o -path '*evaluation_summary.json'`,
found local evaluator batches for Hawkes/SVMHJD, legacy continuous baselines,
S&P500/VIX continuous evaluation, and score-prior summaries for S&P500/VIX.
Generated outputs remain local and uncommitted.

Representative available local batches:

| Experiment | Continuous registered candidate | Discrete registered candidate | Optional research candidate |
| --- | --- | --- | --- |
| `black_scholes` | `outputs/legacy_continuous_evaluation/black_scholes/evaluation_batch.pt` | `outputs/per_experiment_final_evaluation/black_scholes/hidden128_conv_transformer_k3/path_metrics/decoded_paths.pt` | No local score-prior or alternative research batch found. |
| `heston` | `outputs/legacy_continuous_evaluation/heston/evaluation_batch.pt` | `outputs/per_experiment_final_evaluation/heston/standard_vq_additive_ar/path_metrics/decoded_paths.pt` | No local score-prior or alternative research batch found. |
| `pdv` | `outputs/legacy_continuous_evaluation/pdv/evaluation_batch.pt` | `outputs/per_experiment_final_evaluation/pdv/conditional_standard_vq_additive_ar/path_metrics/decoded_paths.pt` | No local score-prior or alternative research batch found. |
| `sp500_vix` | `outputs/per_experiment_final_evaluation/sp500_vix/continuous/evaluation_batch.pt`; also `outputs/sp500_vix_continuous/beta_cvae/evaluation_final/evaluation_batch.pt` | `outputs/per_experiment_final_evaluation/sp500_vix/conditional_standard_vq_additive_ar/path_metrics/discrete_paper_style_batch.pt` | `outputs/per_experiment_final_evaluation/sp500_vix/conditional_hidden128_conv_transformer_k3/path_metrics/discrete_paper_style_batch.pt`; LSGM/score-prior batches such as `outputs/lsgm_score_prior_comparison/sp500_vix_first_eval_ddim50_n1000/score_prior_samples.pt` are also local. |
| `hawkes_jump` | `outputs/hawkes_jump_continuous_logreturn_identity/seed0/evaluation/evaluation_batch.pt`, plus seeds 1 and 2 | `outputs/hawkes_jump_logreturn_robustness/evaluations/conv_transformer_seed0/evaluation_batch.pt`, plus seeds 1 and 2 | Required additive ablation batches are present under `outputs/hawkes_jump_logreturn_robustness/evaluations/additive_seed*/evaluation_batch.pt`; compact candidates are present under `outputs/hawkes_jump_compact_conv_transformer/evaluations/`. |

Payload inspection confirms that the representative path tensors contain usable
real and generated path keys:

- Legacy continuous `evaluation_batch.pt` files contain `real_data`, `fake_data`,
  and `recon_data`, each shaped `(1000, 60, 1)`.
- Final discrete `decoded_paths.pt` files for `black_scholes`, `heston`, and
  `pdv` contain `real_paths` and `decoded_paths`, each shaped `(1000, 60, 1)`.
- S&P500/VIX paper-style discrete batches contain `real_paths` and
  `decoded_paths`, each shaped `(1000, 60, 1)`.
- Hawkes continuous batches contain decoder-space, price-space, and jump tensors;
  the log-return geometry should use `real_decoder_space` and
  `generated_decoder_space`, each shaped `(1024, 60, 1)`.
- Hawkes discrete batches contain `real_decoder_space`,
  `decoded_decoder_space`, price-space tensors, jump tensors, and token tensors;
  the log-return geometry should use `real_decoder_space` and
  `decoded_decoder_space`, each shaped `(1024, 60, 1)`.
- S&P500/VIX score-prior batches contain `real_paths` and `decoded_paths`, each
  shaped `(1000, 60, 1)`.

## Missing Or Partial Outputs

- `black_scholes`, `heston`, and `pdv` have continuous `evaluation_batch.pt`
  files and discrete decoded path tensors, but no discrete
  `evaluation_batch.pt` or `evaluation_summary.json` files under the requested
  scan pattern.
- `black_scholes`, `heston`, and `pdv` do not have local score-prior or LSGM
  generated path batches comparable to the S&P500/VIX LSGM outputs.
- `sp500_vix` has several local LSGM/score-prior sample batches; the final
  notebook should choose one explicitly rather than globbing all ablations.
- `hawkes_jump` has continuous and discrete research-candidate batches, but no
  local LSGM or score-prior comparator output is present.
- The final sample-geometry report notebook is still missing.

## Dependency Status

The t-SNE and KDE dependencies are available in the current Poetry environment:

```text
sklearn True
scipy True
```

No dependency change is required for this checkout. If `sklearn` is missing in a
fresh environment, add `scikit-learn` only to the notebook/report dependency
group, not to the core runtime.
