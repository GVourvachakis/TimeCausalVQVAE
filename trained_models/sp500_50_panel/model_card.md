# S&P500 50-Stock Panel Model Card

## Status

The empirical S&P500 50-stock panel benchmark is experimental and local-only.
It has no public default model and no committed checkpoint weights.

The profile metadata lives in `trained_models/multidim_profiles.yaml`, not in
`trained_models/model_registry.yaml`. The public registry remains unchanged
because no empirical multidimensional model is selected for public default use.

## Profile-Specific Candidates

| Profile | Family | Candidate | Configs | Sampling |
| --- | --- | --- | --- | --- |
| `balanced_empirical` | continuous | `beta_cvae_latent8_hidden64_v3` | `configs/experiments/sp500_50_panel_v3_beta_cvae_latent8_hidden64.yaml` | n/a |
| `discrete_reporting` | discrete | `factor_pca_rvq_q2_cb64_factorised` | `configs/experiments/sp500_50_panel_v3_factor_pca_rvq_q2_cb64_tokenizer_auxloss.yaml`; `configs/experiments/sp500_50_panel_v3_factor_pca_rvq_q2_cb64_prior_factorised_additive.yaml` | `temperature=0.7`, `top_k=20` |

## Metrics Summary

The valid v3 comparison uses the `v3_prefix_market` condition path, `355/89`
train/eval windows, train-only return and label standardisation, train-fitted
PCA for the discrete factor view, and raw-scale diagnostics.

| Metric | BetaCVAE v3 | Calibrated RVQ q2 v3 |
| --- | ---: | ---: |
| MMD | `0.413115` | `0.296044` |
| SWD | `0.005034` | `0.004152` |
| Cov rel Frobenius | `0.421567` | `1.105872` |
| Corr rel Frobenius | `0.838303` | `1.345099` |
| Corr spectrum | `0.236516` | `0.414596` |
| Sector MAE | `0.223918` | `0.430765` |
| Equal-weight vol ratio | `1.071880` | `1.581380` |
| Random-portfolio vol ratio | `1.063057` | `1.566198` |
| Equal-weight VaR q01 error | `0.000311` | `0.018045` |
| Equal-weight ES q01 error | `0.000647` | `0.031749` |

BetaCVAE latent8 hidden64 v3 is the best current one-seed empirical candidate.
The calibrated RVQ q2 variant improves default RVQ sampling but remains
over-dispersed and weak on pair and regime calibration.

## Caveats

- This benchmark is experimental and local-only.
- The v3 comparison is one-seed local evidence, not a seed-robust result.
- Calibrated RVQ q2 has high eval q0/q1 pair L1 and absent-pair mass.
- No empirical multidimensional model is selected for public registry use.
- Downloaded prices, processed panels, checkpoints, token tensors, generated
  paths, local JSON summaries, and rendered notebooks are not committed.

## Local Data And Output Policy

S&P500 panel data must be rebuilt or loaded locally. Expected artefacts remain
under local `data/`, `/tmp/`, and `outputs/` paths and must stay out of git.
