# Multifactor Market Model Card

## Status

The synthetic 50D multifactor benchmark is experimental. It has no public
default model and no committed checkpoint weights.

The profile metadata lives in `trained_models/multidim_profiles.yaml`, not in
`trained_models/model_registry.yaml`. The public registry remains unchanged
because no multidimensional model is selected for public default use.

## Profile-Specific Candidates

| Profile | Family | Candidate | Configs | Sampling |
| --- | --- | --- | --- | --- |
| `correlation_sector` | continuous | `beta_cvae_latent8_hidden64` | `configs/experiments/multifactor_market_standardized_beta_cvae_latent8_hidden64.yaml` | n/a |
| `portfolio_tail` | discrete | `factor_pca_rvq_q2_cb64_factorised` | `configs/experiments/multifactor_market_factor_pca_causal_rvq_tokenizer_q2_cb64_auxloss.yaml`; `configs/experiments/multifactor_market_factor_pca_rvq_q2_cb64_prior_factorised_additive.yaml` | `temperature=1.0`, `top_k=null` |

## Metrics Summary

The no-jump profile split is stable. BetaCVAE latent8 hidden64 is stronger on
correlation relative Frobenius, eigenspectrum, and sector MAE. Factor-PCA RVQ
q2 cb64 with the factorised additive prior is stronger on MMD/SWD, covariance
scale, equal-weight and random-portfolio volatility, and VaR/ES retention.

Representative no-jump three-seed means:

| Metric | BetaCVAE latent8 hidden64 | Factor-PCA RVQ q2 cb64 |
| --- | ---: | ---: |
| MMD | `0.2689` | `0.2236` |
| SWD | `0.003238` | `0.002677` |
| Cov rel Frobenius | `0.3237` | `0.1301` |
| Corr rel Frobenius | `0.2987` | `0.3605` |
| Corr spectrum | `0.1074` | `0.1465` |
| Sector MAE | `0.0942` | `0.1391` |
| Equal-weight vol ratio | `0.8265` | `0.9587` |
| Equal-weight VaR q01 error | `0.003778` | `0.000997` |
| Equal-weight ES q01 error | `0.004349` | `0.002694` |

Under common and sector jumps, the same profile split remains. RVQ q2 keeps the
stronger covariance and portfolio-tail profile. BetaCVAE keeps the stronger
correlation, eigenspectrum, sector, and jump-window conditional-correlation
profile. Neither model fully solves detected jump-window conditional
correlation.

## Caveats

- This benchmark is experimental and profile-specific.
- No model is a selected public default.
- Factor-RVQ q2 trails the continuous baseline on correlation, eigenspectrum,
  sector structure, and jump-window conditional correlation.
- BetaCVAE compresses portfolio volatility on synthetic data.
- Sector-grouped q2 and hierarchical q0-to-q1 priors remain research-only.

## Local Data And Output Policy

No checkpoint weights, token tensors, generated paths, local JSON summaries, or
W&B artefacts are committed. Expected artefacts remain under local `outputs/`
paths after a user runs the corresponding experiments.
