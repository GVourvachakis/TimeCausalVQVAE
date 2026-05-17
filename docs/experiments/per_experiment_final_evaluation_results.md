# Per-Experiment Final-Evaluation Results

## Scope

This document records the final per-experiment evaluation run on branch
`research/per-experiment-model-selection`. Generated arrays, checkpoints, figures, CSV files, and
JSON summaries remain under ignored `outputs/` paths and are not committed. The trained-model
registry is not updated in this pass.

The command was:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread \
poetry run python scripts/run_per_experiment_final_evaluation.py \
  --experiments black_scholes heston pdv sp500_vix \
  --output-dir outputs/per_experiment_final_evaluation \
  --n-sample 1000 \
  --profile balanced_market \
  --no-wandb
```

Aggregate outputs:

- `outputs/per_experiment_final_evaluation/final_evaluation_plan.json`;
- `outputs/per_experiment_final_evaluation/final_evaluation_plan.csv`.

The final run completed with 28 targets. Missing local checkpoints were recorded as
`not_available` before execution. The runner was updated during this pass to accept the requested
`--no-wandb` compatibility flag, discover timestamped continuous `final_model` directories, keep
fallback public candidates in the target set, and preflight required local artefact directories.

A supplemental continuous-baseline evaluation was then run against the cloned
`~/Desktop/TimeCausalVAE` repository. The portable selected configs in this repository already
match the optimal continuous settings from the cloned repository, so the raw legacy
`exp_config.yaml` files were not copied into this branch. Instead, `tcvae-evaluate` used each
cloned `final_model` directory together with the adjacent legacy `exp_config.yaml` already present
beside that checkpoint. The supplemental outputs are local generated artefacts under
`outputs/legacy_continuous_evaluation/` and are not committed.

## Overall Status

| Experiment | Continuous baseline | Public discrete baseline | Provisional discrete candidate | No-leakage status | Notebook or reproduction status |
| --- | --- | --- | --- | --- | --- |
| Black-Scholes | Supplemental cloned-repo `BetaCVAE` evaluation completed; aggregate runner row remains `not_available` because no in-repo `final_model` was supplied. | `not_available`; configured non-smoke public prior and tokenizer artefacts are missing. | `hidden128_conv_transformer_k3` evaluated successfully. | Causal-conv and token-prior checks passed for the provisional candidate; trained unconditioned tokenizer check is not available. | Not rerun. |
| Heston | Supplemental cloned-repo `InfoCVAE` evaluation completed; aggregate runner row remains `not_available` because no in-repo `final_model` was supplied. | Same as provisional `standard_vq_additive_ar`, evaluated successfully. | `standard_vq_additive_ar` evaluated successfully. | Causal-conv and token-prior checks passed; trained unconditioned tokenizer check is not available. | Not rerun. |
| PDV4 | Supplemental cloned-repo `InfoCVAE` evaluation completed; aggregate runner row remains `not_available` because no in-repo `final_model` was supplied. | Same as provisional `conditional_standard_vq_additive_ar`, evaluated successfully. | `conditional_standard_vq_additive_ar` evaluated successfully. | Causal-conv, token-prior, and conditional-tokenizer checks passed. | Not rerun. |
| S&P500/VIX | `sp500_vix_beta_cvae` evaluated successfully from the local timestamped checkpoint and supplementally from the cloned optimal checkpoint. | `conditional_standard_vq_additive_ar` evaluated successfully. | `conditional_hidden128_conv_transformer_k3` evaluated successfully. | Causal-conv, token-prior, and conditional-tokenizer checks passed for both discrete candidates. | Paper-style wrapper was run; notebooks were not rerun. |

The aggregate final-evaluation runner still records Black-Scholes, Heston, and PDV4 continuous
rows as `not_available` because no local `final_model` paths were supplied to that runner. The
separate cloned-repo `tcvae-evaluate` commands show that those continuous baselines are runnable
when the optimal `final_model` directories from `~/Desktop/TimeCausalVAE` are supplied. S&P500/VIX
was the only continuous baseline with a discoverable in-repository checkpoint:
`outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47/final_model`.

The cloned continuous checkpoints used for the supplemental evaluation were:

| Experiment | Config in this repository | Cloned checkpoint |
| --- | --- | --- |
| Black-Scholes | `configs/experiments/black_scholes_beta_cvae.yaml` | `~/Desktop/TimeCausalVAE/trained_models/BSprice_timestep_60/model_BetaCVAE_De_CLSTMRes_En_CLSTMRes_Prior_RealNVP_Con_Id_Dis_None_comment_None/BetaCVAE_training_2024-08-14_14-58-49/final_model` |
| Heston | `configs/experiments/heston_info_cvae.yaml` | `~/Desktop/TimeCausalVAE/trained_models/Hestonprice_timestep_60/model_InfoCVAE_De_CLSTMRes_En_CLSTMRes_Prior_RealNVP_Con_Id_Dis_None_comment_None/InfoCVAE_training_2024-09-16_18-19-18/final_model` |
| PDV4 | `configs/experiments/pdv_info_cvae.yaml` | `~/Desktop/TimeCausalVAE/trained_models/PDVPriceConFeature_timestep_60/model_InfoCVAE_De_CLSTMRes_En_CLSTMRes_Prior_RealNVP_Con_Id_Dis_None_comment_None/InfoCVAE_training_2024-08-21_16-06-50/final_model` |
| S&P500/VIX | `configs/experiments/sp500_vix_beta_cvae.yaml` | `~/Desktop/TimeCausalVAE/trained_models/SP500VIX_timestep_60/model_BetaCVAE_De_CLSTMRes_En_CLSTMRes_Prior_RealNVP_Con_Id_Dis_None_comment_None/BetaCVAE_training_2024-08-22_17-23-52/final_model` |

## Path Metrics

Lower is better for the path metrics below. Blank entries mean that the evaluator did not compute
that metric.

| Experiment | Candidate | Role | Status | Runtime, s | MMD | SWD | Returns W1 | Terminal W1 | Volatility W1 | Drawdown W1 | Return AC L1 | Squared-return AC L1 | Balanced score |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Black-Scholes | `black_scholes_beta_cvae` | continuous selected baseline, supplemental cloned checkpoint | ok | - | 0.089052 | 0.056371 | - | - | - | - | - | - | - |
| Black-Scholes | `public_standard_vq_smoke_baseline` | public discrete baseline | `not_available` | - | - | - | - | - | - | - | - | - | - |
| Black-Scholes | `hidden128_conv_transformer_k3` | provisional best discrete | ok | 77.6 | 0.085255 | 0.056371 | - | 0.067674 | 0.004042 | - | - | - | 0.053335 |
| Heston | `heston_info_cvae` | continuous selected baseline, supplemental cloned checkpoint | ok | - | 0.071840 | 0.066884 | - | - | - | - | - | - | - |
| Heston | `standard_vq_additive_ar` | public and provisional discrete | ok | 68.9 | 0.049702 | 0.074397 | - | 0.051078 | 0.019140 | - | - | - | 0.048579 |
| PDV4 | `pdv_info_cvae` | continuous selected baseline, supplemental cloned checkpoint | ok | - | 0.165295 | 0.010630 | - | - | - | - | - | - | - |
| PDV4 | `conditional_standard_vq_additive_ar` | public and provisional discrete | ok | 69.5 | 1.170725 | 0.051673 | - | 0.057260 | 0.006154 | - | - | - | 0.321453 |
| S&P500/VIX | `sp500_vix_beta_cvae` | continuous selected baseline, local timestamped checkpoint | ok | 4.8 | 0.154422 | 0.008785 | - | - | - | - | - | - | - |
| S&P500/VIX | `sp500_vix_beta_cvae` | continuous selected baseline, supplemental cloned checkpoint | ok | - | 0.131414 | 0.008367 | - | - | - | - | - | - | - |
| S&P500/VIX | `conditional_standard_vq_additive_ar` | public discrete baseline | ok | 70.9 | 0.254177 | 0.013836 | 0.001012 | 0.010252 | 0.001504 | 0.010957 | 0.055395 | 0.039584 | 0.048340 |
| S&P500/VIX | `conditional_hidden128_conv_transformer_k3` | provisional best discrete | ok | 87.5 | 0.501580 | 0.013060 | 0.000976 | 0.018768 | 0.001151 | 0.003914 | 0.051504 | 0.040645 | 0.078950 |

The generic token-prior evaluator used for Black-Scholes, Heston, and PDV4 computed MMD, SWD,
terminal W1, and volatility W1. It did not compute returns W1, drawdown W1, return AC L1, or
squared-return AC L1. S&P500/VIX used the paper-style evaluator and produced the full market
metric set for both discrete candidates.

The supplemental continuous evaluator produced legacy continuous MMD and SWD only. It did not
compute terminal W1, volatility W1, drawdown W1, return AC L1, squared-return AC L1, or condition
bucket diagnostics for the continuous baselines.

## Token Metrics

Token metrics come from the non-smoke per-experiment selection run. Lower CE and perplexity are
better; higher accuracy and active-code count are better.

| Experiment | Candidate | Best eval CE | Best eval perplexity | Best eval accuracy | Tokenizer perplexity | Tokenizer active codes | Token dataset perplexity | Token dataset active codes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Black-Scholes | `standard_vq_additive_ar` | 2.421391 | 11.261929 | 0.178267 | 52.321541 | 64 | 52.045925 | 64 |
| Black-Scholes | `hidden128_conv_transformer_k3` | 2.419572 | 11.241549 | 0.181267 | 54.278244 | 64 | 53.104866 | 64 |
| Heston | `standard_vq_additive_ar` | 2.652545 | 14.195220 | 0.167333 | 50.723011 | 64 | 51.004353 | 64 |
| PDV4 | `conditional_standard_vq_additive_ar` | 1.104750 | 3.019962 | 0.566133 | 51.275726 | 57 | 50.671955 | 57 |
| S&P500/VIX | `conditional_standard_vq_additive_ar` | 0.914807 | 2.507927 | 0.647449 | 29.406109 | 62 | 39.055717 | 63 |
| S&P500/VIX | `conditional_hidden128_conv_transformer_k3` | 0.883431 | 2.426584 | 0.659917 | 51.726196 | 64 | 43.194336 | 60 |

## Experiment Decisions

### Black-Scholes

Selected continuous candidate: `configs/experiments/black_scholes_beta_cvae.yaml`. The aggregate
runner did not evaluate it because no in-repository continuous `final_model` directory was
available, but the supplemental cloned-repo run completed with MMD `0.089052` and SWD `0.056371`.

Selected discrete candidate: `hidden128_conv_transformer_k3`.

Reason: it is the only final-evaluable Black-Scholes discrete candidate in this pass and it also
had the best token-run CE among the approved candidates. The public smoke baseline was recognised
but not evaluated because the configured non-smoke prior directory
`outputs/token_prior/black_scholes_kmeans/prior/black_scholes_causal_token_prior_seed0` and
tokenizer directory `outputs/tokenizer_quality/kmeans/black_scholes_causal_vq_tokenizer_kmeans_seed0/`
were missing.

Remaining uncertainty: public baseline comparison, continuous metrics beyond MMD/SWD, drawdown W1,
returns W1, return AC L1, squared-return AC L1, and trained unconditioned tokenizer no-leakage
remain missing.

### Heston

Selected continuous candidate: `configs/experiments/heston_info_cvae.yaml`. The aggregate runner
did not evaluate it because no in-repository continuous `final_model` directory was available, but
the supplemental cloned-repo run completed with MMD `0.071840` and SWD `0.066884`.

Selected discrete candidate: `standard_vq_additive_ar`.

Reason: the standard additive candidate is both the public comparison candidate and the
provisional token-run selection. It passed path evaluation on the generic metrics and passed the
token-prior no-leakage check. It also had the best token-run CE among Heston candidates.

Remaining uncertainty: continuous metrics beyond MMD/SWD, drawdown W1, returns W1, return AC L1,
squared-return AC L1, and trained unconditioned tokenizer no-leakage remain missing.

### PDV4

Selected continuous candidate: `configs/experiments/pdv_info_cvae.yaml`. The aggregate runner did
not evaluate it because no in-repository continuous `final_model` directory was available, but the
supplemental cloned-repo run completed with MMD `0.165295` and SWD `0.010630`.

Selected discrete candidate: `conditional_standard_vq_additive_ar`.

Reason: the conditional standard additive candidate is both the public comparison candidate and
the provisional token-run selection. It passed generic path evaluation and all available
no-leakage checks. It also had the best token-run CE among PDV4 candidates.

Remaining uncertainty: continuous metrics beyond MMD/SWD, drawdown W1, returns W1, return AC L1,
squared-return AC L1, and PDV4 path-level condition-bucket model-selection criteria remain
missing. The generic evaluator did emit condition-bucket diagnostics, but not the full path metric
set required for registry promotion.

### S&P500/VIX

Selected continuous candidate: `configs/experiments/sp500_vix_beta_cvae.yaml`.

Selected discrete candidate for final promotion consideration: `conditional_standard_vq_additive_ar`
under the current balanced-market profile.

Reason: the hidden128 conv-transformer candidate retained the better token CE and improved SWD,
returns W1, volatility W1, drawdown W1, and return AC L1. However, the public standard additive
baseline had the lower balanced score and lower MMD, terminal W1, and squared-return AC L1 in the
paper-style evaluation. Because the selection rule keeps all metrics visible and does not promote
on token CE alone, the public standard additive baseline remains the better S&P500/VIX discrete
selection in this final pass.

Remaining uncertainty: notebook execution was not rerun. The balanced decision is sensitive to
the profile definition because the hidden128 candidate wins several component metrics while
losing MMD and terminal W1. The registry should not be updated until the profile choice and
notebook/reproduction status are explicitly accepted.

## Registry Metadata Status

`trained_models/model_registry.yaml` now contains lightweight metadata for the selected
continuous and discrete candidates, plus the optional S&P500/VIX hidden128 comparison candidate
needed for metric-sensitive dynamic selection. The registry records config paths, local checkpoint
conventions, sampling policy, metrics, missing metrics, no-leakage status, and caveats only.
Weights and generated outputs remain local and are not committed.

Full public promotion remains blocked by:

- missing full-profile continuous baseline metrics for Black-Scholes, Heston, and PDV4 beyond
  supplemental cloned-repo MMD/SWD;
- unavailable Black-Scholes public discrete checkpoint artefacts;
- missing generic-evaluator path metrics for Black-Scholes, Heston, and PDV4;
- unavailable trained unconditioned tokenizer no-leakage checks for Black-Scholes and Heston;
- notebook execution status not rerun for any experiment.

The S&P500/VIX results are the closest to promotion-ready because the continuous baseline,
public discrete baseline, provisional hidden128 candidate, full paper-style path metrics, and
available no-leakage checks all completed. Even there, the final discrete choice should remain
documented as the public standard additive baseline unless a different profile is explicitly
chosen and justified.
