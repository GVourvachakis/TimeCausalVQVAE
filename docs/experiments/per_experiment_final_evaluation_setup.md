# Per-Experiment Final-Evaluation Setup

## Scope

This document records the final-evaluation setup added after the provisional token-run audit. It
does not train non-smoke models, update `trained_models/model_registry.yaml`, add new model
families, or commit generated outputs.

The setup adds:

- `src/time_causal_vae/experiments/selection_profiles.py`, which scores visible path metrics
  under distributional, tail-risk, sequential-dependence, and balanced-market profiles;
- `scripts/run_per_experiment_final_evaluation.py`, which plans or runs final path-metric and
  no-leakage evaluation for selected continuous baselines, public discrete baselines, and
  provisional best discrete candidates.

## Selection Profiles

All profiles are lower-is-better and keep component metrics visible in the aggregate output.
Missing metrics are reported explicitly as warnings. The profile scorer refuses to select from
MMD alone unless the caller explicitly enables that behaviour.

| Profile | Components |
| --- | --- |
| `distributional` | MMD, SWD, returns W1 where available. |
| `tail_risk` | Terminal W1, volatility W1, drawdown W1. |
| `sequential_dependence` | Return autocorrelation L1 and squared-return autocorrelation L1. |
| `balanced_market` | Average lower-is-better rank over all available path metrics. |

The profile score is a decision aid rather than a replacement for the full metric table. Registry
promotion should still record the selected profile, component metrics, missing metrics, sampling
policy, and no-leakage status.

## Runner Capabilities

`scripts/run_per_experiment_final_evaluation.py` supports:

- reading provisional discrete selections from
  `outputs/per_experiment_selection/selection_results.json`, with documented fallbacks from
  `docs/experiments/per_experiment_model_selection_results.md`;
- planning selected continuous baseline evaluation, public discrete baseline evaluation where
  available, and provisional best discrete evaluation;
- evaluating generic token-prior candidates with `tcvae-evaluate-token-prior`;
- evaluating S&P500/VIX candidates with `scripts/evaluate_sp500_vix_paper_style.py`;
- running no-leakage checks for causal-convolution source logic, token priors, and conditional
  tokenizers;
- writing `final_evaluation_plan.json` and `final_evaluation_plan.csv` under the selected output
  directory;
- preserving `not_available` status when a required evaluator or checkpoint path is absent.

Continuous baseline evaluation is planned through the existing `tcvae-evaluate` CLI. A non-smoke
run must provide local `final_model` directories with
`--continuous-model-dir EXPERIMENT=PATH` unless the runner can discover them under `outputs/`.

## Metrics Supported Per Experiment

| Experiment | Continuous baseline support | Discrete support | Full path metrics supported now | Missing metrics in generic path evaluator |
| --- | --- | --- | --- | --- |
| Black-Scholes | Planned through `tcvae-evaluate` when a local `final_model` path is supplied. | `tcvae-evaluate-token-prior` for the provisional hidden128 conv-transformer candidate. | MMD, SWD, terminal W1, volatility W1. | Returns W1, drawdown W1, return AC L1, squared-return AC L1. |
| Heston | Planned through `tcvae-evaluate` when a local `final_model` path is supplied. | `tcvae-evaluate-token-prior` for the standard additive candidate. | MMD, SWD, terminal W1, volatility W1. | Returns W1, drawdown W1, return AC L1, squared-return AC L1. |
| PDV4 | Planned through `tcvae-evaluate` when a local `final_model` path is supplied. | `tcvae-evaluate-token-prior` for the conditional standard additive candidate. | MMD, SWD, terminal W1, volatility W1. | Returns W1, drawdown W1, return AC L1, squared-return AC L1, and PDV4 path-level condition-bucket diagnostics. |
| S&P500/VIX | Required by `scripts/evaluate_sp500_vix_paper_style.py` for paired paper-style comparison. | Public standard additive and provisional hidden128 conv-transformer candidates. | MMD, SWD, returns W1, terminal W1, volatility W1, drawdown W1, return AC L1, squared-return AC L1, and VIX-bucket path diagnostics. | None in the paper-style evaluator, provided the continuous checkpoint path is available. |

The generic evaluator intentionally records missing metrics rather than inferring them from token
CE or tokenizer summaries. Those token-run quantities remain useful diagnostics, but they are not
path metrics.

## No-Leakage Support

| Experiment | Causal-convolution check | Token-prior check | Tokenizer check |
| --- | --- | --- | --- |
| Black-Scholes | Supported through `scripts/check_causal_conv_no_leakage.py`. | Supported through `scripts/check_conditional_token_prior_no_leakage.py`. | Trained unconditioned tokenizer no-leakage script is not available; runner records `not_available`. |
| Heston | Supported through `scripts/check_causal_conv_no_leakage.py`. | Supported through `scripts/check_conditional_token_prior_no_leakage.py`. | Trained unconditioned tokenizer no-leakage script is not available; runner records `not_available`. |
| PDV4 | Supported through `scripts/check_causal_conv_no_leakage.py`. | Supported through `scripts/check_conditional_token_prior_no_leakage.py`. | Supported through `scripts/check_conditional_vq_tokenizer_no_leakage.py`. |
| S&P500/VIX | Supported through `scripts/check_causal_conv_no_leakage.py`. | Supported through `scripts/check_conditional_token_prior_no_leakage.py`. | Supported through `scripts/check_conditional_vq_tokenizer_no_leakage.py`. |

The causal-convolution check is source-level. The token-prior and conditional-tokenizer checks are
checkpoint-specific once the local ignored checkpoints are present.

## Dry-Run Result

The dry-run command was:

```bash
poetry run python scripts/run_per_experiment_final_evaluation.py \
  --experiments black_scholes heston pdv sp500_vix \
  --output-dir outputs/per_experiment_final_evaluation_dry \
  --dry-run \
  --n-sample 128 \
  --profile balanced_market
```

It completed successfully and wrote:

- `outputs/per_experiment_final_evaluation_dry/final_evaluation_plan.json`;
- `outputs/per_experiment_final_evaluation_dry/final_evaluation_plan.csv`.

The dry-run planned 24 targets:

- 4 continuous selected-baseline targets, all marked `not_available` until local `final_model`
  paths are supplied;
- 5 discrete path-evaluation targets, covering the public or standard comparison candidate and
  the provisional best candidate for each experiment;
- 15 no-leakage targets, including causal-convolution and token-prior checks for all discrete
  candidates, conditional-tokenizer checks for PDV4 and S&P500/VIX candidates, and explicit
  `not_available` tokenizer status for unconditioned Black-Scholes and Heston candidates.

No non-smoke training or path evaluation was executed in the dry-run.

## Next Non-Smoke Command

After confirming the required local checkpoint directories, run:

```bash
poetry run python scripts/run_per_experiment_final_evaluation.py \
  --experiments black_scholes heston pdv sp500_vix \
  --output-dir outputs/per_experiment_final_evaluation \
  --n-sample 1000 \
  --profile balanced_market \
  --continuous-model-dir black_scholes=outputs/<black_scholes_continuous>/final_model \
  --continuous-model-dir heston=outputs/<heston_continuous>/final_model \
  --continuous-model-dir pdv=outputs/<pdv_continuous>/final_model \
  --continuous-model-dir sp500_vix=outputs/<sp500_vix_continuous>/final_model
```

The non-smoke run should remain under ignored `outputs/per_experiment_final_evaluation/`. Registry
promotion should wait until the resulting JSON and CSV contain path metrics, no-leakage statuses,
and any notebook or reproduction evidence required by
`docs/experiments/per_experiment_model_selection_gap_analysis.md`.
