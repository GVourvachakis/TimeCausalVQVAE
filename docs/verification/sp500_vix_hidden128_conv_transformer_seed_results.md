# Hidden128 Conv-Transformer Seed Robustness Results

Status: seed robustness completed for the selected hidden128 causal conv-transformer k3 prior. No
model code was changed, no tokenizer code was changed, and no k5 or `dilations124` run was
started.

## Manifest

- Seed configs:
  `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer_seed1.yaml` and
  `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer_seed2.yaml`
- Prior type: `causal_conv_transformer`
- Conv front-end: kernel size `3`, dilations `[1, 2]`
- Tokenizer:
  `outputs/sp500_vix_discrete/vq_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_hidden128_seed99`
- Output root:
  `outputs/sp500_vix_discrete/token_prior/hidden128_conv_transformer_seed_ablation`
- Aggregate outputs:
  `token_prior_candidate_ablation_summary.json` and
  `token_prior_candidate_ablation_summary.csv`
- Evaluation setting: temperature `1.0`, unrestricted top-k, `n_sample=1000`, evaluation seed `99`

Profile is:

```text
MMD + SWD + terminal_return_wasserstein + volatility_wasserstein
```

Lower is better.

## W&B And Runtime

Both W&B-enabled training subprocesses failed before training with a W&B initialisation timeout:

```text
wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec.
```

The runner retried both configs with `--no-wandb` and recorded
`train_status=passed_no_wandb_fallback`, `wandb_fallback_used=true`, and
`wandb_failure=wandb_training_failed_exit_1`. No W&B URLs were produced.

| Seed | Train runtime | End-to-end runtime | W&B |
| --- | ---: | ---: | --- |
| 1 | 1278.256 s | 1542.798 s | failed, no-W&B fallback |
| 2 | 1418.061 s | 1680.204 s | failed, no-W&B fallback |

The end-to-end runtime includes training, best-checkpoint evaluation, and paper-style diagnostics.

## Token Likelihood

Both seeds selected epoch `100` as the best checkpoint.

| Seed | Eval CE | Eval accuracy | Eval perplexity |
| --- | ---: | ---: | ---: |
| 1 | 1.130656 | 0.557692 | 3.112165 |
| 2 | 1.124913 | 0.557184 | 3.094001 |

The likelihoods are close to each other and close to the seed99 k3 run
(`eval_ce=1.129030`, `eval_accuracy=0.557089`, `eval_perplexity=3.107617`). The training objective
therefore looks stable across seeds.

## Decoded Token-Prior Metrics

| Seed | Profile | MMD | SWD | Terminal W1 | Vol W1 | Active codes | Token perplexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.269447 | 0.246024 | 0.011283 | 0.011060 | 0.001079 | 63/64 | 45.094280 |
| 2 | 0.211065 | 0.197168 | 0.008291 | 0.004360 | 0.001245 | 64/64 | 43.584637 |

Seed2 is close to the selected seed99 k3 result on decoded profile, while seed1 is materially
worse because MMD and terminal W1 both increase. Token usage remains broad for both seeds.

## Transition And Run-Length Diagnostics

| Seed | Marginal code L1 | Transition L1 | Run-length W1 | Paper-style transition L1 | Paper-style run-length W1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.187700 | 0.190801 | 0.130725 | 0.202723 | 0.121859 |
| 2 | 0.132767 | 0.222335 | 0.083499 | 0.190676 | 0.101200 |

Seed1 has better decoded transition L1 than seed2, but worse marginal usage and run-length W1.
Paper-style token diagnostics reverse the transition ordering and keep seed2 better on run length.
Relative to the hidden128 additive prior, both seeds preserve the main k3 advantage on run-length
matching, but the transition improvement is not uniform across evaluation views.

## Paper-Style Metrics

| Seed | Profile | MMD | SWD | Terminal W1 | Vol W1 | Returns W1 | Drawdown W1 | Return AC L1 | Sq-return AC L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.237086 | 0.216701 | 0.010314 | 0.008959 | 0.001113 | 0.000632 | 0.006959 | 0.028718 | 0.051129 |
| 2 | 0.214707 | 0.199984 | 0.009162 | 0.004418 | 0.001143 | 0.000616 | 0.006663 | 0.031265 | 0.048157 |

Across seed99, seed1, and seed2, the k3 paper-style profile range is `0.186725` to `0.237086`,
with an approximate mean of `0.212839`. Both new seeds remain better than the hidden128 additive
profile, but seed1 is only marginally better.

## Comparison

| Model | Profile | MMD | SWD | Terminal W1 | Vol W1 | Drawdown W1 | Sq-return AC L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k3 seed99 selected | 0.186725 | 0.169510 | 0.010918 | 0.005086 | 0.001210 | 0.007687 | 0.050871 |
| k3 seed1 | 0.237086 | 0.216701 | 0.010314 | 0.008959 | 0.001113 | 0.006959 | 0.051129 |
| k3 seed2 | 0.214707 | 0.199984 | 0.009162 | 0.004418 | 0.001143 | 0.006663 | 0.048157 |
| hidden128 additive, temp 0.8, top-k 20 | 0.242020 | 0.229078 | 0.007175 | 0.004509 | 0.001258 | 0.008910 | 0.060885 |
| promoted standard VQ + additive AR | 0.298020 | 0.279341 | 0.007674 | 0.009817 | 0.001188 | 0.010502 | 0.041300 |
| continuous BetaCVAE | 0.172891 | 0.154421 | 0.008785 | 0.009051 | 0.000634 | 0.007667 | 0.029462 |

The seed robustness run supports the core k3 improvement over the hidden128 additive prior on
paper-style profile, MMD, drawdown W1, and squared-return autocorrelation. The evidence is weaker
for SWD, terminal W1, and volatility W1. The continuous BetaCVAE remains stronger on the primary
profile and most stylised diagnostics.

## Secondary Setting

The optional `temperature=1.0, top_k=20` seed1/seed2 evaluation was not run. The ablation runner
does not provide a clean evaluation-only mode for existing best checkpoints, and rerunning it would
retrain the seeds, which was outside the prompt. Keep `top_k=20` as a targeted follow-up only if a
future evaluation-only command is added or run manually against the saved best checkpoints.

## Decision

The k3 conv-transformer is directionally robust but seed-sensitive.

Keep k3 as the best hidden128 discrete research prior for now because both new seeds improve the
paper-style profile over the hidden128 additive prior and remain clearly better than the promoted
standard-VQ additive baseline on the profile. Do not treat the seed99 result as a guaranteed
typical outcome: seed1 regresses close to the additive baseline, and the profile range is driven
mostly by MMD variation.

Run more seeds before any public-baseline promotion or thesis-level claim of robust dominance.
Stopping the prior branch now would be premature because the mean seed profile remains better than
the additive hidden128 prior. If additional seeds confirm the same plateau or reveal larger
instability, the next branch should be a Mamba/SSM package-compatibility check rather than another
tokenizer-family change.
