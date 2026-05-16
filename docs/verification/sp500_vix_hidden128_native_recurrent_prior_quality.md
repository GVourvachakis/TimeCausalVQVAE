# S&P500/VIX Hidden128 Native Recurrent Prior Quality

Status: trained and evaluated on 2026-05-16. No source code was changed for this evaluation, no
dependency was added, no tokenizer code was changed, and no public default was changed.

## Run

Prior config:

```text
configs/experiments/sp500_vix_causal_token_prior_hidden128_native_recurrent.yaml
```

Prior type:

```text
native_recurrent
```

Sampling policy:

```text
temperature=1.0
top_k=none
n_sample=1000
seed=99
```

W&B failed before training began. The requested W&B command entered a network retry loop and then
raised:

```text
wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec.
```

The run was therefore repeated with `--no-wandb`. No W&B URL exists for this run.

Training output:

```text
outputs/sp500_vix_discrete/token_prior/hidden128_native_recurrent/sp500_vix_causal_token_prior_hidden128_native_recurrent_seed99
```

Training runtime was `132.194` seconds on CPU. CUDA initialisation emitted the known local driver
mismatch warning, so this run used CPU execution.

## Token Likelihood

The best checkpoint was also the final checkpoint at epoch 100.

| Metric | Value |
| --- | ---: |
| Train CE | 1.352454 |
| Train accuracy | 0.479962 |
| Train perplexity | 3.867528 |
| Eval CE | 1.351029 |
| Eval accuracy | 0.480260 |
| Eval perplexity | 3.890610 |

The native recurrent prior improves substantially over its source-smoke likelihood, but it remains
weaker than the hidden128 causal conv-transformer k3 likelihood reported previously
(`eval_ce=1.129030`, `eval_accuracy=0.557089`, `eval_perplexity=3.107617`).

## Decoded Token-Prior Metrics

Evaluation output:

```text
outputs/sp500_vix_discrete/token_prior/hidden128_native_recurrent/evaluation_best_temp10_topknone
```

| Metric | Native recurrent |
| --- | ---: |
| Sampled active codes | 64/64 |
| Sampled token perplexity | 46.356960 |
| Real token perplexity | 42.782639 |
| MMD | 0.237231 |
| SWD | 0.013015 |
| Terminal W1 | 0.019814 |
| Volatility W1 | 0.004198 |
| Marginal code L1 | 0.120333 |
| Transition matrix L1 | 0.248042 |
| Run-length W1 | 0.239307 |
| Real run-length mean | 1.614031 |
| Sampled run-length mean | 1.548067 |

The sampled codebook usage is healthy and does not show token collapse. The decoded financial
metrics are weak: volatility and terminal-return errors are much larger than the current hidden128
conv-transformer primary setting, and the run-length distance is especially high.

## Paper-Style Metrics

Paper-style output:

```text
outputs/sp500_vix_discrete/paper_style_hidden128_native_recurrent_temp10_topknone
```

Profile is:

```text
MMD + SWD + terminal_return_wasserstein + volatility_wasserstein
```

| Source | Profile | MMD | SWD | Terminal W1 | Volatility W1 | Returns W1 | Drawdown W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native recurrent hidden128 | 0.250494 | 0.218727 | 0.011493 | 0.016306 | 0.003969 | 0.001509 | 0.019013 |
| continuous BetaCVAE reference | 0.172891 | 0.154421 | 0.008785 | 0.009051 | 0.000634 | 0.000602 | 0.007667 |

Autocorrelation and tail diagnostics:

| Source | Return AC L1 | Squared-return AC L1 | Tail < real q001 | Tail > real q999 |
| --- | ---: | ---: | ---: | ---: |
| native recurrent hidden128 | 0.034853 | 0.049929 | 0.004492 | 0.010983 |
| continuous BetaCVAE reference | 0.025972 | 0.029462 | 0.002525 | 0.003525 |

The recurrent prior produces excessive tails and volatility. The generated path range is
`[0.279571, 1.321818]`; the largest sampled terminal-return outliers include paths with terminal
returns above `1.29` and `2.69`, which is not acceptable for the current hidden128 research claim.

## VIX-Bucket Diagnostics

Paper-style bucket results:

| Bucket | VIX range | Source | MMD | SWD | Vol. W1 | Terminal W1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| very_low | 0.110533-0.139074 | native recurrent | 0.535413 | 0.020481 | 0.003657 | 0.024065 |
| very_low | 0.110533-0.139074 | BetaCVAE | 0.345894 | 0.012328 | 0.001162 | 0.012705 |
| low | 0.139074-0.156730 | native recurrent | 0.269868 | 0.009806 | 0.002666 | 0.009663 |
| low | 0.139074-0.156730 | BetaCVAE | 0.314091 | 0.012184 | 0.000721 | 0.011612 |
| mid | 0.157093-0.177289 | native recurrent | 0.295484 | 0.011358 | 0.003863 | 0.032278 |
| mid | 0.157093-0.177289 | BetaCVAE | 0.423118 | 0.016197 | 0.001040 | 0.014008 |
| high | 0.177410-0.209094 | native recurrent | 0.308235 | 0.015777 | 0.003682 | 0.027946 |
| high | 0.177410-0.209094 | BetaCVAE | 0.244890 | 0.010479 | 0.000495 | 0.009579 |
| very_high | 0.209336-0.492684 | native recurrent | 0.324764 | 0.020315 | 0.005975 | 0.019697 |
| very_high | 0.209336-0.492684 | BetaCVAE | 0.198580 | 0.011254 | 0.000684 | 0.014851 |

The recurrent prior is not uniformly poor by MMD, but the volatility W1 is worse than the
continuous reference in every VIX bucket. Terminal W1 is also weak in the mid, high, and very-high
VIX buckets.

## Transition And Run-Length Diagnostics

Paper-style token diagnostics:

| Metric | Native recurrent |
| --- | ---: |
| Marginal code L1 | 0.107200 |
| Transition matrix L1 | 0.246062 |
| Run-length W1 | 0.238527 |
| Run-length histogram L1 | 0.060635 |
| Real run-length mean | 1.614031 |
| Sampled run-length mean | 1.548587 |
| Sampled active codes | 64/64 |
| Sampled token perplexity | 45.977097 |

The transition and run-length diagnostics are materially worse than hidden128 conv-transformer k3,
which reported transition L1 `0.193243` and run-length W1 `0.050189`. The recurrent prior keeps
broad code usage but does not preserve token persistence.

## Comparison

| Model | Profile | MMD | SWD | Terminal W1 | Vol W1 | Sq-return AC L1 | Transition L1 | Run-length W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| continuous BetaCVAE | 0.172891 | 0.154421 | 0.008785 | 0.009051 | 0.000634 | 0.029462 | n/a | n/a |
| hidden128 conv-transformer k3 | 0.186725 | 0.169510 | 0.010918 | 0.005086 | 0.001210 | 0.050871 | 0.193243 | 0.050189 |
| hidden128 native recurrent | 0.250494 | 0.218727 | 0.011493 | 0.016306 | 0.003969 | 0.049929 | 0.246062 | 0.238527 |
| hidden128 additive | 0.242020 | 0.229078 | 0.007175 | 0.004509 | 0.001258 | 0.060885 | n/a | n/a |
| promoted standard VQ + additive AR | 0.298020 | 0.279341 | 0.007674 | 0.009817 | 0.001188 | 0.041300 | n/a | n/a |

The native recurrent prior is worse than the hidden128 conv-transformer k3 result by the main
paper-style profile and by the transition/run-length diagnostics that motivated the branch. It is
also slightly worse than the former hidden128 additive prior by the main profile. It remains better
than the promoted public baseline on MMD and the main profile, but it is not a good replacement for
the current hidden128 research prior.

## Decision

Reject this exact native recurrent GRU-128, one-layer configuration as the next hidden128 prior.

Do not replace the hidden128 conv-transformer k3 research model. Do not change public defaults.

The useful result is negative but informative: a simple GRU state update can train and sample
without leakage, and it avoids sampled-token collapse, but this configuration overproduces tail
events and fails to improve token persistence. The next most reasonable follow-up is not a
sampling ablation of this checkpoint. The primary failure is broad enough that temperature or top-k
alone is unlikely to fix it without hiding the tail issue. Prefer returning to the conv-transformer
as the current best discrete model.

If the recurrent direction is revisited, tune architecture before sampling: test larger hidden
state, more recurrent layers with dropout, or a more controlled native SSM-style transition, while
keeping the same no-leakage and stepwise-equivalence checks.
