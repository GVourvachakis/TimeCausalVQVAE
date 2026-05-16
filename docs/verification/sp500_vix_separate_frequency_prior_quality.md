# S&P500/VIX Separate Frequency Prior Quality

Status: non-smoke quality evaluation for the separate low/high hierarchical causal token prior.
This run does not change model code and does not add GroupedRVQ, MGVQ, signatures, diffusion,
cross-attention, or new objectives.

## Inputs

Config:

```text
configs/experiments/sp500_vix_separate_frequency_hierarchical_prior_alpha02.yaml
```

Tokenizers:

```text
outputs/sp500_vix_discrete/separate_frequency_tokenizer/low/sp500_vix_freq_low_alpha02_tokenizer
outputs/sp500_vix_discrete/separate_frequency_tokenizer/high/sp500_vix_freq_high_alpha02_tokenizer
```

Paired token data:

```text
outputs/sp500_vix_discrete/token_prior/separate_freq_alpha02_tokens
```

Trained prior:

```text
outputs/sp500_vix_discrete/token_prior/separate_freq_alpha02/sp500_vix_separate_frequency_hierarchical_prior_alpha02_seed0
```

Evaluation output:

```text
outputs/sp500_vix_discrete/token_prior/separate_freq_alpha02_eval
```

The evaluation used `n_sample = 1000`, seed `99`, temperature `0.8`, and `top_k = 40`.

## W&B

The requested online W&B run failed before training began:

```text
wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec.
```

The prior was then trained with `--no-wandb`. No W&B URL is available for this run.

## Training Result

The fallback CPU training completed 100 epochs.

| Epoch | Eval CE | Eval accuracy | Eval perplexity | Low CE | Low accuracy | Low perplexity | High CE | High accuracy | High perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 2.42348392 | 0.61014110 | 11.37620649 | 0.31990123 | 0.87684167 | 1.37801099 | 2.10358272 | 0.34344051 | 8.23364268 |

The best checkpoint is epoch 100:

```text
outputs/sp500_vix_discrete/token_prior/separate_freq_alpha02/sp500_vix_separate_frequency_hierarchical_prior_alpha02_seed0/best_model
```

Teacher-forced evaluation on the first 1000 eval windows gives slightly lower CE because it uses
the evaluation subset used for sampling:

| Metric | Value |
| --- | ---: |
| Aggregate CE | 2.31841135 |
| Aggregate perplexity | 10.15952110 |
| Aggregate accuracy | 0.62465835 |
| Low CE | 0.29097256 |
| Low perplexity | 1.33772790 |
| Low accuracy | 0.88889998 |
| High CE | 2.02743888 |
| High perplexity | 7.59461069 |
| High accuracy | 0.36041668 |
| Same-time true-pair perplexity | 1128.99841309 |

The likelihood result is not enough to clear the prior-quality gate because sampling collapses.

## Token Sampling Diagnostics

Sampled stream shapes:

```text
sampled_low_tokens:  [1000, 60]
sampled_high_tokens: [1000, 60]
sampled_tokens:      [1000, 60, 2]
decoded_paths:       [1000, 60, 1]
```

| Stream | Real active | Sampled active | Real perplexity | Sampled perplexity | Marginal L1 | Transition L1 | Run-length W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | 55 / 64 | 33 / 64 | 40.17523193 | 4.10020781 | 1.86393332 | 1.17757928 | 1.20842087 |
| High | 61 / 64 | 43 / 64 | 43.50838852 | 6.62247515 | 1.76313341 | 1.44845176 | 0.34646890 |
| Same-time pairs | 2277 / 4096 | 404 / 4096 | 1128.99841309 | 25.49054718 | 1.93339992 | 0.64975667 | 0.18156755 |

The sampled low stream is the main bottleneck: low-token sampled perplexity falls from real
`40.1752` to `4.1002`. The high stream also collapses, and same-time pair support drops from
`2277` observed eval pairs to `404` sampled pairs.

## Composed-Path Metrics

Low and high decoded paths were composed as `low_hat + high_hat`.

| Model / setting | MMD | SWD | Volatility W1 | Sq-return AC L1 | Flat sq-return AC L1 | Drawdown W1 | Terminal W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Separate low/high hierarchical, temp 0.8 top-k 40 | 2.33316898 | 0.14535846 | 0.04670410 | 0.04985855 | 0.15767361 | 0.07369982 | 0.71774405 |
| Joint EMA alpha 0.2, temp 0.8 top-k 40 | 0.29730234 | 0.00940700 | 0.00113100 | 0.03540378 | 0.02522561 | 0.00707990 | 0.00666096 |
| Promoted baseline, temp 0.8 top-k 40 | 0.27934083 | 0.00767375 | 0.00118835 | 0.04129972 | 0.12882016 | 0.01050232 | 0.00981713 |
| Hidden128, temp 0.8 top-k 20 | 0.22907834 | 0.00717505 | 0.00125777 | 0.06088475 | 0.07886228 | 0.00891025 | 0.00450928 |
| Continuous BetaCVAE | 0.15442121 | 0.00878550 | 0.00063360 | 0.02946163 | 0.02914374 | 0.00766744 | 0.00905099 |

The separate-frequency prior is not competitive with any retained reference. It is worse than
the joint EMA prior, the promoted discrete baseline, the hidden128 reference, and the continuous
BetaCVAE on MMD, SWD, volatility W1, drawdown W1, and terminal W1. Its terminal-return W1 is two
orders of magnitude worse than the retained discrete references.

## VIX-Bucket Diagnostics

| VIX bucket | Samples | VIX range | MMD | SWD | Volatility W1 | Terminal W1 | Low active | Low perplexity | High active | High perplexity | Pair active | Pair perplexity |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| very_low | 200 | 0.110533 to 0.139074 | 2.44054151 | 0.15119998 | 0.04533847 | 0.62899017 | 25 | 4.24111843 | 37 | 6.48910570 | 245 | 25.27387810 |
| low | 200 | 0.139074 to 0.156730 | 2.35086513 | 0.13903838 | 0.04596070 | 0.68430883 | 25 | 4.18430853 | 40 | 6.52955389 | 246 | 25.24592781 |
| mid | 200 | 0.157093 to 0.177289 | 2.34881544 | 0.14043723 | 0.04665341 | 0.75046510 | 25 | 4.03781462 | 38 | 6.50694990 | 229 | 24.45086861 |
| high | 200 | 0.177410 to 0.209094 | 2.30303454 | 0.14241543 | 0.04677859 | 0.74515718 | 25 | 4.01693296 | 40 | 6.57356071 | 233 | 24.58846474 |
| very_high | 200 | 0.209336 to 0.492684 | 2.31544447 | 0.15536655 | 0.04878933 | 0.77979910 | 26 | 3.94271207 | 38 | 6.87615538 | 232 | 25.35212898 |

The VIX buckets do not reveal a local success case. Composed-path distribution metrics are poor
in every bucket, and low-token sampled usage stays around only 25 to 26 active codes per bucket.

## Continuous Baseline

The requested continuous checkpoint loaded successfully:

```text
outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47/final_model
```

Its metrics match the retained reference:

| Metric | Value |
| --- | ---: |
| MMD | 0.15442121 |
| SWD | 0.00878550 |
| Volatility W1 | 0.00063360 |
| Terminal W1 | 0.00905099 |
| Drawdown W1 | 0.00766744 |
| Squared-return AC L1 | 0.02946163 |

## Decision

Decision: reject the current separate-frequency hierarchical prior quality result.

Rationale:

- The likelihood improved, but sampled token usage collapses in both streams.
- Same-time low/high pair support is far below the real eval support.
- Composed generated paths fail MMD, SWD, volatility W1, drawdown W1, and terminal W1 guardrails.
- The result is worse than the joint EMA alpha 0.2 prior, which was already not promoted as a
  broad replacement.
- The branch does not clear the stage gate that required residual improvement without the large
  MMD/SWD regression of joint EMA.

No sampling ablation is justified as the next default step unless the objective is narrowly to
debug the sampling collapse. Do not move to GroupedRVQ from this result alone. The empirical
conclusion for this branch is that separate low/high tokenizers plus the first hierarchical
causal prior do not improve the S&P500/VIX generation quality.

## Artifacts

The narrow adapter script used for this evaluation is:

```text
scripts/evaluate_sp500_vix_separate_frequency_paper_style.py
```

It writes:

```text
outputs/sp500_vix_discrete/token_prior/separate_freq_alpha02_eval/separate_frequency_prior_summary.json
outputs/sp500_vix_discrete/token_prior/separate_freq_alpha02_eval/separate_frequency_prior_summary.md
outputs/sp500_vix_discrete/token_prior/separate_freq_alpha02_eval/separate_frequency_prior_samples.pt
```
