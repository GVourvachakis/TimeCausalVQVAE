# S&P500/VIX Best Discrete Research Model Decision

Status: current-status decision after hidden128 causal conv-transformer seed robustness. No code was
implemented and no models were trained for this decision.

Profile is:

```text
MMD + SWD + terminal_return_wasserstein + volatility_wasserstein
```

Lower is better.

## Public Baseline

The public discrete baseline remains the promoted standard-VQ tokenizer with the additive VIX-only
causal AR prior:

- tokenizer config: `configs/experiments/sp500_vix_causal_vq_tokenizer.yaml`;
- prior config: `configs/experiments/sp500_vix_causal_token_prior_additive.yaml`;
- interface: one 64-code index per time step;
- condition: scalar VIX, additive injection;
- sampling: temperature `0.8`, `top_k=40`.

| Metric | Standard VQ + additive AR |
| --- | ---: |
| Prior eval CE | 0.914807 |
| Prior eval accuracy | 0.647449 |
| Prior eval perplexity | 2.507927 |
| Paper-style profile | 0.298020 |
| MMD | 0.279341 |
| SWD | 0.007674 |
| Terminal W1 | 0.009817 |
| Volatility W1 | 0.001188 |
| Squared-return AC L1 | 0.041300 |

This remains the public baseline because it is simple, established, and still has the strongest
token likelihood among the compared discrete priors.

## Best Discrete Research Candidate

The best current discrete research candidate is the hidden128 standard-VQ tokenizer with the
causal conv-transformer k3 prior:

- tokenizer config: `configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml`;
- prior config:
  `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml`;
- prior type: `causal_conv_transformer`;
- conv front-end: two causal convolution layers, kernel size `3`, dilations `[1, 2]`;
- transformer trunk: the existing causal additive VIX-conditioned token prior trunk;
- primary sampling: temperature `1.0`, `top_k=none`.

The selected seed99 result is:

| Metric | Hidden128 conv-transformer k3 |
| --- | ---: |
| Prior eval CE | 1.129030 |
| Prior eval accuracy | 0.557089 |
| Prior eval perplexity | 3.107617 |
| Paper-style profile | 0.186725 |
| MMD | 0.169510 |
| SWD | 0.010918 |
| Terminal W1 | 0.005086 |
| Volatility W1 | 0.001210 |
| Returns W1 | 0.000681 |
| Drawdown W1 | 0.007687 |
| Return AC L1 | 0.027889 |
| Squared-return AC L1 | 0.050871 |
| Transition L1 | 0.193243 |
| Run-length W1 | 0.050189 |
| Sampled active codes | 63/64 |
| Sampled token perplexity | 45.280468 |

This is the best discrete research result by the main paper-style profile and by local token
persistence diagnostics. It is not uniformly best on all metrics: hidden128 additive remains better
on SWD and terminal W1, and the public baseline remains better on token likelihood and
squared-return autocorrelation.

## Seed Robustness

Seed robustness was run for seed1 and seed2 at the selected primary sampling setting,
temperature `1.0` and unrestricted top-k. W&B initialisation timed out for both seeds, and the
runner retried both with `--no-wandb`.

| Seed | Eval CE | Eval accuracy | Eval perplexity | Paper profile | MMD | SWD | Terminal W1 | Vol W1 | Sq-return AC L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 99 | 1.129030 | 0.557089 | 3.107617 | 0.186725 | 0.169510 | 0.010918 | 0.005086 | 0.001210 | 0.050871 |
| 1 | 1.130656 | 0.557692 | 3.112165 | 0.237086 | 0.216701 | 0.010314 | 0.008959 | 0.001113 | 0.051129 |
| 2 | 1.124913 | 0.557184 | 3.094001 | 0.214707 | 0.199984 | 0.009162 | 0.004418 | 0.001143 | 0.048157 |

The k3 result holds directionally across seed99, seed1, and seed2. All three k3 seeds beat the
hidden128 additive prior's selected profile of `0.242020`, and all remain clearly ahead of the
promoted public baseline profile of `0.298020`.

The result is still seed-sensitive. The profile range is `0.186725` to `0.237086`, and seed1 is
only marginally better than hidden128 additive. The variation is driven mainly by MMD and terminal
W1, while token likelihood is stable.

## Sampling Policy

Primary reporting policy:

```text
temperature=1.0
top_k=none
```

This setting gives the best conv-transformer profile and the strongest MMD and run-length
diagnostics in the sampling grid.

Secondary policy:

```text
temperature=1.0
top_k=20
```

Use this only when the comparison prioritises volatility W1, transition L1, or squared-return
autocorrelation over the main profile. It improves those diagnostics relative to unrestricted
sampling, but the profile worsens from `0.186725` to `0.205630` because SWD and terminal W1
increase.

## Cross-Model Comparison

| Model | Profile | MMD | SWD | Terminal W1 | Vol W1 | Drawdown W1 | Sq-return AC L1 | Main status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| continuous BetaCVAE | 0.172891 | 0.154421 | 0.008785 | 0.009051 | 0.000634 | 0.007667 | 0.029462 | strongest non-discrete reference |
| hidden128 conv k3 seed99, temp 1.0, top-k none | 0.186725 | 0.169510 | 0.010918 | 0.005086 | 0.001210 | 0.007687 | 0.050871 | best discrete research result |
| hidden128 conv k3 seed2, temp 1.0, top-k none | 0.214707 | 0.199984 | 0.009162 | 0.004418 | 0.001143 | 0.006663 | 0.048157 | robustness seed, still strong |
| hidden128 conv k3 seed1, temp 1.0, top-k none | 0.237086 | 0.216701 | 0.010314 | 0.008959 | 0.001113 | 0.006959 | 0.051129 | robustness seed, close to additive |
| hidden128 additive, temp 0.8, top-k 20 | 0.242020 | 0.229078 | 0.007175 | 0.004509 | 0.001258 | 0.008910 | 0.060885 | former best hidden128 prior |
| promoted standard VQ + additive AR | 0.298020 | 0.279341 | 0.007674 | 0.009817 | 0.001188 | 0.010502 | 0.041300 | public baseline |
| joint EMA alpha02, temp 0.8, top-k 20 | 0.320110 | 0.301908 | 0.009780 | 0.007312 | 0.001110 | 0.005988 | 0.032284 | good drawdown/autocorrelation, weak profile |
| separate-frequency hierarchical prior | 3.242975 | 2.333169 | 0.145358 | 0.717744 | 0.046704 | 0.073700 | 0.049859 | rejected, sampled-token/path collapse |

The conv-transformer k3 prior is the best discrete research model by the current profile evidence.
It improves over hidden128 additive primarily through MMD, drawdown W1, squared-return
autocorrelation, transition matching, and run-length matching. It also avoids the sampled-token
collapse of the separate-frequency hierarchical prior.

The continuous BetaCVAE remains the stronger reference on the overall profile and most stylised
diagnostics. The discrete result should therefore be reported as the best discrete research model,
not as the strongest model overall.

## Decision

Report hidden128 plus causal conv-transformer k3 as the best current discrete research model.

Do not replace the public baseline by default. The public baseline remains the promoted standard-VQ
tokenizer plus additive AR prior unless a later project-level decision explicitly wants the
research model to become the default. The reasons are:

- k3 is seed-sensitive, even though it holds directionally;
- k3 has weaker token likelihood than the public baseline;
- k3 still trails hidden128 additive on SWD and terminal W1;
- k3 still trails the public baseline on squared-return autocorrelation;
- the continuous BetaCVAE remains stronger on the overall profile and most stylised diagnostics.

Use the public baseline for stable default comparisons and use hidden128 conv-transformer k3 for
best-discrete research reporting.

## Next Branch

Because k3 is robust enough to report as the best current discrete research model, the next branch
should prepare report or notebook integration:

- add a table that separates public baseline, best discrete research model, and continuous
  reference;
- state the sampling policy explicitly: temperature `1.0`, `top_k=none`;
- include the seed99, seed1, and seed2 profile range rather than reporting only the best seed;
- retain the hidden128 additive prior as the required ablation reference;
- make clear that public-baseline replacement is not part of this decision.

If later report integration treats the seed spread as too large for the intended claim, run the
Mamba or selective-SSM package-compatibility check next. That check should inspect dependency
support, CUDA/PyTorch compatibility, checkpoint portability, reproducibility, and strict
left-to-right causal generation before implementation.

Do not move to MGVQ or GroupedRVQ unless the single-stream hidden128 prior branch is closed. The
current evidence points to prior calibration and robustness as the active bottleneck, not tokenizer
collapse.
