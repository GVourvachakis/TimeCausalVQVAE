# S&P500/VIX Best Discrete Research Model

Status: report-ready documentation for the current best discrete research model. This note does
not change the public default model, add architecture, or require training.

The reporting distinction is:

- Public discrete baseline: standard VQ tokenizer plus additive VIX-only causal AR prior.
- Best discrete research model: hidden128 VQ tokenizer plus causal conv-transformer k3 prior.
- Continuous reference: continuous BetaCVAE, which remains the strongest overall reference.

Use the phrase "best discrete research model" for the hidden128 conv-transformer result. Do not
describe it as a new public default.

## Model Summary

Profile is:

```text
MMD + SWD + terminal_return_wasserstein + volatility_wasserstein
```

Lower is better.

| Role | Model | Tokenizer | Prior or generator | Sampling | Profile | MMD | SWD | Terminal W1 | Vol W1 | Main status |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Public baseline | Standard VQ + additive AR | `configs/experiments/sp500_vix_causal_vq_tokenizer.yaml` | `configs/experiments/sp500_vix_causal_token_prior_additive.yaml` | temperature `0.8`, `top_k=40` | 0.298020 | 0.279341 | 0.007674 | 0.009817 | 0.001188 | Public default remains here. |
| Best discrete research model | Hidden128 VQ + causal conv-transformer k3 | `configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml` | `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml` | temperature `1.0`, `top_k=none` | 0.186725 | 0.169510 | 0.010918 | 0.005086 | 0.001210 | Best current discrete research result. |
| Continuous reference | Continuous BetaCVAE | n/a | `configs/experiments/sp500_vix_beta_cvae.yaml` | n/a | 0.172891 | 0.154421 | 0.008785 | 0.009051 | 0.000634 | Strongest overall reference. |

The hidden128 research configs listed above are source-branch research artefacts from
`research/stronger-hidden128-prior`; they are not public defaults on this branch.

## Best Discrete Research Model

The current best discrete research model pairs a hidden128 standard-VQ tokenizer with a causal
conv-transformer k3 token prior:

- tokenizer: hidden128 standard VQ, one 64-code index per time step;
- prior type: `causal_conv_transformer`;
- convolutional front end: two causal convolution layers;
- convolutional kernel size: `3`;
- convolutional dilations: `[1, 2]`;
- transformer trunk: the existing causal additive VIX-conditioned token-prior trunk;
- primary reporting policy: temperature `1.0`, unrestricted top-k.

The selected seed99 result is the best discrete result by the current paper-style profile. It
also keeps broad sampled token support, with 63 of 64 sampled active codes and sampled token
perplexity `45.280468`.

## Seed Robustness

The selected k3 sampling policy is:

```text
temperature=1.0
top_k=none
```

Seed robustness was evaluated for seed99, seed1, and seed2 under this policy. W&B initialisation
timed out for seed1 and seed2, so those runs were retried with `--no-wandb`.

| Seed | Eval CE | Eval accuracy | Eval perplexity | Paper profile | MMD | SWD | Terminal W1 | Vol W1 | Sq-return AC L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 99 | 1.129030 | 0.557089 | 3.107617 | 0.186725 | 0.169510 | 0.010918 | 0.005086 | 0.001210 | 0.050871 |
| 1 | 1.130656 | 0.557692 | 3.112165 | 0.237086 | 0.216701 | 0.010314 | 0.008959 | 0.001113 | 0.051129 |
| 2 | 1.124913 | 0.557184 | 3.094001 | 0.214707 | 0.199984 | 0.009162 | 0.004418 | 0.001143 | 0.048157 |

The k3 result holds directionally across the three seeds. All three seeds improve the main
profile relative to the hidden128 additive reference profile of `0.242020` and the public
baseline profile of `0.298020`. The result is nevertheless seed-sensitive: the profile range is
`0.186725` to `0.237086`, mostly driven by MMD and terminal W1 variation.

## Caveats

- The k3 result is seed-sensitive, even though the direction is stable across the available
  seeds.
- The k3 prior has weaker token likelihood than the public baseline: eval CE `1.129030` and
  perplexity `3.107617` for seed99, versus eval CE `0.914807` and perplexity `2.507927` for the
  public baseline.
- The k3 result is weaker than the hidden128 additive reference on SWD and terminal W1. The
  hidden128 additive reference reports SWD `0.007175` and terminal W1 `0.004509`, while k3 seed99
  reports SWD `0.010918` and terminal W1 `0.005086`.
- The continuous BetaCVAE remains strongest overall, with profile `0.172891` versus k3 seed99
  profile `0.186725`.

## Reporting Language

Use:

> The hidden128 VQ tokenizer with a causal conv-transformer k3 prior is the best discrete research
> model under the current S&P500/VIX paper-style profile, using temperature `1.0` and unrestricted
> top-k sampling. It should be reported separately from the public default because it is
> seed-sensitive and still trails the continuous BetaCVAE overall.

Avoid:

> The hidden128 conv-transformer is the new public default.

The public default remains the standard VQ tokenizer with the additive VIX-only AR prior.
