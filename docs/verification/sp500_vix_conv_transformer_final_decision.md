# S&P500/VIX Conv-Transformer Final Decision

Status: final decision note for the hidden128 causal conv-transformer prior family. No code was
implemented and no models were trained for this decision.

Profile is:

```text
MMD + SWD + terminal_return_wasserstein + volatility_wasserstein
```

Lower is better.

## Current Public Baseline

The current public discrete baseline remains the promoted standard-VQ tokenizer with the additive
VIX-only causal AR prior:

- tokenizer config: `configs/experiments/sp500_vix_causal_vq_tokenizer.yaml`;
- prior config: `configs/experiments/sp500_vix_causal_token_prior_additive.yaml`;
- tokenizer interface: one 64-code index per time step;
- prior: additive VIX-only causal AR transformer;
- sampling: temperature `0.8`, `top_k=40`.

| Metric | Promoted standard VQ + additive AR |
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

This remains the public baseline because it is the simplest established causal discrete model and
still has strong token likelihood and squared-return autocorrelation.

## Current Best Discrete Research Baseline

Before the conv-transformer work, the best discrete research baseline was the hidden128 tokenizer
with the unchanged additive VIX-only causal AR prior:

- tokenizer config: `configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml`;
- prior config: `configs/experiments/sp500_vix_causal_token_prior_hidden128_candidate.yaml`;
- selected sampling: temperature `0.8`, `top_k=20`.

| Metric | Hidden128 additive AR |
| --- | ---: |
| Paper-style profile | 0.242020 |
| MMD | 0.229078 |
| SWD | 0.007175 |
| Terminal W1 | 0.004509 |
| Volatility W1 | 0.001258 |
| Maximum drawdown W1 | 0.008910 |
| Return AC L1 | 0.038262 |
| Squared-return AC L1 | 0.060885 |
| Transition L1 | 0.223900 |
| Run-length W1 | 0.390197 |

This setting remains the additive-reference ablation. It is no longer the best hidden128 prior
because the conv-transformer k3 result improves the main profile and local token dynamics.

## Conv-Transformer K3 Result

The selected conv-transformer k3 prior uses the hidden128 tokenizer, two causal convolution layers,
kernel size `3`, dilations `[1, 2]`, and the existing causal transformer trunk.

The best sampling setting from the ablation is:

```text
temperature=1.0, top_k=none
```

| Metric | Conv-transformer k3 |
| --- | ---: |
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

K3 improves over the hidden128 additive prior on profile, MMD, volatility W1, drawdown W1,
return-autocorrelation L1, squared-return-autocorrelation L1, transition L1, and run-length W1. It
does not improve SWD or terminal W1.

## Conv-Transformer K5 Result

The k5 follow-up changed only the local convolution kernel size from `3` to `5`, keeping two
convolution layers and dilations `[1, 2]`.

| Metric | Conv-transformer k5 |
| --- | ---: |
| Train CE / accuracy / perplexity | 1.093099 / 0.573837 / 2.983793 |
| Eval CE / accuracy / perplexity | 1.000186 / 0.617270 / 2.727795 |
| Paper-style profile | 0.275071 |
| MMD | 0.259363 |
| SWD | 0.010101 |
| Terminal W1 | 0.004705 |
| Volatility W1 | 0.000904 |
| Returns W1 | 0.000581 |
| Drawdown W1 | 0.005909 |
| Return AC L1 | 0.032883 |
| Squared-return AC L1 | 0.049123 |
| Transition L1 | 0.181579 |
| Run-length W1 | 0.060747 |
| Sampled active codes | 63/64 |
| Sampled token perplexity | 45.772663 |

K5 improves likelihood and several local/stylised diagnostics relative to k3, including SWD,
terminal W1, volatility W1, drawdown W1, squared-return autocorrelation, and transition L1. It
does not improve the family-level decision because MMD regresses enough to worsen the profile from
`0.186725` to `0.275071`.

## Continuous Reference

The continuous BetaCVAE remains the strongest non-discrete reference:

| Metric | Continuous BetaCVAE |
| --- | ---: |
| Paper-style profile | 0.172891 |
| MMD | 0.154421 |
| SWD | 0.008785 |
| Terminal W1 | 0.009051 |
| Volatility W1 | 0.000634 |
| Returns W1 | 0.000602 |
| Drawdown W1 | 0.007667 |
| Return AC L1 | 0.025972 |
| Squared-return AC L1 | 0.029462 |

K3 is close to the continuous reference on the primary profile and has better terminal W1, but the
continuous model remains stronger on MMD, SWD, volatility W1, returns W1, drawdown W1, and
autocorrelation.

## Summary Comparison

| Model | Profile | MMD | SWD | Terminal W1 | Vol W1 | Drawdown W1 | Sq-return AC L1 | Transition L1 | Run-length W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| continuous BetaCVAE | 0.172891 | 0.154421 | 0.008785 | 0.009051 | 0.000634 | 0.007667 | 0.029462 | n/a | n/a |
| conv-transformer k3, temp 1.0, top-k none | 0.186725 | 0.169510 | 0.010918 | 0.005086 | 0.001210 | 0.007687 | 0.050871 | 0.193243 | 0.050189 |
| conv-transformer k5, temp 1.0, top-k none | 0.275071 | 0.259363 | 0.010101 | 0.004705 | 0.000904 | 0.005909 | 0.049123 | 0.181579 | 0.060747 |
| hidden128 additive, temp 0.8, top-k 20 | 0.242020 | 0.229078 | 0.007175 | 0.004509 | 0.001258 | 0.008910 | 0.060885 | 0.223900 | 0.390197 |
| promoted standard VQ + additive AR | 0.298020 | 0.279341 | 0.007674 | 0.009817 | 0.001188 | 0.010502 | 0.041300 | n/a | n/a |

## Decision

Promote conv-transformer k3 as the best discrete research prior for the hidden128 tokenizer.

Do not promote k5. It improves likelihood and some local diagnostics, but its profile is worse than
k3 and worse than the hidden128 additive prior because MMD regresses materially.

Do not keep the hidden128 additive prior as the selected hidden128 research prior. Keep it as the
required additive-reference ablation because it still has better SWD and terminal W1 than k3.

Do not stop the broader prior research branch. Stop the current conv-transformer capacity branch at
k3 versus k5 for now: k3 is selected, k5 is rejected as the primary variant, and `dilations124`
should not be run automatically unless a future question specifically targets multi-scale local
context.

Do not replace the public promoted standard-VQ baseline in this decision. The conv-transformer k3
is the best discrete research prior, not the new public default.

## Remaining Limitations

The selected k3 result has several limitations:

- SWD is worse than the hidden128 additive prior, the promoted baseline, and the continuous
  BetaCVAE;
- terminal W1 is slightly worse than the hidden128 additive prior;
- volatility W1 is still worse than the continuous BetaCVAE and is only marginally competitive
  with the promoted baseline;
- squared-return autocorrelation improves over hidden128 additive, but remains weaker than the
  promoted baseline and the continuous BetaCVAE;
- continuous BetaCVAE remains stronger on the main profile and most stylised-fact diagnostics.

These limitations argue against public-baseline promotion and against claiming that the discrete
model has replaced the continuous reference.

## Next Research Branch

The next branch should be seed robustness for the selected conv-transformer k3 prior. The seed
study should keep the hidden128 tokenizer fixed, use the selected `temperature=1.0, top_k=none`
paper-style setting, and include the secondary `temperature=1.0, top_k=20` setting only if
volatility W1 or transition L1 is the target diagnostic.

If k3 seed robustness plateaus or reveals instability, the next prior-family branch should be a
Mamba or selective-SSM package-compatibility check. That check should inspect package support,
CUDA/PyTorch constraints, licensing, checkpoint portability, reproducibility, and strict causal
left-to-right generation before any implementation.

Do not move to MGVQ or GroupedRVQ yet. Those tokenizer-family changes should remain deferred until
the single-stream hidden128 prior branch is closed with seed robustness or a clear prior-side
failure mode.
