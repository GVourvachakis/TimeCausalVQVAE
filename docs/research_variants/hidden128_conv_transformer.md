# Hidden128 Causal Conv-Transformer Research Variant

This note records the minimal branch-local surface for the best discrete research model. It is
not a public default change.

## Purpose

The variant pairs a hidden128 standard VQ tokenizer with a causal conv-transformer k3 token prior
for S&P500/VIX research comparisons. The prior keeps the existing BOS-shifted single-code
autoregressive convention and additive scalar VIX conditioning, then inserts a causal residual
convolutional front end before the transformer trunk.

## Config

Prior config:

```text
configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml
```

The config is marked as a research variant and references local ignored artefacts under
`outputs/`:

```text
outputs/sp500_vix_discrete/vq_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_hidden128_seed99
outputs/sp500_vix_discrete/token_prior/sp500_vix_causal_vq_tokenizer_hidden128_tokens
```

Expected trained-prior outputs should also remain under ignored local paths, for example:

```text
outputs/sp500_vix_discrete/token_prior/hidden128_conv_transformer
outputs/sp500_vix_discrete/paper_style/hidden128_conv_transformer
```

## Sampling Policy

Use temperature `1.0` and unrestricted top-k sampling (`top_k=none`) for the selected research
comparison.

## Summary Metrics

The source report in `docs/report_ready_best_discrete_research_model.md` selects seed99 as the
best discrete research result under the main paper-style profile:

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
| Sampled active codes | 63/64 |
| Sampled token perplexity | 45.280468 |

For context, the public standard VQ plus additive AR baseline reports profile `0.298020`, MMD
`0.279341`, SWD `0.007674`, terminal W1 `0.009817`, and volatility W1 `0.001188`.

## Caveat

This hidden128 conv-transformer is optional research infrastructure. The public discrete default
remains the standard VQ tokenizer with the additive VIX-only causal AR prior.
