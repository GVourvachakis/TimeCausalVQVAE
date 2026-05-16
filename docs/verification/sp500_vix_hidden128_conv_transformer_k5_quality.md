# Hidden128 Conv-Transformer K5 Prior Quality

Status: k5 capacity follow-up completed. No source code, tokenizer code, architecture family,
objective, or `dilations124` run was changed or added.

## Manifest

- Config:
  `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer_k5.yaml`
- Prior type: `causal_conv_transformer`
- Capacity change: convolution kernel size `5`, with two causal convolution layers and dilations
  `[1, 2]`
- Tokenizer:
  `outputs/sp500_vix_discrete/vq_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_hidden128_seed99`
- Best checkpoint:
  `outputs/sp500_vix_discrete/token_prior/hidden128_conv_transformer_k5/sp500_vix_causal_token_prior_hidden128_conv_transformer_k5_seed99/best_model`
- Decoded evaluation:
  `outputs/sp500_vix_discrete/token_prior/hidden128_conv_transformer_k5/evaluation_best_temp10_topknone`
- Paper-style evaluation:
  `outputs/sp500_vix_discrete/paper_style_hidden128_conv_transformer_k5_temp10_topknone`
- Sampling: temperature `1.0`, unrestricted top-k, `n_sample=1000`, seed `99`

The `tcvae-evaluate-token-prior` CLI represents unrestricted top-k by omitting `--top-k`. The
paper-style runner accepts `--top-k none`.

Profile is:

```text
MMD + SWD + terminal_return_wasserstein + volatility_wasserstein
```

Lower is better.

## W&B And Runtime

The W&B run did not initialise and produced no run URL. It failed before training with:

```text
wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec.
```

Training was rerun with `--no-wandb` under the same execution profile. The completed run used CPU
and took `1331.52` seconds, from `2026-05-16T14:31:20.822610+00:00` to
`2026-05-16T14:53:32.340083+00:00`.

## Training Metrics

| Split | CE | Accuracy | Perplexity |
| --- | ---: | ---: | ---: |
| Train | 1.093099 | 0.573837 | 2.983793 |
| Eval | 1.000186 | 0.617270 | 2.727795 |

The best checkpoint was epoch `100`, and it matched the final checkpoint. Likelihood improves over
the first conv-transformer run (`eval_ce=1.129030`, `eval_accuracy=0.557089`,
`eval_perplexity=3.107617`), so the wider local kernel helps token prediction. The generated-path
metrics below show that this likelihood gain does not translate into a better primary profile.

## Decoded Token-Prior Metrics

| Metric | Value |
| --- | ---: |
| Decoded profile | 0.288722 |
| MMD | 0.271909 |
| SWD | 0.009946 |
| Terminal W1 | 0.005789 |
| Volatility W1 | 0.001078 |
| Marginal code L1 | 0.127933 |
| Transition matrix L1 | 0.176098 |
| Run-length W1 | 0.033170 |

Token usage remains broad: sampled active codes were `63/64`, sampled token entropy was
`3.818408`, and sampled token perplexity was `45.531654`. The real token entropy and perplexity
were `3.756132` and `42.782639`.

The decoded token diagnostics are locally strong. Transition L1 and run-length W1 are better than
the first conv-transformer primary setting. However, decoded MMD is substantially worse, and that
dominates the decoded profile.

## Paper-Style Metrics

| Metric | Value |
| --- | ---: |
| Paper-style profile | 0.275071 |
| MMD | 0.259363 |
| SWD | 0.010101 |
| Volatility W1 | 0.000904 |
| Terminal W1 | 0.004705 |
| Returns W1 | 0.000581 |
| Drawdown W1 | 0.005909 |
| Return AC L1 | 0.032883 |
| Squared-return AC L1 | 0.049123 |

Paper-style token diagnostics were:

| Metric | Value |
| --- | ---: |
| Marginal code L1 | 0.133967 |
| Transition matrix L1 | 0.181579 |
| Run-length W1 | 0.060747 |
| Run-length histogram L1 | 0.021124 |
| Sampled active codes | 63 |
| Sampled token perplexity | 45.772663 |
| Sampled token entropy | 3.823687 |
| Real run mean | 1.614031 |
| Sampled run mean | 1.622806 |
| Real max run | 24 |
| Sampled max run | 24 |

The paper-style result confirms the same pattern as the decoded evaluation: k5 improves local
token persistence and several stylised-fact metrics, but the generated path distribution moves too
far from the real distribution by MMD.

## Comparison

| Setting | Profile | MMD | SWD | Terminal W1 | Vol W1 | Drawdown W1 | Sq-return AC L1 | Transition L1 | Run-length W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k5, temp 1.0, top-k none | 0.275071 | 0.259363 | 0.010101 | 0.004705 | 0.000904 | 0.005909 | 0.049123 | 0.181579 | 0.060747 |
| first conv, temp 1.0, top-k none | 0.186725 | 0.169510 | 0.010918 | 0.005086 | 0.001210 | 0.007687 | 0.050871 | 0.193243 | 0.050189 |
| first conv, temp 1.0, top-k 20 | 0.205630 | 0.184628 | 0.011476 | 0.008584 | 0.000942 | 0.007858 | 0.050510 | 0.183255 | 0.097464 |
| hidden128 additive, temp 0.8, top-k 20 | 0.242020 | 0.229078 | 0.007175 | 0.004509 | 0.001258 | 0.008910 | 0.060885 | 0.223900 | 0.390197 |
| promoted baseline | 0.298020 | 0.279341 | 0.007674 | 0.009817 | 0.001188 | 0.010502 | 0.041300 | n/a | n/a |
| continuous BetaCVAE | 0.172891 | 0.154421 | 0.008785 | 0.009051 | 0.000634 | 0.007667 | 0.029462 | n/a | n/a |

Relative to the first conv-transformer primary setting, k5 improves SWD, terminal W1, volatility
W1, drawdown W1, squared-return autocorrelation, and transition L1. It is worse on the profile,
MMD, and run-length W1. The MMD regression is large enough that k5 is not the better primary
hidden128 prior.

Relative to the hidden128 additive prior, k5 improves profile, volatility W1, drawdown W1,
squared-return autocorrelation, transition L1, and run-length W1. It remains worse on SWD and
terminal W1. This means k5 is still a useful stronger-prior ablation, but not the best
conv-transformer variant.

The optional secondary `temperature=1.0, top_k=20` k5 evaluation was not run. The primary
unrestricted setting was clearly worse than the first conv-transformer primary setting on the
paper-style profile because of MMD, and the prompt limited the secondary run to cases that were not
clearly worse.

## Decision

K5 does not improve the best hidden128 conv-transformer prior.

Keep the first conv-transformer with kernel size `3`, temperature `1.0`, and unrestricted top-k as
the best hidden128 prior setting. K5 should not be promoted despite its stronger likelihood and
cleaner transition/run-length diagnostics, because the primary generated-path profile regresses
from `0.186725` to `0.275071`.

Do not stop the conv-transformer branch solely from this result. The k5 result is informative: it
shows that a larger local receptive field improves token likelihood and local persistence but can
over-shape the path distribution. A future `dilations124` run is justified only if the next question
is whether longer multi-scale local context can recover MMD without losing the improved local
diagnostics. It was not run in this prompt.
