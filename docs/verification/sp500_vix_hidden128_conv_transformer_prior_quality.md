# Hidden128 Causal Conv-Transformer Prior Quality

Status: full source-quality run completed for the hidden128 single-stream causal conv-transformer
prior. No model code was changed during this training and evaluation pass.

## Run Manifest

- Prior config:
  `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml`
- Prior type: `causal_conv_transformer`
- Tokenizer:
  `outputs/sp500_vix_discrete/vq_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_hidden128_seed99`
- Full prior output:
  `outputs/sp500_vix_discrete/token_prior/hidden128_conv_transformer/sp500_vix_causal_token_prior_hidden128_conv_transformer_seed99`
- Best checkpoint:
  `outputs/sp500_vix_discrete/token_prior/hidden128_conv_transformer/sp500_vix_causal_token_prior_hidden128_conv_transformer_seed99/best_model`
- Decoded evaluation output:
  `outputs/sp500_vix_discrete/token_prior/hidden128_conv_transformer/evaluation_best`
- Paper-style output:
  `outputs/sp500_vix_discrete/paper_style_hidden128_conv_transformer_temp08_topk20`
- Sampling: temperature `0.8`, top-k `20`, seed `99`, `n_sample=1000`

The requested command named the run as `seed0`, but the experiment config sets `seed: 99`.
Training therefore wrote the actual checkpoint under the `seed99` run directory above.

## W&B And Runtime

The W&B launch failed before training began. The run did not produce a W&B URL.

```text
wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec.
```

The training command was rerun with `--no-wandb`, preserving the requested W&B execution
environment. Training completed on CPU in `1316.12` seconds, from
`2026-05-16T13:07:40.272773+00:00` to `2026-05-16T13:29:36.394204+00:00`.

## Token-Prior Likelihood

| Split | CE | Accuracy | Perplexity |
| --- | ---: | ---: | ---: |
| Train | 1.192685 | 0.532112 | 3.296098 |
| Eval | 1.129030 | 0.557089 | 3.107617 |

The best checkpoint was selected at epoch `100`. The promoted hidden128 additive prior remains the
stronger likelihood reference in the robustness report (`eval_ce=0.914807`,
`eval_accuracy=0.647449`, `eval_perplexity=2.507927`), but the conv-transformer improves over the
source-smoke run and gives a usable calibrated-token candidate for decoded evaluation.

## Decoded Token-Prior Metrics

Decoded evaluation used the best checkpoint with temperature `0.8` and top-k `20`.

| Metric | Value |
| --- | ---: |
| MMD | 0.203557 |
| SWD | 0.008292 |
| Terminal W1 | 0.009631 |
| Volatility W1 | 0.001375 |
| Marginal code L1 | 0.230467 |
| Transition matrix L1 | 0.228723 |
| Run-length W1 | 0.486822 |

Token diversity stayed broad rather than collapsing: sampled active codes were `63/64`, sampled
token entropy was `3.736032`, and sampled token perplexity was `41.931255`. The corresponding real
token entropy and perplexity were `3.756132` and `42.782639`.

The sampled run mean was `1.788802` versus real `1.614031`; the sampled maximum run length was
`29` versus real `24`. This indicates mild run over-persistence rather than the severe
sampled-token collapse seen in the separate-frequency experiment.

## Decoded VIX-Bucket Diagnostics

| Bucket | MMD | SWD | Volatility W1 | Terminal W1 | Active Codes | Token Perplexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| very_low | 0.372362 | 0.015731 | 0.001489 | 0.009708 | 60 | 35.991627 |
| low | 0.376618 | 0.010241 | 0.001097 | 0.013799 | 59 | 36.474762 |
| mid | 0.487009 | 0.014843 | 0.002096 | 0.021721 | 60 | 37.079155 |
| high | 0.270933 | 0.010479 | 0.001629 | 0.016275 | 60 | 43.067135 |
| very_high | 0.235972 | 0.013480 | 0.000976 | 0.009188 | 62 | 47.614971 |

The mid-VIX bucket remains the weakest decoded bucket, especially on MMD and terminal W1. The
very-high bucket is comparatively well behaved, which is a useful sign for the VIX-conditioned
generation objective.

## Paper-Style Metrics

| Model | MMD | SWD | Vol W1 | Terminal W1 | Returns W1 | Drawdown W1 | Return AC L1 | Sq-Return AC L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hidden128 conv-transformer | 0.200300 | 0.007683 | 0.001409 | 0.008033 | 0.001190 | 0.011835 | 0.036597 | 0.063155 |
| continuous BetaCVAE | 0.154421 | 0.008785 | 0.000634 | 0.009051 | 0.000602 | 0.007667 | 0.025972 | 0.029462 |

The paper-style token diagnostics were:

| Metric | Value |
| --- | ---: |
| Marginal code L1 | 0.201533 |
| Transition matrix L1 | 0.211512 |
| Run-length W1 | 0.468005 |
| Run-length histogram L1 | 0.068086 |
| Real active codes | 59 |
| Sampled active codes | 62 |
| Real token perplexity | 42.782639 |
| Sampled token perplexity | 41.816635 |
| Real run mean | 1.614031 |
| Sampled run mean | 1.779148 |
| Real max run | 24 |
| Sampled max run | 44 |

Paper-style VIX-bucket diagnostics for the discrete conv-transformer prior were:

| Bucket | MMD | SWD | Volatility W1 | Terminal W1 |
| --- | ---: | ---: | ---: | ---: |
| very_low | 0.411785 | 0.013111 | 0.001382 | 0.004964 |
| low | 0.389261 | 0.011575 | 0.001301 | 0.016179 |
| mid | 0.342435 | 0.012066 | 0.001929 | 0.017717 |
| high | 0.297423 | 0.012504 | 0.001878 | 0.018066 |
| very_high | 0.265518 | 0.019455 | 0.001008 | 0.016086 |

## Comparison To Prior References

| Reference | MMD | SWD | Vol W1 | Terminal W1 | Drawdown W1 | Sq-Return AC L1 | Token Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hidden128 conv-transformer, temp 0.8/top-k 20 | 0.200300 | 0.007683 | 0.001409 | 0.008033 | 0.011835 | 0.063155 | sampled PPL 41.816635, active 62 |
| hidden128 additive prior, temp 0.8/top-k 20 | 0.229078 | 0.007175 | 0.001258 | 0.004509 | 0.008910 | 0.060885 | sampled PPL 42.898106, active 63 |
| promoted baseline | 0.279341 | 0.007674 | 0.001188 | 0.009817 | 0.010502 | 0.041300 | stronger likelihood reference |
| joint EMA alpha02, temp 0.8/top-k 20 | 0.301908 | 0.009780 | 0.001110 | 0.007312 | 0.005988 | 0.032284 | transition L1 0.277110 |
| separate-frequency hierarchical prior | 2.333169 | 0.145358 | 0.046704 | 0.717744 | 0.073700 | 0.049859 | low/high sampled PPL 4.100208/6.622475 |
| continuous BetaCVAE | 0.154421 | 0.008785 | 0.000634 | 0.009051 | 0.007667 | 0.029462 | continuous reference |

The conv-transformer improves global MMD over the hidden128 additive prior and the promoted
baseline at the selected sampling point. It does not improve SWD, volatility W1, terminal W1,
drawdown W1, or squared-return autocorrelation relative to the hidden128 additive prior. Relative
to the separate-frequency hierarchical prior, it avoids the sampled-token collapse and composed
path failure by a wide margin.

## Decision

Continue the conv-transformer line, but do not promote it from this run alone.

The immediate next step should be a sampling ablation over temperature, top-k, and nucleus
sampling. The model has broad sampled token support and improved global MMD, so its current
weakness looks more like residual calibration and persistence than architectural failure. The
sampling ablation should optimise jointly for MMD, SWD, terminal W1, volatility W1, drawdown W1,
squared-return autocorrelation, token entropy/perplexity, transition L1, and run-length distance.

If sampling cannot recover terminal and persistence metrics, the next architecture study should
tune convolution depth and dilation before testing a wider transformer. The separate low/high
hierarchical prior remains rejected for this line because its sampled composed paths are far outside
the single-stream and continuous references.
