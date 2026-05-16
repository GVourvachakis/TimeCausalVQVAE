# Hidden128 Conv-Transformer Sampling Ablation

Status: sampling ablation completed for the hidden128 causal conv-transformer prior. No tokenizer
code, model code, new architecture family, or objective was changed.

## Manifest

- Prior config:
  `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml`
- Prior checkpoint:
  `outputs/sp500_vix_discrete/token_prior/hidden128_conv_transformer/sp500_vix_causal_token_prior_hidden128_conv_transformer_seed99/best_model`
- Tokenizer:
  `outputs/sp500_vix_discrete/vq_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_hidden128_seed99`
- Continuous reference:
  `outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47/final_model`
- Evaluation: `n_sample=1000`, seed `99`, paper-style metrics

The grid covered temperatures `0.6`, `0.8`, and `1.0` crossed with unrestricted sampling,
`top_k=20`, and `top_k=40`. The previous `temperature=0.8, top_k=20` paper-style output was reused
from the source-quality run because it used the same checkpoint, seed, sample count, and evaluator.

Profile is:

```text
MMD + SWD + terminal_return_wasserstein + volatility_wasserstein
```

## Sampling Grid

| Temp | Top-k | Profile | MMD | SWD | Terminal W1 | Vol W1 | Returns W1 | Drawdown W1 | Return AC L1 | Sq-return AC L1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6 | none | 0.344756 | 0.321481 | 0.010541 | 0.010696 | 0.002039 | 0.002028 | 0.019997 | 0.057046 | 0.077696 |
| 0.6 | 20 | 0.312048 | 0.291516 | 0.009039 | 0.009550 | 0.001944 | 0.001909 | 0.018390 | 0.053411 | 0.075847 |
| 0.6 | 40 | 0.369216 | 0.344858 | 0.011499 | 0.010782 | 0.002076 | 0.002068 | 0.020943 | 0.056564 | 0.079857 |
| 0.8 | none | 0.214753 | 0.199570 | 0.007088 | 0.006755 | 0.001340 | 0.001152 | 0.010240 | 0.036593 | 0.062583 |
| 0.8 | 20 | 0.217425 | 0.200300 | 0.007683 | 0.008033 | 0.001409 | 0.001190 | 0.011835 | 0.036597 | 0.063155 |
| 0.8 | 40 | 0.224663 | 0.207723 | 0.007753 | 0.007833 | 0.001355 | 0.001192 | 0.011396 | 0.036888 | 0.064231 |
| 1.0 | none | 0.186725 | 0.169510 | 0.010918 | 0.005086 | 0.001210 | 0.000681 | 0.007687 | 0.027889 | 0.050871 |
| 1.0 | 20 | 0.205630 | 0.184628 | 0.011476 | 0.008584 | 0.000942 | 0.000620 | 0.007858 | 0.025463 | 0.050510 |
| 1.0 | 40 | 0.199009 | 0.179137 | 0.010718 | 0.008025 | 0.001128 | 0.000643 | 0.007347 | 0.027389 | 0.050987 |

The best profile setting is `temperature=1.0, top_k=none`. It also gives the best MMD and
run-length distance. The best volatility W1, transition L1, and squared-return autocorrelation L1
come from `temperature=1.0, top_k=20`, but that setting gives a weaker profile because terminal W1
and SWD increase.

## Transition And Run-Length Diagnostics

| Temp | Top-k | Marginal L1 | Transition L1 | Run-length W1 | Run hist L1 | Sampled PPL | Entropy | Active | Run mean | Run max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6 | none | 0.294033 | 0.304338 | 0.819219 | 0.123072 | 37.587696 | 3.626677 | 62 | 2.001267 | 42 |
| 0.6 | 20 | 0.256000 | 0.259420 | 0.775614 | 0.122716 | 38.409771 | 3.648312 | 60 | 1.970314 | 33 |
| 0.6 | 40 | 0.333767 | 0.315339 | 0.838724 | 0.130635 | 36.519661 | 3.597851 | 62 | 2.015113 | 42 |
| 0.8 | none | 0.196967 | 0.213996 | 0.437022 | 0.063240 | 41.679924 | 3.730020 | 62 | 1.763254 | 34 |
| 0.8 | 20 | 0.201533 | 0.211512 | 0.468005 | 0.068086 | 41.816635 | 3.733294 | 62 | 1.779148 | 44 |
| 0.8 | 40 | 0.231233 | 0.232225 | 0.481943 | 0.070572 | 41.277710 | 3.720323 | 63 | 1.786299 | 26 |
| 1.0 | none | 0.145833 | 0.193243 | 0.050189 | 0.022480 | 45.280468 | 3.812876 | 63 | 1.620746 | 26 |
| 1.0 | 20 | 0.157900 | 0.183255 | 0.097464 | 0.028427 | 45.641308 | 3.820813 | 63 | 1.630745 | 30 |
| 1.0 | 40 | 0.146800 | 0.208838 | 0.093801 | 0.025697 | 45.238525 | 3.811949 | 64 | 1.629903 | 28 |

The low-temperature settings over-persist token runs. Raising temperature to `1.0` brings the
sampled run mean close to the real run mean of `1.614031` and sharply reduces run-length W1. There
is no sampled-token collapse: all `1.0` settings keep `63-64` active codes and sampled perplexity
around `45`.

## Comparison To Hidden128 Additive Top-k20

| Setting | Profile | MMD | SWD | Terminal W1 | Vol W1 | Drawdown W1 | Sq-return AC L1 | Transition L1 | Run-length W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| conv-transformer, temp 1.0, top-k none | 0.186725 | 0.169510 | 0.010918 | 0.005086 | 0.001210 | 0.007687 | 0.050871 | 0.193243 | 0.050189 |
| conv-transformer, temp 1.0, top-k 20 | 0.205630 | 0.184628 | 0.011476 | 0.008584 | 0.000942 | 0.007858 | 0.050510 | 0.183255 | 0.097464 |
| hidden128 additive, temp 0.8, top-k 20 | 0.242020 | 0.229078 | 0.007175 | 0.004509 | 0.001258 | 0.008910 | 0.060885 | 0.223900 | 0.390197 |

The conv-transformer best profile setting improves MMD, volatility W1, drawdown W1, squared-return
autocorrelation, transition L1, and run-length W1 versus the hidden128 additive `temperature=0.8,
top_k=20` reference. It does not improve SWD or terminal W1. The `temperature=1.0, top_k=20`
setting gives the best volatility W1 and transition L1, but the terminal W1 penalty is too large to
make it the primary setting.

## Capacity Ablation Configs

The first conv-transformer run improved some target metrics but did not dominate. Two small
capacity configs were therefore created for later controlled training:

- `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer_k5.yaml`
- `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer_dilations124.yaml`

No additional non-smoke capacity training was run in this prompt. The sampling ablation already
recovered a stronger profile and substantially better run-length diagnostics without changing
capacity, so launching another full CPU training run would not be the most diagnostic next step.

## Decision

Continue the hidden128 conv-transformer prior and use `temperature=1.0, top_k=none` as the current
paper-style profile setting. Keep `temperature=1.0, top_k=20` as a secondary calibration setting
when volatility W1, transition L1, or squared-return autocorrelation is prioritised.

Do not promote the conv-transformer as a public baseline yet. It now beats the hidden128 additive
top-k20 profile and improves the local token persistence diagnostics, but it still trails the
additive reference on SWD and terminal W1. The next non-smoke capacity run, if needed, should start
with the `k5` config because it increases local receptive field with the lowest change in depth and
dependency risk.
