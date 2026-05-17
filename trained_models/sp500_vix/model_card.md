# S&P500/VIX Model Card

## Selection

| Family | Selected candidate | Configs | Profile |
| --- | --- | --- | --- |
| Continuous | `beta_cvae` | `configs/experiments/sp500_vix_beta_cvae.yaml` | `balanced_market` |
| Discrete | `conditional_standard_vq_additive_ar` | `configs/experiments/sp500_vix_causal_vq_tokenizer.yaml`; `configs/experiments/sp500_vix_causal_token_prior_additive.yaml` | `balanced_market` |

## Metrics

Lower is better except accuracy and active-code counts.

| Family | Candidate | MMD | SWD | Returns W1 | Terminal W1 | Volatility W1 | Drawdown W1 | Return AC L1 | Squared-return AC L1 | Balanced score | Prior CE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Continuous | `beta_cvae` | 0.154422 | 0.008785 | - | - | - | - | - | - | - | - |
| Discrete | `conditional_standard_vq_additive_ar` | 0.254177 | 0.013836 | 0.001012 | 0.010252 | 0.001504 | 0.010957 | 0.055395 | 0.039584 | 0.048340 | 0.914807 |
| Optional discrete | `conditional_hidden128_conv_transformer_k3` | 0.501580 | 0.013060 | 0.000976 | 0.018768 | 0.001151 | 0.003914 | 0.051504 | 0.040645 | 0.078950 | 0.883431 |

## Local Checkpoint Convention

The registry stores metadata only. Continuous checkpoints should remain under a local path such
as `outputs/sp500_vix_continuous/beta_cvae/<training-run>/final_model`. Public discrete tokenizer,
token data, prior, paper-style outputs, and latent-geometry artefacts should remain under
`outputs/sp500_vix_discrete/`.

## Caveats

The public standard additive discrete baseline is selected by the balanced-market path profile. The hidden128 conv-transformer candidate wins several component metrics and token CE, but loses MMD, terminal W1, squared-return AC L1, and the balanced score. Notebook reproduction remains missing before final public registry promotion.
