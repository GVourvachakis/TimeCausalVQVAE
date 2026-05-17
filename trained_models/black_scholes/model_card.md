# Black-Scholes Model Card

## Selection

| Family | Selected candidate | Configs | Profile |
| --- | --- | --- | --- |
| Continuous | `beta_cvae` | `configs/experiments/black_scholes_beta_cvae.yaml` | `balanced_market` |
| Discrete | `hidden128_conv_transformer_k3` | `configs/experiments/black_scholes_causal_vq_tokenizer_hidden128.yaml`; `configs/experiments/black_scholes_causal_token_prior_hidden128_conv_transformer.yaml` | `balanced_market` |

## Metrics

Lower is better except accuracy and active-code counts.

| Family | Candidate | MMD | SWD | Terminal W1 | Volatility W1 | Balanced score | Prior CE | Prior perplexity | Prior accuracy | Active codes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Continuous | `beta_cvae` | 0.089052 | 0.056371 | - | - | - | - | - | - | - |
| Discrete | `hidden128_conv_transformer_k3` | 0.085255 | 0.056371 | 0.067674 | 0.004042 | 0.053335 | 2.419572 | 11.241549 | 0.181267 | 64 |

## Local Checkpoint Convention

The registry stores metadata only. Continuous checkpoints should remain under a local path such as `outputs/black_scholes_continuous/beta_cvae/<training-run>/final_model`. Discrete tokenizer, token data, and prior artefacts should remain under `outputs/per_experiment_selection/black_scholes/hidden128_conv_transformer_k3/`.

## Caveats

The continuous baseline has only legacy MMD/SWD from the supplemental cloned-repository evaluation. The Black-Scholes public standard discrete artefacts were unavailable in the final
evaluation pass. Drawdown W1, returns W1, return AC L1, squared-return AC L1, trained unconditioned tokenizer no-leakage, and notebook reproduction remain missing.
