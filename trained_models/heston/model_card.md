# Heston Model Card

## Selection

| Family | Selected candidate | Configs | Profile |
| --- | --- | --- | --- |
| Continuous | `info_cvae` | `configs/experiments/heston_info_cvae.yaml` | `balanced_market` |
| Discrete | `standard_vq_additive_ar` | `configs/experiments/heston_causal_vq_tokenizer.yaml`; `configs/experiments/heston_causal_token_prior_additive.yaml` | `balanced_market` |

## Metrics

Lower is better except accuracy and active-code counts.

| Family | Candidate | MMD | SWD | Terminal W1 | Volatility W1 | Balanced score | Prior CE | Prior perplexity | Prior accuracy | Active codes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Continuous | `info_cvae` | 0.071840 | 0.066884 | - | - | - | - | - | - | - |
| Discrete | `standard_vq_additive_ar` | 0.049702 | 0.074397 | 0.051078 | 0.019140 | 0.048579 | 2.652545 | 14.195220 | 0.167333 | 64 |

## Local Checkpoint Convention

The registry stores metadata only. Continuous checkpoints should remain under a local path such
as `outputs/heston_continuous/info_cvae/<training-run>/final_model`. Discrete tokenizer, token
data, and prior artefacts should remain under
`outputs/per_experiment_selection/heston/standard_vq_additive_ar/`.

## Caveats

The continuous baseline has only legacy MMD/SWD from the supplemental cloned-repository
evaluation. The generic discrete evaluator did not compute returns W1, drawdown W1, return AC L1,
or squared-return AC L1. Trained unconditioned tokenizer no-leakage and notebook reproduction
remain missing.
