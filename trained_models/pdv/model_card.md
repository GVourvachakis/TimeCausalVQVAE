# PDV4 Model Card

## Selection

| Family | Selected candidate | Configs | Profile |
| --- | --- | --- | --- |
| Continuous | `info_cvae` | `configs/experiments/pdv_info_cvae.yaml` | `balanced_market` |
| Discrete | `conditional_standard_vq_additive_ar` | `configs/experiments/pdv_causal_vq_tokenizer_codebook64_codebookdim16.yaml`; `configs/experiments/pdv_causal_token_prior_additive_seed1.yaml` | `balanced_market` |

## Metrics

Lower is better except accuracy and active-code counts.

| Family | Candidate | MMD | SWD | Terminal W1 | Volatility W1 | Balanced score | Prior CE | Prior perplexity | Prior accuracy | Active codes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Continuous | `info_cvae` | 0.165295 | 0.010630 | - | - | - | - | - | - | - |
| Discrete | `conditional_standard_vq_additive_ar` | 1.170725 | 0.051673 | 0.057260 | 0.006154 | 0.321453 | 1.104750 | 3.019962 | 0.566133 | 57 |

## Local Checkpoint Convention

The registry stores metadata only. Continuous checkpoints should remain under a local path such
as `outputs/pdv_continuous/info_cvae/<training-run>/final_model`. Discrete tokenizer, token data,
and prior artefacts should remain under
`outputs/per_experiment_selection/pdv/conditional_standard_vq_additive_ar/`.

## Caveats

The continuous baseline has only legacy MMD/SWD from the supplemental cloned-repository
evaluation. The generic discrete evaluator did not compute returns W1, drawdown W1, return AC L1,
squared-return AC L1, or the full PDV4 condition-bucket path profile. Notebook reproduction
remains missing.
