# Hawkes/SVMHJD Compact Conv-Transformer Prior

## Scope

This one-seed comparison tests whether the selected hidden128 log-return cb64
causal conv-transformer prior can be reduced without losing the Hawkes/SVMHJD
rare-event advantage.

The tokenizer and extracted token-data paths are unchanged:

- Tokenizer: `outputs/hawkes_jump_tokenizer_ablation/tokenizers/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64_seed0`
- Token data: `outputs/hawkes_jump_tokenizer_ablation/tokens/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64`

The comparison uses seed `0`, `n_sample=1024`, sampling temperature `1.0`,
and unrestricted `top_k`. No tokenizer weights, Ogata simulator parameters, or
registry entries were changed.

## Parameter Counts

Reported with `scripts/report_token_prior_parameter_counts.py`.

| Candidate | Parameters | Embed | Layers | MLP | Conv blocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Additive AR | 289,472 | 128 | 2 | 256 | 0 |
| Conv-transformer k3 | 388,544 | 128 | 2 | 256 | 2 |
| Small | 298,624 | 96 | 3 | 192 | 2 |
| Tiny | 91,712 | 64 | 2 | 128 | 1 |

## Runtime

Runtime is local CPU wall-clock time. CUDA was unavailable because the local
driver is too old for the installed PyTorch build.

| Candidate | Training s | Evaluation s | Best epoch | Best eval CE |
| --- | ---: | ---: | ---: | ---: |
| Additive AR | 89.81 | 34.18 | 32 | 3.6963 |
| Conv-transformer k3 | 105.13 | 47.23 | 20 | 3.6836 |
| Small | 115.54 | 43.39 | 30 | 3.6780 |
| Tiny | 53.34 | 23.67 | 50 | 3.6634 |

## Smooth Metrics

Lower is better. Log-return decoder outputs are converted to normalised prices
before these Hawkes-specific smooth diagnostics are computed.

| Candidate | MMD | SWD | Terminal W1 | Volatility W1 | Drawdown W1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Additive AR | 0.0877 | 0.0150 | 0.0189 | 0.0007 | 0.0114 |
| Conv-transformer k3 | 0.0734 | 0.0127 | 0.0134 | 0.0005 | 0.0084 |
| Small | 0.0859 | 0.0139 | 0.0152 | 0.0007 | 0.0062 |
| Tiny | 0.0918 | 0.0141 | 0.0163 | 0.0005 | 0.0097 |

## Jump Metrics

Lower W1 is better. Jump detection uses thresholds fitted on the real
evaluation paths.

| Candidate | Jump-count W1 | Inter-arrival W1 | Jump-size W1 | Generated jump count | Negative jump fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Additive AR | 0.0352 | 13.8182 | 0.0128 | 0.2070 | 1.0000 |
| Conv-transformer k3 | 0.0430 | 18.8333 | 0.0121 | 0.2246 | 1.0000 |
| Small | 0.0459 | 15.8824 | 0.0147 | 0.1963 | 1.0000 |
| Tiny | 0.0244 | 14.4583 | 0.0127 | 0.2178 | 1.0000 |

## VaR and ES

Lower-tail risk is computed from generated one-step log returns after
normalised-price conversion.

| Candidate | VaR 1\% | ES 1\% | VaR 5\% | ES 5\% |
| --- | ---: | ---: | ---: | ---: |
| Additive AR | -0.0717 | -0.0988 | -0.0431 | -0.0618 |
| Conv-transformer k3 | -0.0746 | -0.1006 | -0.0443 | -0.0636 |
| Small | -0.0712 | -0.0973 | -0.0437 | -0.0617 |
| Tiny | -0.0729 | -0.0993 | -0.0438 | -0.0625 |

## Token Diagnostics

| Candidate | Active codes | Sampled perplexity | Marginal L1 | Transition L1 | Run-length W1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Additive AR | 64 | 44.93 | 0.0460 | 0.4201 | 0.0022 |
| Conv-transformer k3 | 64 | 45.18 | 0.0542 | 0.4151 | 0.0015 |
| Small | 64 | 44.30 | 0.0540 | 0.4203 | 0.0007 |
| Tiny | 64 | 45.12 | 0.0463 | 0.4067 | 0.0028 |

## Decision

Keep the current conv-transformer k3 as the selected reference for now. The
small variant is not compelling on this seed: it reduces parameters only
modestly, trains slightly slower than k3, and does not improve the rare-event
metrics.

Do not promote tiny from a single seed, but run seed robustness for tiny next.
Tiny is the only compact candidate that merits the follow-up: it has 23.6\% of
the k3 parameter count, roughly halves training and evaluation runtime, improves
best eval cross-entropy, keeps all 64 codes active, and has the best seed-0
jump-count W1. Its smooth MMD/SWD and tail-risk estimates are weaker than k3,
so the next check should determine whether the jump advantage is stable across
seeds `1` and `2`.

## Artefacts

- Parameter count JSON: `outputs/hawkes_jump_compact_conv_transformer/parameter_counts.json`
- Small prior: `outputs/hawkes_jump_compact_conv_transformer/priors/small/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer_small_seed0`
- Tiny prior: `outputs/hawkes_jump_compact_conv_transformer/priors/tiny/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer_tiny_seed0`
- Evaluations: `outputs/hawkes_jump_compact_conv_transformer/evaluations/`
