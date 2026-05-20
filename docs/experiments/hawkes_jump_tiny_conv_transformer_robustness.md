# Hawkes/SVMHJD Tiny Conv-Transformer Robustness

## Scope

This robustness pass tests whether the tiny causal conv-transformer prior can
replace the selected k3 conv-transformer prior for the Hawkes/SVMHJD log-return
token setting. The run keeps the prior family, sampling policy, tokenizer
configuration, codebook size, condition convention, and simulator outputs
unchanged.

Tiny seeds `1` and `2` use the matching tokenizer and token artefacts from the
existing Hawkes log-return robustness run. Tiny seed `0` is the compact
comparison run from
`outputs/hawkes_jump_compact_conv_transformer/evaluations/tiny_seed0`. Evaluation
uses `n_sample=1024`, temperature `1.0`, unrestricted `top_k`, and sampling
seeds `0`, `1`, and `2`, matching the existing three-seed robustness convention
rather than using a fixed sampling seed of `99`.

## Tiny Seed Results

| Seed | Train s | Eval s | MMD | SWD | Terminal W1 | Volatility W1 | Drawdown W1 | Jump-count W1 | Inter-arrival W1 | Jump-size W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 53.34 | 23.67 | 0.0918 | 0.0141 | 0.0163 | 0.0005 | 0.0097 | 0.0244 | 14.4583 | 0.0127 |
| 1 | 55.42 | 26.94 | 0.1495 | 0.0247 | 0.0297 | 0.0021 | 0.0131 | 0.0664 | 2.4324 | 0.0294 |
| 2 | 62.11 | 21.12 | 0.2289 | 0.0347 | 0.0464 | 0.0006 | 0.0073 | 0.0195 | 20.1765 | 0.0130 |

| Seed | Negative jump fraction | VaR 1\% | ES 1\% | Active codes | Sampled perplexity | Transition L1 | Run-length W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.0000 | -0.0729 | -0.0993 | 64 | 45.1205 | 0.4067 | 0.0028 |
| 1 | 0.9964 | -0.0765 | -0.1140 | 64 | 43.9053 | 0.4306 | 0.0046 |
| 2 | 1.0000 | -0.0749 | -0.1089 | 64 | 43.6864 | 0.4138 | 0.0022 |

## Aggregate Comparison

Values are mean ± sample standard deviation over seeds `0`, `1`, and `2`.
Reference evaluation runtimes are unavailable because the existing additive AR
and k3 robustness summaries pre-date evaluator runtime recording.

| Prior | Parameters | Train s | Eval s | MMD | SWD | Terminal W1 | Volatility W1 | Drawdown W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Additive AR | 289,472 | 93.19 ± 0.26 | n/a | 0.1567 ± 0.0644 | 0.0238 ± 0.0085 | 0.0320 ± 0.0152 | 0.0011 ± 0.0008 | 0.0106 ± 0.0059 |
| Conv-transformer k3 | 388,544 | 109.69 ± 0.20 | n/a | 0.1141 ± 0.0355 | 0.0186 ± 0.0060 | 0.0217 ± 0.0120 | 0.0010 ± 0.0010 | 0.0111 ± 0.0052 |
| Tiny conv-transformer | 91,712 | 56.95 ± 4.59 | 23.91 ± 2.92 | 0.1567 ± 0.0688 | 0.0245 ± 0.0103 | 0.0308 ± 0.0151 | 0.0011 ± 0.0009 | 0.0100 ± 0.0029 |

## Jump-Regime Profile

| Prior | Jump-count W1 | Inter-arrival W1 | Jump-size W1 | Negative jump fraction |
| --- | ---: | ---: | ---: | ---: |
| Additive AR | 0.0469 ± 0.0319 | 12.7327 ± 8.2921 | 0.0180 ± 0.0101 | 0.9955 ± 0.0078 |
| Conv-transformer k3 | 0.0576 ± 0.0324 | 16.2591 ± 10.3617 | 0.0177 ± 0.0101 | 0.9989 ± 0.0019 |
| Tiny conv-transformer | 0.0368 ± 0.0258 | 12.3557 ± 9.0570 | 0.0184 ± 0.0096 | 0.9988 ± 0.0021 |

The tiny prior has the best mean jump-count W1 and inter-arrival W1, while its
jump-size W1 is effectively tied with the additive and k3 references under the
observed seed variability. This preserves the rare-event advantage that made
the compact candidate worth testing.

## Tail-Risk Profile

| Prior | VaR 1\% | ES 1\% |
| --- | ---: | ---: |
| Additive AR | -0.0745 ± 0.0028 | -0.1068 ± 0.0080 |
| Conv-transformer k3 | -0.0748 ± 0.0026 | -0.1069 ± 0.0080 |
| Tiny conv-transformer | -0.0747 ± 0.0018 | -0.1074 ± 0.0075 |

The 1\% lower-tail estimates are materially unchanged across the three priors.
Tiny does not introduce a visible tail-risk degradation, but it also does not
improve the VaR/ES profile.

## Token Diagnostics

| Prior | Active codes | Sampled perplexity | Transition L1 | Run-length W1 |
| --- | ---: | ---: | ---: | ---: |
| Additive AR | 64.0 ± 0.0 | 44.3647 ± 0.7028 | 0.4341 ± 0.0153 | 0.0031 ± 0.0033 |
| Conv-transformer k3 | 64.0 ± 0.0 | 44.4198 ± 0.8026 | 0.4345 ± 0.0177 | 0.0037 ± 0.0022 |
| Tiny conv-transformer | 64.0 ± 0.0 | 44.2374 ± 0.7726 | 0.4170 ± 0.0123 | 0.0032 ± 0.0012 |

All models retain full sampled code coverage. Tiny slightly improves transition
L1 relative to both references and keeps run-length W1 within the same range.

## Runtime and Capacity

Tiny uses 91,712 parameters, which is 23.6\% of the selected k3 conv-transformer
and 31.7\% of the additive AR reference. Its mean training runtime is 56.95 s,
or about 52\% of the k3 runtime and 61\% of the additive AR runtime on the local
CPU setup.

The efficiency gain is therefore substantial. The cost is a weaker smooth
profile: k3 remains better on MMD, SWD, terminal W1, and volatility W1, while
drawdown W1 is similar across all three candidates.

## Decision

Keep the current k3 conv-transformer as the selected research prior, and record
tiny as the efficiency ablation. Tiny is compelling for parameter and runtime
efficiency and preserves the jump-count/inter-arrival advantage, but it does not
match the k3 smooth-distribution profile across seeds.

Do not promote tiny as the selected research candidate in this pass. Do not keep
the additive AR as the selected prior either: additive remains a useful
jump-only reference, but it does not improve the overall smooth profile relative
to k3. A further tiny-capacity variant is optional only if the next question is
efficiency at fixed jump fidelity; it is not required before retaining k3 as the
balanced selected model.

## Artefacts

- Parameter counts:
  `outputs/hawkes_jump_compact_conv_transformer/parameter_counts_final.json`
- Aggregate JSON:
  `outputs/hawkes_jump_compact_conv_transformer/aggregate_prior_comparison.json`
- Aggregate CSV:
  `outputs/hawkes_jump_compact_conv_transformer/aggregate_prior_runs.csv`
- Tiny seed1 prior:
  `outputs/hawkes_jump_compact_conv_transformer/priors/tiny_seed1/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer_tiny_seed1`
- Tiny seed2 prior:
  `outputs/hawkes_jump_compact_conv_transformer/priors/tiny_seed2/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer_tiny_seed2`
- Tiny seed1 evaluation:
  `outputs/hawkes_jump_compact_conv_transformer/evaluations/tiny_seed1`
- Tiny seed2 evaluation:
  `outputs/hawkes_jump_compact_conv_transformer/evaluations/tiny_seed2`
