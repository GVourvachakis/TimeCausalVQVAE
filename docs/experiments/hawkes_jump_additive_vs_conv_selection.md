# Hawkes/SVMHJD Additive AR versus Conv-Transformer Selection

## Status

This note decides whether `hidden128_logreturn_cb64_additive_ar` should replace
`hidden128_logreturn_cb64_conv_transformer_k3` as the selected Hawkes/SVMHJD
discrete research candidate.

No models were trained for this decision. The simulator parameters are
unchanged, `main` was not updated, and `trained_models/model_registry.yaml`
should not be changed by this prompt.

Evidence was read from:

- `trained_models/hawkes_jump/model_card.md`;
- `trained_models/model_registry.yaml`;
- `docs/experiments/hawkes_jump_final_model_decision.md` from
  `research/hawkes-jump-discrete-benchmark`;
- `docs/experiments/hawkes_jump_continuous_ablation_results.md` from
  `research/hawkes-continuous-ablation`;
- `outputs/hawkes_jump_matched_continuous_comparison/aggregate_summary.json`;
- `outputs/hawkes_jump_continuous_ablation_comparison/aggregate_summary.json`.

The proposed registry note
`docs/benchmarks/hawkes_jump_model_registry_proposal.md` was not present on this
branch and was not present at the checked path on
`research/hawkes-jump-discrete-benchmark`.

## Scoring Protocol

The local scoring script is `scripts/score_hawkes_prior_selection.py`. It
prefers the aggregate JSON files when available. For tail errors, it uses the
per-seed aggregate rows and their matched `evaluation_summary.json` references
to compute absolute VaR 1%, ES 1%, and negative-jump-fraction errors against the
real Ogata evaluation paths. If aggregate outputs are unavailable, the script
falls back to hard-coded model-card metric rows.

Profiles are lower-is-better rank averages:

- `smooth_profile`: MMD, SWD, terminal W1, volatility W1, and drawdown W1;
- `jump_timing_profile`: jump-count W1, inter-arrival W1, and jump-size W1;
- `tail_profile`: absolute reference error for VaR 1%, ES 1%, and negative jump
  fraction;
- `token_profile`: transition L1 and run-length W1. Sampled perplexity is
  reported descriptively rather than ranked.

The `balanced_profile` is the average rank over the smooth, jump-timing, and
tail profiles.

## Metric Table

Mean / standard deviation across seeds `0/1/2`. Lower is better except for raw
VaR 1%, ES 1%, negative jump fraction, and sampled perplexity, which are
included to make the candidate behaviour visible.

| Candidate | MMD | SWD | Terminal W1 | Volatility W1 | Drawdown W1 | Jump-count W1 | Inter-arrival W1 | Jump-size W1 | VaR 1% | ES 1% | Negative jump fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hidden128_logreturn_cb64_additive_ar` | 0.156719 / 0.064426 | 0.023802 / 0.008500 | 0.032023 / 0.015202 | 0.001132 / 0.000779 | 0.010593 / 0.005852 | 0.046875 / 0.031929 | 6.307987 / 4.223949 | 0.018033 / 0.010062 | -0.074451 / 0.002772 | -0.106841 / 0.007976 | 0.995480 / 0.007828 |
| `hidden128_logreturn_cb64_conv_transformer_k3` | 0.114114 / 0.035514 | 0.018607 / 0.005952 | 0.021678 / 0.011993 | 0.000964 / 0.001021 | 0.011108 / 0.005156 | 0.057617 / 0.032374 | 8.188788 / 6.726952 | 0.017676 / 0.010061 | -0.074836 / 0.002578 | -0.106911 / 0.008011 | 0.998900 / 0.001905 |

## Tail Reference Errors

| Candidate | Abs VaR 1% error | Abs ES 1% error | Abs negative jump fraction error |
| --- | ---: | ---: | ---: |
| `hidden128_logreturn_cb64_additive_ar` | 0.002842 / 0.002090 | 0.007310 / 0.003448 | 0.030132 / 0.017451 |
| `hidden128_logreturn_cb64_conv_transformer_k3` | 0.002756 / 0.002164 | 0.006181 / 0.005426 | 0.033552 / 0.013083 |

## Profile Winners

| Candidate | Smooth rank | Jump-timing rank | Tail rank | Token rank | Balanced rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hidden128_logreturn_cb64_additive_ar` | 1.800 | 1.333 | 1.667 | 1.000 | 1.600 |
| `hidden128_logreturn_cb64_conv_transformer_k3` | 1.200 | 1.667 | 1.333 | 2.000 | 1.400 |

The smooth-profile winner is
`hidden128_logreturn_cb64_conv_transformer_k3`. It wins MMD, SWD, terminal W1,
and volatility W1, while additive AR only has a small drawdown W1 advantage.

The jump-timing-profile winner is `hidden128_logreturn_cb64_additive_ar`. It
wins jump-count W1 and inter-arrival W1, while the conv-transformer is slightly
better on jump-size W1.

The tail-profile winner is
`hidden128_logreturn_cb64_conv_transformer_k3` under matched per-seed reference
errors. It has lower VaR 1% and ES 1% absolute errors, while additive AR has the
lower negative-jump-fraction error. The tail gap is small, so this profile does
not justify switching the overall selected model.

The token-profile rank winner is `hidden128_logreturn_cb64_additive_ar` on
transition L1 and run-length W1:

| Candidate | Transition L1 | Run-length W1 | Sampled perplexity |
| --- | ---: | ---: | ---: |
| `hidden128_logreturn_cb64_additive_ar` | 0.434098 / 0.015278 | 0.003103 / 0.003255 | 44.364707 / 0.702811 |
| `hidden128_logreturn_cb64_conv_transformer_k3` | 0.434527 / 0.017726 | 0.003693 / 0.002232 | 44.419824 / 0.802644 |

Sampled perplexity is almost identical and should be read as a utilisation
diagnostic, not as an independent lower-is-better selection objective.

## Balanced Decision

The balanced-profile winner is
`hidden128_logreturn_cb64_conv_transformer_k3` with rank score `1.400` versus
`1.600` for additive AR. This satisfies the explicit rule for retaining the
conv-transformer: it wins the balanced and smooth profiles, and it does not
materially degrade the tail profile. Its weaker jump-count and inter-arrival W1
results are real but application-specific rather than decisive for the overall
research candidate.

Final decision:

- keep `hidden128_logreturn_cb64_conv_transformer_k3` as the selected
  Hawkes/SVMHJD discrete research candidate;
- keep `hidden128_logreturn_cb64_additive_ar` as the required ablation and the
  specialised candidate for jump-count and inter-arrival diagnostics;
- do not switch the registry to additive AR in this prompt.

## Exact Public Wording

README wording:

```text
Hawkes/SVMHJD has an optional research-candidate registry entry. The
S&P500/VIX workflow remains the public default demo. The selected Hawkes/SVMHJD
research candidate is the hidden128 log-return cb64 tokenizer + causal
conv-transformer k3 prior. The required ablation is the hidden128 log-return
cb64 tokenizer + additive AR prior. Continuous comparators use the log-return
BetaCVAE and InfoCVAE configurations.

The benchmark remains a scenario-data stress test, not an arbitrage-free
pricing model. No Hawkes/SVMHJD trained weights, checkpoints, token tensors,
generated samples, W&B exports, or output summaries are committed.
```

Model-card wording:

```text
`hidden128_logreturn_cb64_conv_transformer_k3` remains the selected
Hawkes/SVMHJD research candidate. It wins the balanced and smooth profiles,
preserves the lower-tail reference errors at least as well as additive AR, and
does not materially degrade sparse-jump evidence for the current
research-candidate use case. `hidden128_logreturn_cb64_additive_ar` remains the
required ablation and the preferred specialised baseline for jump-count and
inter-arrival diagnostics.
```
