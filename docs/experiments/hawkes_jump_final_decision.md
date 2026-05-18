# Hawkes-Jump Final Decision

## Status

The Hawkes/SVMHJD benchmark should be split into two tracks:

- merge the simulator and diagnostic infrastructure to `main` through a
  simulator-only branch;
- keep model-selection and registry work on the research branch until the
  continuous log-return baseline is repaired and rerun.

This is a positive infrastructure decision, but not a model-registry decision.

## Simulator Decision

Merge the Ogata/fixed-grid Hawkes-jump dataset infrastructure to `main`, subject
to a clean simulator-only review. The Ogata backend is research-quality: it uses
continuous-time Ogata modified thinning, exponential Hawkes intensity decay,
branching-ratio validation, asymmetric folded-normal marks, mark-dependent
excitation, jump-excited volatility, exact Brownian variance integration within
event-free sub-intervals, and O(n_events + n_timestep) projection onto the model
grid.

Keep the fixed-grid backend as a smoke and throughput backend. It is useful for
fast deterministic checks, but research-quality comparisons should use
`simulation_scheme: ogata`.

The simulator merge should include the dataset, smoke script, visual diagnostic
script, jump diagnostics, no-leakage check, and documentation. It should not
include generated outputs, trained checkpoints, or registry entries. The
no-arbitrage caveat remains: these are scenario paths observed on a regular
grid, not an arbitrage-free risk-neutral pricing model.

## Model Decision

Do not add Hawkes/SVMHJD to `trained_models/model_registry.yaml` now. The current
registry has no Hawkes entry, and that should remain true until a matched
continuous log-return baseline trains successfully and the discrete advantage is
confirmed against it.

The discrete log-return result is promising, but the benchmark is not
registry-ready because the continuous log-return BetaCVAE failed before
training on all three robustness seeds. A registry update would overstate the
evidence.

## Best Continuous Model

There is no valid best continuous log-return model yet. The planned matched
continuous BetaCVAE baseline failed for seeds `0/1/2` with
`ArithmeticError: NaN detected in train loss` before completing epoch 1. The
likely cause is the current log-return continuous config applying `transform:
log` and `inverse_transform: exp` to signed log-return data.

The earlier price-level BetaCVAE result remains useful as a diagnostic
reference, but it is not a matched log-return baseline. It reported MMD
`0.9638`, SWD `0.1133`, terminal W1 `0.2263`, mean detected jumps per path
`16.8164`, and jump-count W1 `16.5742`, which made it weak on the
jump-regime profile. That result should not be used as a registry comparator
for the log-return discrete priors.

## Best Discrete Model

The best overall discrete candidate is
`hidden128_logreturn_cb64 + causal conv-transformer k3`. Across three seeds, it
has the strongest smooth profile and full sampled code usage:

| Metric | Mean | Std |
|---|---:|---:|
| MMD | 0.1141 | 0.0355 |
| SWD | 0.0186 | 0.0060 |
| Terminal W1 | 0.0217 | 0.0120 |
| Volatility W1 | 0.0010 | 0.0010 |
| Drawdown W1 | 0.0111 | 0.0052 |
| Sampled active codes | 64.0 | 0.0 |
| Sampled perplexity | 44.42 | 0.80 |
| Best prior CE | 3.6503 | 0.0292 |

The cb64 additive AR prior remains the strongest jump-count baseline. It has
slightly better mean jump-count W1 and inter-arrival W1:

| Metric | cb64 additive mean/std | cb64 conv-transformer mean/std |
|---|---:|---:|
| Jump-count W1 | 0.0469 / 0.0319 | 0.0576 / 0.0324 |
| Inter-arrival W1 | 6.3442 / 4.3467 | 8.1075 / 6.6507 |
| Jump-size W1 | 0.0180 / 0.0101 | 0.0177 / 0.0101 |
| Negative jump fraction | 0.9955 / 0.0078 | 0.9989 / 0.0019 |
| VaR 1% | -0.0745 / 0.0028 | -0.0748 / 0.0026 |
| ES 1% | -0.1068 / 0.0080 | -0.1069 / 0.0080 |

Use the conv-transformer as the leading registry candidate once a valid
continuous comparator exists. Keep additive AR as the mandatory ablation because
it is competitive on jump diagnostics.

## Robustness

The tokenizer-utilisation result is robust. Extracted token usage across seeds
was:

| Seed | Extracted active codes | Extracted perplexity |
|---:|---:|---:|
| 0 | 63 / 64 | 44.80 |
| 1 | 64 / 64 | 44.00 |
| 2 | 64 / 64 | 43.16 |

Both priors sampled all 64 codes on all seeds. This fixes the first comparison's
price-tokenizer collapse, where the standard tokenizer extracted `6/64` active
codes and the hidden128 tokenizer extracted `4/64`.

The remaining robustness weakness is not token collapse. It is comparator
coverage: the matched continuous log-return baseline has no usable metrics.

## Metric Profile

The smooth profile favours the cb64 conv-transformer. It has lower mean MMD,
SWD, and terminal W1 than cb64 additive AR, with similar drawdown and volatility
W1.

The jump-regime profile is mixed but healthy. Both cb64 priors reproduce the
sparse downside-jump scale and preserve negative jump fractions near the real
Ogata reference. Additive AR is slightly better on jump-count and inter-arrival
W1; conv-transformer is similar on jump-size W1 and stronger on smooth path
metrics.

The VaR/ES profile is close to the simulator reference. The seed-0 real Ogata
reference had VaR 1% `-0.0724` and ES 1% `-0.1064`. Across seeds, additive AR
had VaR 1% `-0.0745 +/- 0.0028` and ES 1% `-0.1068 +/- 0.0080`; the
conv-transformer had VaR 1% `-0.0748 +/- 0.0026` and ES 1%
`-0.1069 +/- 0.0080`.

Token diagnostics are registry-plausible but not sufficient alone. The
conv-transformer sampled all 64 codes, with sampled perplexity `44.42 +/- 0.80`,
transition L1 `0.4345 +/- 0.0177`, run-length W1 `0.0037 +/- 0.0022`, and best
CE `3.6503 +/- 0.0292`. Transition fidelity remains an improvement target, but
it is no longer masked by code collapse.

## Decision

Do not update the registry now.

Proceed with a simulator-only public merge plan.

Run more experiments before selecting a model. The required experiment is a
matched continuous log-return baseline with a signed-data-safe transform,
followed by the same three-seed comparison against cb64 additive AR and cb64
conv-transformer k3.

Do not stop the branch. The branch has produced useful infrastructure and
credible discrete evidence, but one blocker remains before a public model
selection claim.

## Exact Next Plan

1. Create a simulator-only merge branch from `main`, for example
   `feature/hawkes-jump-simulator`.
2. Cherry-pick only infrastructure and documentation needed for public use:
   the Hawkes dataset backends, jump diagnostics, smoke and plotting scripts,
   no-leakage check, configs needed for smoke, and simulator documentation.
3. Exclude trained outputs, robustness outputs, model-selection claims, and
   `trained_models/model_registry.yaml` changes.
4. On the research branch, create a follow-up branch for the matched continuous
   baseline, for example `research/hawkes-logreturn-continuous-baseline`.
5. Fix the continuous log-return config to use a signed-data-safe transform
   without changing Ogata simulator parameters.
6. Rerun the three-seed comparison for continuous BetaCVAE or InfoCVAE, cb64
   additive AR, and cb64 conv-transformer k3.
7. Reconsider the registry only if the cb64 conv-transformer or additive AR
   keeps its jump-regime advantage without unacceptable smooth-metric
   regression against the repaired continuous baseline.
