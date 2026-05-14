# Model Selection Profiles for S&P500/VIX Market Generators

Status: architecture and evaluation policy only. This document does not train
models, change source code, add dependencies, or implement new metrics.

## 1. Motivation

There is no universal metric for market generators. A generator can improve one
view of path quality while degrading another: for example, it can match a
global distributional distance while missing tail risk, or improve terminal
return and volatility functionals while worsening aggregate MMD.

MMD and SWD remain useful because they provide compact distributional checks on
generated paths. They are not sufficient by themselves. Market use cases care
about different failure modes:

- broad distribution matching for visual and statistical realism;
- tail risk for VaR, expected shortfall, stress testing, and drawdown-sensitive
  workflows;
- sequential dependence for volatility clustering, hedging, and adapted
  portfolio decisions.

Model selection should therefore report a profile, not just a scalar score. A
single score may be used as a tie-breaker or ranking summary, but it must never
hide the component metrics.

## 2. Profiles

### A. Distributional profile

Use this profile when the goal is broad path-distribution matching.

Primary metrics:

- MMD;
- SWD;
- one-step returns W1.

Required companion diagnostics:

- lower-tail exceedance rates;
- upper-tail exceedance rates;
- path bounds and non-positive path counts;
- extreme-path plots where available.

This profile should not promote a model that has strong MMD/SWD but clearly
underrepresents severe return tails.

### B. Tail-risk profile

Use this profile when the generated paths will support VaR, expected shortfall,
stress testing, drawdown analysis, or scenario generation for risk review.

Primary metrics:

- terminal-return W1;
- volatility W1;
- maximum-drawdown W1;
- tail exceedance rates.

Required companion diagnostics:

- extreme return diagnostics;
- maximum absolute return per path;
- maximum rolling-volatility diagnostics;
- volatility-tail comparison plots;
- VIX-bucket terminal and volatility distances.

This profile should prefer models that preserve path-functional risk even if
their aggregate MMD is not the best.

### C. Sequential-dependence profile

Use this profile when the generated paths will be used in hedging, adapted
portfolio allocation, or other sequential decision problems.

Primary metrics:

- return autocorrelation L1;
- squared-return autocorrelation L1;
- volatility clustering diagnostics;
- VIX-bucket stability of those diagnostics where available.

Future optional metric:

- signature-kernel distance, once the optional `sigkernel` path is installed,
  smoke-tested, and numerically stable.

This profile should not select a model only because marginal returns look good.
It should explicitly check whether the path order and squared-return dependence
remain plausible.

### D. Balanced market profile

Use this profile as the default research ranking when no single downstream use
case has priority.

The balanced profile should rank or weight:

- SWD;
- returns W1;
- terminal-return W1;
- volatility W1;
- maximum-drawdown W1;
- return autocorrelation L1;
- squared-return autocorrelation L1;
- tail penalty.

The tail penalty should be documented rather than hidden. A simple version is a
rank penalty for underrepresenting real-data lower or upper tail exceedance
rates. A later implementation may formalise this, but the current policy is to
report the component tail rates next to any balanced score.

Balanced ranking is useful for comparing ablations, but the component metrics
must remain visible in verification documents.

## 3. Current S&P500/VIX Reading

Current same-setting paper-style diagnostics used:

- `n_sample=1000`;
- `seed=99`;
- `temperature=0.8`;
- `top_k=40`.

The current evidence is mixed:

- VIX-only is best among the discrete models for global MMD and SWD.
- `logsig_l3_ctx20` is best among the discrete models for the balanced
  market-style view used in the paper-style comparison. It is strongest on
  returns W1, terminal-return W1, volatility W1, maximum-drawdown W1, return
  autocorrelation L1, and squared-return autocorrelation L1.
- `logsig_l3_ctx10` is strongest among the discrete models on flattened
  squared-return autocorrelation, nearly matching the continuous reference in
  the recorded comparison.
- The continuous BetaCVAE reference remains strong on MMD and several
  autocorrelation and return metrics, but it is not the promoted discrete
  token-prior baseline.
- No single model dominates every metric.

Profile interpretation:

| Profile | Current preferred model | Rationale |
| --- | --- | --- |
| Distributional | VIX-only | Best discrete MMD and SWD under the current paper-style setting. |
| Tail-risk | `logsig_l3_ctx20` | Best terminal-return, volatility, drawdown, and most tail-sensitive path-functional metrics among the discrete variants. |
| Sequential-dependence | `logsig_l3_ctx20`, with `logsig_l3_ctx10` as a check | `logsig_l3_ctx20` is best on within-path return and squared-return autocorrelation L1; `logsig_l3_ctx10` is strongest on flattened squared-return autocorrelation. |
| Balanced market | `logsig_l3_ctx20` | Best rank across the currently reported market-style path-functional diagnostics. |

The resulting interpretation is not that signatures have won globally. Rather,
signature conditioning has produced a promising risk/path-function ablation
that still needs robustness checks before promotion.

## 4. Decision Rule

All model-selection reports should keep the full metric table visible. Do not
promote a model solely from:

- MMD;
- SWD;
- a balanced scalar score;
- a token-prior likelihood;
- a single bucket result.

Promotion should require:

1. the selected profile to match the intended application;
2. no severe regression in the other profiles;
3. stable behaviour under at least one additional seed or an equivalent
   robustness check;
4. explicit reporting of tail exceedance rates and VIX-bucket diagnostics.

For VaR, expected shortfall, hedging, or risk scenario generation, use the
tail-risk and balanced market profiles as the primary selectors. The
distributional profile should remain visible as a guardrail, but it should not
override materially better tail and path-functional diagnostics by itself.

For public default selection, prefer the simplest robust model unless a more
complex ablation improves the relevant profile and remains competitive on the
others.

## 5. Next Ablation Target

The next target is `logsig_l3_ctx20` robustness:

- repeat at another seed;
- inspect epoch selection around the best checkpoint;
- run a small learning-rate ablation if the second seed confirms the pattern;
- preserve the same tokenizer, additive prior architecture, and sampling
  settings during the comparison.

The VIX-only additive prior remains the public default until the
`logsig_l3_ctx20` improvements are robust across seed, checkpoint, and sampling
checks.
