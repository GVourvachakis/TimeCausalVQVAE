# Signature Conditioning Final Decision V2

## Purpose

This record updates the S&P500/VIX signature-conditioning decision after three
additional checks beyond the first final decision:

- standardised `logsig_l3_ctx20` additive conditioning;
- evaluation-only signature-kernel MMD on saved paper-style batches;
- FiLM/AdaLN-style `adaln_lite` conditioning with standardised
  `logsig_l3_ctx20` features.

No code was implemented and no model was trained for this document.

## 1. VIX-Only Baseline Status

The promoted public discrete baseline remains:

- standard causal VQ tokenizer;
- additive scalar VIX-only causal AR token prior;
- market-calibrated sampling with `temperature=0.8` and `top_k=40`;
- paper-style diagnostics and latent-geometry diagnostics as the main
  verification path.

The VIX-only additive prior remains the best discrete model for the
distributional guardrail metrics. Under the same paper-style setting it records
MMD `0.27934083` and SWD `0.00767375`, both stronger than the current
signature-conditioned discrete variants.

## 2. Raw `logsig_l3_ctx20` Status

The raw additive `logsig_l3_ctx20` model remains the strongest
signature-conditioning research candidate.

Setup:

- context length: `20`;
- log-signature depth: `3`;
- lead-lag transform;
- time and VIX channels;
- raw `iisignature` log-signature features;
- feature dimension: `385`;
- total condition dimension after scalar VIX concatenation: `386`;
- additive condition embedding.

Same-setting paper-style metrics:

| Metric | Raw `logsig_l3_ctx20` |
| --- | ---: |
| MMD | 0.34163502 |
| SWD | 0.01082886 |
| Returns W1 | 0.00099378 |
| Terminal W1 | 0.00451245 |
| Volatility W1 | 0.00080577 |
| Drawdown W1 | 0.00549569 |
| Return AC L1 | 0.04789805 |
| Squared-return AC L1 | 0.03506229 |

Interpretation: raw `logsig_l3_ctx20` regresses the distributional profile
relative to VIX-only, but it is the strongest discrete signature result for
tail-risk and sequential-dependence metrics. It remains valuable research
evidence, but its robustness grid did not justify replacing the public default.

## 3. Standardised Additive Logsig Status

The standardised additive `logsig_l3_ctx20_std` run verifies that train-set
standardisation of log-signature features is feasible and numerically stable.
It does not improve enough profiles to replace the raw signature candidate.

Same-setting paper-style metrics:

| Metric | Standardised additive `logsig_l3_ctx20` |
| --- | ---: |
| MMD | 0.37982193 |
| SWD | 0.01154852 |
| Returns W1 | 0.00108519 |
| Terminal W1 | 0.00272364 |
| Volatility W1 | 0.00111281 |
| Drawdown W1 | 0.00570303 |
| Return AC L1 | 0.05253890 |
| Squared-return AC L1 | 0.03923871 |

Interpretation: standardisation improves terminal-return W1 relative to the
raw signature and VIX-only runs, and it is narrowly favoured by no-lead-lag
signature-kernel MMD among the discrete signature models. It does not beat raw
`logsig_l3_ctx20` on returns W1, volatility W1, drawdown W1, return
autocorrelation, squared-return autocorrelation, or balanced-market rank.

## 4. FiLM/AdaLN Logsig Status

The FiLM/AdaLN run used the existing `adaln_lite` conditioning path with scalar
VIX plus standardised `logsig_l3_ctx20` features.

Token-prior training improved substantially:

| Metric | FiLM/AdaLN `logsig_l3_ctx20_std` |
| --- | ---: |
| Train CE | 0.69486250 |
| Train accuracy | 0.73076924 |
| Train perplexity | 2.00356812 |
| Eval CE | 0.56481865 |
| Eval accuracy | 0.78709130 |
| Eval perplexity | 1.76141225 |

However, decoded and paper-style diagnostics did not improve:

| Metric | FiLM/AdaLN `logsig_l3_ctx20_std` |
| --- | ---: |
| MMD | 0.36974820 |
| SWD | 0.01028688 |
| Returns W1 | 0.00126779 |
| Terminal W1 | 0.00780297 |
| Volatility W1 | 0.00119087 |
| Drawdown W1 | 0.01071865 |
| Return AC L1 | 0.05361288 |
| Squared-return AC L1 | 0.04005349 |

Interpretation: stronger in-block conditioning improves teacher-forced token
prediction, but that improvement does not transfer to market path quality under
the current sampling policy. FiLM/AdaLN is worse than raw additive
`logsig_l3_ctx20` on tail-risk, sequential-dependence, balanced-market rank,
and no-lead-lag signature-kernel MMD. Do not continue to a FiLM/AdaLN seed
ablation yet.

## 5. Signature-Kernel Metric Status

`sigkernel` is available only as an optional local dependency. It is not part
of the default project dependency set.

No-lead-lag signature-kernel MMD, lower is better:

| Model | Signature-kernel MMD |
| --- | ---: |
| Continuous BetaCVAE | 0.00087344 |
| Standardised `logsig_l3_ctx20` additive | 0.00224031 |
| Raw `logsig_l3_ctx20` additive | 0.00238908 |
| VIX-only additive | 0.00555772 |
| FiLM/AdaLN `logsig_l3_ctx20_std` | 0.00650810 |

Lead-lag 64-sample signature-kernel MMD:

| Model | Signature-kernel MMD |
| --- | ---: |
| VIX-only additive | 0.00232711 |
| FiLM/AdaLN `logsig_l3_ctx20_std` | 0.00540422 |
| Standardised `logsig_l3_ctx20` additive | 0.00623763 |
| Raw `logsig_l3_ctx20` additive | 0.01525021 |
| Continuous BetaCVAE | 0.01637310 |

Interpretation: no-lead-lag signature-kernel MMD supports keeping signature
conditioning as a research direction because it favours the raw and
standardised additive signature variants over VIX-only. It does not support
FiLM/AdaLN promotion. The smaller lead-lag pass gives a different ranking and
therefore remains a feasibility check, not a promotion gate.

Signature-kernel MMD should remain exploratory. It may inform future
model-selection reports, but it must not be used as the sole selection metric.

## 6. Metrics By Selection Profile

The profile scores below are lower-is-better average ranks across the visible
paper-style component metrics in `docs/architecture/model_selection_profiles.md`.

| Model | Distributional | Tail-risk | Sequential-dependence | Balanced-market |
| --- | ---: | ---: | ---: | ---: |
| VIX-only additive | 2.333 | 4.333 | 4.000 | 3.714 |
| Raw `logsig_l3_ctx20` additive | 3.000 | 1.667 | 2.000 | 2.143 |
| Standardised `logsig_l3_ctx20` additive | 4.333 | 2.000 | 3.500 | 3.000 |
| FiLM/AdaLN `logsig_l3_ctx20_std` | 4.000 | 4.333 | 4.500 | 4.286 |
| Continuous BetaCVAE | 1.333 | 2.667 | 1.000 | 1.857 |

### Distributional Profile

VIX-only remains the strongest discrete model by MMD and SWD. Continuous
BetaCVAE remains the strongest overall reference by MMD and returns W1. The
signature-conditioned variants do not justify default promotion under this
profile.

### Tail-Risk Profile

Raw `logsig_l3_ctx20` is the strongest discrete model by the profile rank and
by the combination of volatility W1 and drawdown W1. Standardised additive
`logsig_l3_ctx20` is best on terminal W1, but does not dominate the whole
tail-risk profile. FiLM/AdaLN does not improve tail-risk diagnostics.

### Sequential-Dependence Profile

Raw `logsig_l3_ctx20` is the strongest discrete candidate by return and
squared-return autocorrelation rank. Continuous BetaCVAE remains the strongest
overall reference. FiLM/AdaLN regresses despite better token likelihood.

### Balanced-Market Profile

Raw `logsig_l3_ctx20` remains the strongest discrete signature variant by
balanced-market rank. VIX-only remains the default because it is simpler,
stronger on distributional guardrails, and the raw signature gains were not
robust enough across follow-up ablations.

## 7. Decision

Decision:

```text
keep VIX-only default; retain raw additive logsig_l3_ctx20 as the optional
signature-conditioning research candidate; do not continue FiLM/AdaLN seed
ablation yet; do not defer signatures entirely.
```

Detailed choices:

- do not promote signature conditioning as the public default;
- keep the VIX-only additive prior as the public default;
- keep raw additive `logsig_l3_ctx20` as the best signature-conditioning
  research reference;
- retain standardised additive features as optional infrastructure and an
  ablation reference;
- stop FiLM/AdaLN escalation for now because token-likelihood gains did not
  improve path metrics;
- continue to treat signature-kernel MMD as exploratory evidence, not a gate.

## 8. Merge Scope

### Remain Only On `research/signature-conditioning`

The following should remain on the research branch unless a later promotion
decision is made:

- signature-conditioned experiment configs;
- trained signature-conditioned checkpoint paths and output artifacts;
- robustness ablation results;
- FiLM/AdaLN signature-conditioning quality report;
- optional `sigkernel` install state and generated signature-kernel outputs;
- any result that changes the promoted model recommendation.

### Candidate Merge Scope For `feat/causal-vq-vae`

The following can be merged as optional infrastructure if they are default-off
and do not change VIX-only behaviour:

- documentation of model-selection profiles;
- optional condition-feature loading support where config defaults are `null`;
- optional log-signature feature extraction utilities and scripts, with clear
  optional-dependency errors;
- optional signature-kernel evaluation utilities, if `sigkernel` remains
  optional and absent from default dependencies;
- W&B execution-profile documentation for restricted environments;
- verification reports that clarify why VIX-only remains the public default.

Do not merge mandatory `iisignature` or `sigkernel` dependency changes into the
default project path without a separate dependency decision.

## 9. Future Work

- Revisit cross-attention only if the condition becomes a sequential/raw
  context path, a covariate-token sequence, or a set of time-indexed market
  features. It is not needed for a fixed log-signature vector.
- Revisit Gumbel-Softmax only after selecting a differentiable path-level
  objective, such as a stable signature-kernel loss. It is not needed for the
  current teacher-forced token cross-entropy objective.
- Revisit adapted or causal Wasserstein only after a final signature variant is
  selected. It remains a heavy downstream metric, not a blocker for the current
  default.
- If signature conditioning is revisited, start from raw additive
  `logsig_l3_ctx20` and investigate checkpoint/seed stability before adding
  stronger conditioning mechanisms.

## Check Status

Completed checks:

- `poetry run ruff format docs`;
- `poetry run ruff check docs --fix`;
- `poetry check`.
