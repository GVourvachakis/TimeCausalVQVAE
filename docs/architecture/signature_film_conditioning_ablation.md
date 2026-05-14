# Signature FiLM/AdaLN Conditioning Ablation

This note specifies a design-only ablation for stronger conditioning of the
causal token prior on VIX plus log-signature features. It does not implement a
new architecture, train models, add cross-attention, or introduce
Gumbel-Softmax.

## 1. Motivation

The current signature-conditioning branch concatenates historical
log-signature features to the scalar VIX label and injects the resulting vector
through the existing additive condition embedding. This is intentionally simple
and keeps the promoted prior close to the VIX-only baseline, but it may be too
weak for high-dimensional log-signature features.

The strongest current signature candidate, `logsig_l3_ctx20`, uses a
386-dimensional condition vector after concatenating scalar VIX with the
truncated log-signature features. Additive injection asks one projected
condition vector to shift token embeddings uniformly before the transformer
stack. That may underuse path-shape information, especially for tail-risk,
volatility, drawdown, and sequential-dependence diagnostics where different
transformer layers may need different condition-dependent behaviour.

FiLM or AdaLN-style modulation is the next architecture ablation because it
keeps the condition as a fixed vector, preserves the causal token model, and can
make the condition act inside transformer blocks without introducing a
sequence-to-sequence conditioning mechanism.

## 2. Method

The proposed variant modulates normalised hidden states inside causal
transformer blocks using scale and shift parameters predicted from the
condition vector:

```text
gamma, beta = MLP(condition)
h = norm(h) * (1 + gamma) + beta
```

For this repository, the first implementation target should be the existing
FiLM/AdaLN family already represented by the token-prior configuration value
`condition_injection: adaln_lite`. The design intent is:

- input condition: `[scalar VIX, standardised log-signature features]`;
- condition dimension: `386` for `logsig_l3_ctx20`;
- modulation network: an MLP from the full condition vector to per-hidden-unit
  scale and shift parameters;
- transformer location: apply modulation after layer normalisation and before
  attention or MLP sublayers;
- sampling: keep the same market-calibrated settings used in current
  comparisons, `temperature=0.8` and `top_k=40`.

The ablation should not alter tokenizer weights, token sequence construction,
causal attention masks, sampling rules, or the metric definitions. Any code
change, when a future implementation prompt requests it, should be narrowly
scoped to selecting or extending the existing single-code token-prior
conditioning mode.

## 3. Causality

The condition must remain admissible at the start of the generated window.
Specifically:

- log-signature features must be computed only from the historical context path
  before the target window;
- generated target path values must not enter the condition;
- future VIX sequences must not enter the condition;
- the token prior must keep the same causal mask over generated token
  positions;
- evaluation must use the same context alignment convention as the existing
  log-signature feature extraction reports.

The FiLM/AdaLN mechanism changes how an admissible condition vector modulates
the hidden states. It does not permit the model to inspect future target tokens
or future path values.

## 4. Variants

The controlled ablation should compare four variants under identical data,
tokenizer, sampling, and paper-style evaluation settings.

| Variant | Condition features | Injection | Purpose |
| --- | --- | --- | --- |
| VIX-only additive baseline | scalar VIX only | additive | promoted public baseline and regression guard |
| Raw logsig additive | scalar VIX plus raw `logsig_l3_ctx20` | additive | current high-dimensional signature conditioning without feature standardisation |
| Standardised logsig additive | scalar VIX plus standardised `logsig_l3_ctx20_std` | additive | tests whether train-set feature scaling is enough |
| Standardised logsig FiLM/AdaLN | scalar VIX plus standardised `logsig_l3_ctx20_std` | FiLM/AdaLN-style modulation | tests whether stronger in-block conditioning improves path metrics |

The FiLM/AdaLN variant should keep the same tokenizer artefacts, token data,
single-code prior family, context length, log-signature depth, optimiser
defaults, and sampling settings unless a later robustness prompt explicitly
varies seeds, epochs, or learning rate.

## 5. Why Not Cross-Attention Yet

Cross-attention is deferred because the current log-signature condition is a
fixed finite-dimensional vector, not a temporal sequence or set of covariate
tokens. Treating it as a sequence would add architectural complexity without a
clear representation need.

Cross-attention becomes a better candidate only if the condition changes from a
summary vector to one of the following:

- a raw historical context path;
- a sequence of market covariates;
- per-time-step VIX or realised-volatility tokens;
- a set of multi-scale context features with meaningful token identities.

Until then, FiLM/AdaLN is the more direct conditioning ablation.

## 6. Success Criteria

The FiLM/AdaLN variant should be judged with the model-selection profiles in
`docs/architecture/model_selection_profiles.md`, not by a single scalar metric.
It is successful only if it satisfies all of the following:

- improves the tail-risk or balanced-market profile relative to the
  standardised additive log-signature variant;
- avoids a severe regression in MMD or SWD relative to the VIX-only and
  additive log-signature baselines;
- maintains stable token-prior training diagnostics, including cross-entropy,
  accuracy, and perplexity;
- remains numerically stable in decoded-path evaluation;
- shows consistent behaviour across at least two seeds.

Signature-kernel MMD can be reported as an exploratory evaluation metric once
the optional `sigkernel` workflow is stable, but it should not be used as a
training objective or sole promotion criterion for this ablation.

## 7. Non-Goals

This ablation does not include:

- Gumbel-Softmax or another token relaxation;
- signature-kernel training loss;
- diffusion;
- MGVQ;
- GroupedRVQ;
- cross-attention;
- tokenizer architecture changes;
- adapted or causal Wasserstein implementation.

The promoted default remains the VIX-only additive prior until a
signature-conditioned variant demonstrates robust improvements under the
selection profiles and seed checks.
