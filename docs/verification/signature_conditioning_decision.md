# Signature Conditioning Decision

## Purpose

This note records the current decision after the first S&P500/VIX non-smoke
log-signature conditioning grid. It is documentation only: no model code,
dependencies, tokenisers, priors, Gumbel-Softmax relaxations, or
signature-kernel metrics are introduced here.

## Current Promoted Baseline

The promoted public baseline remains:

- standard causal VQ tokenizer;
- additive scalar VIX-only causal autoregressive token prior;
- market-calibrated sampling with `temperature=0.8` and `top_k=40`;
- S&P500/VIX paper-style diagnostics and latent-geometry diagnostics as the
  main verification stack.

This baseline remains preferred because it retains the best aggregate decoded
MMD among the current same-setting comparisons, while preserving the simpler
one-scalar condition interface.

## Signature-Conditioning Implementation

The implemented signature-conditioning ablation uses only historical
information available before the target generation window:

- build a historical context path for each aligned sample;
- include price, return, cumulative-return, time, and optional VIX channels;
- apply a lead-lag transform to expose order and quadratic-variation-like
  information;
- compute truncated log-signatures with optional `iisignature` support;
- concatenate the log-signature feature vector to the scalar VIX label;
- keep the same additive condition embedding in the causal token prior.

No cross-attention, AdaLN, diffusion model, new tokeniser, or new prior family
was used.

## Package Decision

The package status is:

- `iisignature` is accepted as the optional CPU feature extractor for offline
  truncated signature and log-signature features;
- `signatory` remains deferred and reference-only because it is incompatible
  with the current Python/PyTorch path without a legacy or custom environment;
- `sigkernel` is the next candidate for an evaluation-only signature-kernel
  metric;
- `KSig` remains deferred until a GPU-compatible and Python/NumPy-compatible
  environment is available.

No signature package is promoted into the default dependency path by this
decision.

## Ablation Summary

The current same-setting comparison uses `n_sample=1000`, `temperature=0.8`,
and `top_k=40`.

| Model | MMD | SWD | Terminal W1 | Volatility W1 | Token diagnostic note |
| --- | ---: | ---: | ---: | ---: | --- |
| VIX-only baseline | 0.28880176 | 0.00889640 | 0.01076925 | 0.00132492 | Best aggregate MMD. |
| `logsig_l2_ctx20` | 0.31721902 | 0.00859528 | 0.00708540 | 0.00134660 | Best SWD among the grid. |
| `logsig_l3_ctx10` | 0.31865206 | 0.01090581 | 0.00425231 | 0.00103497 | Stronger terminal and volatility W1. |
| `logsig_l3_ctx20` | 0.32337123 | 0.01074053 | 0.00472903 | 0.00091678 | Best token-space diagnostics in the grid. |

The decision is mixed:

- VIX-only remains best on decoded MMD;
- `logsig_l2_ctx20` is best on SWD;
- depth-3 log-signature variants are best on terminal-return Wasserstein,
  volatility Wasserstein, marginal code usage, transition distance, and
  run-length distance;
- no signature-conditioned variant is promoted as the new default yet.

## Gumbel-Softmax Decision

Do not implement Gumbel-Softmax yet.

Gumbel-Softmax, straight-through estimators, or other soft-token relaxations
should only be reconsidered for differentiable path-level objectives after an
evaluation-only signature-kernel metric is stable, useful, and reproducible.
The current evidence supports evaluation and model-selection work before any
objective-level signature loss.

## Next Work

The next work items are:

1. Train or reproduce the continuous S&P500/VIX baseline so paper-style
   continuous-versus-discrete comparisons are complete.
2. Run paper-style diagnostics for the VIX-only baseline and the depth-3
   log-signature variants under identical sampling settings.
3. Add an evaluation-only signature-kernel metric using `sigkernel` as the next
   candidate package.
4. Revisit promotion only after signature-kernel evaluation, paper-style
   diagnostics, and at least one additional seed support the same conclusion.
