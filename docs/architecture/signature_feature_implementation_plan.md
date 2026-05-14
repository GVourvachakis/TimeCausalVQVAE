# Signature Feature Implementation Plan

Status: implementation plan only. This document does not add dependencies, change promoted model
defaults, modify tokenizers or priors, train models, or implement signature conditioning.

## Scope

Implement optional signature feature extraction and evaluation metrics without changing promoted
model defaults. The promoted S&P500/VIX method remains the standard causal VQ tokenizer with the
additive scalar-conditioned causal AR prior. Signature tools should be opt-in diagnostics and
ablation support until a later implementation thread explicitly enables them.

The package decision is:

- use `iisignature`, if installed, for offline CPU truncated signature and log-signature feature
  extraction;
- use `sigkernel`, if installed, for evaluation-only signature-kernel metrics;
- do not add `signatory`;
- do not add `KSig`;
- defer `pathsig`.

## Module layout

Proposed project-native modules and scripts:

- `src/time_causal_vae/evaluation/signature_features.py`
- `src/time_causal_vae/evaluation/signature_kernel_metrics.py`
- `scripts/extract_signature_features.py`
- `scripts/evaluate_signature_kernel_metrics.py`

These modules should live outside `src/time_causal_vae/evaluation/external/`. The existing
`external/signatures.py` file is an upstream optional expected-signature diagnostic and should not
be extended for new project-native signature work.

## Feature extraction

`signature_features.py` should use `iisignature` only when the caller requests signature features.
If `iisignature` is missing, it should raise a clear optional-dependency error that explains that
the package is not installed by default and points to the manual install notes in `README.md`.

The first implementation should support:

- time channel;
- returns path;
- cumulative log-return path;
- optional lead-lag transform;
- truncation levels 2 and 3 initially;
- finite-value checks for every feature array;
- deterministic ordering of feature names and channels;
- JSON summary output;
- NPZ feature output.

The feature extractor should record a manifest with:

- source path and split;
- preprocessing choices;
- channels used;
- lead-lag status;
- truncation level;
- feature dimension;
- number of paths;
- number of finite rows;
- package version, if available;
- runtime.

The first smoke test should operate on a small S&P500/VIX batch and should not train or sample a
model.

## Signature-kernel metric

`signature_kernel_metrics.py` should use `sigkernel` only when the caller requests
signature-kernel metrics. If `sigkernel` is missing, it should raise a clear optional-dependency
error that explains that the package is not installed by default and points to the manual install
notes in `README.md`.

The first implementation should support:

- small-batch CPU MMD smoke tests;
- real versus generated S&P500/VIX path batches;
- path preprocessing consistent with the feature extractor;
- preprocessing manifest output;
- static-kernel, dyadic-order, dtype, and batch-size recording;
- runtime logging;
- finite-value checks on Gram matrices and scalar distances;
- symmetric-Gram and positive-diagonal checks for smoke runs.

The first metric script should compare already available real and generated path arrays. It should
not train models, alter checkpoints, or select a promoted model by itself.

## Conditioning ablation

A later implementation phase can define a controlled conditioning ablation:

- VIX-only baseline;
- VIX plus log-signature features;
- additive condition embedding only at first.

This ablation should keep the tokenizer, token-prior family, sampling settings, and evaluation
protocol fixed. Cross-attention should remain deferred unless the condition becomes a temporal
sequence or a set of covariate tokens.

## Non-goals

This implementation plan does not include:

- `signatory` dependency;
- `KSig` dependency;
- objective-level signature loss;
- cross-attention;
- diffusion;
- MGVQ;
- GroupedRVQ;
- tokenizer or prior changes;
- training runs.

Objective-level signature-kernel losses should remain future work until evaluation-only metrics
are stable and useful for model selection.
