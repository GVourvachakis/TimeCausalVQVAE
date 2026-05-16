# Stronger Hidden128 Prior Roadmap

Status: planning note only. No source code was changed, no configuration files were modified, and
no models were trained for this roadmap.

## Motivation

The `hidden128` tokenizer is the strongest current single-stream standard-VQ research candidate
for the S&P500/VIX setting. It keeps the one-token-per-time-step interface while improving
tokenizer reconstruction, latent geometry, and paper-style generated-market profile relative to
the promoted public standard-VQ baseline. The main unresolved weakness is not tokenizer collapse:
the full latent-geometry diagnostics show all 64 codes active, with global perplexity `50.358673`,
and broad VIX-bucket support.

The failed separate-frequency direction should therefore be treated as a warning against adding
tokenizer structure before the generation bottleneck is understood. That branch failed because the
sampled tokens collapsed under the low/high hierarchical prior, not because the hidden128
single-stream tokenizer lacked active code usage. The current bottleneck is prior calibration:
the additive prior can generate competitive decoded paths after sampling calibration, but it
models hidden128 tokens with materially worse likelihood than the promoted tokenizer.

The next architecture work should keep the hidden128 tokenizer fixed and compare stronger causal
priors against the current additive AR prior. The objective is to improve local volatility
clustering, squared-return autocorrelation, and token-transition realism while preserving the
strict left-to-right causal generation contract.

## Baseline

The baseline for this roadmap is the hidden128 single-stream standard-VQ tokenizer:

- tokenizer config: `configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml`;
- prior config:
  `configs/experiments/sp500_vix_causal_token_prior_hidden128_candidate.yaml`;
- tokenizer family: standard causal VQ with wider encoder/decoder hidden capacity;
- prior family: additive VIX-only causal autoregressive transformer;
- condition: scalar VIX, `condition_dim=1`;
- sampling policy: temperature `0.8`, `top_k=20`.

This baseline is the selected hidden128 research setting. Its paper-style profile is `0.242020`,
with MMD `0.229078`, SWD `0.007175`, terminal W1 `0.004509`, volatility W1 `0.001258`, and
squared-return autocorrelation L1 `0.060885`. It improves the promoted discrete baseline on the
overall paper-style profile, MMD, SWD, returns W1, terminal W1, maximum-drawdown W1, and return
autocorrelation, but it remains weaker on token likelihood, volatility W1, and squared-return
autocorrelation.

## Candidate Priors

### A. Causal Convolution And Transformer Hybrid

Add a causal temporal-convolutional preprocessor before the transformer trunk. The preprocessor
should operate only on the prefix-visible token and conditioning stream, using strictly causal
padding and optional dilation. The transformer remains the global autoregressive component, while
the local convolutional stack provides an inductive bias for short-horizon token patterns.

Intended benefits:

- improve local volatility clustering and squared-return autocorrelation;
- make short repeated token patterns easier to represent than with attention alone;
- preserve the existing single-token interface and left-to-right sampling loop;
- avoid new package dependencies if implemented with native PyTorch modules.

This direction is consistent with the financial time-series motivation for dilated causal
convolutions recorded in `docs/references.md` under `[deepvol_2022]`. It should be evaluated
first because it targets the observed residual dynamics without expanding the tokenizer or adding
multi-code compatibility constraints.

### B. Wider Or Deeper Transformer

Scale the existing causal transformer prior while keeping its interface unchanged. Candidate
ablations should vary depth, attention heads, embedding width, and MLP expansion. The comparison
must include dropout and regularisation settings because the hidden128 token stream is more
diverse than the promoted-tokenizer stream and may expose overfitting or miscalibrated logits.

Recommended ablations:

- additional transformer layers at fixed width;
- wider embeddings and MLPs at fixed depth;
- more attention heads where the embedding dimension supports them cleanly;
- dropout on token embeddings, attention, residual paths, and MLP blocks;
- weight decay and early-stopping sensitivity.

This candidate has low conceptual risk, but its cost is higher than the convolutional hybrid. It
should be treated as the second implementation candidate unless diagnostics show that the current
prior is capacity-limited rather than locally mis-specified.

### C. Entropy And Calibration-Aware Sampling Diagnostics

The hidden128 result already depends on sampling calibration: `temperature=0.8`, `top_k=20` is
stronger than the original `top_k=40` policy by the paper-style profile. The next prior sweep
should therefore log sampling behaviour as an architectural diagnostic rather than as an
afterthought.

Required diagnostics:

- temperature, top-k, and nucleus-sampling grids for each trained prior;
- sampled token entropy and perplexity against held-out tokenizer-token statistics;
- sampled marginal code usage versus real marginal usage;
- sampled transition matrices or sparse transition summaries versus real transitions;
- VIX-bucket marginal and transition shifts;
- run-length distribution matching, including excessive persistence and under-persistence.

The purpose is to distinguish a genuinely stronger learned prior from a model whose logits require
a narrow sampling policy to avoid collapse or excessive dispersion.

### D. Optional Mamba Or Selective-SSM Prior

A Mamba-style or selective-state-space prior is optional future work, not the first implementation
candidate. It should be considered only after package compatibility, CUDA/PyTorch constraints,
licensing, checkpoint portability, and reproducibility have been inspected on this repository.

Any selective-SSM prior must preserve causal left-to-right generation:

- logits at time `t` may depend only on tokens and conditions available up to `t`;
- recurrent state updates must be prefix-only during sampling;
- batching optimisations must not introduce bidirectional context;
- evaluation must include the same no-future-leakage checks as the transformer prior.

No new reference key is currently recorded for Mamba in `docs/references.md`; add one only after a
specific implementation dependency and paper are selected.

## Rejected For Now

The following directions should remain deferred for this branch:

- VQ-Diffusion, including mask-and-replace discrete diffusion, because the promoted method needs a
  simple causal generator and diffusion would change the generation contract;
- GroupedRVQ, because sparse same-time compatibility remains a known risk for multi-code
  generation;
- MGVQ, because it would require a dedicated financial time-series adaptation and a stable
  multi-code prior interface before it can be meaningfully evaluated;
- signatures, because previous signature-conditioning work improved likelihood in some settings
  without providing enough generated-market evidence to replace VIX-only conditioning;
- separate low/high hierarchical prior, because the separate-frequency branch failed through
  sampled-token collapse rather than solving the volatility and autocorrelation residuals.

These rejections are scope controls, not permanent architecture judgements. They should be
revisited only after a stronger single-stream hidden128 prior has been evaluated.

## Evaluation

Every candidate must be evaluated against the hidden128 paper-style protocol and against token
diagnostics that can expose prior-calibration failure.

Path-level diagnostics:

- hidden128 paper-style profile;
- MMD and SWD;
- volatility W1;
- squared-return autocorrelation L1;
- drawdown W1, including maximum-drawdown W1;
- terminal-return W1;
- returns W1 and return-autocorrelation L1 as supporting checks.

Token-level diagnostics:

- sampled token entropy and perplexity;
- active-code count and active-code ratio;
- marginal code-usage distance from real tokenizer tokens;
- transition-matrix or sparse-transition distance;
- VIX-bucket marginal and transition diagnostics;
- run-length distribution distance.

Selection should require improvement in generated-market diagnostics and no new sampled-token
collapse. Lower cross-entropy alone is insufficient, and lower MMD alone is insufficient if
volatility clustering, squared-return autocorrelation, or token-transition realism regress.

## First Implementation Recommendation

Implement the causal convolution and transformer hybrid prior first.

This is the lowest-risk next prior because it keeps the hidden128 tokenizer, single-token
interface, VIX-only condition, and left-to-right sampling path intact. It introduces no necessary
external dependency, directly targets local residual dynamics, and provides a clear diagnostic
contrast with the current additive transformer prior. If it improves squared-return
autocorrelation and volatility W1 without narrowing the sampling-stability region, it becomes the
preferred stronger hidden128 prior candidate. If it does not, the wider/deeper transformer ablation
should follow before any dependency-heavy selective-SSM or multi-code architecture is reopened.
