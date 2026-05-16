# S&P500/VIX Separate Frequency Final Decision

Status: final decision note for the separate low/high frequency-tokenizer branch. This document
does not implement code, train models, or run sampling ablations.

## What Was Tested

This branch tested a narrow decomposition hypothesis for the S&P500/VIX discrete path:

- deterministic causal EMA decomposition with `alpha = 0.2`;
- a separate standard VQ tokenizer for the low EMA component;
- a separate standard VQ tokenizer for the high residual component;
- paired low/high token extraction for prior training;
- a hierarchical causal autoregressive prior with the factorisation:

```text
p(low_t | low_<t, high_<t, VIX)
p(high_t | low_<=t, high_<t, VIX)
```

The design kept the generation order causal: `low_t` is sampled first, `high_t` may condition on
the sampled current low token, and neither stream may condition on future low or high tokens.
Decoded scalar paths were composed as:

```text
low_hat + high_hat
```

No GroupedRVQ, MGVQ, signatures, diffusion, cross-attention, learned filters, bidirectional
prior, or new objective was added for this branch.

## What Passed

The deterministic split and tokenizer stages passed their intended gates.

Tokenizer and decomposition checks:

- causal EMA low/high decomposition preserved the no-future-leakage contract;
- separate low/high tokenizer no-leakage checks passed;
- untrained low/high tokenizer encoder-prefix checks passed;
- deterministic token-prefix checks passed after warm-up.

Tokenizer quality:

| Component | Eval shape | Recon L1 | Recon L2 | Volatility error | Active codes | Perplexity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Low | `[512, 60, 1]` | 0.00759831 | 0.00911571 | 0.00124286 | 55 / 64 | 41.87561798 |
| High | `[512, 60, 1]` | 0.00076673 | 0.00096738 | 0.00014078 | 61 / 64 | 46.23094559 |

Paired token extraction also passed. The paired dataset contains matched tensors:

```text
train_low_tokens.pt   -> [2457, 60]
train_high_tokens.pt  -> [2457, 60]
eval_low_tokens.pt    -> [2457, 60]
eval_high_tokens.pt   -> [2457, 60]
train_labels.pt       -> [2457, 1]
eval_labels.pt        -> [2457, 1]
train_data.pt         -> [2457, 60, 1]
eval_data.pt          -> [2457, 60, 1]
```

Extraction retained complete codebook support:

| Stream | Split | Active codes | Perplexity |
| --- | --- | ---: | ---: |
| Low | Combined | 64 / 64 | 48.94255066 |
| High | Combined | 64 / 64 | 50.19335175 |

Same-time pair support was also broad before prior training:

| Split | Active pairs | Pair perplexity |
| --- | ---: | ---: |
| Combined | 3044 / 4096 | 1602.10266113 |

The hierarchical prior source checks passed:

- paired loader smoke passed with `tokens: [batch, 60, 2]` and `conditions: [batch, 1]`;
- source no-leakage passed for future low/high perturbations;
- the allowed same-time edge `low_t -> high_logits_t` was verified;
- one-epoch smoke training completed end to end.

These results establish that the failure below is not a basic data-shape, leakage, tokenizer, or
checkpointing failure.

## What Failed

The non-smoke hierarchical prior did not generate usable S&P500/VIX scalar paths.

Training likelihood improved, but sampling collapsed:

| Stream | Real active | Sampled active | Real perplexity | Sampled perplexity | Marginal L1 | Transition L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | 55 / 64 | 33 / 64 | 40.17523193 | 4.10020781 | 1.86393332 | 1.17757928 |
| High | 61 / 64 | 43 / 64 | 43.50838852 | 6.62247515 | 1.76313341 | 1.44845176 |
| Same-time pairs | 2277 / 4096 | 404 / 4096 | 1128.99841309 | 25.49054718 | 1.93339992 | 0.64975667 |

The low stream collapsed most severely: sampled low-token perplexity fell from `40.1752` in the
real eval tokens to `4.1002` in generated samples. The high stream also collapsed, and sampled
same-time pair support fell from `2277` observed eval pairs to `404` generated pairs.

The composed scalar paths failed the main distributional guardrails:

| Model / setting | MMD | SWD | Volatility W1 | Sq-return AC L1 | Flat sq-return AC L1 | Drawdown W1 | Terminal W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Separate low/high hierarchical, temp 0.8 top-k 40 | 2.33316898 | 0.14535846 | 0.04670410 | 0.04985855 | 0.15767361 | 0.07369982 | 0.71774405 |
| Joint EMA alpha 0.2, temp 0.8 top-k 40 | 0.29730234 | 0.00940700 | 0.00113100 | 0.03540378 | 0.02522561 | 0.00707990 | 0.00666096 |
| Promoted baseline, temp 0.8 top-k 40 | 0.27934083 | 0.00767375 | 0.00118835 | 0.04129972 | 0.12882016 | 0.01050232 | 0.00981713 |
| Hidden128, temp 0.8 top-k 20 | 0.22907834 | 0.00717505 | 0.00125777 | 0.06088475 | 0.07886228 | 0.00891025 | 0.00450928 |
| Continuous BetaCVAE | 0.15442121 | 0.00878550 | 0.00063360 | 0.02946163 | 0.02914374 | 0.00766744 | 0.00905099 |

The separate-frequency hierarchical prior is worse than every retained reference on MMD, SWD,
volatility W1, drawdown W1, and terminal W1. The terminal-return W1 degradation is especially
large.

VIX-bucket diagnostics did not reveal a local success case. Every bucket had MMD above `2.30`,
SWD above `0.139`, volatility W1 around `0.045` to `0.049`, and terminal W1 above `0.628`.

## Main Bottleneck

The main bottleneck is sampling-time token-prior calibration and exposure bias, not tokenizer
codebook collapse.

The evidence is:

- tokenizer evaluation retained broad active-code support and healthy perplexity for both
  component tokenizers;
- paired extraction retained complete low/high codebook support and broad same-time pair support;
- teacher-forced prior likelihood became plausible by epoch 100;
- autoregressive sampling nevertheless collapsed low-token usage, high-token usage, and
  same-time pair support;
- the decoded scalar paths then failed the paper-style market diagnostics.

This points to a prior sampling failure. The model can learn useful teacher-forced next-token
statistics, but it does not remain calibrated when it must consume its own sampled low/high
history. The current branch therefore diagnoses an exposure-bias and calibration problem in the
separate-stream prior, rather than a failure of deterministic EMA decomposition or of the
separate tokenizer codebooks themselves.

## Decision

Final decision: reject the current separate-frequency hierarchical prior as a generation model.

Operational decisions:

- Do not promote the separate-frequency hierarchical prior.
- Do not run more sampling ablations by default.
- Do not move directly to MGVQ from this result.
- Do not move directly to VQ-Diffusion from this result.
- Do not merge this branch into the public S&P500/VIX baseline.

Additional sampling ablations are justified only if the explicit goal is to debug token-prior
calibration. They are not justified as the next default thesis path because the current default
setting already fails by a large margin against all retained baselines.

## Recommended Next Branch

Return to the single-stream hidden128 tokenizer as the practical discrete baseline for the next
prior-modelling branch. Hidden128 remains the stronger retained discrete reference on broad
distributional quality, especially MMD, SWD, and terminal-return W1.

The next branch should test stronger causal prior variants on the single-stream hidden128 token
interface before adding new tokenisation machinery:

1. Causal convolution plus transformer prior.
   Use causal convolutional local context before the transformer trunk, while preserving the
   left-to-right token factorisation.

2. Optional selective state-space or Mamba-style prior.
   Consider this only after a dependency and reproducibility check passes. It should remain a
   causal token prior, not a bidirectional sequence model.

3. Entropy and calibration-aware prior diagnostics.
   Add diagnostics that compare teacher-forced entropy, sampled entropy, marginal token support,
   transition support, run-lengths, VIX-bucket support, and calibration under free-running
   sampling. These diagnostics should be used before interpreting decoded path metrics.

This recommendation keeps the next experiment focused on the observed failure mode: robust
causal sampling from a learned token prior.

## Merge Scope

Keep the separate-frequency documents and optional research infrastructure on the research
branch. They are useful as a negative result and as reusable tooling for paired-token diagnostics.

Do not merge this branch into the public baseline. In particular, do not promote:

- the separate low/high hierarchical prior as a generation model;
- the alpha `0.2` separate-frequency config as a baseline config;
- the sampling adapter as part of the public evaluation path;
- any claim that separate low/high tokenization improves S&P500/VIX generation quality.

If the research branch is preserved, its role should be clearly labelled as archival and
diagnostic: separate tokenizers passed their local checks, but the first hierarchical prior
failed the generation-quality gate.
