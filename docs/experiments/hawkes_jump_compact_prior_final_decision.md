# Hawkes/SVMHJD Compact Prior Final Decision

## Evidence Base

This note fixes the current decision for the Hawkes/SVMHJD compact
conv-transformer prior family. It synthesises:

- `docs/experiments/hawkes_jump_compact_conv_transformer_results.md`;
- `docs/experiments/hawkes_jump_tiny_conv_transformer_robustness.md`;
- `trained_models/hawkes_jump/model_card.md`.

`docs/experiments/hawkes_jump_continuous_ablation_results.md` was not present
in this checkout. The continuous comparator evidence is therefore taken from
the existing model card, where the log-return BetaCVAE remains much weaker than
the discrete candidates on jump and lower-tail diagnostics.

No additional models were trained for this decision.

## Current Selected Model

Keep the hidden128 log-return cb64 tokenizer with the k3 causal
conv-transformer prior as the selected Hawkes/SVMHJD discrete research
candidate under the balanced/smooth profile.

Selected configuration:

- Tokenizer:
  `configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64.yaml`
- Prior:
  `configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer.yaml`
- Prior family: `causal_conv_transformer`
- Parameters: 388,544
- Sampling policy: temperature `1.0`, unrestricted `top_k`
- Evaluation convention: Ogata-backed log-return Hawkes/SVMHJD, `n_sample=1024`,
  seeds `0/1/2`

The k3 prior remains the best balanced-profile candidate because it has the
strongest three-seed smooth profile among the discrete references:

| Candidate | MMD | SWD | Terminal W1 | Volatility W1 | Drawdown W1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Additive AR | 0.1567 ± 0.0644 | 0.0238 ± 0.0085 | 0.0320 ± 0.0152 | 0.0011 ± 0.0008 | 0.0106 ± 0.0059 |
| Conv-transformer k3 | 0.1141 ± 0.0355 | 0.0186 ± 0.0060 | 0.0217 ± 0.0120 | 0.0010 ± 0.0010 | 0.0111 ± 0.0052 |
| Tiny conv-transformer | 0.1567 ± 0.0688 | 0.0245 ± 0.0103 | 0.0308 ± 0.0151 | 0.0011 ± 0.0009 | 0.0100 ± 0.0029 |

## Required Ablation

Keep the hidden128 log-return cb64 additive AR prior as the required ablation.
It is not the balanced-profile selection, but it remains essential because it
is competitive or slightly stronger on application-specific jump timing.

Required ablation configuration:

- Tokenizer:
  `configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64.yaml`
- Prior:
  `configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_additive.yaml`
- Prior family: additive autoregressive single-code prior
- Parameters: 289,472

The additive AR prior should be reported as a jump-profile specialist, not as
the selected balanced model. It does not improve the smooth profile relative to
k3, and it provides no compute advantage over tiny.

## Tiny Candidate

The tiny conv-transformer is the only compact candidate that merits continued
reporting. It uses the same tokenizer, token data convention, prior family,
condition convention, codebook size, sequence length, sampling temperature, and
unrestricted top-k policy as the selected k3 prior.

Tiny configuration:

- Prior:
  `configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer_tiny.yaml`
- Token embedding dimension: 64
- Transformer layers: 2
- MLP hidden dimension: 128
- Convolution blocks: 1
- Convolution kernel size: 3
- Parameters: 91,712

### Parameter Count and Runtime

| Candidate | Parameters | Train s | Eval s |
| --- | ---: | ---: | ---: |
| Additive AR | 289,472 | 93.19 ± 0.26 | n/a |
| Conv-transformer k3 | 388,544 | 109.69 ± 0.20 | n/a |
| Tiny conv-transformer | 91,712 | 56.95 ± 4.59 | 23.91 ± 2.92 |

Tiny uses 23.6% of the k3 parameter count and trains in about 52% of the k3
local CPU wall-clock time. Existing additive AR and k3 robustness summaries do
not contain evaluator runtimes, so only tiny has three-seed evaluation runtime
statistics from the current evaluator.

### Seed Robustness

| Seed | Train s | Eval s | MMD | SWD | Jump-count W1 | Inter-arrival W1 | VaR 1% | ES 1% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 53.34 | 23.67 | 0.0918 | 0.0141 | 0.0244 | 14.4583 | -0.0729 | -0.0993 |
| 1 | 55.42 | 26.94 | 0.1495 | 0.0247 | 0.0664 | 2.4324 | -0.0765 | -0.1140 |
| 2 | 62.11 | 21.12 | 0.2289 | 0.0347 | 0.0195 | 20.1765 | -0.0749 | -0.1089 |

The tiny prior is stable in code usage and tail scale, but its smooth metrics
vary substantially across seeds and degrade relative to k3 on the three-seed
mean.

### Smooth Profile

Tiny loses the balanced smooth profile to k3. Its mean MMD, SWD, terminal W1,
and volatility W1 are all weaker than k3 and approximately match the additive
AR level rather than improving it. Drawdown W1 is similar across all three
discrete candidates.

### Jump Profile

| Candidate | Jump-count W1 | Inter-arrival W1 | Jump-size W1 | Negative jump fraction |
| --- | ---: | ---: | ---: | ---: |
| Additive AR | 0.0469 ± 0.0319 | 12.7327 ± 8.2921 | 0.0180 ± 0.0101 | 0.9955 ± 0.0078 |
| Conv-transformer k3 | 0.0576 ± 0.0324 | 16.2591 ± 10.3617 | 0.0177 ± 0.0101 | 0.9989 ± 0.0019 |
| Tiny conv-transformer | 0.0368 ± 0.0258 | 12.3557 ± 9.0570 | 0.0184 ± 0.0096 | 0.9988 ± 0.0021 |

Tiny has the best mean jump-count W1 and inter-arrival W1 among the compared
priors. Jump-size W1 is effectively tied under the observed seed variability.
This is a real efficiency-and-jump-count result, but it is not sufficient to
replace k3 for the balanced profile.

### Tail Profile

| Candidate | VaR 1% | ES 1% |
| --- | ---: | ---: |
| Additive AR | -0.0745 ± 0.0028 | -0.1068 ± 0.0080 |
| Conv-transformer k3 | -0.0748 ± 0.0026 | -0.1069 ± 0.0080 |
| Tiny conv-transformer | -0.0747 ± 0.0018 | -0.1074 ± 0.0075 |

Tiny does not introduce an unacceptable tail regression. Its VaR and ES profile
is materially aligned with additive AR and k3. The blocker is therefore the
smooth-profile regression, not lower-tail risk.

## Decision

Keep the hidden128 log-return cb64 + k3 conv-transformer prior as the current
selected Hawkes/SVMHJD research candidate.

Report the tiny conv-transformer as an efficiency candidate: it is substantially
smaller and faster, and it improves mean jump-count and inter-arrival distances,
but it does not win the balanced profile because it loses the smooth-distribution
metrics to k3. Its tail profile is acceptable, but that is not enough to
override the smooth robustness gap.

Keep additive AR as the required ablation and as a jump-profile specialist. It
is useful for interpreting jump-count and inter-arrival diagnostics, but it is
not the balanced-profile selection.

## Public Documentation Update Plan

### README

Use the following exact sentence in the Hawkes/SVMHJD benchmark section, either
as a replacement for the current selection sentence or as the final sentence of
that paragraph:

> The selected Hawkes/SVMHJD discrete research candidate remains the hidden128
> log-return cb64 tokenizer + causal conv-transformer k3 prior; the additive AR
> prior is the required jump-profile ablation, and the tiny conv-transformer is
> reported only as a compute-efficient candidate because it improves jump-count
> and inter-arrival means but loses the balanced smooth profile.

### Model Card

Use the following exact update in `trained_models/hawkes_jump/model_card.md`
under `## Selection` after the existing paragraph beginning with
`hidden128_logreturn_cb64_conv_transformer_k3`:

> The compact-prior follow-up confirms that
> `hidden128_logreturn_cb64_conv_transformer_tiny` is an efficiency candidate,
> not the selected balanced model. Tiny uses 91,712 parameters versus 388,544
> for k3 and improves mean jump-count/inter-arrival W1, but its three-seed MMD,
> SWD, terminal W1, and volatility W1 are weaker than k3. The tail profile is
> materially aligned with k3, so the decision is driven by smooth-profile
> robustness rather than VaR/ES failure.

Also add a row to the model-card selection table:

| Family | Role | Candidate | Configs |
| --- | --- | --- | --- |
| Discrete | Efficiency candidate | `hidden128_logreturn_cb64_conv_transformer_tiny` | `configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64.yaml`; `configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer_tiny.yaml` |

### Registry

Do not switch `trained_models/model_registry.yaml` from k3 to tiny. The registry
should continue to point at the k3 conv-transformer as the selected
balanced-profile research candidate. Tiny may be mentioned in documentation as
an efficiency candidate, but it should not become the registered selected model
unless a later experiment closes the smooth-profile gap.

No merge to `main` is required from this research branch for the compact-prior
ablation close-out.
