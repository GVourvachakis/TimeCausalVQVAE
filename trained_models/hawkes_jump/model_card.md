# Hawkes/SVMHJD Model Card

## Status

Hawkes/SVMHJD is registered as a `research_candidate` with
`public_default: false`. The registry stores metadata only. No checkpoints,
weights, token tensors, generated paths, or output summaries are committed.

## Selection

| Family | Role | Candidate | Configs |
| --- | --- | --- | --- |
| Continuous | Comparator | `beta_cvae_logreturn_identity` | `configs/experiments/hawkes_jump_beta_cvae_logreturn_identity.yaml` |
| Discrete | Required ablation | `hidden128_logreturn_cb64_additive_ar` | `configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64.yaml`; `configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_additive.yaml` |
| Discrete | Selected research candidate | `hidden128_logreturn_cb64_conv_transformer_k3` | `configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64.yaml`; `configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer.yaml` |
| Discrete | Efficiency candidate | `hidden128_logreturn_cb64_conv_transformer_tiny` | `configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64.yaml`; `configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer_tiny.yaml` |

Sampling uses `temperature=1.0` and `top_k=null`. Evaluation uses the Ogata
backend, `data_output: log_return`, `n_sample=1024`, and seeds `0/1/2`.

`hidden128_logreturn_cb64_conv_transformer_k3` remains the selected
Hawkes/SVMHJD discrete research candidate under the balanced/smooth profile.
`hidden128_logreturn_cb64_additive_ar` remains the required ablation and is
slightly stronger on jump-count and inter-arrival diagnostics. This selection
does not imply that either prior dominates every metric.

The compact-prior follow-up confirms that
`hidden128_logreturn_cb64_conv_transformer_tiny` is an efficiency candidate,
not the selected balanced model. Tiny uses 91,712 parameters versus 388,544 for
k3 and trains in about 52% of the k3 local CPU wall-clock time. Tiny has the
best mean jump-count and inter-arrival W1 in the compact-prior robustness
comparison, but k3 remains better on MMD, SWD, terminal W1, and volatility W1.
The tail profile is materially aligned with k3, so the decision is driven by
smooth-profile robustness rather than VaR/ES failure. The registry therefore
remains selected on `hidden128_logreturn_cb64_conv_transformer_k3`.

## Compact-Prior Follow-Up

Mean / std across seeds. Lower is better. Reference evaluation runtimes for
additive AR and k3 are unavailable because the existing robustness summaries
pre-date evaluator runtime recording.

| Candidate | Parameters | Train s | Eval s | MMD | SWD | Terminal W1 | Volatility W1 | Jump-count W1 | Inter-arrival W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Additive AR ablation | 289,472 | 93.19 / 0.26 | n/a | 0.1567 / 0.0644 | 0.0238 / 0.0085 | 0.0320 / 0.0152 | 0.0011 / 0.0008 | 0.0469 / 0.0319 | 12.7327 / 8.2921 |
| Conv-transformer k3 | 388,544 | 109.69 / 0.20 | n/a | 0.1141 / 0.0355 | 0.0186 / 0.0060 | 0.0217 / 0.0120 | 0.0010 / 0.0010 | 0.0576 / 0.0324 | 16.2591 / 10.3617 |
| Tiny conv-transformer | 91,712 | 56.95 / 4.59 | 23.91 / 2.92 | 0.1567 / 0.0688 | 0.0245 / 0.0103 | 0.0308 / 0.0151 | 0.0011 / 0.0009 | 0.0368 / 0.0258 | 12.3557 / 9.0570 |

## Smooth Metrics

Mean / std across seeds. Lower is better.

| Candidate | MMD | SWD | Terminal W1 | Volatility W1 | Drawdown W1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Continuous BetaCVAE | 1.3153 / 0.1405 | 0.1320 / 0.0231 | 0.1682 / 0.0316 | 0.0243 / 0.0003 | 0.1486 / 0.0754 |
| Additive AR ablation | 0.1567 / 0.0644 | 0.0238 / 0.0085 | 0.0320 / 0.0152 | 0.0011 / 0.0008 | 0.0106 / 0.0059 |
| Conv-transformer k3 | 0.1141 / 0.0355 | 0.0186 / 0.0060 | 0.0217 / 0.0120 | 0.0010 / 0.0010 | 0.0111 / 0.0052 |

## Jump Metrics

| Candidate | Jump-count W1 | Inter-arrival W1 | Jump-size W1 | Generated jump count | Negative jump fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Continuous BetaCVAE | 0.2236 / 0.0144 | 15.6138 / 2.9303 | 0.0844 / 0.0775 | 0.0026 / 0.0030 | 0.6667 / 0.5774 |
| Additive AR ablation | 0.0469 / 0.0319 | 6.3080 / 4.2239 | 0.0180 / 0.0101 | 0.2314 / 0.0492 | 0.9955 / 0.0078 |
| Conv-transformer k3 | 0.0576 / 0.0324 | 8.1888 / 6.7270 | 0.0177 / 0.0101 | 0.2357 / 0.0555 | 0.9989 / 0.0019 |

## VaR/ES

| Candidate | VaR 1% | ES 1% |
| --- | ---: | ---: |
| Continuous BetaCVAE | -0.0061 / 0.0020 | -0.0113 / 0.0018 |
| Additive AR ablation | -0.0745 / 0.0028 | -0.1068 / 0.0080 |
| Conv-transformer k3 | -0.0748 / 0.0026 | -0.1069 / 0.0080 |

## Token Diagnostics

| Candidate | Sampled active codes | Sampled perplexity | Transition L1 | Run-length W1 |
| --- | ---: | ---: | ---: | ---: |
| Additive AR ablation | 64.00 / 0.00 | 44.36 / 0.70 | 0.4341 / 0.0153 | 0.0031 / 0.0033 |
| Conv-transformer k3 | 64.00 / 0.00 | 44.42 / 0.80 | 0.4345 / 0.0177 | 0.0037 / 0.0022 |

## No-Leakage Status

| Check | Status |
| --- | --- |
| Dataset model-visible fields | Passed: data and constant labels only |
| Causal convolution | Passed |
| Hidden128 tokenizer prefix check | Passed |
| Token-prior causality check | Passed |

## Caveats

- This is a scenario-data benchmark, not an arbitrage-free pricing model.
- Hawkes/SVMHJD is not a public default experiment.
- No weights or generated outputs are committed.
- No LSGM, score-prior, or diffusion-style continuous comparator has been run.
- The repaired continuous comparator is valid but much weaker on jump and lower-tail diagnostics.
