# Hawkes-Jump First Model Comparison

This note records the first local non-smoke Hawkes-jump comparison on the
Ogata simulator. It is an exploratory benchmark pass, not a registry result:
all runs used one seed, two epochs, CPU execution, and W&B disabled. Generated
model, token, plot, and metric artefacts remain under `outputs/` and are not
part of git.

## Run Setup

The public Hawkes-jump configs currently use `n_samples=512`. The first
continuous training attempt reached epoch-one evaluation and failed because the
continuous trainer samples 1000 generated paths during evaluation while only
512 conditional labels were available. To keep the run matched without changing
public defaults or simulator dynamics, local generated run configs were written
under `outputs/hawkes_jump_first_model_comparison/run_configs/` with
`n_samples=1024`.

Matched settings:

| Field | Value |
| --- | ---: |
| Simulation scheme | `ogata` |
| Train and eval samples | 1024 |
| Timesteps | 60 |
| Seed | 0 |
| Epochs | 2 |
| Evaluation sample count | 1024 |
| W&B | disabled |
| Device | CPU |

Ogata simulator parameters were otherwise unchanged:

| Parameter | Value |
| --- | ---: |
| `dt` | 0.0166666667 |
| `drift` | 0.0 |
| `brownian_volatility` | 0.18 |
| `baseline_intensity` | 3.0 |
| `excitation` | 2.0 |
| `decay` | 12.0 |
| `mark_excitation` | 20.0 |
| `negative_jump_probability` | 0.7 |
| `severe_jump_probability` | 0.08 |
| `volatility_excitation` | true |
| `data_output` | `price` |

## Candidates

| Candidate | Config source | Local run config | Status |
| --- | --- | --- | --- |
| Continuous BetaCVAE | `configs/experiments/hawkes_jump_beta_cvae.yaml` | `hawkes_jump_beta_cvae_1024.yaml` | trained and evaluated |
| Standard VQ + additive AR | `configs/experiments/hawkes_jump_causal_vq_tokenizer.yaml` and `hawkes_jump_causal_token_prior_additive.yaml` | `*_1024.yaml` | trained and evaluated |
| Hidden128 VQ + additive AR | no Hawkes config currently exists | not run | skipped |
| Hidden128 VQ + causal conv-transformer k3 | `configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128.yaml` and `hawkes_jump_causal_token_prior_hidden128_conv_transformer.yaml` | `*_1024.yaml` | trained and evaluated |

## Runtime

| Component | Runtime seconds |
| --- | ---: |
| BetaCVAE training | 1.72 |
| Standard tokenizer training | 0.74 |
| Standard additive prior training | 4.50 |
| Hidden128 tokenizer training | 1.73 |
| Hidden128 conv-transformer prior training | 5.23 |

The standard prior evaluation took roughly one minute for 1024 autoregressive
samples; the hidden128 conv-transformer evaluation took roughly forty seconds.

## Smooth Path Profile

Lower is better for all metrics in this table.

| Model | MMD | SWD | Terminal W1 | Volatility W1 | Drawdown W1 | Return ACF L1 | Squared-return ACF L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BetaCVAE | 0.9638 | 0.1133 | 0.2263 | 0.0915 | 0.2784 | 0.0967 | 0.0596 |
| Standard VQ + additive AR | 1.8068 | 0.1806 | 1.9214 | 0.0592 | 0.1799 | 0.1245 | 0.0635 |
| Hidden128 VQ + conv-transformer k3 | 1.6380 | 0.1446 | 1.1902 | 0.0424 | 0.1198 | 0.1446 | 0.0894 |

The continuous BetaCVAE is still strongest on MMD, SWD, and terminal-return
W1. The hidden128 conv-transformer prior improves over the standard additive
prior on MMD, SWD, terminal W1, volatility W1, and drawdown W1, but both
discrete decoded generators remain weak on terminal distribution.

## Jump-Regime Profile

Detected jumps used a common robust threshold fitted on the real evaluation
paths: median return `-0.000679`, robust scale `0.024978`, and absolute
threshold `0.099911` from a `4.0` multiplier. Real detected jumps averaged
`0.2422` per path, with `248` detected jumps across `1024` paths and
`19.43%` of paths containing at least one detected jump.

| Model | Mean detected jumps/path | Jump-count W1 | Inter-arrival W1 | Jump-size W1 | Paths with jumps | Negative jump fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real Ogata eval paths | 0.2422 | 0.0000 | 0.0000 | 0.0000 | 0.1943 | 0.9718 |
| BetaCVAE | 16.8164 | 16.5742 | 13.9184 | 0.4990 | 1.0000 | 0.5262 |
| Standard VQ + additive AR | 2.1445 | 1.9023 | 13.9184 | 0.2766 | 1.0000 | 0.0096 |
| Hidden128 VQ + conv-transformer k3 | 2.0000 | 1.7734 | 13.9184 | 0.4300 | 1.0000 | 0.0000 |

The continuous model generates too many threshold exceedances, mostly because
its decoded paths are noisy relative to the real one-step return scale. The
discrete priors are closer on jump-count W1, but they still place detected
events on every path and do not reproduce the strongly negative detected jump
sign profile. The identical inter-arrival W1 for all generated models reflects
poor burst timing rather than a meaningful tie.

Risk metrics also show over-smoothing in the decoded discrete paths:

| Model | VaR 1% | ES 1% | VaR 5% | ES 5% |
| --- | ---: | ---: | ---: | ---: |
| BetaCVAE | -0.3876 | -0.5325 | -0.1925 | -0.3098 |
| Standard VQ + additive AR | -0.0233 | -0.0459 | -0.0110 | -0.0208 |
| Hidden128 VQ + conv-transformer k3 | -0.0299 | -0.0352 | -0.0138 | -0.0215 |

The discrete decoded paths severely understate left-tail losses in this first
pass. The continuous model has heavier tails but does not match event sparsity.

## Token Diagnostics

| Metric | Standard VQ + additive AR | Hidden128 VQ + conv-transformer k3 |
| --- | ---: | ---: |
| Extracted active codes | 6 / 64 | 4 / 64 |
| Extracted perplexity | 2.8179 | 2.6268 |
| Sampled active codes | 62 / 64 | 29 / 64 |
| Marginal code L1 | 0.4263 | 0.1371 |
| Transition matrix L1 | 0.9287 | 0.4246 |
| Run-length W1 | 8.9019 | 15.6489 |
| Run-length histogram L1 | 0.4005 | 0.4677 |
| Tokenizer reconstruction L1 | 0.1505 | 0.1084 |
| Tokenizer volatility reconstruction error | 0.0651 | 0.0445 |

The hidden128 tokenizer reconstructs better and the conv-transformer prior is
closer on marginal token usage and transition-matrix distance. However, the
token alphabets are already collapsed before prior training, especially for the
hidden128 tokenizer. This makes the current token-regime comparison
underpowered: low transition distance is easier to achieve when only a few real
codes are active.

## Decision

The expected discrete advantage is not demonstrated in this first comparison.
The hidden128 conv-transformer prior is the strongest discrete candidate and
beats the standard additive prior on several smooth decoded-path metrics and on
token transition distance. Nevertheless, it does not yet recover rare-event
structure: detected jumps occur on every generated path, signs are not
asymmetric in the right direction, left-tail VaR/ES are too mild, and run-length
diagnostics remain poor.

The useful result is diagnostic rather than leaderboard-style:

- BetaCVAE remains best on MMD, SWD, and terminal distribution.
- Hidden128 conv-transformer is the best discrete candidate among those run.
- Both VQ tokenizers need better code utilisation before a fair rare-event
  regime test.
- The current two-epoch benchmark is too short to make a registry decision.

## Next Steps

1. Fix or parameterise the continuous trainer's evaluation sample count so the
   public `512` sample Hawkes configs train without local `1024` run configs.
2. Improve tokenizer utilisation before training longer priors, for example by
   revisiting commitment weight, codebook size, codebook dimension, or
   frequency-aware initialisation in a separate prompt.
3. Re-run at multiple seeds after token usage is healthy, then report jump
   diagnostics with confidence intervals.
4. Add an oracle-real-data table beside detected-jump diagnostics so simulator
   event labels validate the benchmark while generated paths remain evaluated
   only from observable prices.
5. Keep the registry unchanged until a discrete model improves jump/regime
   metrics without unacceptable smooth-metric regression.
