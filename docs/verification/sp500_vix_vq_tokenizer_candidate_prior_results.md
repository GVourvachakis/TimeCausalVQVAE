# S&P500/VIX VQ Tokenizer Candidate Prior Results

## Scope

This note records additive causal AR prior training for the two standard-VQ tokenizer
candidates selected from `docs/verification/sp500_vix_vq_tokenizer_ablation_results.md`.
No prior architecture, conditioning mechanism, objective, RVQ, GRVQ, MGVQ, diffusion, or
signature-conditioning change was introduced.

All candidate priors used:

- S&P500/VIX data from `data/processed`;
- `condition_dim=1`;
- additive VIX-only conditioning;
- `n_sample=1000`, `seed=99`, `temperature=0.8`, `top_k=40` for decoded and paper-style
  evaluation;
- continuous baseline path
  `outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47/final_model`.

## Selected Tokenizers

| Candidate | Tokenizer checkpoint | Selection reason from tokenizer ablation |
| --- | --- | --- |
| `hidden128` | `outputs/sp500_vix_discrete/vq_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_hidden128_seed99` | Best reconstruction and volatility candidate. It improved reconstruction L1 from the promoted baseline `0.01120215` to `0.00592622`, improved terminal error from `0.00556620` to `0.00472229`, and essentially matched baseline volatility error. |
| `cb64_dim32` | `outputs/sp500_vix_discrete/vq_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_cb64_dim32_seed99` | Conservative codebook-dimension candidate. It improved reconstruction L1 to `0.00750538`, kept terminal error close to baseline, and increased tokenizer perplexity to `37.57756424`. |

Token extraction wrote:

| Candidate | Token data dir | Train tokens | Eval tokens | Combined active codes | Combined perplexity |
| --- | --- | ---: | ---: | ---: | ---: |
| `hidden128` | `outputs/sp500_vix_discrete/token_prior/sp500_vix_causal_vq_tokenizer_hidden128_tokens` | `2457 x 60` | `2457 x 60` | 64/64 | 50.358673 |
| `cb64_dim32` | `outputs/sp500_vix_discrete/token_prior/sp500_vix_causal_vq_tokenizer_cb64_dim32_tokens` | `2457 x 60` | `2457 x 60` | 64/64 | 45.641132 |

## W&B Execution

The required live W&B profile was attempted with:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread
```

for project `time-causal-token-prior` and entity `tc_vae`. The first live run failed before
training with `wandb.errors.errors.CommError` after the 90 second `wandb.init()` timeout. An
elevated live HTTP attempt was requested to keep active W&B transmission, but the execution
policy rejected the request because it would export run metadata and training metrics to W&B.

No W&B URLs were produced. Both candidate priors were therefore rerun with `--no-wandb`, as
required by the fallback rule, and `runtime_summary.json` records `wandb_enabled=false`.

## Prior Training

| Model | Config | Runtime s | Best epoch | Eval CE | Eval accuracy | Eval perplexity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Promoted baseline, seed0 | `configs/experiments/sp500_vix_causal_token_prior_additive.yaml` | 1264.156 | 100 | 0.914807 | 0.647449 | 2.507927 |
| `hidden128` candidate | `configs/experiments/sp500_vix_causal_token_prior_hidden128_candidate.yaml` | 1234.886 | 100 | 1.237114 | 0.509476 | 3.467770 |
| `cb64_dim32` candidate | `configs/experiments/sp500_vix_causal_token_prior_cb64_dim32_candidate.yaml` | 1185.728 | 100 | 0.957563 | 0.620574 | 2.615031 |

The promoted tokenizer remains easiest for the unchanged additive prior to model in token space.
`cb64_dim32` is close to the promoted baseline on likelihood, while `hidden128` is materially
harder to predict. The decoded diagnostics below are therefore essential, since token likelihood
alone does not rank generated-market behaviour.

## Decoded Token-Prior Metrics

Decoded evaluation used `tcvae-evaluate-token-prior` with the candidate best checkpoints.

| Model | MMD | SWD | Terminal W1 | Volatility W1 | Active sampled codes | Sampled perplexity | Marginal code L1 | Transition L1 | Run-length W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Promoted baseline | 0.288802 | 0.008896 | 0.010769 | 0.001325 | 62/64 | 30.110167 | 0.259400 | 0.343222 | 0.982533 |
| `hidden128` | 0.257473 | 0.007442 | 0.003729 | 0.001386 | 63/64 | 42.205532 | 0.130467 | 0.215735 | 0.335666 |
| `cb64_dim32` | 0.229372 | 0.008934 | 0.011263 | 0.001750 | 61/64 | 37.559692 | 0.094033 | 0.298198 | 0.694828 |

`cb64_dim32` gives the best decoded MMD, but it regresses terminal-return and volatility
Wasserstein errors. `hidden128` improves the baseline on MMD, SWD, terminal-return W1, marginal
code L1, transition L1, and run-length W1, with only a small volatility-W1 regression.

## VIX-Bucket Code Usage

Rows report sampled active code count and sampled token perplexity by paired evaluation VIX bucket.

| Model | Very low | Low | Mid | High | Very high |
| --- | ---: | ---: | ---: | ---: | ---: |
| Promoted baseline | 53 / 21.72 | 54 / 24.50 | 54 / 27.09 | 55 / 31.79 | 59 / 39.77 |
| `hidden128` | 58 / 34.54 | 57 / 35.46 | 60 / 40.29 | 60 / 41.46 | 62 / 48.69 |
| `cb64_dim32` | 48 / 27.69 | 51 / 30.36 | 54 / 32.03 | 57 / 34.22 | 59 / 43.90 |

`hidden128` preserves broader sampled code support across all VIX buckets and is especially
strong in the low-to-high regimes. `cb64_dim32` keeps high-regime coverage but loses support in
the very-low and low buckets.

## Paper-Style Metrics

Paper-style evaluation used
`scripts/evaluate_sp500_vix_paper_style.py` with the same sample count and sampling settings.

| Model | MMD | SWD | Returns W1 | Terminal W1 | Volatility W1 | Max drawdown W1 | Return autocorr L1 | Squared-return autocorr L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Continuous beta-CVAE reference | 0.154421 | 0.008785 | 0.000602 | 0.009051 | 0.000634 | 0.007667 | 0.025972 | 0.029462 |
| Promoted baseline | 0.279341 | 0.007674 | 0.001242 | 0.009817 | 0.001188 | 0.010502 | 0.051741 | 0.041300 |
| `hidden128` | 0.253583 | 0.007210 | 0.001097 | 0.004496 | 0.001301 | 0.010136 | 0.041318 | 0.064390 |
| `cb64_dim32` | 0.266204 | 0.007364 | 0.001697 | 0.008913 | 0.001730 | 0.010066 | 0.039507 | 0.037091 |

The continuous beta-CVAE still has the strongest path-level MMD and volatility W1. Within the
standard-VQ token-prior family, `hidden128` has the best overall generated-market profile: it
improves promoted-baseline MMD, SWD, returns W1, terminal-return W1, maximum-drawdown W1, and
return-autocorrelation L1. Its main regressions are volatility W1 and squared-return
autocorrelation L1. `cb64_dim32` improves MMD and some autocorrelation statistics but loses on
returns W1 and volatility W1.

## Model-Selection Profiles

For this tokenizer-prior gate, the primary profile score is:

```text
profile = MMD + SWD + terminal_return_wasserstein + volatility_wasserstein
```

Lower is better. The paper-style profile is used for the decision because it comes from the
paper-style diagnostic pipeline.

| Model | Decoded profile | Paper-style profile | Decision rank |
| --- | ---: | ---: | ---: |
| `hidden128` | 0.270029 | 0.266589 | 1 |
| `cb64_dim32` | 0.251319 | 0.284212 | 2 |
| Promoted baseline | 0.309792 | 0.298020 | 3 |

The decoded-only profile prefers `cb64_dim32`, driven mostly by its lower decoded MMD. The
paper-style profile and the broader metric table prefer `hidden128`, because it balances MMD,
terminal-return error, and transition/run-length diagnostics more consistently.

## Latent Geometry Recommendation

`hidden128` should receive the next latent-geometry pass before any architecture promotion. The
prior-facing evidence is stronger than the promoted baseline, but the promoted baseline still has
the mature geometry record. The geometry check should focus on:

- VIX-bucket occupancy and trajectory continuity;
- transition concentration relative to the improved transition L1 result;
- whether the higher hidden width creates local code neighbourhoods that are stable under
  same-condition sampling;
- whether the squared-return autocorrelation regression is visible in code-space dynamics.

`cb64_dim32` remains useful as a secondary diagnostic candidate, especially for testing whether
wider codebook embeddings improve token likelihood without improving decoded-market fidelity.

## Decision

Do not replace the official promoted tokenizer yet. The immediate decision is to run more tuning
and validation, with `hidden128` as the leading candidate for a full latent-geometry diagnostic
and repeat-seed prior run.

Rationale:

- `hidden128` improves the promoted VIX-only discrete baseline on the main paper-style profile
  (`0.266589` versus `0.298020`) and on most generated-market diagnostics.
- Its token likelihood is worse than the promoted baseline, so promotion should not be based on a
  single generated-sample seed.
- It still needs the same full latent-geometry record that supports the current promoted
  tokenizer.
- `cb64_dim32` is not promoted because its stronger token likelihood does not translate into
  better terminal/volatility reconstruction after decoding.

The promoted architecture therefore remains standard VQ plus additive VIX-only causal AR prior,
with `hidden128` promoted only to the next validation candidate.
