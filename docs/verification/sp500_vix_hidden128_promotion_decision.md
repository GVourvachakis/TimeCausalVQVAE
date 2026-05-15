# S&P500/VIX Hidden128 Promotion Decision

Status: decision recorded. No code was implemented, no models were trained, and no configs were
modified.

## Decision

Do not replace the current promoted standard VQ tokenizer with `hidden128` yet.

`hidden128` should remain the leading standard-VQ tokenizer ablation and the preferred tokenizer
candidate for the next validation branch. It has stronger reconstruction, stronger latent geometry,
and stronger generated-market profile scores than the current promoted tokenizer. However, the
unchanged additive AR prior models its tokens materially worse, and the generated diagnostics still
show squared-return autocorrelation and volatility-W1 regressions relative to the promoted
baseline. Those limitations are important enough to keep the promoted baseline as the default.

The project should therefore keep the current promoted baseline, treat `hidden128` as a strong
ablation rather than a replacement, and run more tokenizer-side tuning only if the next branch stays
inside the standard-VQ family.

## Current Promoted Baseline

The current promoted discrete architecture remains:

- standard causal VQ tokenizer;
- one 64-entry codebook with 16-dimensional codebook embeddings;
- scalar VIX condition, `condition_dim=1`;
- additive VIX-only causal AR token prior;
- S&P500/VIX benchmark;
- paper-style diagnostics and latent-geometry diagnostics.

The promoted tokenizer has the most mature baseline record:

| Diagnostic | Promoted baseline |
| --- | ---: |
| Tokenizer reconstruction L1 | 0.011202 |
| Tokenizer reconstruction L2 | 0.012552 |
| Tokenizer terminal error | 0.005566 |
| Tokenizer volatility error | 0.000810 |
| Ablation-slice active codes | 57/64 |
| Ablation-slice perplexity | 33.296204 |
| Full geometry active codes | 63/64 |
| Full geometry perplexity | 39.055717 |
| Prior eval CE | 0.914807 |
| Prior eval accuracy | 0.647449 |
| Prior eval perplexity | 2.507927 |
| Paper-style profile | 0.298020 |

This baseline also has the simplest prior-facing representation and remains the public default in
the discrete architecture notes.

## Hidden128 Tokenizer Evidence

The `hidden128` tokenizer changes encoder/decoder hidden capacity while keeping the standard
single-code VQ interface. It does not require a multi-code prior.

On the tokenizer ablation slice, it improves the promoted tokenizer on the main reconstruction and
market-reconstruction metrics:

| Diagnostic | Promoted baseline | `hidden128` |
| --- | ---: | ---: |
| Reconstruction L1 | 0.011202 | 0.005926 |
| Reconstruction L2 | 0.012552 | 0.007156 |
| Terminal error | 0.005566 | 0.004722 |
| Volatility error | 0.000810 | 0.000797 |
| Active codes | 57/64 | 58/64 |
| Perplexity | 33.296204 | 44.763039 |

Its full latent geometry is also stronger than the promoted tokenizer:

| Geometry diagnostic | Promoted baseline | `hidden128` |
| --- | ---: | ---: |
| Active codes | 63/64 | 64/64 |
| Global perplexity | 39.055717 | 50.358673 |
| Very-low VIX bucket | 53 / 28.49 | 61 / 40.77 |
| Low VIX bucket | 56 / 32.34 | 61 / 44.07 |
| Mid VIX bucket | 58 / 35.60 | 62 / 46.61 |
| High VIX bucket | 60 / 40.41 | 61 / 49.44 |
| Very-high VIX bucket | 63 / 43.82 | 64 / 48.22 |

This is positive tokenizer evidence: `hidden128` gives broader code usage without changing the
standard VQ token interface.

## Prior Robustness

The unchanged additive VIX-only AR prior is consistently worse at predicting `hidden128` tokens
than promoted-baseline tokens.

| Prior run | Best epoch | Eval CE | Eval accuracy | Eval perplexity |
| --- | ---: | ---: | ---: | ---: |
| Promoted baseline seed0 | 100 | 0.914807 | 0.647449 | 2.507927 |
| `hidden128` seed99 reference | 100 | 1.237114 | 0.509476 | 3.467770 |
| `hidden128` seed1 | 100 | 1.235018 | 0.512169 | 3.459926 |
| `hidden128` seed2 | 100 | 1.232192 | 0.512020 | 3.449989 |

The hidden128 likelihood is stable across three prior seeds, but it is not competitive with the
promoted tokenizer likelihood. This is the main reason not to promote hidden128 as the default.

Generated-market diagnostics are more favourable:

| Prior run | Paper profile | MMD | SWD | Terminal W1 | Volatility W1 | Squared-return AC L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Promoted baseline, temp 0.8, top-k 40 | 0.298020 | 0.279341 | 0.007674 | 0.009817 | 0.001188 | 0.041300 |
| `hidden128` seed99, temp 0.8, top-k 40 | 0.266589 | 0.253583 | 0.007210 | 0.004496 | 0.001301 | 0.064390 |
| `hidden128` seed1, temp 0.8, top-k 40 | 0.271278 | 0.250218 | 0.008251 | 0.011366 | 0.001443 | 0.065187 |
| `hidden128` seed2, temp 0.8, top-k 40 | 0.203517 | 0.190422 | 0.006152 | 0.005626 | 0.001317 | 0.063873 |

All hidden128 seeds beat the promoted baseline on the paper-style profile. The improvement is
driven mainly by MMD and, for seed99 and seed2, terminal-return W1.

## Sampling Robustness

For the existing hidden128 reference checkpoint, the sampling grid shows that `temperature=0.8`
is the reliable region. Low temperature under-diversifies, while temperature `1.0` improves some
transition/run-length statistics at the cost of terminal-return and SWD degradation.

| Temperature | Top-k | Profile | MMD | SWD | Terminal W1 | Volatility W1 | Active codes | Token ppl |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6 | none | 0.335593 | 0.316651 | 0.009160 | 0.007839 | 0.001944 | 60/64 | 38.701260 |
| 0.6 | 20 | 0.347565 | 0.328484 | 0.009363 | 0.007759 | 0.001959 | 60/64 | 38.322296 |
| 0.6 | 40 | 0.376030 | 0.355912 | 0.010093 | 0.008014 | 0.002011 | 59/64 | 37.668682 |
| 0.8 | none | 0.245956 | 0.233525 | 0.006649 | 0.004415 | 0.001367 | 63/64 | 41.970245 |
| 0.8 | 20 | 0.242020 | 0.229078 | 0.007175 | 0.004509 | 0.001258 | 63/64 | 42.898106 |
| 0.8 | 40 | 0.266589 | 0.253583 | 0.007210 | 0.004496 | 0.001301 | 64/64 | 41.683281 |
| 1.0 | none | 0.251476 | 0.227252 | 0.012425 | 0.010530 | 0.001269 | 64/64 | 45.341309 |
| 1.0 | 20 | 0.289511 | 0.265435 | 0.011384 | 0.011758 | 0.000935 | 63/64 | 45.527821 |
| 1.0 | 40 | 0.303367 | 0.276092 | 0.013663 | 0.012337 | 0.001275 | 64/64 | 45.475639 |

The best hidden128 sampling setting is `temperature=0.8, top_k=20`, with paper-style profile
`0.242020`. This is better than the promoted baseline profile `0.298020`, but it is a tuned
sampling policy rather than a default-tokenizer replacement by itself.

## Continuous BetaCVAE Comparison

The continuous BetaCVAE remains the stronger reference on broad path-level distributional fidelity.

| Model | Profile | MMD | SWD | Returns W1 | Terminal W1 | Volatility W1 | Max drawdown W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Continuous BetaCVAE reference | 0.172891 | 0.154421 | 0.008785 | 0.000602 | 0.009051 | 0.000634 | 0.007667 |
| Promoted discrete baseline | 0.298020 | 0.279341 | 0.007674 | 0.001242 | 0.009817 | 0.001188 | 0.010502 |
| `hidden128`, temp 0.8, top-k 20 | 0.242020 | 0.229078 | 0.007175 | 0.001044 | 0.004509 | 0.001258 | 0.008910 |

Hidden128 improves the discrete baseline and has better terminal-return W1 than the continuous
reference at its best sampling setting. It still trails the continuous model on MMD, returns W1,
volatility W1, maximum drawdown, and autocorrelation diagnostics.

## Model-Selection Scores

The model-selection profile used in these notes is:

```text
MMD + SWD + terminal_return_wasserstein + volatility_wasserstein
```

`docs/architecture/model_selection_profiles.md` is not present on this branch, so the score above
is taken from the verification notes.

| Model or setting | Decoded profile | Paper-style profile |
| --- | ---: | ---: |
| Continuous BetaCVAE reference | n/a | 0.172891 |
| Promoted discrete baseline | 0.309792 | 0.298020 |
| `hidden128` seed99, temp 0.8, top-k 40 | 0.270029 | 0.266589 |
| `hidden128` seed1, temp 0.8, top-k 40 | 0.281735 | 0.271278 |
| `hidden128` seed2, temp 0.8, top-k 40 | 0.265540 | 0.203517 |
| `hidden128` seed99, temp 0.8, top-k 20 | 0.242020 | 0.242020 |

The profile scores support hidden128 as the leading standard-VQ candidate. They do not remove the
likelihood and volatility/autocorrelation caveats.

## Limitations

The main limitations are:

- Squared-return autocorrelation regression: hidden128 is worse than the promoted baseline across
  the prior seeds at `top_k=40`, and it remains worse at the best `top_k=20` setting
  (`0.060885` versus `0.041300`).
- Volatility W1 regression: hidden128 is slightly worse than the promoted baseline in paper-style
  evaluation (`0.001258` at best sampling versus baseline `0.001188`; `0.001301` at the original
  `top_k=40` setting).
- Token likelihood regression: hidden128 prior CE is about `1.23-1.24`, versus `0.914807` for the
  promoted tokenizer under the same additive AR prior family.
- W&B visibility remains unresolved for these non-smoke runs; the live W&B attempts failed or were
  not approved, so the evidence is local-output based.
- The strongest hidden128 result depends on a sampling-policy adjustment from `top_k=40` to
  `top_k=20`.

## Promotion Path Not Taken

Because hidden128 is not promoted here, no README minimal-config change should be made in this
decision. If a later decision promotes it, the README minimal config list should add:

- `configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml`;
- `configs/experiments/sp500_vix_causal_token_prior_hidden128_candidate.yaml`;
- the selected seed robustness configs only as verification references, not minimal default
  configs:
  - `configs/experiments/sp500_vix_causal_token_prior_hidden128_candidate_seed1.yaml`;
  - `configs/experiments/sp500_vix_causal_token_prior_hidden128_candidate_seed2.yaml`.

The report and notebooks would also need updates to:

- replace promoted-tokenizer reconstruction tables;
- replace or add latent-geometry figures from `outputs/latent_geometry/sp500_vix_hidden128_vq`;
- record the selected sampling setting, currently `temperature=0.8, top_k=20`;
- discuss the token-likelihood and squared-return autocorrelation regressions explicitly;
- update the discrete model-selection table and any S&P500/VIX paper-style comparison tables.

## Next Recommended Branch

The next branch should not jump directly to MGVQ. Two reasonable paths remain:

- causal frequency decomposition, if the aim is to improve volatility and squared-return
  autocorrelation without introducing a multi-code token interface;
- grouped tokenizers, only after defining a stable multi-code prior interface and a concrete
  failure mode that standard VQ cannot address.

Given the current evidence, causal frequency decomposition is the cleaner next branch. It targets
the residual volatility/autocorrelation weaknesses while avoiding the sparse same-time code-pair
problem already observed for RVQ q2.

