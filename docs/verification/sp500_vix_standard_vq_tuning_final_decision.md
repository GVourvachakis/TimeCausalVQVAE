# S&P500/VIX Standard VQ Tuning Final Decision

Status: final standard-VQ tuning decision for the `hidden128` tokenizer candidate. No models were
trained, no source code was changed, and no signatures, GroupedRVQ, MGVQ, diffusion, or
cross-attention components were added.

## Decision

Do not replace the promoted public standard-VQ baseline yet.

`hidden128` is the leading standard-VQ research candidate. It should be used in future
tokenizer/prior experiments where improved standard-VQ capacity is desired, and it should be
reported as the best current standard-VQ ablation when the report needs the strongest discrete
research model. The public baseline remains unchanged because the unchanged additive AR prior models
the promoted tokenizer tokens much more easily, and hidden128 still regresses squared-return
autocorrelation and volatility W1.

The next branch should test causal frequency decomposition before revisiting GroupedRVQ or MGVQ.

## Current Promoted Public Baseline

The current public discrete baseline remains:

- tokenizer config: `configs/experiments/sp500_vix_causal_vq_tokenizer.yaml`;
- prior config: `configs/experiments/sp500_vix_causal_token_prior_additive.yaml`;
- tokenizer family: standard causal VQ;
- codebook interface: one 64-code index per time step, 16-dimensional codebook embeddings;
- condition: scalar VIX, `condition_dim=1`;
- prior: additive VIX-only causal AR token prior;
- promoted sampling setting: temperature `0.8`, `top_k=40`.

Main baseline metrics:

| Metric | Promoted baseline |
| --- | ---: |
| Tokenizer reconstruction L1 | 0.011202 |
| Tokenizer reconstruction L2 | 0.012552 |
| Tokenizer terminal error | 0.005566 |
| Tokenizer volatility error | 0.000810 |
| Ablation-slice active codes | 57/64 |
| Ablation-slice perplexity | 33.296204 |
| Full latent-geometry active codes | 63/64 |
| Full latent-geometry perplexity | 39.055717 |
| Prior eval CE | 0.914807 |
| Prior eval accuracy | 0.647449 |
| Prior eval perplexity | 2.507927 |
| Paper-style profile | 0.298020 |
| Paper-style MMD | 0.279341 |
| Paper-style SWD | 0.007674 |
| Paper-style terminal W1 | 0.009817 |
| Paper-style volatility W1 | 0.001188 |
| Squared-return autocorrelation L1 | 0.041300 |

This baseline keeps the simplest causal interface and remains the safest public default.

## Hidden128 Candidate

The leading standard-VQ candidate is:

- tokenizer config: `configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml`;
- prior config: `configs/experiments/sp500_vix_causal_token_prior_hidden128_candidate.yaml`;
- tokenizer family: standard causal VQ with wider encoder/decoder hidden capacity;
- codebook interface: unchanged one-code-per-time-step standard VQ;
- condition: scalar VIX, `condition_dim=1`;
- prior: unchanged additive VIX-only causal AR token prior;
- selected sampling setting: temperature `0.8`, `top_k=20`.

Main hidden128 metrics:

| Metric | Hidden128 |
| --- | ---: |
| Tokenizer reconstruction L1 | 0.005926 |
| Tokenizer reconstruction L2 | 0.007156 |
| Tokenizer terminal error | 0.004722 |
| Tokenizer volatility error | 0.000797 |
| Ablation-slice active codes | 58/64 |
| Ablation-slice perplexity | 44.763039 |
| Full latent-geometry active codes | 64/64 |
| Full latent-geometry perplexity | 50.358673 |
| Prior eval CE, seed99 | 1.237114 |
| Prior eval CE, seed1 | 1.235018 |
| Prior eval CE, seed2 | 1.232192 |
| Best paper-style profile | 0.242020 |
| Best paper-style MMD | 0.229078 |
| Best paper-style SWD | 0.007175 |
| Best paper-style terminal W1 | 0.004509 |
| Best paper-style volatility W1 | 0.001258 |
| Best squared-return autocorrelation L1 | 0.060885 |

The selected sampling policy is part of the hidden128 result and must be reported with it.

## Evidence Summary

Tokenizer reconstruction favours hidden128. Relative to the promoted baseline, hidden128 roughly
halves reconstruction L1, improves reconstruction L2, improves terminal reconstruction error, and
essentially matches the tokenizer-level volatility error.

Latent geometry also favours hidden128. It uses all 64 codes with global perplexity `50.358673`,
compared with 63/64 active codes and perplexity `39.055717` for the promoted baseline. Its VIX
bucket support remains broad, using at least 61 active codes in every bucket.

Prior token likelihood favours the promoted baseline. The promoted additive AR prior reaches eval
CE `0.914807` and perplexity `2.507927`. Hidden128 is stable across seeds, but its eval CE remains
around `1.23-1.24` with perplexity around `3.45-3.47`.

Paper-style diagnostics favour hidden128 on the primary generated-market profile. At the original
`temperature=0.8`, `top_k=40` setting, all three hidden128 prior seeds beat the promoted baseline
paper-style profile. The best hidden128 sampling setting, `temperature=0.8`, `top_k=20`, improves
the profile to `0.242020` versus the promoted baseline `0.298020`.

Seed robustness is positive but not promotion-complete. Hidden128 seed99, seed1, and seed2 all
beat the promoted baseline on paper-style profile, but the same seeds consistently remain worse on
token likelihood and squared-return autocorrelation.

Sampling robustness identifies a narrow stable region. Temperature `0.8` is the reliable setting.
`top_k=20` is best by profile, `top_k=40` preserves full sampled code coverage, and temperature
`1.0` improves some transition/run-length statistics while worsening terminal-return and SWD
diagnostics.

## Model-Selection Profiles

The verification notes use:

```text
MMD + SWD + terminal_return_wasserstein + volatility_wasserstein
```

Lower is better.

| Model or setting | Decoded profile | Paper-style profile |
| --- | ---: | ---: |
| Continuous BetaCVAE reference | n/a | 0.172891 |
| Promoted standard-VQ baseline, temp 0.8, top-k 40 | 0.309792 | 0.298020 |
| Hidden128 seed99, temp 0.8, top-k 40 | 0.270029 | 0.266589 |
| Hidden128 seed1, temp 0.8, top-k 40 | 0.281735 | 0.271278 |
| Hidden128 seed2, temp 0.8, top-k 40 | 0.265540 | 0.203517 |
| Hidden128 seed99, temp 0.8, top-k 20 | 0.242020 | 0.242020 |

The profile score makes hidden128 the leading standard-VQ research model, not the new public
baseline.

## Reasons Not To Promote Yet

Hidden128 is not promoted because:

- token likelihood is weaker under the unchanged additive AR prior;
- squared-return autocorrelation regresses, including at the selected `top_k=20` sampling setting;
- volatility W1 is slightly worse than the promoted baseline in paper-style evaluation;
- the strongest hidden128 result depends on a changed sampling policy, `top_k=20` instead of
  `top_k=40`;
- the continuous BetaCVAE is still stronger on MMD, returns W1, volatility W1, maximum drawdown,
  and autocorrelation diagnostics.

These are not failures of hidden128 as an ablation. They are enough to avoid replacing the public
default.

## Reporting Guidance

If a future paper, report, or notebook wants the best discrete research model, present hidden128 as
an improved standard-VQ ablation:

- name it as `hidden128`, not as the public baseline;
- state the tokenizer and prior configs explicitly;
- state the sampling setting explicitly: temperature `0.8`, `top_k=20`;
- keep the promoted baseline table visible for continuity;
- report both the stronger paper-style profile and the weaker token likelihood;
- call out the squared-return autocorrelation and volatility-W1 regressions.

This framing lets the report show the best standard-VQ research evidence without moving the
baseline goalposts.

## Next Branch Recommendation

Proceed to causal frequency decomposition before GroupedRVQ or MGVQ.

The standard-VQ tuning stage has shown that extra capacity can improve reconstruction, geometry,
and generated-market profile, but the residual weaknesses are volatility and squared-return
autocorrelation. A strictly causal trend/residual or low/high-frequency decomposition targets those
weaknesses while preserving a simpler prior interface than grouped or multi-code tokenizers.

GroupedRVQ and MGVQ remain deferred until there is a stable multi-code prior interface and a
specific measured failure mode that standard VQ plus causal decomposition cannot address.
