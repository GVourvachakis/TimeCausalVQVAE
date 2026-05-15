# S&P500/VIX Hidden128 Prior Robustness Results

Status: completed. No model code, tokenizer architecture, signature conditioning,
GroupedRVQ, MGVQ, diffusion, or objective change was made.

## Execution

The requested live W&B run was attempted with:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread
```

W&B initialisation failed before training with:

```text
wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec.
```

Two live-HTTP retry requests with escalated network permissions were submitted, but the automatic
approval review timed out both times. No W&B run URLs were produced. The robustness training was
therefore rerun with `--no-wandb`, as specified by the setup protocol. The runner flag used for
paper-style diagnostics was the implemented `--paper-style` flag.

The first combined no-W&B run completed seed 1, then the seed 2 subprocess stopped after epoch 24
without a final aggregate row. Seed 2 was rerun alone and completed successfully under:

```text
outputs/sp500_vix_discrete/token_prior/hidden128_seed_ablation_seed2_retry
```

## Seed Robustness

All seed robustness runs used the hidden128 tokenizer artifacts, additive VIX-only conditioning,
`n_sample=1000`, evaluation seed `99`, temperature `0.8`, and `top_k=40`.

| Prior run | Train seed | Runtime (s) | Best epoch | Eval CE | Eval acc | Eval ppl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hidden128 seed99 reference | 99 | n/a | 100 | 1.237114 | 0.509476 | 3.467770 |
| hidden128 seed1 | 1 | 1384.591 | 100 | 1.235018 | 0.512169 | 3.459926 |
| hidden128 seed2 retry | 2 | 2387.273 | 100 | 1.232192 | 0.512020 | 3.449989 |

The token-likelihood results are stable across the three trained priors. Seed 2 is the strongest
likelihood run, but the spread is small enough that the main decision should be based on decoded
and paper-style diagnostics.

## Decoded Metrics

Profile score is:

```text
MMD + SWD + terminal_return_wasserstein + volatility_wasserstein
```

Lower is better.

| Prior run | Profile | MMD | SWD | Terminal W1 | Volatility W1 | Active codes | Token ppl | Transition L1 | Run-length dist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hidden128 seed99 reference | 0.270029 | 0.257473 | 0.007442 | 0.003729 | 0.001386 | 63/64 | 42.205532 | 0.215735 | 0.335666 |
| hidden128 seed1 | 0.281735 | 0.260540 | 0.008614 | 0.011167 | 0.001413 | 62/64 | 40.644531 | 0.193840 | 0.396695 |
| hidden128 seed2 retry | 0.265540 | 0.246141 | 0.008311 | 0.009669 | 0.001418 | 62/64 | 40.166172 | 0.208921 | 0.446848 |

Seed 1 is slightly weaker than the seed99 reference on decoded profile score, mainly due to
terminal-return W1. Seed 2 improves the decoded profile versus seed99, with better MMD and a similar
volatility error. Code usage remains broad at 62-63 active codes.

## Paper-Style Metrics

| Prior run | Profile | MMD | SWD | Returns W1 | Terminal W1 | Volatility W1 | Max drawdown W1 | Return AC L1 | Squared-return AC L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hidden128 seed99, temp 0.8, top-k 40 | 0.266589 | 0.253583 | 0.007210 | 0.001097 | 0.004496 | 0.001301 | 0.010136 | 0.041318 | 0.064390 |
| hidden128 seed1, temp 0.8, top-k 40 | 0.271278 | 0.250218 | 0.008251 | 0.001208 | 0.011366 | 0.001443 | 0.012349 | 0.043973 | 0.065187 |
| hidden128 seed2, temp 0.8, top-k 40 | 0.203517 | 0.190422 | 0.006152 | 0.001104 | 0.005626 | 0.001317 | 0.009787 | 0.041848 | 0.063873 |
| promoted baseline, temp 0.8, top-k 40 | 0.298020 | 0.279341 | 0.007674 | 0.001242 | 0.009817 | 0.001188 | 0.010502 | 0.051741 | 0.041300 |
| continuous BetaCVAE reference | 0.172891 | 0.154421 | 0.008785 | 0.000602 | 0.009051 | 0.000634 | 0.007667 | 0.025972 | 0.029462 |

All three hidden128 prior seeds beat the promoted standard-VQ baseline on the paper-style profile.
Seed 2 is substantially stronger than both the seed99 reference and the promoted baseline. The
continuous BetaCVAE remains the strongest reference on MMD and volatility, so the discrete model is
not a replacement for the continuous benchmark.

## Seed0 Sampling Grid

The existing hidden128 reference checkpoint was evaluated at:

```text
outputs/sp500_vix_discrete/token_prior/candidate_priors/sp500_vix_causal_token_prior_hidden128_candidate_seed99/best_model
```

The requested `seed0` path was not present; this is the previously documented hidden128 candidate
checkpoint trained with experiment seed `99`.

| Temp | Top-k | Profile | MMD | SWD | Terminal W1 | Volatility W1 | Active codes | Token ppl | Transition L1 | Run-length dist |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6 | none | 0.335593 | 0.316651 | 0.009160 | 0.007839 | 0.001944 | 60/64 | 38.701260 | 0.280399 | 0.780362 |
| 0.6 | 20 | 0.347565 | 0.328484 | 0.009363 | 0.007759 | 0.001959 | 60/64 | 38.322296 | 0.271303 | 0.768069 |
| 0.6 | 40 | 0.376030 | 0.355912 | 0.010093 | 0.008014 | 0.002011 | 59/64 | 37.668682 | 0.303856 | 0.781734 |
| 0.8 | none | 0.245956 | 0.233525 | 0.006649 | 0.004415 | 0.001367 | 63/64 | 41.970245 | 0.223241 | 0.372016 |
| 0.8 | 20 | 0.242020 | 0.229078 | 0.007175 | 0.004509 | 0.001258 | 63/64 | 42.898106 | 0.223900 | 0.390197 |
| 0.8 | 40 | 0.266589 | 0.253583 | 0.007210 | 0.004496 | 0.001301 | 64/64 | 41.683281 | 0.240341 | 0.372891 |
| 1.0 | none | 0.251476 | 0.227252 | 0.012425 | 0.010530 | 0.001269 | 64/64 | 45.341309 | 0.199133 | 0.123151 |
| 1.0 | 20 | 0.289511 | 0.265435 | 0.011384 | 0.011758 | 0.000935 | 63/64 | 45.527821 | 0.177118 | 0.112202 |
| 1.0 | 40 | 0.303367 | 0.276092 | 0.013663 | 0.012337 | 0.001275 | 64/64 | 45.475639 | 0.199302 | 0.150293 |

The best decoded sampling setting is `temperature=0.8, top_k=20`. It improves the profile from
`0.266589` at `top_k=40` to `0.242020`, mainly through lower MMD. A paper-style rerun at
`temperature=0.8, top_k=20` produced the same profile components:

| Setting | Profile | MMD | SWD | Returns W1 | Terminal W1 | Volatility W1 | Max drawdown W1 | Return AC L1 | Squared-return AC L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hidden128 seed99, temp 0.8, top-k 20 | 0.242020 | 0.229078 | 0.007175 | 0.001044 | 0.004509 | 0.001258 | 0.008910 | 0.038262 | 0.060885 |

## Comparison

Relative to hidden128 seed99 at the original `top_k=40`, `top_k=20` improves MMD, maximum-drawdown
W1, return-autocorrelation L1, and squared-return-autocorrelation L1, with a negligible terminal W1
change and slightly lower active-code coverage. Relative to the promoted baseline, hidden128 is
better on paper-style profile, MMD, SWD, returns W1, terminal W1, maximum-drawdown W1, and return
autocorrelation. The promoted baseline remains better on squared-return autocorrelation and slightly
better on volatility W1.

Relative to the continuous BetaCVAE, hidden128 remains weaker on the profile score and most
distributional diagnostics, but it has a better terminal-return W1 at the best sampling setting.

## Decision

Do not change the promoted baseline in this prompt. The robustness evidence is positive enough to
continue hidden128 as the leading standard-VQ tokenizer candidate, but not enough to make a final
architecture promotion without recording the sampling policy and resolving the W&B visibility gap.

Recommended next action:

- keep the promoted baseline as the current default;
- carry hidden128 forward as the preferred tokenizer candidate for the next prior comparison;
- use `temperature=0.8, top_k=20` for hidden128 sampling diagnostics unless a later sweep replaces
  it;
- run one more hidden128 prior seed only if the next branch needs stronger seed-level confidence;
- do not switch to another tokenizer, GroupedRVQ, MGVQ, or signatures before closing this
  standard-VQ robustness decision.

