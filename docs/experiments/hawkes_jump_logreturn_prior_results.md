# Hawkes-Jump Log-Return Prior Results

## Status

The hidden128 log-return tokenizer priors were trained and evaluated locally on
the Ogata Hawkes-jump benchmark. This run is a non-registry experimental result:
it does not change model families, simulator parameters, public defaults, or the
model registry.

The prior configs still specify two epochs, which is appropriate for smoke runs
but too short for a meaningful token-prior comparison. All four candidates were
therefore trained with the same controlled override, `--epochs 50`, with W&B
disabled.

## Run Setup

| Candidate | Config | Output directory |
|---|---|---|
| hidden128 log-return cb64 + additive AR | `hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_additive.yaml` | `outputs/hawkes_jump_logreturn_priors/hidden128_logreturn_cb64_additive` |
| hidden128 log-return cb64 + conv-transformer k3 | `hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer.yaml` | `outputs/hawkes_jump_logreturn_priors/hidden128_logreturn_cb64_conv_transformer` |
| hidden128 log-return cb32 + additive AR | `hawkes_jump_causal_token_prior_hidden128_logreturn_cb32_additive.yaml` | `outputs/hawkes_jump_logreturn_priors/hidden128_logreturn_cb32_additive` |
| hidden128 log-return cb32 + conv-transformer k3 | `hawkes_jump_causal_token_prior_hidden128_logreturn_cb32_conv_transformer.yaml` | `outputs/hawkes_jump_logreturn_priors/hidden128_logreturn_cb32_conv_transformer` |

Evaluation used `n_sample=1024`, `seed=99`, `temperature=1.0`, and
`top_k=none`. Decoded log returns were converted to normalised price paths before
market and jump diagnostics. The secondary sampling setting
`temperature=0.8, top_k=20` was not run because the primary pass did not show a
material jump-count or tail failure that required immediate sampling-temperature
triage.

## Prior Fit

| Candidate | Runtime s | Best epoch | Best CE | Best acc. | Best perplexity | Final CE |
|---|---:|---:|---:|---:|---:|---:|
| cb64 additive AR | 89.8 | 32 | 3.6963 | 0.0640 | 40.30 | 3.7284 |
| cb64 conv-transformer k3 | 105.1 | 20 | 3.6836 | 0.0626 | 39.79 | 3.8404 |
| cb32 additive AR | 90.6 | 18 | 3.1712 | 0.0716 | 23.84 | 3.2313 |
| cb32 conv-transformer k3 | 105.6 | 13 | 3.1740 | 0.0713 | 23.90 | 3.3256 |

All models overfit after their best validation epoch, so evaluation used the
saved `best_model` checkpoint. Cross-entropy is not directly comparable between
the 32-code and 64-code tokenizers, but within the 64-code setting the
conv-transformer obtains the better best validation CE.

## Token Sampling

| Candidate | Sampled active codes | Sampled perplexity | Real active codes | Real perplexity | Marginal L1 | Transition L1 | Run W1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cb64 additive AR | 64 | 44.78 | 63 | 44.82 | 0.0493 | 0.4082 | 0.0048 |
| cb64 conv-transformer k3 | 64 | 44.95 | 63 | 44.82 | 0.0514 | 0.4131 | 0.0047 |
| cb32 additive AR | 32 | 25.79 | 31 | 25.78 | 0.0424 | 0.2786 | 0.0017 |
| cb32 conv-transformer k3 | 32 | 25.76 | 31 | 25.78 | 0.0504 | 0.2768 | 0.0024 |

The code-collapse blocker from the first comparison is resolved in sampled
tokens. The previous price-tokenizer comparison had only `6/64` extracted active
codes for standard VQ and `4/64` for hidden128 VQ; the log-return tokenizers now
use essentially the full available alphabet before prior sampling, and the
sampled priors preserve that usage.

## Smooth Path Metrics

| Candidate | MMD | SWD | Terminal W1 | Volatility W1 | Drawdown W1 |
|---|---:|---:|---:|---:|---:|
| cb64 additive AR | 0.1296 | 0.0207 | 0.0223 | 0.0008 | 0.0157 |
| cb64 conv-transformer k3 | 0.1015 | 0.0163 | 0.0211 | 0.0006 | 0.0120 |
| cb32 additive AR | 0.3641 | 0.0488 | 0.0736 | 0.0005 | 0.0185 |
| cb32 conv-transformer k3 | 0.4275 | 0.0580 | 0.0927 | 0.0004 | 0.0334 |

The 64-code log-return priors are much stronger than the earlier price-tokenizer
discrete runs. In the first comparison, hidden128 price-tokenizer
conv-transformer k3 reported MMD `1.6380`, SWD `0.1446`, and terminal W1
`1.1902`. The log-return cb64 conv-transformer reduces these to `0.1015`,
`0.0163`, and `0.0211`, respectively.

Against the first-comparison continuous BetaCVAE result, the log-return discrete
priors are also stronger on these diagnostics: BetaCVAE reported MMD `0.9638`,
SWD `0.1133`, and terminal W1 `0.2263`. This comparison should still be treated
as experimental because the representation changed from price-level tokenisation
to log-return tokenisation.

## Jump And Tail Diagnostics

The real Ogata evaluation paths had mean detected jumps per path `0.2422`,
paths-with-jumps fraction `0.1943`, negative detected jump fraction `0.9718`,
VaR 1% `-0.0724`, ES 1% `-0.1064`, VaR 5% `-0.0447`, and ES 5% `-0.0646`.

| Candidate | Mean jumps | Jump-count W1 | Inter-arrival W1 | Jump-size W1 | Paths with jumps | Negative jump frac. |
|---|---:|---:|---:|---:|---:|---:|
| cb64 additive AR | 0.2041 | 0.0381 | 5.2964 | 0.0139 | 0.1855 | 1.0000 |
| cb64 conv-transformer k3 | 0.1934 | 0.0488 | 3.9862 | 0.0147 | 0.1738 | 1.0000 |
| cb32 additive AR | 0.2559 | 0.0547 | 6.7908 | 0.0179 | 0.2285 | 0.9962 |
| cb32 conv-transformer k3 | 0.2461 | 0.0430 | 2.7420 | 0.0177 | 0.2178 | 1.0000 |

| Candidate | VaR 1% | ES 1% | VaR 5% | ES 5% |
|---|---:|---:|---:|---:|
| cb64 additive AR | -0.0717 | -0.0975 | -0.0428 | -0.0612 |
| cb64 conv-transformer k3 | -0.0727 | -0.0979 | -0.0436 | -0.0622 |
| cb32 additive AR | -0.0752 | -0.1038 | -0.0459 | -0.0654 |
| cb32 conv-transformer k3 | -0.0749 | -0.1024 | -0.0456 | -0.0651 |

The log-return priors now reproduce the detected jump-count scale, downside
asymmetry, and tail quantiles much better than the price-tokenizer priors. The
first hidden128 price-tokenizer conv-transformer had jump-count W1 `1.7734`,
jump-size W1 `0.4300`, negative detected jump fraction `0.0000`, VaR 1%
`-0.0299`, and ES 1% `-0.0352`. The best log-return candidates reduce the
jump-count error below `0.05`, preserve almost all detected jumps as negative,
and match VaR closely, although ES remains mildly less severe for the 64-code
models.

Compared with the first-comparison BetaCVAE, the log-return priors show a clear
rare-event diagnostic improvement. BetaCVAE produced too many detected jumps
with mean detected jumps per path `16.8164` and jump-count W1 `16.5742`; the
log-return discrete priors stay near the real mean of `0.2422`. BetaCVAE had
heavier left-tail VaR/ES than the simulator, while the log-return priors track
the Ogata tail scale closely.

## Decision

The best overall candidate is `hidden128_logreturn_cb64 + conv-transformer k3`.
It has the strongest smooth path profile, full sampled code usage, the best
64-code validation CE, and acceptable jump diagnostics. The cb64 additive AR is
a close baseline because it has the lowest jump-count W1 and jump-size W1, while
the cb32 conv-transformer is useful as a jump-timing sensitivity point because it
has the lowest inter-arrival W1.

A discrete rare-event advantage is now visible in this log-return setting. The
advantage was not present in the first price-tokenizer comparison because the
tokenizer alphabet had collapsed before prior training. After moving to
log-return tokenisation, the discrete priors use the codebooks, recover the
simulator's sparse jump-count scale, retain downside jump asymmetry, and improve
substantially over both the earlier price-tokenizer discrete priors and the first
continuous BetaCVAE diagnostic result on jump-specific metrics.

This is not yet registry-quality evidence. The result should be repeated across
seeds and evaluated against a matched continuous log-return baseline before any
registry update. The next experiment should compare the cb64 conv-transformer and
cb64 additive AR priors against a continuous baseline trained on the same
log-return representation and then rerun the strongest pair across multiple
seeds.
