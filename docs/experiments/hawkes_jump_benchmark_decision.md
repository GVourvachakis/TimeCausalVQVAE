# Hawkes-Jump Benchmark Decision

## Status

The Hawkes-jump branch should continue. The benchmark has moved from a useful
diagnostic idea to an integrated rare-event testbed with a research-quality
simulator, visual diagnostics, no-leakage checks, tokenizer utilisation evidence,
and a first credible discrete log-return prior result.

No registry update should be made yet. The current evidence is strong enough to
continue the branch, but not yet robust enough to promote a trained model.

## Simulator Status

The Ogata backend is research-quality and ready for an eventual public merge. It
uses continuous-time Ogata modified thinning for marked Hawkes event times,
exponential intensity decay, mark-dependent excitation, asymmetric folded-normal
jump marks, jump-excited volatility, and exact sub-interval Brownian variance
integration before projecting events onto the regular model grid.

The fixed-grid backend should remain available, but only as a smoke and
throughput backend. It is useful for fast integration checks because it is simple
and deterministic, but research comparisons should use `simulation_scheme:
ogata`.

The visual diagnostics support this status. The Ogata dataset passes positivity,
finiteness, shape, jump-incidence, downside-asymmetry, intensity, and volatility
checks. The generated paths are still scenario data observed on a regular grid;
the simulator does not claim arbitrage-free risk-neutral generation.

## Tokenizer Status

The first price-level tokenizers collapsed before prior training. The diagnostic
comparison extracted only `6/64` active codes for the standard VQ tokenizer and
`4/64` active codes for the hidden128 tokenizer. That made the first discrete
prior comparison unfair for rare-event structure.

The log-return tokenizers fixed the utilisation problem. The main hidden128
log-return tokenizer used `63/64` codes with perplexity `44.82`, reconstruction
L1 `0.001375`, and meaningful jump/non-jump code separation. The
hidden128 log-return cb32 tokenizer used `31/32` codes and had the strongest
rare-code lift near jumps.

The conclusion is representation-specific: Hawkes jumps should be tokenised on
log returns by default. Price-tokenizer results remain useful as a failure mode,
not as the benchmark's preferred discrete representation.

## Prior And Model Status

The best current prior candidate is `hidden128_logreturn_cb64 +
conv-transformer k3`. It used all `64` sampled codes, had sampled token
perplexity `44.95`, and gave the strongest smooth profile among the log-return
priors:

| Metric | cb64 conv-transformer k3 |
| --- | ---: |
| MMD | 0.1015 |
| SWD | 0.0163 |
| Terminal W1 | 0.0211 |
| Volatility W1 | 0.0006 |
| Drawdown W1 | 0.0120 |

The cb64 additive AR prior remains an important baseline. It was slightly weaker
on smooth metrics, but had the lowest jump-count W1 (`0.0381`) and jump-size W1
(`0.0139`). The cb32 conv-transformer is useful as a timing diagnostic because
it had the lowest inter-arrival W1 (`2.7420`).

Compared with the first continuous BetaCVAE run, the log-return discrete priors
now show a discrete rare-event advantage. The first BetaCVAE result had MMD
`0.9638`, SWD `0.1133`, terminal W1 `0.2263`, mean detected jumps per path
`16.8164`, and jump-count W1 `16.5742`. The cb64 conv-transformer log-return
prior reduced these smooth metrics to MMD `0.1015`, SWD `0.0163`, terminal W1
`0.0211`, and kept mean detected jumps near the real Ogata scale at `0.1934`
versus real `0.2422`.

This is visible evidence for the intended discrete advantage, but it is not yet
registry-quality evidence because the comparison is single-seed and the
continuous baseline has not yet been retrained on the same log-return
representation.

## Metrics

The smooth path profile favours the hidden128 cb64 conv-transformer. It improves
substantially over the earlier price-tokenizer hidden128 conv-transformer, which
reported MMD `1.6380`, SWD `0.1446`, and terminal W1 `1.1902`.

The jump-regime profile now matches the sparse jump scale. Real Ogata evaluation
paths had mean detected jumps `0.2422`, paths-with-jumps fraction `0.1943`, and
negative detected jump fraction `0.9718`. The cb64 conv-transformer produced
mean detected jumps `0.1934`, paths-with-jumps fraction `0.1738`, negative jump
fraction `1.0000`, jump-count W1 `0.0488`, inter-arrival W1 `3.9862`, and
jump-size W1 `0.0147`.

The VaR/ES profile is close but still worth monitoring. Real Ogata evaluation
paths had VaR 1% `-0.0724`, ES 1% `-0.1064`, VaR 5% `-0.0447`, and ES 5%
`-0.0646`. The cb64 conv-transformer produced VaR 1% `-0.0727`, ES 1%
`-0.0979`, VaR 5% `-0.0436`, and ES 5% `-0.0622`. VaR is well matched; ES is
slightly less severe for the cb64 priors.

Token diagnostics are healthy enough for prior work. The cb64 conv-transformer
sampled all `64` codes, with sampled perplexity `44.95`, real perplexity
`44.82`, marginal L1 `0.0514`, transition L1 `0.4131`, and run W1 `0.0047`.
The transition metric remains a weakness, but it is now a genuine prior-learning
issue rather than a collapsed-tokenizer artefact.

## Decision

Continue the Hawkes branch. The benchmark is exposing the intended distinction
between smooth path quality and rare-event structure.

Update the registry later, not now. A registry update requires seed robustness,
a matched continuous log-return baseline, and confirmation that the discrete
advantage survives beyond this single local run.

Merge only the simulator and dataset infrastructure to `main` when ready. The
Ogata backend, fixed-grid smoke backend, no-leakage checks, and diagnostics are
valuable independently of the experimental model results. Training outputs,
single-seed conclusions, and registry changes should stay on the research branch
until robustness is established.

Run seed robustness next. The immediate scientific risk is not architecture
choice, but whether the log-return discrete advantage is stable across seeds.

Tune the tokenizer further only after seed robustness. Current code utilisation
is healthy; further tokenizer tuning should target ES calibration,
transition/run-length fidelity, and rare-code interpretability rather than basic
anti-collapse.

Do not switch to continuous LSGM or another branch yet. A stronger continuous
baseline is valuable, but the current branch has just produced the first
evidence that the discrete representation can help on the intended rare-event
benchmark.

## Next Exact Step

Run a three-seed robustness comparison with unchanged Ogata simulator
parameters:

1. hidden128 log-return cb64 tokenizer plus additive AR prior;
2. hidden128 log-return cb64 tokenizer plus conv-transformer k3 prior;
3. a matched continuous BetaCVAE or InfoCVAE trained and evaluated on
   `data_output: log_return`, with generated log returns converted to
   normalised prices before diagnostics.

Use the same metrics as this decision note: MMD, SWD, terminal W1, volatility
W1, drawdown W1, jump-count W1, inter-arrival W1, jump-size W1, negative jump
fraction, VaR/ES, sampled active codes, sampled perplexity, transition L1, and
run-length diagnostics. If the cb64 conv-transformer remains best or tied on
smooth metrics while preserving the jump-count and tail advantage, then prepare
a registry candidate and a separate simulator-only merge plan.
