# Hawkes-Jump Benchmark Plan

Status: planning note only. This document proposes a future synthetic benchmark. It does
not implement code, train models, add dependencies, generate artefacts, or update the model
registry.

## Motivation

The current discrete pipeline is competitive, but it is not yet dominant on the existing
diffusion and stochastic-volatility benchmarks. The public standard VQ plus additive AR prior
remains the stable S&P500/VIX discrete baseline, while the hidden128 causal conv-transformer is
the stronger research candidate for several component diagnostics. Existing research notes still
show that continuous TC-VAE baselines can remain ahead on smooth path-profile metrics.

This motivates a complementary benchmark rather than another smooth-diffusion comparison. Discrete
latents may be more useful when paths contain regime switches, discontinuous jumps, rare-event
clusters, and black-swan-like drawdowns. These behaviours are awkward to summarise with only MMD,
SWD, or volatility distance, but they are precisely the kind of structured temporal events that a
tokeniser and causal token prior might encode as persistent states or sharp transitions.

Hawkes processes are a natural source for this benchmark because they model self-exciting event
arrivals: one event temporarily increases the probability of further nearby events. They are common
in market microstructure models for clustered order arrivals, trades, cancellations, and price
moves. A Hawkes-jump price process therefore gives a controlled way to stress-test whether discrete
latent models capture event clustering and tail regimes better than a continuous TC-VAE baseline.

## Synthetic Process

The benchmark should use one-dimensional normalised price paths with the existing-compatible shape
`[n_sample, 60, 1]`. Each path starts at one and evolves through log returns,

```text
r_t = drift_t + sqrt(v_t * dt) * eps_t + sum_j Y_{t,j}
S_t = S_{t-1} * exp(r_t)
```

where the first term is the diffusion drift, the second term is the Brownian or
stochastic-volatility component, and the final term aggregates all jump marks arriving in the
time interval.

The default event-arrival mechanism should be a stationary marked Hawkes process. Its intensity
should have baseline arrival rate, exponential mean reversion, and excitation after previous
jumps. The branching ratio must stay below one so the event process is stable. Mark-dependent
excitation should be allowed, so a large negative jump can create stronger subsequent jump
intensity than a small positive jump. A simpler univariate Hawkes version should be retained as a
dataset smoke variant before adding marks and volatility feedback.

Jump sizes should be asymmetric. A practical default is a two-sided mixture in which downside
jumps are either more frequent, larger in magnitude, or both. The downside distribution should
include a rare severe-loss component so that a small fraction of paths exhibit black-swan-like
terminal losses or maximum drawdowns. Upside jumps should remain possible to avoid making the
benchmark a trivial one-sided detector.

An optional volatility-excitation channel should be included as a controlled difficulty setting.
After a jump, especially a large negative jump, short-run volatility can increase and then decay
back toward its baseline level. This creates clustered jumps and clustered large returns, making
the benchmark sensitive to both event timing and post-event volatility persistence.

## No-Arbitrage Caveat

The generated discounted price paths should be treated as scenario data, not automatically as an
arbitrage-free pricing model. A physical-measure Hawkes jump process can be useful for stress
testing and generative benchmarking without defining a risk-neutral measure.

The benchmark should not claim arbitrage-free generation unless a risk-neutral construction is
introduced explicitly. If the process is later used for pricing claims, the implementation must
specify the money-market account, compensator, jump-risk premia, and martingale condition for
discounted prices.

Relevant diagnostics can still be reported:

- empirical drift of discounted returns;
- mean terminal discounted return and confidence interval;
- pathwise martingale-residual proxy, such as conditional next-return averages by state or
  intensity bucket;
- effect of drift normalisation, with and without jump compensator terms.

These diagnostics are checks on scenario calibration. They are not sufficient to prove absence of
arbitrage.

## Candidate Models

The benchmark should compare the same model families used in the current repository, with matched
sample counts, path length, and evaluation budget:

- continuous `BetaCVAE` or `InfoCVAE` baseline, selected consistently with
  `trained_models/model_registry.yaml` and the continuous objective used for the closest existing
  experiment;
- standard causal VQ tokenizer plus additive AR token prior;
- hidden128 causal VQ tokenizer plus additive AR token prior, recovering the S&P500/VIX-style
  hidden128 additive config from research history or defining the equivalent config before any
  future training;
- hidden128 causal VQ tokenizer plus causal conv-transformer k3 token prior.

The comparison should keep the continuous baseline visible rather than treating the discrete
models as the default winner. The current evidence supports a sharper hypothesis: continuous
TC-VAE may retain the best smooth-path fit, while the hidden128 conv-transformer may be better on
jump timing, rare-regime transitions, and event clustering.

## Metrics

The first evaluation group should preserve the existing path diagnostics:

- MMD;
- SWD;
- terminal W1;
- volatility W1;
- drawdown W1;
- return autocorrelation L1;
- squared-return autocorrelation L1.

The second group should be jump-specific. For generated paths, jumps should be detected from
returns because the trained models output prices rather than simulator event labels. Oracle
simulator event labels should be used only to validate the synthetic dataset and jump detector.

- detected jump-count distribution per path;
- jump-size distribution, with separate downside and upside summaries;
- inter-arrival time distribution;
- jump-clustering or burst diagnostics, such as counts of adjacent jump windows and Hawkes-style
  over-dispersion;
- tail exceedance against real-simulator quantile thresholds;
- VaR and ES at severe left-tail levels;
- black-swan path frequency, defined by extreme terminal loss, maximum drawdown, or clustered
  jump criteria.

The third group should diagnose discrete-token behaviour:

- marginal code usage and rare-code usage;
- transition matrix distance;
- run-length distance;
- transition and run-length diagnostics conditioned on detected regimes;
- alignment between jump windows and token changes or rare-token activations.

The final selection table should show at least two profiles. A smooth-path profile can aggregate
MMD, SWD, terminal W1, volatility W1, and drawdown W1. A jump-regime profile can aggregate jump
count W1, jump-size W1, inter-arrival W1, clustering distance, tail exceedance error, VaR error,
ES error, and token/regime transition diagnostics where applicable. The benchmark should not
select from MMD alone.

## Stage Gates

1. Dataset smoke

   Verify positive normalised paths, shape `[n_sample, 60, 1]`, plausible jump counts, stable
   Hawkes intensity, asymmetric jump tails, and visible volatility excitation when that option is
   enabled. Confirm that the univariate Hawkes smoke variant is stable before using the marked
   variant.

2. No-leakage

   Ensure dataset conditions, jump labels, intensity summaries, and volatility states do not leak
   future information into model inputs. Existing causal-conv, tokenizer, and token-prior
   no-leakage checks should be run for the applicable discrete candidates.

3. Continuous and discrete training

   Train only after the dataset smoke and no-leakage checks pass. Use matched sample counts,
   seeds, path length, batch budget, and evaluation sample count across continuous and discrete
   candidates. Do not add new dependencies for the Hawkes simulator unless a later implementation
   note justifies them.

4. Path and jump-specific evaluation

   Evaluate both smooth path metrics and jump-specific metrics. Separate in-distribution Hawkes
   evaluation from optional tail-stress evaluation so a model is not rewarded for overproducing
   black-swan paths in the ordinary test split.

5. Registry update

   Update `trained_models/model_registry.yaml` only if the result is robust across seeds and the
   winning model improves the intended profile without unacceptable regression on the other
   profile. Registry metadata should include missing metrics and caveats rather than presenting a
   single score as universally optimal.

## Selection Hypothesis

The expected outcome is not that one model dominates all metrics. Continuous TC-VAE may win smooth
distributional metrics such as MMD, SWD, and volatility W1 because the Brownian or
stochastic-volatility component remains continuous and low dimensional.

The discrete hidden128 causal conv-transformer may win jump and regime metrics because the token
sequence can represent persistent high-intensity regimes, abrupt transitions, rare states, and
clustered event arrivals. Its strongest evidence would be better jump-count, inter-arrival,
clustering, tail, VaR, ES, transition, and run-length diagnostics without severe degradation of
smooth path fit.

The benchmark should therefore be reported as a two-profile comparison:

- smooth path fidelity, where the continuous baseline may remain strongest;
- jump and regime fidelity, where the hidden128 conv-transformer is the main discrete candidate.

## Implementation Scope For A Later Branch

This planning note deliberately stops before implementation. A later branch should add the
dataset, configs, smoke checks, and evaluation code in small stages. The first implementation
should avoid new dependencies, reuse the existing dataset factory shape conventions, and keep
oracle simulator labels separate from model-visible features.
