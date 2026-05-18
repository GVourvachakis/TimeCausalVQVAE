# Hawkes-Jump Ogata Simulator

## Purpose

The Hawkes-jump benchmark now exposes two simulator backends behind the same
`HawkesJumpDataset` tensor interface:

- `fixed_grid`: the original fast discrete-time approximation.
- `ogata`: a continuous-time marked Hawkes jump simulator using Ogata modified
  thinning, followed by projection onto the model's regular observation grid.

The fixed-grid backend remains useful for quick smoke tests because it is simple,
fast, deterministic, and preserves the original implementation used when the
benchmark was first introduced. It is intentionally retained as
`simulation_scheme: fixed_grid`.

The Ogata backend is the preferred research-quality default for new Hawkes-jump
experiments because it simulates event arrival times in continuous time rather
than approximating self-exciting arrivals on the output grid. That distinction is
important for the benchmark target: jump detection, volatility clustering, tail
shape, and rare-event clustering should reflect the Hawkes dynamics, not a
discretisation artefact.

## Numerical Construction

The Ogata backend uses modified thinning for a univariate marked Hawkes process.
Between accepted events, the intensity follows exact exponential decay towards
the baseline:

```text
lambda(t) = lambda_0 + (lambda_current - lambda_0) exp(-beta Delta t)
```

Accepted events use asymmetric folded-normal marks. Upside marks are sampled as
positive folded normals; ordinary downside and severe downside marks are sampled
as negative folded normals. The absolute value is applied before assigning the
sign, so the downside branches cannot accidentally produce positive marks.

The intensity excitation is mark-dependent:

```text
Q(y) = excitation + mark_excitation * |y|
```

The volatility state is also jump-excited and decays exponentially between
events. During projection to the grid, each interval is partitioned by the exact
event times. On sub-intervals without jumps, volatility is deterministic:

```text
sigma(t) = v_base + vs exp(-kappa_v (t - a))
```

The Brownian variance is therefore integrated exactly:

```text
Var = v_base^2 Delta t
    + 2 v_base vs (1 - alpha) / kappa_v
    + vs^2 (1 - alpha^2) / (2 kappa_v)
```

where `alpha = exp(-kappa_v Delta t)`. This avoids treating a decaying
jump-excited volatility state as constant over the full grid step.

The Hawkes stability gate `excitation / decay < 1` is enforced before
simulation.

## Output Compatibility

Both backends return the same tensor layout:

```text
prices:          [n_sample, n_timestep, 1]
log_returns:     [n_sample, n_timestep, 1]
jump_indicators: [n_sample, n_timestep, 1]
jump_counts:     [n_sample, n_timestep, 1]
jump_sizes:      [n_sample, n_timestep, 1]
intensities:     [n_sample, n_timestep, 1]
volatilities:    [n_sample, n_timestep, 1]
```

The event simulation in the Ogata backend is continuous-time, but the model still
observes the generated path only on the regular grid. Oracle event metadata is
provided for diagnostics and dataset validation; model training remains based on
the configured path tensor.

## Smoke Results

The following smoke commands were run with `n_samples=256`, `n_timesteps=60`, and
`seed=99`.

| Scheme | Total jumps | Mean jumps per path | Paths with jumps | Negative jump fraction | Max intensity | Max volatility | Adjacent jump pairs | Lower-tail VaR q01 | Lower-tail ES q01 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_grid` | 1027 | 4.0117 | 0.9570 | 0.7111 | 16.7161 | 0.5577 | 117 | -0.076687 | -0.105185 |
| `ogata` | 917 | 3.5820 | 0.9375 | 0.7013 | 17.2248 | 0.4643 | 84 | -0.071069 | -0.101811 |

Both runs passed the smoke assertions for positive prices, expected
batch-time-channel shape, finite paths and returns, at least one jump, at least
one negative jump step, and finite intensities.

## Interpretation

The fixed-grid backend draws per-step Poisson arrivals and applies jump and
volatility feedback at grid times. This is adequate for fast integration tests,
but it can distort the timing relationship between jump arrivals, variance
bursts, and Brownian increments inside each step.

The Ogata backend separates the event process from the observation grid. It
first generates exact continuous-time Hawkes event times, then performs a single
O(n_events + n_timestep) projection onto the grid. This gives the benchmark a
cleaner target for rare-event structure while keeping all downstream tensor
contracts unchanged.

## No-Arbitrage Caveat

The generated discounted prices should be treated as scenario data for model
diagnostics. The simulator is not, by itself, an arbitrage-free pricing model,
and this integration does not claim risk-neutral generation. Any pricing use
would require an explicit risk-neutral construction and martingale or
drift-normalisation diagnostics.
