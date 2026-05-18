# Hawkes-Jump Dataset Visual Diagnostics

## Scope

This note records the first visual diagnostics for the synthetic Hawkes-jump
benchmark after adding the Ogata simulator backend. No model training was run.
The generated figures and JSON summaries live under `outputs/hawkes_jump_plots/`
and are not intended to be committed.

The primary smoke command used the research-quality Ogata backend with
`n_samples=512`, `n_timesteps=60`, and `seed=99`:

```bash
poetry run python scripts/plot_hawkes_jump_dataset.py \
  --simulation-scheme ogata \
  --n-samples 512 \
  --n-timesteps 60 \
  --seed 99 \
  --output-dir outputs/hawkes_jump_plots/ogata
```

## Figure Inventory

The plotting script writes the following files for a single simulator scheme:

- `sample_price_paths.png`: sample normalised price trajectories.
- `sample_log_return_paths.png`: sample realised log-return paths.
- `jump_indicator_raster.png`: path-by-time binary jump raster.
- `intensity_trajectories.png`: Hawkes intensity trajectories.
- `volatility_trajectories.png`: jump-excited volatility trajectories.
- `jump_count_histogram.png`: per-path jump-count distribution.
- `inter_arrival_histogram.png`: within-path inter-arrival gaps on the grid.
- `jump_size_distribution.png`: aggregate signed jump-size distribution.
- `return_tail_histogram.png`: one-step return histogram with tail quantiles.
- `var_es_summary.png`: lower-tail VaR and ES bars.

The same run also writes `summary.json` and `summary.md`.

## Key Ogata Statistics

The Ogata run produced tensors with price and log-return shape `[512, 60, 1]`.
All basic checks passed: prices were positive, paths and returns were finite,
the shape was batch-time-channel, at least one jump occurred, at least one
negative jump step occurred, and intensity and volatility tensors were finite.

| Statistic | Value |
| --- | ---: |
| Total jumps | 1861 |
| Mean jumps per path | 3.6348 |
| Paths with jumps fraction | 0.9453 |
| Negative jump fraction | 0.6993 |
| Branching ratio proxy | 0.1667 |
| Max intensity | 17.2248 |
| Max volatility | 0.4643 |
| Count over-dispersion | 1.5329 |
| Adjacent jump pairs | 175 |
| Paths with adjacent jumps fraction | 0.2598 |
| Return q001 | -0.147284 |
| Return q999 | 0.079048 |
| Lower-tail VaR q01 | -0.070816 |
| Lower-tail ES q01 | -0.101872 |

These numbers are consistent with a rare-event dataset that is not dominated by
jumps at every time step, but has frequent enough event clusters and downside
asymmetry to exercise jump-specific diagnostics.

## Fixed-Grid Versus Ogata

The comparison command used the same high-level parameters:

```bash
poetry run python scripts/plot_hawkes_jump_dataset.py \
  --compare-schemes \
  --n-samples 512 \
  --n-timesteps 60 \
  --seed 99 \
  --output-dir outputs/hawkes_jump_plots/comparison
```

| Scheme | Runtime | Mean jumps/path | Paths with jumps | Count over-dispersion | VaR q01 | ES q01 | Max drawdown mean | Max drawdown q99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_grid` | 0.1354 | 3.8965 | 0.9492 | 1.7471 | -0.071260 | -0.101269 | 0.226479 | 0.504623 |
| `ogata` | 0.0843 | 3.6348 | 0.9453 | 1.5329 | -0.070816 | -0.101872 | 0.223963 | 0.520748 |

The two backends are close on first-order tail risk and path drawdown summaries,
while the fixed-grid backend produces slightly more jumps and higher count
over-dispersion in this seed. The Ogata backend remains the preferred research
backend because event times are simulated in continuous time and then projected
onto the output grid.

## Training Readiness

The dataset is ready for the first small training run. The visual and statistical
checks support the intended benchmark profile:

- positive, finite one-dimensional paths with shape `[n_sample, 60, 1]`;
- downside-asymmetric marks;
- non-trivial but not overwhelming jump incidence;
- observable event clustering and count over-dispersion;
- heavy lower-tail behaviour visible through VaR, ES, and return quantiles;
- bounded Hawkes intensity and bounded jump-excited volatility.

The outputs should still be treated as scenario data rather than an
arbitrage-free pricing model. Registry updates should wait until results are
robust across seeds and both smooth path metrics and jump-specific diagnostics
are reported.

## Recommended First Training Parameters

Use the existing small smoke configs as the first pass, with the Ogata backend:

- `simulation_scheme: ogata`
- `n_samples: 512`
- `n_timesteps: 60`
- `seed: 0` for config-driven training, with additional seeds for robustness;
- `data_output: price`;
- `dt: 0.0166666667`;
- `brownian_volatility: 0.18`;
- `baseline_intensity: 3.0`;
- `excitation: 2.0`;
- `decay: 12.0`;
- `mark_excitation: 20.0`;
- `negative_jump_probability: 0.7`;
- `severe_jump_probability: 0.08`;
- `volatility_excitation: true`.

For model order, start with the continuous BetaCVAE smoke config and the
standard VQ tokenizer smoke config, then move to the hidden128 tokenizer and the
hidden128 causal conv-transformer prior only after the dataset smoke is stable
across at least two seeds.
