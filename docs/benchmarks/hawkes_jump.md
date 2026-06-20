# Hawkes/SVMHJD Synthetic Benchmark

The Hawkes/SVMHJD benchmark provides synthetic one-dimensional market paths
with clustered, asymmetric jumps. It is intended for rare-event dataset smoke
tests, leakage checks, and discrete-latent benchmark studies where jump timing
and tail behaviour matter.

## Formulation

The fixed-grid approximation uses log returns

```text
r_t = drift dt + sigma_t sqrt(dt) z_t + Y_t,
S_t = S_{t-1} exp(r_t),
```

where `z_t` is standard normal noise and `Y_t` is the aggregate marked jump
size in the grid cell. The jump count is sampled from a clipped Poisson rate
based on the current Hawkes intensity:

```text
N_t ~ Poisson(lambda_t dt),
Y_t = sum_{j=1}^{N_t} mark_{t,j}.
```

After observing the grid cell, intensity and optional jump-excited volatility
are updated by

```text
lambda_{t+1}
  = lambda_0 + (lambda_t - lambda_0) exp(-decay dt)
    + excitation N_t + mark_excitation abs(Y_t),

vol_state_{t+1}
  = vol_state_t exp(-volatility_decay dt)
    + volatility_excitation_scale abs(Y_t).
```

The Ogata backend simulates continuous-time event arrivals by thinning with
the same baseline, excitation, decay, mark-excitation, and mark distribution,
then projects the resulting jump-diffusion path onto the regular observation
grid. This is the preferred research-quality event-timing backend.

Marks are asymmetric: negative jumps are more likely than positive jumps, with
an additional severe-negative component. The simulator is synthetic stress-test
data, not a calibrated no-arbitrage pricing model.

## Tensor Convention

`HawkesJumpDataset` exposes:

- `data`: either prices or log returns, selected by `data_output`, with shape
  `[n_sample, 60, 1]` in the public configs;
- `labels`: constant-one scalar labels with shape `[n_sample, 1]`;
- `prices` and `log_returns`: full simulated path tensors;
- `jump_indicators`, `jump_counts`, `jump_sizes`, `intensities`, and
  `volatilities`: oracle simulator metadata tensors with shape
  `[n_sample, 60, 1]`;
- `metadata`: aggregate simulator diagnostics such as total jumps and jump
  fractions.

The supported simulator backends are:

- `simulation_scheme: fixed_grid`, a fast discrete-time Hawkes approximation;
- `simulation_scheme: ogata`, continuous-time marked Hawkes arrivals observed
  on the fixed model grid.

## Preprocessing And No-Leakage Convention

No empirical data is loaded and no train-only standardisation is applied by the
dataset. Eval paths use a shifted seed in `DataPipeline` when a seed is
configured, so held-out synthetic samples are independent of train samples.

Oracle jump metadata is diagnostic-only. Jump counts, jump indicators, jump
sizes, intensities, and volatility states should be used for smoke summaries,
leakage checks, plots, and post-hoc jump diagnostics, not as model-visible
conditions or targets unless a later benchmark explicitly changes that
contract.

## Smoke Checks

```bash
poetry run python scripts/smoke_hawkes_jump_dataset.py \
  --simulation-scheme ogata \
  --n-samples 128 \
  --n-timesteps 60 \
  --seed 99 \
  --output-dir outputs/hawkes_jump_public_smoke

poetry run python scripts/check_hawkes_jump_dataset_no_leakage.py \
  --config configs/experiments/hawkes_jump_causal_vq_tokenizer.yaml
```

## Public Status

Hawkes/SVMHJD has an optional registry entry with `status:
research_candidate` and `public_default: false`. The selected discrete research
candidate under the balanced/smooth profile remains the hidden128 log-return
cb64 tokenizer plus causal conv-transformer k3 prior. The hidden128 log-return
cb64 tokenizer plus additive AR prior remains the required jump-profile
ablation.

No weights, token tensors, generated paths, W&B artefacts, or local result grids
are committed.
