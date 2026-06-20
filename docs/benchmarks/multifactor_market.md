# Synthetic Multifactor Market Benchmark

The multifactor market benchmark is an experimental 50-dimensional synthetic
market panel for cross-sectional generative modelling. It extends the public
one-dimensional path benchmarks with controlled factor, sector, covariance, and
portfolio diagnostics while keeping oracle simulator metadata out of model
inputs.

## Formulation

The no-jump return model is

```text
r_t = B f_t + epsilon_t,
f_t = sigma^f_t z^f_t,
epsilon_t = sigma^epsilon z^epsilon_t.
```

Here `r_t` is a 50-dimensional log-return vector, `B` is a sector-structured
loading matrix, `f_t` contains five latent factor returns, and `epsilon_t` is
asset-specific idiosyncratic noise. Factor volatilities are persistent and
regime-dependent:

```text
log sigma^f_{t+1}
  = log_vol_ar log sigma^f_t + regime_term_t + innovation_t.
```

The optional jump stress augments the return equation as

```text
r_t = B f_t + epsilon_t + J_t,
J_t = J^common_t + J^sector_t.
```

The implemented stress profile supports common and sector jumps. The first
jump profile deliberately keeps `idiosyncratic_jumps=false`.

## Tensor Convention

The standard dataset exposes:

- `data`: 50D log-return paths with shape `[n_sample, 60, 50]`;
- `labels`: prefix-safe conditions with shape `[n_sample, 1]` under
  `condition_mode: constant`;
- `metadata`: simulator loadings, sector labels, factor-volatility paths,
  covariance summaries, and optional jump indicators for diagnostics only.

The default synthetic structure is `n_assets=50`, `n_factors=5`, and
`n_sectors=5`. `structure_seed` controls persistent market structure such as
loadings and sector volatilities. `path_seed` controls realised regimes,
factor shocks, idiosyncratic shocks, and jump events.

## Preprocessing And Splits

The research configs use train-only per-asset return standardisation when
`standardize_returns: true`. `DataPipeline` fits mean and standard deviation on
the train split, stores the statistics, and reuses them for eval. The raw
returns remain available on the dataset for inverse transformation and
raw-scale diagnostics.

Eval paths use `path_seed + 1` when `path_seed` is configured. This keeps the
same market structure while drawing independent realised train and eval paths.

Oracle metadata is not model-visible unless a later experiment explicitly adds
a prefix-safe condition. Loadings, sector labels, realised covariance summaries,
jump indicators, and generated jump masks are valid for post-hoc diagnostics,
not for encoder, tokenizer, or prior conditioning.

## Factor-Projected And Sector-Projected Views

`multifactor_market_factor_projected` exposes low-dimensional factor
coordinates, usually `[n_sample, 60, 5]`. The empirical-compatible projection
mode is `train_pca`, which fits the basis on train data and reuses it for eval.
`oracle_loadings` exists only for synthetic diagnostic ablations.

`multifactor_market_sector_projected` exposes concatenated global and
sector-residual coordinates for research-only multi-stream tokenizer work. The
view records group boundaries and inverse-transform metadata so decoded
coordinates can reconstruct raw 50D log returns. It is not a selected public
model path.

Generated or reconstructed factor paths should be evaluated after inverse
projection when the question concerns asset covariance, sector correlation, or
portfolio risk.

## Diagnostics

The cross-sectional diagnostics include:

- covariance and correlation matrices;
- covariance and correlation relative Frobenius errors;
- correlation eigenspectrum distance;
- sector-block correlation summaries;
- equal-weight and random-portfolio volatility, VaR, and ES;
- factor-loading or factor-subspace comparisons when valid loadings exist;
- generated jump and tail diagnostics for common/sector jump stress tests.

Reference oracle jump masks remain held-out simulator diagnostics only.

## Smoke Commands

Run a no-jump dataset smoke:

```bash
poetry run python scripts/smoke_multifactor_market_dataset.py \
  --n-samples 256 \
  --n-assets 50 \
  --n-factors 5 \
  --n-timesteps 60 \
  --seed 99 \
  --standardize-returns \
  --output-dir outputs/multifactor_market_public_smoke
```

Run the common/sector jump data smoke:

```bash
poetry run python scripts/smoke_multifactor_market_dataset.py \
  --n-samples 256 \
  --n-assets 50 \
  --n-factors 5 \
  --n-timesteps 60 \
  --seed 99 \
  --with-jumps \
  --standardize-returns \
  --output-dir outputs/multifactor_market_jump_public_smoke
```

## Public Status

This benchmark is public experimental infrastructure. It is suitable for
dataset checks, shape checks, no-leakage checks, and local research runs. No
multidimensional model is selected in `trained_models/model_registry.yaml`, and
no checkpoints, token tensors, generated samples, W&B artefacts, or result grids
are committed.
