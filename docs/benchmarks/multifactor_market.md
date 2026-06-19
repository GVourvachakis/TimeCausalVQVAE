# Synthetic Multifactor Market Benchmark

The multifactor market benchmark is an experimental 50-dimensional synthetic
market panel for cross-sectional generative modelling. It extends the public
one-dimensional path benchmarks with controlled factor, sector, covariance, and
portfolio diagnostics while keeping oracle simulator metadata out of the model
inputs.

The default tensor convention is:

- `data`: log-return paths with shape `[n_sample, 60, 50]`;
- `labels`: prefix-safe conditions, initially a constant scalar with shape
  `[n_sample, 1]`;
- `metadata`: simulator loadings, sector labels, factor-volatility paths,
  covariance summaries, and optional jump indicators for diagnostics only.

The simulator supports:

- `n_assets=50`, `n_factors=5`, and `n_sectors=5` by default;
- a low-rank factor loading matrix with sector structure;
- autoregressive stochastic factor volatility;
- idiosyncratic asset noise;
- deterministic `structure_seed` and `path_seed` separation;
- train-only per-asset return standardisation;
- optional common and sector jumps for synthetic stress tests.

Oracle metadata is not model-visible unless a later experiment explicitly adds
a prefix-safe condition. This means loadings, realised covariance summaries,
and jump indicators are valid for post-hoc diagnostics, not for encoder,
tokenizer, or prior conditioning.

## Factor-Projected View

`multifactor_market_factor_projected` exposes low-dimensional factor
coordinates with shape `[n_sample, 60, 5]`. The public empirical-compatible
projection mode is `train_pca`, which fits the basis on the train split and
reuses it for eval data. `oracle_loadings` exists only for synthetic diagnostic
ablations and must not be used as an empirical-data convention.

Factor coordinates can be inverse-projected to 50D returns for raw-scale
diagnostics. Generated or reconstructed factor paths should be evaluated after
inverse projection when the question concerns asset covariance, sector
correlation, or portfolio risk.

## Diagnostics

The cross-sectional diagnostics include:

- covariance and correlation matrices;
- covariance and correlation relative Frobenius errors;
- correlation eigenspectrum distance;
- sector-block correlation summaries;
- equal-weight and random-portfolio volatility, VaR, and ES;
- factor-loading or factor-subspace comparisons when valid loadings exist;
- generated jump and tail diagnostics for common/sector jump stress tests.

Unconditional generated jump samples are evaluated using detected jump and tail
windows in the generated paths. Reference oracle jump masks remain
`reference_oracle_jump_diagnostics` for held-out simulator paths only.

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

The corresponding configs under `configs/experiments/` are smoke and
experimental entry points. RVQ q2 configs are included to exercise the
experimental multi-code tokenizer and prior infrastructure, not to declare a
selected multidimensional generator.

## Public Status

This benchmark is public experimental infrastructure. It is suitable for
dataset checks, shape checks, no-leakage checks, and local research runs. No
multidimensional model is selected in `trained_models/model_registry.yaml`, and
no checkpoints, token tensors, generated samples, W&B artefacts, or result grids
are committed.
