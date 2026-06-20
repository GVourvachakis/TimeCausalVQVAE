# S&P500 50-Stock Panel Benchmark

The S&P500 50-stock panel is an experimental local-only empirical benchmark for
50-dimensional daily equity returns. It follows the 60-step S&P500/VIX
conditioning convention while extending the model-visible tensor from one path
to a static sector-stratified 50-stock cross-section.

The benchmark is not packaged with downloaded data. Raw and processed market
data must remain local under `data/raw`, `data/processed`, or `outputs`.

## Data Access

The planned downloader uses `yfinance`, which is optional and isolated in the
`data` dependency group:

```bash
poetry install --with data
```

`yfinance` is unaffiliated with Yahoo. Yahoo-backed data is intended here for
local research and educational use, subject to Yahoo's terms. Downloaded price
data must not be redistributed or committed.

## Tensor Convention

The processed dataset exposes:

- `data`: 50D adjusted-close log returns with shape `[n_window, 60, 50]`;
- `labels`: prefix-safe conditions with shape `[n_window, 2]`;
- condition names: `spy_log_return_start` and `log_vix_level_start`;
- metadata: ticker order, sector labels, date range, split boundaries,
  condition names, and missing-data handling.

The v2 empirical convention uses separate train and eval counts, train-only
per-asset standardisation, and train-fitted PCA for factor-projected discrete
experiments. Eval data reuses the train standardisation statistics and train
PCA basis. No oracle synthetic factor loading matrix exists for the empirical
panel.

## Universe

The first public universe is a static, sector-stratified 50-stock list recorded
by the downloader metadata under `universe_id:
sp500_50_liquid_sector_v0`. It is deliberately not scraped dynamically in the
first pass. If a ticker materially shortens the joined panel, update the
versioned universe rather than silently substituting another asset.

The downloader aligns the 50 stocks plus `SPY` and `^VIX` by date, uses an
inner join for the first pass, and computes log returns after alignment. It
does not forward-fill asset prices inside the benchmark panel.

## Local Downloader

Example local-only download and processing command:

```bash
poetry run python scripts/download_sp500_50_panel.py \
  --start 2020-01-01 \
  --end 2021-12-31 \
  --output-root data \
  --include-sector-etfs
```

This writes local raw and processed files under `data/raw/sp500_50_panel/` and
`data/processed/sp500_50_panel/`. These paths are ignored by policy and must
not be committed.

## Smoke Commands

After local processing, run the dataset smoke:

```bash
poetry run python scripts/smoke_sp500_50_panel_dataset.py \
  --train-n-samples 355 \
  --eval-n-samples 89 \
  --standardize-returns \
  --base-data-dir data/processed \
  --output-dir outputs/sp500_50_panel_public_smoke
```

The empirical configs under `configs/experiments/sp500_50_panel*.yaml` are
smoke and experimental entry points. The factor-PCA RVQ q2 configs are included
only as experimental infrastructure and are not registry-selected.

## Diagnostics

Use the same cross-sectional diagnostics as the synthetic benchmark:
covariance and correlation errors, eigenspectrum distance, sector-block
correlations, equal-weight and random-portfolio VaR/ES, marginal asset
statistics, and condition-bucket summaries where available.

## Public Status

The S&P500 50-stock panel is an experimental benchmark, not a public default.
S&P500/VIX remains the stable public workflow. No empirical 50D model is
selected in `trained_models/model_registry.yaml`, and no downloaded data,
processed tensors, checkpoints, generated samples, or non-smoke result logs are
committed.
