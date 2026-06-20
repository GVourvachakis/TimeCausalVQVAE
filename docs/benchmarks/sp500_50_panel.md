# S&P500 50-Stock Panel Benchmark

The S&P500 50-stock panel is an experimental local-only empirical benchmark for
50-dimensional daily equity returns. It follows the 60-step market-condition
convention while extending the model-visible tensor from one path to a static
sector-stratified 50-stock cross-section.

The benchmark is not packaged with downloaded data. Raw and processed market
data must remain local under `data/raw`, `data/processed`, or `outputs`.

## Data Source Convention

The downloader uses optional `yfinance` access to Yahoo-backed adjusted-close
data:

```bash
poetry install --with data
```

`yfinance` is unaffiliated with Yahoo. Yahoo-backed data is intended here for
local research and educational use, subject to Yahoo's terms. Downloaded price
data must not be redistributed or committed.

The fixed universe is recorded as `sp500_50_liquid_sector_v0`. It is a static,
sector-stratified 50-stock list plus `SPY` and `^VIX`; it is deliberately not
scraped dynamically during processing.

## Return And Condition Construction

The processor aligns all required adjusted-close columns by date, drops any row
with missing prices, and computes log returns after alignment:

```text
r_{t,a} = log P_{t,a} - log P_{t-1,a}.
```

For a generated stock-return window beginning at index `s`, the raw data tensor
contains

```text
stock_returns.iloc[s : s + 60].
```

The legacy v2 condition mode, `spy_vix_level`, uses two start-of-window
features:

- `spy_log_return_start`;
- `log_vix_level_start`.

The v3 condition mode, `v3_prefix_market`, uses six prefix-safe features:

- `spy_log_return_start`;
- `log_vix_level_start`;
- `previous_window_spy_realized_volatility`;
- `previous_window_equal_weight_realized_volatility`;
- `previous_window_average_correlation`;
- `previous_window_equal_weight_log_return`.

The previous-window statistics are computed from
`stock_returns.iloc[s - 60 : s]`, so the previous window ends strictly before
the generated 60-day stock-return window starts.

## Tensor Convention

The processed dataset exposes:

- `data`: 50D adjusted-close log-return windows with shape
  `[n_window, 60, 50]`;
- `labels`: train-standardised condition vectors with shape `[n_window, 2]`
  for v2 `spy_vix_level`, or `[n_window, 6]` for v3 `v3_prefix_market`;
- `raw_data.pt`: raw log returns;
- `standardized_data.pt`: train-standardised returns written by the
  downloader;
- metadata: ticker order, sector labels, date range, split boundaries,
  condition names, lag conventions, missing-data handling, and data-use caveats.

The dataset loader can expose raw returns or train-standardised returns. When
`standardize_returns: true`, train statistics are fitted only on train windows
and reused for eval. If `projection_mode: train_pca` is enabled, the train PCA
basis is fitted on train model-visible returns and reused for eval.

## Train/Eval Split

The processed metadata records the leading train split and trailing eval split.
The v3 local comparison uses:

- `train_n_samples=355`;
- `eval_n_samples=89`;
- `n_timestep=60`;
- train-only return standardisation;
- train-only label standardisation;
- `condition_mode: v3_prefix_market`.

No future generated-window realised volatility, covariance, correlation,
drawdown, or tail label is model-visible. Sector labels are metadata for
diagnostics and universe auditing, not conditioning.

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
