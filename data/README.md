# Data directory

Do not commit large raw market data, generated synthetic datasets, or checkpoint artifacts.

Expected layout:

```text
data/
├── raw/
├── processed/
└── external/
```

For the S&P500/VIX experiment, place the normalized array at:

```text
data/processed/sp500vix/sp500vix_normalized.npy
```

This file is local input data and is ignored by Git. If the upstream
`sp500vix/` directory was copied wholesale, make sure the array is not nested one
level deeper as `data/processed/sp500vix/sp500vix/sp500vix_normalized.npy`.
Verification scripts use `data/processed` as the local `base_data_dir` for this
dataset.
