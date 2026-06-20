# S&P500/VIX Empirical Benchmark

The S&P500/VIX benchmark is the stable public one-dimensional empirical
workflow. It uses a local processed S&P500 price series, a scalar VIX
condition, and 60-step path windows.

## Data Source Convention

The package does not redistribute market data. The legacy processed array is
expected locally at:

```text
data/processed/sp500vix/sp500vix_normalized.npy
```

The array contains two aligned columns:

- column 0: S&P500 price or index level, already normalised by the local
  preprocessing convention;
- column 1: VIX level used as the scalar condition.

Users are responsible for obtaining and storing this local file consistently
with their data licence. The repository and package do not include the raw
prices or processed tensor.

## Tensor Convention

`SP500VIXDataset` constructs sliding windows from the S&P500 column:

- `data`: price windows divided by their first value, with shape
  `[n_sample, 60, 1]` for the public configs;
- `labels`: VIX scalar labels with shape `[n_sample, 1]`;
- condition feature: `vix`.

The public continuous config declares `transform: log` and
`inverse_transform: exp`. Tokenizer and token-prior configs use
`condition_dim: 1` for VIX-conditioned discrete modelling.

## Preprocessing And Splits

The dataset uses a stride-one sliding-window convention. It does not fit
standardisation statistics. The model-visible condition is the VIX value from
the aligned local series at the window start in the legacy convention.

No future generated-window statistic, realised volatility label, or ex-post
tail label is exposed by the standard dataset. Condition-bucket diagnostics may
use VIX post hoc for evaluation, but they do not change the training tensor.

## Configs

Representative configs:

- `configs/experiments/sp500_vix_beta_cvae.yaml`;
- `configs/experiments/sp500_vix_causal_vq_tokenizer.yaml`;
- `configs/experiments/sp500_vix_causal_token_prior_additive.yaml`.

S&P500/VIX is the public default empirical workflow. Registry metadata records
local checkpoint conventions and caveats, not weights.
