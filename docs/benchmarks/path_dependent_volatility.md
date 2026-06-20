# Path-Dependent Volatility Synthetic Benchmark

The path-dependent volatility benchmark is a one-dimensional conditional
synthetic baseline. It uses a volatility state that depends on filtered past
returns, then exposes a prefix volatility feature as the model condition.

## Formulation

The simulator keeps two return filters and two squared-return filters. At each
fine time step,

```text
r1_t = (1 - theta1) r10_t + theta1 r11_t,
r2_t = (1 - theta2) r20_t + theta2 r21_t,
sigma_t = beta0 + beta1 r1_t + beta2 sqrt(r2_t),
Delta X_{t+1} = mu dt + sigma_t Delta W_t.
```

The internal filters update as

```text
r10_{t+1} = r10_t + lambda10 (sigma_t Delta W_t - r10_t dt),
r11_{t+1} = r11_t + lambda11 (sigma_t Delta W_t - r11_t dt),
r20_{t+1} = r20_t + lambda20 (sigma_t^2 - r20_t) dt,
r21_{t+1} = r21_t + lambda21 (sigma_t^2 - r21_t) dt.
```

Prices are reconstructed by cumulative exponentiation of the simulated log
returns.

## Tensor Convention

Configs using `dataset: path_dependent_volatility` expose:

- `data`: price windows normalised by the first price in the window, with shape
  `[n_sample, 60, 1]` for the public configs;
- `labels`: one prefix volatility feature with shape `[n_sample, 1]`;
- condition feature: `r2_volatility_feature`.

The continuous configs declare `transform: log` and `inverse_transform: exp`.
The dataset stores normalised prices, while model code may transform them to log
space.

## Preprocessing And No-Leakage Convention

`PDVPriceFeatureDataset` first simulates one long path, builds sliding price
windows, discards the first `100` windows, and normalises each retained window by
its first price. The label for a generated window starting at index `s` is
aligned to the feature at `s - 1`, so the condition is available before the
window starts.

No train-only standardisation is applied by the dataset. The volatility filters
and simulated latent states are process internals; only the scalar prefix
feature is model-visible in the standard benchmark.

## Configs

Representative configs:

- `configs/experiments/pdv_info_cvae.yaml`;
- `configs/experiments/pdv_causal_vq_tokenizer_hidden128.yaml`;
- `configs/experiments/pdv_causal_token_prior_additive_seed1.yaml`.

The benchmark is stable synthetic infrastructure. It does not ship weights,
generated samples, or local outputs.
