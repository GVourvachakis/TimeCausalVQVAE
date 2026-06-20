# Black-Scholes Synthetic Benchmark

The Black-Scholes benchmark is the smallest one-dimensional synthetic path
benchmark. It is used for smoke tests, registry metadata checks, and baseline
continuous and discrete generation workflows.

## Formulation

The simulator uses the geometric Brownian motion model

```text
dS_t = mu S_t dt + sigma S_t dW_t,
S_0 = 1.
```

The exact discrete path used by `simulate_BS` is

```text
S_t = exp(sigma W_t + (mu - 0.5 sigma^2) t),
```

where Brownian increments are sampled on the configured grid. The default
dataset parameters are `mu=0.1`, `sigma=0.2`, and `dt=1/12`, unless overridden
by config `data_params`.

## Tensor Convention

Through `DataPipeline`, configs using `dataset: black_scholes` expose:

- `data`: normalised price paths with shape `[n_sample, 60, 1]` for the public
  configs;
- `labels`: a constant-one scalar condition with shape `[n_sample, 1]` for
  continuous conditional models;
- tokenizer configs may set `condition_dim: 0` and ignore the constant label.

The continuous configs declare `transform: log` and `inverse_transform: exp`, so
model code may train in log space while the dataset itself stores positive
normalised prices.

## Preprocessing And Splits

No empirical data is loaded. Train and eval datasets are simulated locally from
the configured synthetic process. No train-only standardisation is applied in
the dataset. The only model-visible condition is the constant scalar used by
legacy conditional model wiring.

There is no oracle metadata beyond simulator parameters. Any future diagnostic
metadata should remain post-hoc unless a benchmark explicitly adds a
prefix-safe condition.

## Configs

Representative configs:

- `configs/experiments/black_scholes_beta_cvae.yaml`;
- `configs/experiments/black_scholes_causal_vq_tokenizer.yaml`;
- `configs/experiments/black_scholes_causal_token_prior_additive.yaml`.

The benchmark is stable synthetic infrastructure. It does not ship weights,
generated samples, or local outputs.
