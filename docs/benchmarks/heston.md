# Heston Synthetic Benchmark

The Heston benchmark is a one-dimensional stochastic-volatility synthetic
baseline. It tests whether continuous and discrete models can represent price
paths whose volatility is latent, persistent, and negatively correlated with
price shocks.

## Formulation

The intended continuous-time model is

```text
dS_t = r S_t dt + sqrt(v_t) S_t dW^S_t,
dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW^v_t,
corr(dW^S_t, dW^v_t) = rho.
```

The implementation uses an Euler scheme for the variance channel and applies an
absolute-value positivity guard:

```text
S_{t+1} = S_t exp((r - 0.5 v_t) dt + sqrt(v_t) Delta W^S_t),
v_{t+1} = abs(v_t + kappa (theta - v_t) dt + xi sqrt(v_t) Delta W^v_t).
```

The defaults are `r=0.02`, `kappa=1`, `theta=0.2`, `v_0=0.2`,
`rho=-0.9`, `xi=0.5`, and `dt=1/12`.

## Tensor Convention

Through `DataPipeline`, configs using `dataset: heston` expose:

- `data`: the price channel only, with shape `[n_sample, 60, 1]` for the public
  configs;
- `labels`: a constant-one scalar condition with shape `[n_sample, 1]`;
- `paths`: available on the raw dataset object as `[price, variance]`, but the
  variance channel is not model-visible in the standard benchmark.

The continuous configs declare `transform: log` and `inverse_transform: exp`.
The dataset therefore stores positive prices, while the model workflow may use
log-transformed values internally.

## Preprocessing And Splits

No empirical data is loaded. Train and eval datasets are generated locally from
the synthetic process. The benchmark does not apply train-only
standardisation, and latent variance is diagnostic-only unless a later config
explicitly makes it prefix-safe.

## Configs

Representative configs:

- `configs/experiments/heston_info_cvae.yaml`;
- `configs/experiments/heston_causal_vq_tokenizer.yaml`;
- `configs/experiments/heston_causal_token_prior_additive.yaml`.

The benchmark is stable synthetic infrastructure. It does not ship weights,
generated samples, or local outputs.
