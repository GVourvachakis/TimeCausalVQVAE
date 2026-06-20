# Hawkes/SVMHJD Synthetic Benchmark

The Hawkes/SVMHJD benchmark provides synthetic one-dimensional market paths with clustered,
asymmetric jumps. It is intended for rare-event dataset smoke tests, leakage checks, and future
discrete-latent benchmark studies where jump timing and tail behaviour matter.

Two simulator backends are available through `HawkesJumpDataset`:

- `simulation_scheme: fixed_grid` uses a fast discrete-time Hawkes approximation. It is useful for
  quick smoke tests and local debugging.
- `simulation_scheme: ogata` simulates continuous-time marked Hawkes arrivals by Ogata thinning and
  then observes the jump-diffusion path on the requested fixed grid.

The Ogata backend is preferred for research-quality event timing because jump arrivals are sampled
in continuous time before projection to the observation grid. This avoids making the Hawkes process
depend on the arbitrary grid step used by the model input tensor, while retaining the same
batch-time-channel tensor interface.

The dataset exposes the following tensors:

- `data`: configured model-visible tensor, either normalised prices or log returns, with shape
  `[n_sample, n_timestep, 1]`;
- `labels`: constant scalar labels with shape `[n_sample, 1]`;
- `prices` and `log_returns`: full simulated path tensors with shape `[n_sample, n_timestep, 1]`;
- `jump_indicators`, `jump_counts`, `jump_sizes`, `intensities`, and `volatilities`: oracle
  simulator metadata tensors with shape `[n_sample, n_timestep, 1]`;
- `metadata`: aggregate simulator diagnostics such as total jumps and jump fractions.

Oracle metadata is diagnostic-only. It should be used for smoke summaries, leakage checks, plots,
and post-hoc jump diagnostics, not as a model-visible condition or target unless a later benchmark
explicitly changes that contract.

This simulator is not a no-arbitrage pricing model. The generated paths are synthetic stress-test
data for generative modelling diagnostics, not calibrated tradable dynamics or derivative-pricing
inputs.

Smoke checks:

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

Hawkes/SVMHJD has an optional registry entry with `status: research_candidate` and
`public_default: false`. The selected discrete research candidate under the balanced/smooth profile
remains the hidden128 log-return cb64 tokenizer + causal conv-transformer k3 prior. The hidden128
log-return cb64 tokenizer + additive AR prior remains the required jump-profile ablation. The tiny
conv-transformer is an optional efficiency candidate: it improves mean jump-count and inter-arrival
distances in the compact-prior follow-up, but it loses the balanced smooth profile and is not the
registered selected model.
