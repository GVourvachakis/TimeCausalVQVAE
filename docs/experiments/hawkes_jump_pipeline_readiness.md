# Hawkes-Jump Pipeline Readiness

## Scope

This readiness pass verifies that the Ogata Hawkes-jump dataset is integrated
with the continuous TC-VAE path, the causal VQ tokenizer path, and the existing
no-future-leakage checks. It does not train full models and does not commit any
generated outputs.

The active Hawkes-jump configs use `simulation_scheme: ogata` for the continuous
BetaCVAE and tokenizer smoke configs. The fixed-grid simulator remains available
through `simulation_scheme: fixed_grid`.

## Pipeline Smoke Results

The continuous dry run completed successfully:

```bash
poetry run tcvae-train \
  --config configs/experiments/hawkes_jump_beta_cvae.yaml \
  --output-dir outputs/hawkes_jump_smoke/continuous \
  --epochs 1 \
  --dry-run \
  --no-wandb
```

Result:

- Dataset resolved as `HawkesJump`.
- Model resolved as `BetaCVAE`.
- Train data shape: `512x60x1`, labels: `512x1`.
- Eval data shape: `512x60x1`, labels: `512x1`.
- Model class: `BetaConditionalVAE`.
- Parameter count: `202989`.
- No training was started.

The standard tokenizer smoke completed one epoch:

```bash
poetry run tcvae-train-tokenizer \
  --config configs/experiments/hawkes_jump_causal_vq_tokenizer.yaml \
  --output-dir outputs/hawkes_jump_smoke/tokenizer_standard \
  --epochs 1 \
  --no-wandb
```

Result:

- Final loss: `0.54092515`.
- Reconstruction loss: `0.53898244`.
- Active codes: `63`.

The hidden128 tokenizer smoke also completed one epoch:

```bash
poetry run tcvae-train-tokenizer \
  --config configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128.yaml \
  --output-dir outputs/hawkes_jump_smoke/tokenizer_hidden128 \
  --epochs 1 \
  --no-wandb
```

Result:

- Final loss: `0.45104744`.
- Reconstruction loss: `0.44773971`.
- Active codes: `64`.

Both tokenizer runs emitted a local CUDA driver warning and then used the
available runtime successfully. The warning does not affect CPU smoke
readiness.

## Dataset No-Leakage Result

A dataset-specific check was added:

```bash
poetry run python scripts/check_hawkes_jump_dataset_no_leakage.py \
  --config configs/experiments/hawkes_jump_causal_vq_tokenizer.yaml
```

Result: `PASS Hawkes-jump dataset no-leakage check`.

The check verified:

- configured visible data is the selected price tensor;
- the pipeline exposes only the configured data tensor and labels;
- oracle fields are present on the direct simulator dataset only, not on the
  wrapped `BaseDataset` returned by `DataPipeline`;
- visible data is not identical to jump indicators, jump counts, jump sizes,
  intensities, or volatilities;
- labels are constant scalar conditions with shape `[n_sample, 1]`;
- train and eval samples differ because the eval split offsets the configured
  seed;
- scalar constant conditions are prefix-safe.

## Visible And Oracle Fields

Model-visible fields:

- `data`: `[n_sample, 60, 1]`, configured as price paths for the current smoke
  configs;
- `labels`: `[n_sample, 1]`, constant non-informative scalar labels.

Oracle-only direct dataset attributes:

- `prices`;
- `log_returns`;
- `jump_indicators`;
- `jump_counts`;
- `jump_sizes`;
- `intensities`;
- `volatilities`;
- `metadata`.

Only `data` and `labels` are passed through the training/evaluation pipeline.
Jump indicators, jump counts, jump sizes, intensities, volatilities, and
metadata remain diagnostic fields.

## Tokenizer And Prior No-Leakage

The source-level causal convolution no-leakage check passed:

```bash
poetry run python scripts/check_causal_conv_no_leakage.py
```

Result:

- Output shape: `(4, 24, 5)`.
- Cutoff: `11`.
- Status: pass.

The fresh standard tokenizer check passed on Hawkes-jump data:

```bash
poetry run python scripts/check_conditional_vq_tokenizer_no_leakage.py \
  --config configs/experiments/hawkes_jump_causal_vq_tokenizer.yaml \
  --batch-size 8 \
  --cutoff 29 \
  --seed 99 \
  --device cpu
```

Result:

- Input shape: `(8, 60, 1)`.
- Conditions shape: `(8, 1)`.
- Indices shape: `(8, 60)`.
- Reconstruction shape: `(8, 60, 1)`.
- `max_prefix_diff=0.00000000e+00`.

The fresh hidden128 tokenizer check also passed with the same tensor shapes and
`max_prefix_diff=0.00000000e+00`.

The trained one-epoch smoke tokenizers were checked as well:

- Standard tokenizer checkpoint:
  `outputs/hawkes_jump_smoke/tokenizer_standard/hawkes_jump_causal_vq_tokenizer_seed0`,
  `max_prefix_diff=0.00000000e+00`.
- Hidden128 tokenizer checkpoint:
  `outputs/hawkes_jump_smoke/tokenizer_hidden128/hawkes_jump_causal_vq_tokenizer_hidden128_seed0`,
  `max_prefix_diff=0.00000000e+00`.

The token-prior no-leakage checks passed at source level for the Hawkes prior
configs:

- `configs/experiments/hawkes_jump_causal_token_prior_additive.yaml`;
- `configs/experiments/hawkes_jump_causal_token_prior_hidden128_conv_transformer.yaml`.

Both checks used synthetic random tokens because this readiness pass did not run
token extraction for Hawkes-jump token datasets. Each check used scalar additive
conditions with shape `(16, 1)` and reported `max_prefix_diff=0.00000000e+00`.

## Remaining Work

Before registry-quality results, run the full non-smoke workflow:

- train continuous and tokenizer candidates across at least two seeds;
- extract Hawkes-jump token datasets from the trained tokenizers;
- run token-prior no-leakage checks against extracted Hawkes token datasets;
- train the additive and hidden128 conv-transformer priors;
- evaluate both smooth path metrics and jump-specific diagnostics;
- update the registry only if results are robust across seeds.

## Readiness Decision

The Ogata Hawkes-jump dataset is ready for the first non-smoke training pass.
The pipeline recognises `hawkes_jump`, continuous and tokenizer smoke paths run
successfully, model-visible tensors exclude oracle jump metadata, scalar
conditions are non-informative and prefix-safe, and existing tokenizer/prior
causality checks pass for the Hawkes configs.
