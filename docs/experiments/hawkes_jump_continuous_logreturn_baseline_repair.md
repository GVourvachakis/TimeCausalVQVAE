# Hawkes-Jump Continuous Log-Return Baseline Repair

## Status

The matched continuous Hawkes/SVMHJD log-return baseline failure was repaired at
the configuration level. No Ogata simulator parameters, discrete tokenizer
results, token-prior results, model families, or registry entries were changed.

The selected repair is an identity/no-transform continuous config using the
legacy-supported transform name `id`.

## Root Cause

The failed robustness run used:

```yaml
data:
  transform: log
  inverse_transform: exp
  params:
    data_output: log_return
```

This is inappropriate for signed log-return data. The Hawkes dataset wrapper
returns `simulation.log_returns` when `data_output: log_return`, and these values
can be negative. The continuous objective applies `self.transform(x0)` before
encoding. The `log` transform is implemented as `x.log()`, so negative returns
produce NaNs before the first BetaCVAE loss is computed.

The selected-config adapter passes `data.transform` and `data.inverse_transform`
through to the continuous backend. The backend transform registry accepts `id`
or an empty string as the identity transform; it does not accept the literal
string `identity`. Therefore the identity-named Hawkes configs use:

```yaml
data:
  transform: id
  inverse_transform: id
```

## Configs Created

| Config | Seed | Transform | Samples |
|---|---:|---|---:|
| `configs/experiments/hawkes_jump_beta_cvae_logreturn_identity.yaml` | 0 | `id` / `id` | 1024 |
| `configs/experiments/hawkes_jump_beta_cvae_logreturn_identity_seed1.yaml` | 1 | `id` / `id` | 1024 |
| `configs/experiments/hawkes_jump_beta_cvae_logreturn_identity_seed2.yaml` | 2 | `id` / `id` | 1024 |

The simulator parameters are unchanged from the previous log-return configs:
`simulation_scheme: ogata`, `dt: 0.0166666667`, `brownian_volatility: 0.18`,
`baseline_intensity: 3.0`, `excitation: 2.0`, `decay: 12.0`,
`mark_excitation: 20.0`, `negative_jump_probability: 0.7`,
`severe_jump_probability: 0.08`, and `volatility_excitation: true`.

The new configs use `n_samples: 1024` so the existing continuous trainer's fixed
1000-sample evaluation path has enough condition labels. A first 512-sample
attempt trained without NaNs but failed during evaluation because generation
requested 1000 samples while only 512 labels were available.

## Smoke Result

Dry run:

```bash
poetry run tcvae-train \
  --config configs/experiments/hawkes_jump_beta_cvae_logreturn_identity.yaml \
  --output-dir outputs/hawkes_jump_continuous_logreturn_identity_smoke \
  --epochs 1 \
  --dry-run \
  --no-wandb
```

Result: passed. The dry run built a `BetaConditionalVAE` with 1024 train samples,
1024 eval samples, data shape `[1024, 60, 1]`, labels shape `[1024, 1]`, and
202,989 parameters.

One-epoch smoke:

```bash
poetry run tcvae-train \
  --config configs/experiments/hawkes_jump_beta_cvae_logreturn_identity.yaml \
  --output-dir outputs/hawkes_jump_continuous_logreturn_identity_smoke \
  --epochs 1 \
  --no-wandb
```

Result: passed. Training completed without NaN loss, saved a checkpoint, and
wrote:

```text
outputs/hawkes_jump_continuous_logreturn_identity_smoke/
  BetaCVAE_training_2026-05-18_17-33-39/final_model
```

The one-epoch smoke printed total loss `11.35`, reconstruction `11.15`, and
regularisation `4.86` during the training loop. These smoke values only verify
that the baseline can train; they are not benchmark results.

## Evaluator Status

`scripts/evaluate_hawkes_jump_continuous.py` already exists and supports the
required Hawkes-specific evaluation behaviour:

- loads a continuous `final_model` directory and selected config;
- reads `data_output`;
- converts generated and real log returns to normalised price paths when
  `data_output: log_return`;
- fits jump-detection thresholds on real Ogata evaluation paths;
- writes `evaluation_summary.json`, `evaluation_summary.md`, and
  `evaluation_batch.pt`.

Smoke evaluation command:

```bash
poetry run python scripts/evaluate_hawkes_jump_continuous.py \
  --config configs/experiments/hawkes_jump_beta_cvae_logreturn_identity.yaml \
  --model-dir outputs/hawkes_jump_continuous_logreturn_identity_smoke/BetaCVAE_training_2026-05-18_17-33-39/final_model \
  --output-dir outputs/hawkes_jump_continuous_logreturn_identity_smoke/evaluation \
  --n-sample 128 \
  --seed 99
```

Result: passed. The evaluator reported `data_output: log_return`,
`generated_prices_shape: [128, 60, 1]`, MMD `0.35188922`, SWD
`1080387.25000000`, and detected jump-count W1 `29.62500000`. The SWD and jump
metrics are expected to be poor after a one-epoch smoke and should not be
interpreted as final model quality.

## Standardisation Decision

No standardised transform family was added. Identity training no longer NaNs and
the evaluator can consume the resulting checkpoint. Standardisation may still
be useful for model quality, but it is not required for the immediate baseline
repair.

## Next Non-Smoke Command

Run the repaired three-seed continuous baseline with the same high-level budget
used for the discrete robustness comparison:

```bash
for config in \
  configs/experiments/hawkes_jump_beta_cvae_logreturn_identity.yaml \
  configs/experiments/hawkes_jump_beta_cvae_logreturn_identity_seed1.yaml \
  configs/experiments/hawkes_jump_beta_cvae_logreturn_identity_seed2.yaml
do
  poetry run tcvae-train \
    --config "$config" \
    --output-dir outputs/hawkes_jump_continuous_logreturn_identity \
    --epochs 50 \
    --no-wandb
done
```

Then evaluate each final model with `scripts/evaluate_hawkes_jump_continuous.py`
using `n_sample=1024`, seed-matched evaluation, and log-return-to-price
conversion enabled by the config.
