# S&P500/VIX Continuous BetaCVAE Baseline Reproduction

## Purpose

This note records the local reproduction of the continuous S&P500/VIX BetaCVAE
baseline needed for paper-style comparison against the discrete token-prior
variants. No model architecture, signature feature code, tokenizer code, or
discrete prior code was modified.

## Inputs

The required local data file was present:

```text
data/processed/sp500vix/sp500vix_normalized.npy
```

The run used:

- config: `configs/experiments/sp500_vix_beta_cvae.yaml`;
- data root: `data/processed`;
- model family: BetaCVAE with RealNVP prior;
- device: CPU.

## W&B Live Attempt

Requested live W&B command:

```bash
env -u WANDB_MODE WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread poetry run tcvae-train \
  --config configs/experiments/sp500_vix_beta_cvae.yaml \
  --output-dir outputs/sp500_vix_continuous/beta_cvae \
  --base-data-dir data/processed \
  --wandb \
  --wandb-project time-causal-continuous-baseline \
  --wandb-entity tc_vae \
  --wandb-run-name sp500_vix_beta_cvae_continuous_seed0
```

W&B initialisation succeeded and produced a live run:

```text
https://wandb.ai/tc_vae/time-causal-continuous-baseline/runs/zqf9vu0q
```

The live run did not complete. It stopped at epoch 27 with a Tcl/Tk threading
failure after earlier Matplotlib/Tk warnings:

```text
Tcl_AsyncDelete: async handler deleted by the wrong thread
```

The partial run directory was:

```text
outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-09
```

That partial run contains only an early checkpoint and is not used as the
continuous baseline.

## Completed Reproduction

The complete reproduction used the no-W&B fallback:

```bash
poetry run tcvae-train \
  --config configs/experiments/sp500_vix_beta_cvae.yaml \
  --output-dir outputs/sp500_vix_continuous/beta_cvae \
  --base-data-dir data/processed \
  --no-wandb
```

The run completed all 500 epochs.

| Field | Value |
| --- | --- |
| Training run directory | `outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47` |
| Final model path | `outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47/final_model` |
| Runtime | 350.112 seconds |
| Start time | `2026-05-14T15:13:47.140817+00:00` |
| End time | `2026-05-14T15:19:37.252394+00:00` |
| Terminal-reported final train loss | 1.0309 |

The final model directory contains:

```text
model.pt
model_config.json
training_config.json
```

The parent training directory contains `exp_config.yaml`, which is required by
the paper-style loader.

## Continuous Evaluator

The standard continuous evaluator was available and was run on the reproduced
checkpoint:

```bash
poetry run tcvae-evaluate \
  --config configs/experiments/sp500_vix_beta_cvae.yaml \
  --model-dir outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47/final_model \
  --output-dir outputs/sp500_vix_continuous/beta_cvae/evaluation_final \
  --base-data-dir data/processed \
  --n-sample-test 1000 \
  --seed 99
```

The evaluator produced:

```text
outputs/sp500_vix_continuous/beta_cvae/evaluation_final/summary.json
outputs/sp500_vix_continuous/beta_cvae/evaluation_final/evaluation_batch.pt
outputs/sp500_vix_continuous/beta_cvae/evaluation_final/hyper_metric.pkl
```

Persisted evaluation summary:

| Metric | Value |
| --- | ---: |
| Real data shape | `1000x60x1` |
| Fake data shape | `1000x60x1` |
| Reconstruction data shape | `1000x60x1` |
| MMD | 0.15442199 |
| SWD | 0.00878550 |

The evaluator emitted a CUDA-driver warning and used CPU-compatible execution.
This does not affect checkpoint readiness.

## Paper-Style Readiness

The checkpoint is ready for paper-style comparison. The expected loader inputs
are present:

- `final_model/model.pt`;
- `final_model/model_config.json`;
- `final_model/training_config.json`;
- parent `exp_config.yaml`.

Use the continuous model directory below for future paper-style runs:

```text
outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47/final_model
```
