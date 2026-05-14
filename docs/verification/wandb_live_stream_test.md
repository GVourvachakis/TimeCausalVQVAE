# W&B Live Stream Test

## Purpose

This note records a one-epoch W&B live-streaming diagnostic for token-prior
training. The goal was to keep live HTTP telemetry active while bypassing the
local loopback service socket that fails in this environment.

No model code, configs, tokeniser code, or prior architecture were changed.

## Configuration

The validated live configuration is:

```bash
unset WANDB_MODE
export WANDB_DISABLE_SERVICE=true
export WANDB_START_METHOD=thread
```

`WANDB_DISABLE_SERVICE=true` bypasses the local W&B background service daemon.
`WANDB_START_METHOD=thread` keeps telemetry in process threads. `WANDB_MODE`
must be unset for live cloud streaming.

## Diagnostic Command

The requested one-epoch diagnostic was:

```bash
env -u WANDB_MODE WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread poetry run tcvae-train-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l2_ctx10.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/diagnostics/live_stream_test \
  --epochs 1 \
  --wandb \
  --wandb-project time-causal-token-prior \
  --wandb-entity tc_vae \
  --wandb-run-name live_stream_verification
```

The same command inside the restricted sandbox reached W&B initialisation but
entered a network retry loop with `ConnectionError`. Re-running the same command
with network access succeeded.

## Result

Live W&B streaming was validated.

| Field | Value |
| --- | --- |
| Project | `https://wandb.ai/tc_vae/time-causal-token-prior` |
| Run | `https://wandb.ai/tc_vae/time-causal-token-prior/runs/rgkvsjpz` |
| Run name | `live_stream_verification` |
| Local run directory | `wandb/run-20260514_175016-rgkvsjpz` |
| Epochs | 1 |
| Runtime | 16.120 s |
| Eval cross-entropy | 3.32649058 |
| Eval accuracy | 0.22741826 |
| Eval perplexity | 29.08112798 |

W&B reported that it synced five W&B files, two artifact files, and no media
files. The run printed a live dashboard URL and completed without entering
offline mode.

## Runner Update

`scripts/run_sp500_vix_signature_conditioning_ablation.py` now applies the live
configuration to W&B-enabled subprocesses:

- remove `WANDB_MODE` from the subprocess environment;
- set `WANDB_DISABLE_SERVICE=true` when unset;
- set `WANDB_START_METHOD=thread` when unset;
- preserve the requested W&B project and entity.

This makes subsequent multi-config ablation runs use live dashboard telemetry
without requiring `WANDB_MODE=offline`.
