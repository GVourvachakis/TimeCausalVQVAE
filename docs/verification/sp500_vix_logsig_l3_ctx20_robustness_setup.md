# S&P500/VIX `logsig_l3_ctx20` Robustness Setup

## Purpose

This note records setup-only robustness ablations for the best
signature-conditioned candidate from the current S&P500/VIX paper-style
comparison, `logsig_l3_ctx20`. No non-smoke training was run, no tokeniser
architecture was changed, and no Gumbel-Softmax or signature-kernel objective
was added.

## Base Candidate

All robustness variants preserve the promoted discrete-token setup used by the
depth-3, context-20 signature-conditioned prior:

- tokenizer and token data are unchanged;
- condition feature directory:
  `outputs/sp500_vix_discrete/signature_features/logsig_l3_ctx20`;
- signature feature dimension: `385`;
- total condition dimension: `386`;
- condition injection: `additive`;
- log-signature preprocessing: lead-lag path, time channel, and VIX channel.

## Configs Created

| Config | Intended change | Seed | Learning rate | Epochs |
| --- | --- | ---: | ---: | ---: |
| `configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_seed1.yaml` | seed robustness | 1 | 0.0003 | 100 |
| `configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_seed2.yaml` | seed robustness | 2 | 0.0003 | 100 |
| `configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_lr1e4.yaml` | lower learning rate | 0 | 0.0001 | 100 |
| `configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_lr5e4.yaml` | higher learning rate | 0 | 0.0005 | 100 |
| `configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_e200.yaml` | longer training horizon | 0 | 0.0003 | 200 |

Each config keeps `condition_dim=386`, the same signature feature directory,
the same token data, and additive scalar-vector condition injection. Only the
intended seed, learning-rate, or epoch field is changed.

## Runner Update

`scripts/run_sp500_vix_signature_conditioning_ablation.py` now records the
model-selection profile source from
`docs/architecture/model_selection_profiles.md`. When `--run-paper-style` is
enabled and `paper_style_summary.json` is available, it also extracts separate
paper-style scalar metrics and writes profile rank-score summaries to the
aggregate JSON/CSV outputs.

The profile rank scores are reporting summaries only. They use lower-is-better
average ranks over already-computed paper-style diagnostics and do not add a new
evaluation metric. Tail exceedance fields remain visible separately.

The runner also captures W&B cloud run URLs from subprocess output when W&B is
enabled, and it sets `MPLBACKEND=Agg` for W&B-enabled child processes to avoid
graphical backend collisions during non-smoke runs.

## Dry Run

Command:

```bash
MPLBACKEND=Agg poetry run python scripts/run_sp500_vix_signature_conditioning_ablation.py \
  --configs \
    configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_seed1.yaml \
    configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_seed2.yaml \
    configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_lr1e4.yaml \
    configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_lr5e4.yaml \
    configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_e200.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/signature_conditioning_robustness_dry \
  --base-data-dir data/processed \
  --epochs 1 \
  --n-sample 128 \
  --seed 99 \
  --dry-run \
  --no-wandb
```

Result: all five configs returned `dry_run_validated` with return code `0`.
The dry run did not train checkpoints or run model evaluation.

Common loaded shapes:

| Field | Shape |
| --- | --- |
| train tokens | `(2457, 60)` |
| train conditions | `(2457, 386)` |
| eval tokens | `(2457, 60)` |
| eval conditions | `(2457, 386)` |

Dry-run output files:

- `outputs/sp500_vix_discrete/token_prior/signature_conditioning_robustness_dry/ablation_results.json`;
- `outputs/sp500_vix_discrete/token_prior/signature_conditioning_robustness_dry/ablation_results.csv`.

Observed non-fatal warnings:

- Matplotlib used a temporary cache directory because
  `/home/georgios-vourvachakis/.config/matplotlib` was not writable;
- CUDA initialisation warned that the local NVIDIA driver is older than the
  installed PyTorch CUDA build, so the dry run used CPU;
- PyTorch emitted the existing nested-tensor warning for
  `nn.TransformerEncoder`.

## W&B Live Settings

Use the following environment wrapper for the subsequent non-smoke robustness
runs:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread
```

This keeps W&B in live cloud-streaming mode while disabling the local W&B
service daemon and forcing Matplotlib onto a headless backend. The target W&B
project and entity remain:

- project: `time-causal-token-prior`;
- entity: `tc_vae`.

## Next Step

Run the five configs as non-smoke jobs with W&B enabled, then run paper-style
diagnostics for the completed best checkpoints. Promotion remains deferred
until at least the seed variants confirm the `logsig_l3_ctx20` path-functional
gains.
