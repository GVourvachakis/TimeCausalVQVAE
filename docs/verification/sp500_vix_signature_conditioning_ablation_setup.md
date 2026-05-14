# S&P500/VIX Signature-Conditioning Ablation Setup

## Purpose

This note records the setup for log-signature conditioning ablations over
context length and log-signature depth. No full training run was started in this
setup pass, and the tokeniser and token-prior architecture were left unchanged.

The ablation grid uses the existing additive scalar-conditioned causal token
prior with optional precomputed condition features concatenated to the scalar
VIX label.

## W&B Socket Bypass

The local execution environment blocks W&B's default loopback service socket.
The requested pre-flight checks gave the following result:

| Method | Result |
| --- | --- |
| `WANDB_START_METHOD=thread` | Failed with `listen tcp 127.0.0.1:0: socket: operation not permitted`. |
| `WANDB_MODE=offline` | Failed with the same local socket error. |
| `WANDB_DISABLE_SERVICE=true WANDB_MODE=offline` | Succeeded for offline logging only. |
| `WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread`, with `WANDB_MODE` unset | Succeeded for live cloud streaming when the process had network access. |

Future live non-smoke runs in this environment should therefore use:

```bash
export WANDB_DISABLE_SERVICE=true
export WANDB_START_METHOD=thread
unset WANDB_MODE
```

The intended W&B project and entity remain:

- project: `time-causal-token-prior`;
- entity: `tc_vae`.

## Feature Sets

All feature sets were extracted from
`data/processed/sp500vix/sp500vix_normalized.npy` with:

- lead-lag transform enabled;
- time channel enabled;
- VIX channel enabled;
- sample alignment preserved with left padding of the historical context using
  the first available value;
- `iisignature` version `0.24`;
- finite-value validation passed.

| Feature directory | Depth | Context length | Feature dimension | Feature shape | Total condition dimension | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `outputs/sp500_vix_discrete/signature_features/logsig_l2_ctx10` | 2 | 10 | 55 | `(2457, 55)` | 56 | 0.248 s |
| `outputs/sp500_vix_discrete/signature_features/logsig_l2_ctx20` | 2 | 20 | 55 | `(2457, 55)` | 56 | 0.258 s |
| `outputs/sp500_vix_discrete/signature_features/logsig_l3_ctx10` | 3 | 10 | 385 | `(2457, 385)` | 386 | 1.203 s |
| `outputs/sp500_vix_discrete/signature_features/logsig_l3_ctx20` | 3 | 20 | 385 | `(2457, 385)` | 386 | 1.266 s |

The extraction outputs are generated artefacts under `outputs/` and are not part
of the committed setup.

## Configs

The following configs were created:

| Config | Condition feature directory | `model.condition_dim` |
| --- | --- | ---: |
| `configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l2_ctx10.yaml` | `outputs/sp500_vix_discrete/signature_features/logsig_l2_ctx10` | 56 |
| `configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l2_ctx20.yaml` | `outputs/sp500_vix_discrete/signature_features/logsig_l2_ctx20` | 56 |
| `configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx10.yaml` | `outputs/sp500_vix_discrete/signature_features/logsig_l3_ctx10` | 386 |
| `configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20.yaml` | `outputs/sp500_vix_discrete/signature_features/logsig_l3_ctx20` | 386 |

Each config preserves the promoted token-prior architecture:

- causal token prior;
- `condition_injection: additive`;
- codebook size `64`;
- sequence length `60`;
- token embedding dimension `128`;
- four transformer layers;
- no cross-attention, AdaLN, diffusion, GroupedRVQ, MGVQ, or tokeniser changes.

## Ablation Runner

The new runner is:

```text
scripts/run_sp500_vix_signature_conditioning_ablation.py
```

It coordinates existing commands rather than introducing new model code. For
each config it:

1. loads the YAML config;
2. reads `signature_feature_summary.json`;
3. verifies `model.condition_dim == 1 + feature_dimension`;
4. trains the token prior, or calls the training CLI in `--dry-run` mode;
5. evaluates the best checkpoint for non-dry runs;
6. optionally runs paper-style diagnostics;
7. writes aggregate `ablation_results.json` and `ablation_results.csv`.

When `--wandb` is enabled, the runner passes:

- `--wandb-project time-causal-token-prior`;
- `--wandb-entity tc_vae`;
- a run name matching the experiment name.

In this socket-restricted environment, the runner also defaults W&B-enabled
subprocesses to `WANDB_DISABLE_SERVICE=true` and `WANDB_START_METHOD=thread`,
and removes `WANDB_MODE` from the subprocess environment so live cloud tracking
is not silently downgraded to offline mode.

## Dry Run

Command:

```bash
WANDB_DISABLE_SERVICE=true WANDB_MODE=offline poetry run python scripts/run_sp500_vix_signature_conditioning_ablation.py \
  --configs \
    configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l2_ctx10.yaml \
    configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l2_ctx20.yaml \
    configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx10.yaml \
    configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/signature_conditioning_ablation_dry \
  --base-data-dir data/processed \
  --epochs 1 \
  --n-sample 128 \
  --seed 99 \
  --dry-run \
  --no-wandb
```

The dry run completed successfully. It did not train checkpoints and did not run
decoded evaluation.

| Experiment | Status | Train tokens | Train conditions | Eval conditions | Parameters | W&B |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `sp500_vix_causal_token_prior_additive_logsig_l2_ctx10` | `dry_run_validated` | `(2457, 60)` | `(2457, 56)` | `(2457, 56)` | 561472 | false |
| `sp500_vix_causal_token_prior_additive_logsig_l2_ctx20` | `dry_run_validated` | `(2457, 60)` | `(2457, 56)` | `(2457, 56)` | 561472 | false |
| `sp500_vix_causal_token_prior_additive_logsig_l3_ctx10` | `dry_run_validated` | `(2457, 60)` | `(2457, 386)` | `(2457, 386)` | 603712 | false |
| `sp500_vix_causal_token_prior_additive_logsig_l3_ctx20` | `dry_run_validated` | `(2457, 60)` | `(2457, 386)` | `(2457, 386)` | 603712 | false |

Aggregate dry-run artefacts were written to:

- `outputs/sp500_vix_discrete/token_prior/signature_conditioning_ablation_dry/ablation_results.json`;
- `outputs/sp500_vix_discrete/token_prior/signature_conditioning_ablation_dry/ablation_results.csv`.

These aggregate files are generated verification artefacts and are not committed.

## Decision

The setup is ready for the next non-smoke ablation stage. The next run should
compare depth `2` and `3` at context lengths `10` and `20`, with W&B live
streaming through service disablement plus threaded start mode.
