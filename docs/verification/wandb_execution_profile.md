# W&B Execution Profile For Signature Conditioning

## Purpose

This note codifies the execution profile for non-smoke
signature-conditioning runs in socket-restricted environments. It follows the
`Infrastructure and Telemetry Execution Profile` recorded in
`docs/verification/sp500_vix_logsig_l3_ctx20_robustness_results.md`.

No model architecture, tokenizer, prior, or training configuration is changed
by this document.

## Required Environment Wrapper

For live W&B runs, use the following wrapper:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread <command>
```

Example:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread poetry run python scripts/run_sp500_vix_signature_conditioning_ablation.py \
  --configs <configs> \
  --output-dir <outputs-dir> \
  --base-data-dir data/processed \
  --wandb \
  --wandb-project time-causal-token-prior \
  --wandb-entity tc_vae
```

The variables have distinct roles:

- `env -u WANDB_MODE`: prevents inherited offline mode from disabling live
  cloud tracking;
- `MPLBACKEND=Agg`: forces Matplotlib onto a headless backend and avoids
  Tkinter/Tcl shutdown crashes;
- `WANDB_DISABLE_SERVICE=true`: disables the local W&B service daemon that may
  attempt loopback socket binds;
- `WANDB_START_METHOD=thread`: keeps W&B startup in the in-process thread path.

The signature-conditioning ablation runner applies these defaults to
W&B-enabled subprocesses. Parent-shell values for `MPLBACKEND`,
`WANDB_DISABLE_SERVICE`, and `WANDB_START_METHOD` are preserved when explicitly
set. `WANDB_MODE` is removed for W&B-enabled subprocesses so that live mode is
not silently replaced by offline mode.

## Known Failure Modes

Observed failure modes in this environment include:

- local loopback socket failure, reported as `socket: operation not permitted`;
- upstream W&B `CommError` or connection timeout during run initialisation;
- Tcl/Tk backend crashes during metric or plot cleanup when Matplotlib uses an
  interactive backend.

These failures are infrastructure constraints, not model-quality failures.

## Accepted Fallback

If live W&B still fails, rerun the same command with `--no-wandb`. Do not switch
to `WANDB_MODE=offline` unless a future prompt explicitly requests an offline
sync workflow.

The accepted fallback is:

- keep all model, data, sampling, and output settings unchanged;
- replace `--wandb` with `--no-wandb`;
- preserve local JSON, CSV, and Markdown outputs under ignored `outputs/`
  paths;
- document the missing W&B URL in the verification report.

This policy keeps non-smoke experiments reproducible while avoiding hidden
telemetry-mode changes.
