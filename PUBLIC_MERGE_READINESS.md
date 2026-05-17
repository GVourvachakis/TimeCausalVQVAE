# Public Merge Readiness

Branch: `cleanup/public-final-minimal`

Date: 2026-05-17

## Summary

Status: ready for merge into `main` after normal review of the current branch diff.

No full model training was run. Smoke outputs and executed notebook copies were written only under
ignored `outputs/` paths.

## Checks

| Check | Result | Notes |
| --- | --- | --- |
| `git status --short` | Pass | Clean before this report was added. |
| Forbidden tracked artefact scan | Pass | No tracked `outputs/`, `wandb/`, `data/processed/`, `.npy`, `.pt`, `.pkl`, `.pyc`, or `__pycache__` entries. |
| Editor/Codex/lock tracked-file scan | Pass | No tracked `.codex/`, `.vscode/`, `.editorconfig`, or `poetry.lock`. |
| Required public configs | Pass | `sp500_vix_beta_cvae.yaml`, `sp500_vix_causal_vq_tokenizer.yaml`, and `sp500_vix_causal_token_prior_additive.yaml` exist. |
| Research config policy | Pass | Hidden128 conv-transformer configs are absent from public configs and referenced only as optional research-branch metadata/notebook paths. |
| Package import | Pass | `import time_causal_vae` and `build_token_prior_model` import succeeded; version reported `0.1.0`. |
| Continuous dry run | Pass | `tcvae-train` dry run completed for S&P500/VIX BetaCVAE; local data found with 2,457 samples of shape `60x1`. |
| Tokenizer smoke | Pass | One-epoch tokenizer smoke completed; final loss `0.55719300`, active codes `64`. |
| Notebook light execution | Pass | `notebooks/discrete/sp500_vix.ipynb` executed successfully to ignored `outputs/notebook_checks/`. A sandboxed attempt failed because local socket creation was blocked; rerun with local Jupyter socket access passed. |
| `poetry check` | Pass | Reported `All set!`. |
| `poetry run ruff check src scripts configs --fix` | Pass | Reported `All checks passed!`. |
| `poetry run mypy src/time_causal_vae` | Pass | Reported no issues in 97 source files. |

## Local Data

S&P500/VIX processed local data is present in ignored `data/processed/` and was sufficient for the
continuous dry run and tokenizer smoke check. No processed data is tracked.

## Generated Outputs

The following validation commands wrote only ignored local outputs:

- `outputs/smoke_merge_continuous/`
- `outputs/smoke_merge_tokenizer/`
- `outputs/notebook_checks/sp500_vix_discrete_merge_check.ipynb`

These paths are intentionally not tracked.

## Readiness Statement

This branch is merge-ready from the public-minimal validation perspective. The remaining action is a
human review of the branch diff, followed by the normal merge procedure into `main`.
