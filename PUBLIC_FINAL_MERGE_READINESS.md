# Public Final Merge Readiness

Branch: `cleanup/public-final-registry-namespaces`

Date: 2026-05-17

## Summary

The branch is ready to merge to `main`, subject to the caveats below. No merge was performed.
No full model training was run. Generated smoke and notebook-check artefacts were kept out of git
and removed after validation.

## Checks Run

| Check | Result | Notes |
| --- | --- | --- |
| `git status --short` | Pass | Clean before validation. |
| Forbidden tracked artefact scan | Pass | No tracked `outputs/`, `wandb/`, `data/processed/`, arrays, checkpoints, pickles, bytecode, or `__pycache__` entries found. |
| Forbidden tracked workspace file scan | Pass | No tracked `.codex/`, `.agents/`, `.vscode/`, `.editorconfig`, or `poetry.lock` entries found. |
| Namespace import check | Pass | New discrete prior import and old `time_causal_vae.token_prior` compatibility import both resolved. |
| Registry selector: `black_scholes` continuous | Pass | Selected `beta_cvae`; local checkpoint paths remain metadata-only. |
| Registry selector: `heston` discrete | Pass | Selected `standard_vq_additive_ar`; registry reports expected missing optional metrics. |
| Registry selector: `pdv` discrete | Pass | Selected `conditional_standard_vq_additive_ar`; registry reports expected missing optional metrics. |
| Registry selector: `sp500_vix` discrete | Pass | Selected `conditional_standard_vq_additive_ar`; no selector warnings. |
| Continuous smoke dry run | Pass | `tcvae-train` completed dry-run setup for `sp500_vix_beta_cvae` without starting training. |
| Tokenizer smoke run | Pass | `tcvae-train-tokenizer` completed one short S&P500/VIX smoke epoch using local `data/processed`. Generated output was removed. |
| Discrete S&P500/VIX notebook light check | Pass | Initial sandbox run failed because Jupyter could not open a local kernel socket; rerun with local-kernel permissions passed and the executed notebook output was removed. |
| `poetry check` | Pass | Package metadata valid. |
| `poetry run ruff check src scripts configs --fix` | Pass | No remaining Ruff issues. |
| `poetry run mypy src/time_causal_vae` | Pass | No type-checking issues reported. |

## Remaining Caveats

- The trained-model registry is metadata only. It intentionally records local checkpoint
  conventions rather than committed weights.
- Several registry entries intentionally report missing optional metrics such as notebook
  reproduction or additional path metrics. These are visible in `trained_models/model_registry.yaml`
  and do not block the public merge.
- Import checks produced a Matplotlib cache warning because the default user cache directory was
  not writable in this environment; Matplotlib used `/tmp`.
- The tokenizer smoke run emitted a CUDA driver warning and ran successfully on the available
  runtime.
- The notebook execution needed permission to start a local Jupyter kernel socket. The sandbox
  failure was environmental; the rerun passed.

## Merge Readiness

Ready to merge to `main`: **yes**.

