# Hawkes Public Merge Readiness

Date: 2026-05-18

Repository checked: `/home/georgios-vourvachakis/Desktop/TimeCausalVQVAE`

Branch checked: `feature/hawkes-jump-registry-notebooks`

## Summary

The Hawkes/SVMHJD public integration branch is ready for merge to `main`, subject to the usual
maintainer review. The integration keeps Hawkes/SVMHJD as optional research-candidate metadata and
does not make it a public default.

Generated notebook outputs were written under `outputs/notebook_checks/` during validation. These
files are generated artefacts and are not committed.

## Checks Run

| Check | Result | Notes |
| --- | --- | --- |
| `git status --short` | Pass | Clean before creating this readiness note. |
| Forbidden generated files grep | Pass | No tracked `outputs/`, `wandb/`, `data/processed/`, tensor, pickle, Python cache, or NumPy artefacts matched. |
| Local metadata grep | Pass | No tracked `.codex/`, `.agents/`, `.vscode/`, `.editorconfig`, or `poetry.lock` files matched. |
| `poetry run python scripts/select_registered_model.py --experiment hawkes_jump --family continuous` | Pass | Selected `beta_cvae_logreturn_identity` with `status: research_candidate` and `public_default: false`. |
| `poetry run python scripts/select_registered_model.py --experiment hawkes_jump --family discrete` | Pass | Selected `hidden128_logreturn_cb64_conv_transformer_k3` with `status: research_candidate` and `public_default: false`. |
| `poetry run python scripts/select_registered_model.py --experiment sp500_vix --family discrete` | Pass | Selected the existing S&P500/VIX public discrete baseline. |
| `poetry run jupyter nbconvert --to notebook --execute notebooks/benchmarks/hawkes_jump_dataset.ipynb --output-dir outputs/notebook_checks --output hawkes_jump_dataset_public_check.ipynb` | Pass | Required escalated execution because sandboxed Jupyter could not open local kernel sockets. |
| `poetry run jupyter nbconvert --to notebook --execute notebooks/report/hawkes_jump_model_comparison.ipynb --output-dir outputs/notebook_checks --output hawkes_jump_model_comparison_public_check.ipynb` | Pass | Executed copy written under `outputs/notebook_checks/`. |
| `poetry run jupyter nbconvert --to notebook --execute notebooks/continuous/hawkes_jump.ipynb --output-dir outputs/notebook_checks --output hawkes_jump_continuous_public_check.ipynb` | Pass | Executed copy written under `outputs/notebook_checks/`. |
| `poetry run jupyter nbconvert --to notebook --execute notebooks/discrete/hawkes_jump.ipynb --output-dir outputs/notebook_checks --output hawkes_jump_discrete_public_check.ipynb` | Pass | Executed copy written under `outputs/notebook_checks/`. |
| `find notebooks -name "*.ipynb" -print0 | xargs -0 poetry run pre-commit run nbstripout --files || true` | Pass | `nbstripout` passed. |
| `poetry check` | Pass | Poetry project metadata is valid. |
| `poetry run ruff check src scripts configs --fix` | Pass | Ruff reported all checks passed. |
| `poetry run mypy src/time_causal_vae` | Pass | Mypy reported no issues in 110 source files. |

## Merge Assessment

Ready for main: yes.

Residual notes:

- No full model training was run.
- No generated outputs are staged for commit.
- The Hawkes/SVMHJD registry entry remains a research candidate rather than a public default.
