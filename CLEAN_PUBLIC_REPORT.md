# Clean Public Branch Audit

Branch audited: `cleanup/public-final-minimal`

Initial worktree status: clean.

## Tracked Top-Level Files

- `.editorconfig`
- `.gitignore`
- `.pre-commit-config.yaml`
- `CONTRIBUTING.md`
- `README.md`
- `poetry.lock`
- `pyproject.toml`

## Tracked Docs Files

- `docs/final_deliverable_plan.md`
- `docs/references.md`
- `docs/report_branch_merge_note.md`
- `docs/report_ready_best_discrete_research_model.md`
- `docs/report_ready_sp500_vix_promoted_method.md`
- `docs/verification/discrete_latent_geometry_decision.md`
- `docs/verification/latent_geometry_source_smoke.md`
- `docs/verification/sp500_vix_rvq_q2_latent_geometry.md`
- `docs/verification/sp500_vix_standard_vq_latent_geometry.md`

## Tracked Scripts

- `scripts/analyze_discrete_latent_geometry.py`
- `scripts/check_causal_conv_no_leakage.py`
- `scripts/check_conditional_token_prior_no_leakage.py`
- `scripts/check_conditional_vq_tokenizer_no_leakage.py`
- `scripts/check_multicode_token_prior_no_leakage.py`
- `scripts/check_vq_family_tokenizer_shapes.py`
- `scripts/check_vq_tokenizer_shapes.py`
- `scripts/evaluate_sp500_vix_paper_style.py`
- `scripts/extract_token_indices.py`
- `scripts/inspect_selected_configs.py`
- `scripts/reproduce_black_scholes.py`
- `scripts/reproduce_heston.py`
- `scripts/reproduce_pdv.py`
- `scripts/reproduce_sp500_vix.py`
- `scripts/reproduction_common.py`
- `scripts/run_pdv_tokenizer_ablation.py`
- `scripts/run_sp500_vix_conditional_token_prior_ablation.py`
- `scripts/run_sp500_vix_tokenizer_ablation.py`
- `scripts/run_token_prior_sampling_ablation.py`

## Tracked Configs

- `configs/experiments/black_scholes_beta_cvae.yaml`
- `configs/experiments/black_scholes_causal_token_prior.yaml`
- `configs/experiments/black_scholes_causal_vq_tokenizer.yaml`
- `configs/experiments/heston_info_cvae.yaml`
- `configs/experiments/pdv_causal_token_prior_additive_seed1.yaml`
- `configs/experiments/pdv_causal_vq_tokenizer_codebook64_codebookdim16.yaml`
- `configs/experiments/pdv_info_cvae.yaml`
- `configs/experiments/sp500_vix_beta_cvae.yaml`
- `configs/experiments/sp500_vix_causal_rvq_token_prior_q2.yaml`
- `configs/experiments/sp500_vix_causal_rvq_tokenizer_q2.yaml`
- `configs/experiments/sp500_vix_causal_token_prior_additive.yaml`
- `configs/experiments/sp500_vix_causal_vq_tokenizer.yaml`

## Tracked Notebooks

- `notebooks/README.md`
- `notebooks/continuous/black_scholes.ipynb`
- `notebooks/continuous/heston.ipynb`
- `notebooks/continuous/pdv.ipynb`
- `notebooks/continuous/sp500_vix.ipynb`
- `notebooks/discrete/black_scholes.ipynb`
- `notebooks/discrete/discrete_latent_geometry_demo.ipynb`
- `notebooks/discrete/heston.ipynb`
- `notebooks/discrete/pdv.ipynb`
- `notebooks/discrete/sp500_vix.ipynb`
- `notebooks/report/sp500_vix_report_figures.ipynb`

## Tracked Generated Artefacts

Command:

```bash
git ls-files | grep -E '(^outputs/|^wandb/|^data/processed/|\.npy$|\.pt$|\.pkl$|\.pyc$|__pycache__)' || true
```

Result: no tracked generated artefacts were found.

Local generated directories were observed but are untracked: `outputs/`, `wandb/`,
`data/processed/`, `scripts/__pycache__/`, and `src/time_causal_vae/__pycache__/`.

## Tracked Codex, VSCode, And Editor Files

Command:

```bash
git ls-files | grep -E '(^\.codex/|^\.vscode/|^\.editorconfig$|^poetry.lock$)' || true
```

Result:

- `.codex/MIGRATION_CHECKLIST.md`
- `.codex/PROJECT_CONTEXT.md`
- `.codex/PROMPTS.md`
- `.codex/REFACTOR_GUIDE.md`
- `.editorconfig`
- `poetry.lock`

No tracked `.vscode/` files were found.

## Lock File Status

`poetry.lock` was tracked at audit time. Public-minimal policy removes it from this branch.

## Cleanup Actions Applied

The cleanup retains the package source, public configs, notebooks, reproduction wrappers, token
extraction, paper-style evaluation, latent-geometry diagnostics, and public no-leakage checks.

The cleanup removes:

- tracked Codex/editor/lock files: `.codex/`, `.editorconfig`, and `poetry.lock`;
- RVQ q2 public configs under `configs/experiments/`;
- the tracked `docs/` tree, with compact final references moved into `README.md`;
- research-only ablation runners under `scripts/`;
- VQ-family and multi-code smoke scripts that were only referenced by the removed RVQ q2 public
  surface.

No generated artefacts were added.
