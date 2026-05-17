# Public Registry Cleanup Report

## Branch

- `cleanup/public-final-registry-namespaces`

## Initial Audit

The branch was clean before cleanup.

### Tracked Top-Level Files

- `.gitignore`
- `.pre-commit-config.yaml`
- `CONTRIBUTING.md`
- `README.md`
- `pyproject.toml`

### Tracked Documentation

- `CONTRIBUTING.md`
- `README.md`
- `assets/figures/README.md`
- `data/README.md`
- `docs/experiments/per_experiment_final_evaluation_results.md`
- `docs/experiments/per_experiment_final_evaluation_setup.md`
- `docs/experiments/per_experiment_model_selection_gap_analysis.md`
- `docs/experiments/per_experiment_model_selection_plan.md`
- `docs/experiments/per_experiment_model_selection_results.md`
- `docs/experiments/per_experiment_model_selection_setup.md`
- `notebooks/README.md`
- `src/time_causal_vae/evaluation/external/README.md`
- `trained_models/README.md`
- `trained_models/black_scholes/model_card.md`
- `trained_models/heston/model_card.md`
- `trained_models/pdv/model_card.md`
- `trained_models/sp500_vix/model_card.md`

### Tracked Scripts

- `scripts/analyze_discrete_latent_geometry.py`
- `scripts/check_causal_conv_no_leakage.py`
- `scripts/check_conditional_token_prior_no_leakage.py`
- `scripts/check_conditional_vq_tokenizer_no_leakage.py`
- `scripts/check_vq_tokenizer_shapes.py`
- `scripts/evaluate_sp500_vix_paper_style.py`
- `scripts/extract_token_indices.py`
- `scripts/inspect_selected_configs.py`
- `scripts/reproduce_black_scholes.py`
- `scripts/reproduce_heston.py`
- `scripts/reproduce_pdv.py`
- `scripts/reproduce_sp500_vix.py`
- `scripts/reproduction_common.py`
- `scripts/run_per_experiment_final_evaluation.py`
- `scripts/run_per_experiment_model_selection.py`
- `scripts/select_registered_model.py`

### Tracked Configs

- `configs/experiments/black_scholes_beta_cvae.yaml`
- `configs/experiments/black_scholes_causal_token_prior.yaml`
- `configs/experiments/black_scholes_causal_token_prior_additive.yaml`
- `configs/experiments/black_scholes_causal_token_prior_hidden128_additive.yaml`
- `configs/experiments/black_scholes_causal_token_prior_hidden128_conv_transformer.yaml`
- `configs/experiments/black_scholes_causal_vq_tokenizer.yaml`
- `configs/experiments/black_scholes_causal_vq_tokenizer_codebook64_codebookdim16.yaml`
- `configs/experiments/black_scholes_causal_vq_tokenizer_hidden128.yaml`
- `configs/experiments/heston_causal_token_prior_additive.yaml`
- `configs/experiments/heston_causal_token_prior_hidden128_additive.yaml`
- `configs/experiments/heston_causal_token_prior_hidden128_conv_transformer.yaml`
- `configs/experiments/heston_causal_vq_tokenizer.yaml`
- `configs/experiments/heston_causal_vq_tokenizer_hidden128.yaml`
- `configs/experiments/heston_info_cvae.yaml`
- `configs/experiments/pdv_causal_token_prior_additive_seed1.yaml`
- `configs/experiments/pdv_causal_token_prior_hidden128_additive.yaml`
- `configs/experiments/pdv_causal_token_prior_hidden128_conv_transformer.yaml`
- `configs/experiments/pdv_causal_vq_tokenizer_codebook64_codebookdim16.yaml`
- `configs/experiments/pdv_causal_vq_tokenizer_hidden128.yaml`
- `configs/experiments/pdv_info_cvae.yaml`
- `configs/experiments/sp500_vix_beta_cvae.yaml`
- `configs/experiments/sp500_vix_causal_token_prior_additive.yaml`
- `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml`
- `configs/experiments/sp500_vix_causal_vq_tokenizer.yaml`
- `configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml`

### Tracked Notebooks

- `notebooks/continuous/black_scholes.ipynb`
- `notebooks/continuous/heston.ipynb`
- `notebooks/continuous/pdv.ipynb`
- `notebooks/continuous/sp500_vix.ipynb`
- `notebooks/discrete/black_scholes.ipynb`
- `notebooks/discrete/discrete_latent_geometry.ipynb`
- `notebooks/discrete/heston.ipynb`
- `notebooks/discrete/pdv.ipynb`
- `notebooks/discrete/sp500_vix.ipynb`
- `notebooks/report/sp500_vix_report_figures.ipynb`

### Tracked Generated Artefacts

The required generated-artefact audit command found no tracked paths matching:

```bash
git ls-files | grep -E '(^outputs/|^wandb/|^data/processed/|\.npy$|\.npz$|\.pt$|\.pkl$|\.pyc$|__pycache__)' || true
```

Small curated public figure assets are tracked under `assets/figures/` and are referenced by
`README.md`:

- `assets/figures/sp500_vix_best_generated_paths.png`
- `assets/figures/sp500_vix_best_research_paths.png`
- `assets/figures/sp500_vix_hidden128_codebook_voronoi.png`
- `assets/figures/sp500_vix_vix_bucket_code_usage.png`

### Tracked Local Tooling Files

The required local-tooling audit command found no tracked paths matching:

```bash
git ls-files | grep -E '(^\.codex/|^\.agents/|^\.vscode/|^\.editorconfig$|^poetry.lock$)' || true
```

An untracked empty `.agents/` directory existed in the working tree. Removal was attempted with
`rmdir`, but the environment re-exposes it as an empty read-only `tmpfs` mount:

```text
/home/georgios-vourvachakis/Desktop/TimeCausalVQVAE/.agents tmpfs ro,mode=555
```

It is not tracked and is not reported by `git status --short --untracked-files=all`.

## Cleanup Decisions

### Removed From The Public Branch

- `docs/experiments/per_experiment_final_evaluation_results.md`
- `docs/experiments/per_experiment_final_evaluation_setup.md`
- `docs/experiments/per_experiment_model_selection_gap_analysis.md`
- `docs/experiments/per_experiment_model_selection_plan.md`
- `docs/experiments/per_experiment_model_selection_results.md`
- `docs/experiments/per_experiment_model_selection_setup.md`
- `scripts/run_per_experiment_final_evaluation.py`
- `scripts/run_per_experiment_model_selection.py`
- untracked empty `.agents/` removal attempted; it remains only as an environment-provided
  read-only `tmpfs` mount, not as a tracked repository path

### Retained Public Registry Surface

- `trained_models/model_registry.yaml`
- `trained_models/README.md`
- `trained_models/black_scholes/model_card.md`
- `trained_models/heston/model_card.md`
- `trained_models/pdv/model_card.md`
- `trained_models/sp500_vix/model_card.md`
- `src/time_causal_vae/experiments/model_registry.py`
- `src/time_causal_vae/experiments/selection_profiles.py`
- `src/time_causal_vae/experiments/config.py`
- `scripts/select_registered_model.py`
- `scripts/inspect_selected_configs.py`
- `scripts/extract_token_indices.py`
- `scripts/evaluate_sp500_vix_paper_style.py`
- `scripts/analyze_discrete_latent_geometry.py`
- no-leakage scripts referenced by the public documentation
- reproduction wrappers referenced by the public documentation
- notebooks that use registry selection
- selected configs referenced by `trained_models/model_registry.yaml`

## Verification

Registry selector checks passed:

```bash
poetry run python scripts/select_registered_model.py --experiment sp500_vix --family discrete || true
poetry run python scripts/select_registered_model.py --experiment black_scholes --family continuous || true
```

Repository checks passed:

```bash
poetry check
poetry run ruff check src scripts configs --fix
poetry run ruff format --preview src scripts configs README.md CONTRIBUTING.md trained_models
poetry run mypy src/time_causal_vae
```

The requested formatter command without `--preview` was also attempted:

```bash
poetry run ruff format src scripts configs README.md CONTRIBUTING.md trained_models
```

It formatted Python/YAML targets, then failed for `README.md` and `CONTRIBUTING.md` because this
Ruff version requires preview mode for Markdown. The same target set was rerun successfully with
`--preview`.

Final generated-artefact and local-tooling audits found no tracked matches:

```bash
git ls-files | grep -E '(^outputs/|^wandb/|^data/processed/|\.npy$|\.npz$|\.pt$|\.pkl$|\.pyc$|__pycache__)' || true
git ls-files | grep -E '(^\.codex/|^\.agents/|^\.vscode/|^\.editorconfig$|^poetry.lock$)' || true
```
