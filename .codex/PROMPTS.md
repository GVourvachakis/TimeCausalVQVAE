# Codex thread prompts

## Prompt 01: upstream audit

You are in a local clone of `justinhou95/TimeCausalVAE`. Do not edit files yet. Audit the repository and create `docs/audit/upstream_inventory.md`. Include current directory tree, `src/tsvae` modules, `src/evaluations` modules, config files and selections, trained checkpoint directories and selected paper config values, generated artifacts to omit, borrowed external modules, and exact source files needed for Black-Scholes, Heston, PDV, and S&P500/VIX.

## Prompt 02: scaffold the new repository

Create a new professional package scaffold named `time-causal-vae` with import package `time_causal_vae`. Add `pyproject.toml`, `.pre-commit-config.yaml`, `.editorconfig`, `.gitignore`, `README.md`, `CONTRIBUTING.md`, `docs/migration_log.md`, `docs/repository_structure.md`, and `.codex` context files. Do not migrate implementation files yet. Do not add pytest gates.

## Prompt 03: migrate package namespace

Move `src/tsvae` into `src/time_causal_vae`. Update imports from `tsvae.` to `time_causal_vae.`. Preserve behavior. Do not rename classes yet. Run Poetry/Ruff checks and report unresolved imports.

## Prompt 04: model API cleanup

Refactor `time_causal_vae.models` into objectives, encoders, decoders, priors, conditioners, `factory.py`, and `config.py`. Rename `CLSTMResEncoder` to `ConditionalResidualLSTMEncoder`, `CLSTMResDecoder` to `ConditionalResidualLSTMDecoder`, `FlowPrior` to `RealNVPPrior`, `BetaCVAE` to `BetaConditionalVAE`, and `InfoCVAE` to `InfoConditionalVAE`. Add NumPy-style docstrings and type hints. Preserve tensor shapes and loss formulas.

## Prompt 05: conditional TC-VAE documentation

Write `docs/architecture/conditional_tcvae.md`. Explain condition labels in `CVAE.forward`, identity conditioner, repeated condition concatenated to encoder input and decoder latent path, unconditional RealNVP prior, and PDV/SP500 condition definitions.

## Prompt 06: selected paper configs

Create `docs/experiments/selected_paper_configs.md` and YAML files under `configs/experiments/`. Document selected checkpoint configurations without calling them globally optimal.

## Prompt 07: dataset cleanup

Refactor datasets into `data/base.py`, `black_scholes.py`, `heston.py`, `path_dependent_volatility.py`, `market.py`, `pipeline.py`, and `windows.py`. Add docstrings explaining `data`, `labels`, path normalization, and condition construction. Preserve behavior.

## Prompt 08: evaluation cleanup

Move `src/evaluations` into `time_causal_vae.evaluation`; separate core distances, unconditional evaluation, conditional evaluation, downstream finance evaluations, plotting helpers, and optional borrowed algorithms. Do not rewrite numerical algorithms unless needed for imports.

## Prompt 09: final quality pass

Run Poetry, Ruff, and mypy. Fix only low-risk issues. Update `docs/migration_log.md` with deviations from upstream.
