# Codex project context

You are refactoring a cloned copy of `justinhou95/TimeCausalVAE` into a new professional Python package.

## Goal

Create a new repository with distribution name `time-causal-vae` and import package `time_causal_vae`. Use Poetry, Ruff, mypy, pre-commit, Commitizen, stripped notebooks, typed configs, docstrings, and clearer module names. Do not add pytest gates in the initial migration.

## Preserve scientific behavior

Preserve behavior before improving it. The selected paper backbone is conditional residual LSTM encoder, conditional residual LSTM decoder, identity conditioner, and RealNVP prior. Selected checkpoints use BetaCVAE for Black-Scholes and S&P500/VIX, and InfoCVAE for Heston and PDV.

## Important constraints

1. Do not silently change tensor shapes.
2. Do not change loss formulas unless explicitly requested.
3. Do not remove evaluation utilities required for paper figures until a replacement exists.
4. Do not commit trained models, generated metric files, notebook outputs, or W&B directories.
5. Keep migration notes in `docs/migration_log.md`.
6. Prefer small, reviewable commits.

## Original repository pain points

The package name `tsvae` is vague, root layout mixes experiments/artifacts/source, `__init__.py` files are empty, public APIs lack docstrings and type hints, configs use inconsistent conventions and absolute paths, notebooks hard-code checkpoint paths, and generated artifacts are committed.
