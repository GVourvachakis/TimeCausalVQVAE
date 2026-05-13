# Codex refactor guide

## Phase 0: Audit

Inspect the cloned upstream repository. Create `docs/audit/upstream_inventory.md` with file tree, modules to keep, generated artifacts, borrowed code, selected checkpoint configs, and risky files. Do not edit source code.

## Phase 1: New repository scaffold

Add package scaffold, `pyproject.toml`, pre-commit, README, CONTRIBUTING, migration docs, and Codex context. Do not migrate implementation files yet. Commit scope: `refactor(scaffold)`.

## Phase 2: Namespace migration

Move `src/tsvae` to `src/time_causal_vae`; update imports mechanically. Preserve behavior. Commit scope: `refactor(package)`.

## Phase 3: Model modules

Restructure model code into objectives, encoders, decoders, priors, conditioners, factory, and config modules. Rename classes only after references are updated. Commit scope: `refactor(models)`.

## Phase 4: Config modernization

Replace ad hoc config access with typed dataclasses and YAML configs under `configs/experiments/`. Remove absolute paths. Commit scope: `refactor(config)`.

## Phase 5: Dataset cleanup

Move datasets to `data/base.py`, `black_scholes.py`, `heston.py`, `path_dependent_volatility.py`, `market.py`, `pipeline.py`, and `windows.py`. Document `data` and `labels`. Commit scope: `refactor(data)`.

## Phase 6: Evaluation cleanup

Move evaluation code to `time_causal_vae.evaluation`, separating metrics, conditional/unconditional evaluation, downstream finance tasks, plotting helpers, and optional borrowed algorithms. Commit scope: `refactor(evaluation)`.

## Phase 7: CLI and scripts

Add CLI entry points for training, evaluation, and checkpoint inspection. Commit scope: `feat(cli)`.

## Phase 8: Documentation

Document architecture, conditioning, selected paper configurations, and migration deviations. Commit scope: `docs`.
