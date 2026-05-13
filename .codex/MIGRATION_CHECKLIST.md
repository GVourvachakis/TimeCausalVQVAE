# Migration checklist

- [ ] Create new repository.
- [ ] Copy scaffold files.
- [ ] Initialize Git.
- [ ] Install Poetry environment.
- [ ] Install pre-commit and commit-msg hooks.
- [ ] Confirm `poetry check` passes.
- [ ] Create `docs/audit/upstream_inventory.md`.
- [ ] Identify core files, generated artifacts, and borrowed code.
- [ ] Migrate `src/tsvae` to `src/time_causal_vae`.
- [ ] Update imports.
- [ ] Migrate evaluation modules.
- [ ] Create typed config loaders.
- [ ] Add CLI entry points.
- [ ] Add docstrings and type hints.
- [ ] Document architecture, conditioning, selected configs, and deviations.
- [ ] Run Ruff and mypy.
- [ ] Strip notebook outputs.
- [ ] Use Commitizen for commits.
