# Contributing

This project is a packaging and maintainability rewrite of the Time-Causal VAE implementation.

## Development setup

```bash
poetry install
poetry shell
poetry run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

## Code style

- Use NumPy-style docstrings for public functions, classes, and modules.
- Add type hints to public APIs.
- Prefer dataclasses or typed configuration models over unstructured dictionaries.
- Keep public APIs small and explicit.
- Avoid environment-specific absolute paths.
- Avoid committing generated notebook outputs or trained model artifacts.

## Formatting and linting

```bash
poetry run pre-commit run --all-files
```

The standard hooks use Ruff for formatting and linting, mypy for gradual type checking, nbstripout for notebooks, Poetry checks, and Commitizen for conventional commits.

## Commit process

Use Conventional Commits through Commitizen:

```bash
poetry run cz commit
```

Examples:

```text
refactor(models): rename CLSTMRes encoder and decoder modules
feat(config): add typed experiment configuration loader
docs(readme): document selected TC-VAE paper checkpoints
```

## Refactor rules

1. Preserve behavior before improving behavior.
2. Keep one migration concern per commit.
3. Rename modules only after creating compatibility notes.
4. Document every public class that remains.
5. Convert notebooks into scripts or reusable functions where possible.
6. Keep generated artifacts out of Git unless they are small reference fixtures.
7. Record substantial deviations from the original repository in the relevant change notes.

## No pytest requirement

This rewrite does not initially require pytest. Do not add a pre-push pytest gate until deterministic test fixtures are available.
