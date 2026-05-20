# Contributing

Thank you for working on `time-causal-vae`. This repository is intentionally kept small on public
branches: package source, configs, notebooks, and lightweight scripts belong here; generated
artefacts and detailed research evidence belong in local training/evaluation directories or
research branches.

## Development Setup

Prerequisites:

- Python 3.11 or 3.12.
- [Poetry](https://python-poetry.org/) for dependency management.
- Git and a working shell environment.

Install the full development environment:

```bash
poetry install
```

Install pre-commit hooks when you are doing regular development:

```bash
poetry run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

Use a lean runtime environment only when you do not need development, notebook, or tracking tools:

```bash
poetry install --only main
```

## Poetry

Common commands:

- Install dependencies: `poetry install`
- Install only runtime dependencies: `poetry install --only main`
- Run a command inside the environment: `poetry run <command>`
- Build the package: `poetry build`
- Check package metadata: `poetry check`

Dependency groups are declared in `pyproject.toml` for `dev`, `notebooks`, and `tracking`.
Before adding a dependency, prefer existing scientific Python packages already used in the
project, and keep optional tooling out of the core runtime when possible.

## Running Checks

This repository currently uses linting, formatting, type checking, and package metadata checks.
It does not currently advertise a pytest or coverage gate.

Run the standard public checks:

```bash
poetry check
poetry run ruff check src scripts configs
poetry run ruff format --check src scripts
poetry run mypy src/time_causal_vae
```

For a local auto-fix pass:

```bash
poetry run ruff check src scripts configs --fix
poetry run ruff format src scripts
```

Pre-commit can run the configured hooks:

```bash
poetry run pre-commit run --all-files
```

## Documentation

Keep public documentation concise and runnable. The root `README.md` should describe the public
pipeline, public configs, notebooks, scripts, and references without becoming a research log.

Use Markdown links rather than bare URLs. Do not invent citations, authors, DOIs, or publication
details. If a reference is uncertain, cite the title and known arXiv or repository link only.

Detailed experiment notes, verification write-ups, and branch-specific evidence belong on
research branches, not on public-minimal branches.

## Notebook Policy

Committed notebooks must remain output-stripped. Notebooks should default to safe, guarded flags
such as `RUN_TRAINING = False` and `RUN_EVALUATION = False`.

Notebook examples may print commands, inspect configs, or display existing local figures, but
they should not train models, evaluate checkpoints, or write report figures merely by being
opened.

When a notebook needs generated figures, read them from local `outputs/` paths. Do not commit
executed notebooks or generated notebook-check artefacts.

## Generated Artefact Policy

Do not commit:

- `outputs/`
- `wandb/`
- `data/processed/`
- model checkpoints or weights
- `.pt`, `.npy`, `.npz`, `.pkl`, `.pyc`, or cache files
- executed notebooks
- local logs and paper-style JSON summaries

The `outputs/` directory is for local training and evaluation results. It should stay out of Git.

Small curated public images may be committed under `assets/figures/` when they are intentionally
selected for the README or a demo. Trained model directories should contain only lightweight
metadata such as `trained_models/README.md` or `trained_models/model_registry.yaml`, not weights.

## Model Registry Contributions

The trained-model registry is metadata only. To add or promote a new candidate:

1. Train the candidate locally and keep checkpoints under local `outputs/` paths.
2. Evaluate it with the relevant path metrics, token metrics, no-leakage checks, and notebook or
   reproduction workflow.
3. Add or update metadata under `trained_models/model_registry.yaml` and the relevant
   `trained_models/<experiment>/model_card.md`.
4. Record config paths, local checkpoint conventions, selection profile, visible metrics, missing
   metrics, sampling policy, and caveats. Use `local_outputs_only` or relative `outputs/...`
   conventions rather than absolute paths.
5. Do not commit weights, token tensors, processed data, generated summaries, figures, or W&B
   exports.
6. Run the selector before committing, for example:

```bash
poetry run python scripts/select_registered_model.py --experiment sp500_vix --family discrete
poetry run python scripts/select_registered_model.py --experiment sp500_vix --family discrete --metric mmd
```

Do not change public defaults unless the registry contains explicit metadata justifying the
selection and the corresponding documentation explains the trade-off.

## Code Quality

Prefer narrow, local changes. Preserve public commands and notebook workflows when refactoring.

Guidelines:

- Use NumPy-style docstrings for public modules, classes, and functions.
- Add type hints to public APIs and avoid unnecessary `Any`.
- Prefer dataclasses or typed config objects over unstructured dictionaries.
- Keep no-anticipation assumptions explicit in causal models and checks.
- Use repository helpers and established local patterns before adding new abstractions.
- Avoid absolute local paths in committed code or documentation.

## Versioning / Branches

The package version is configured in `pyproject.toml`, and Commitizen is configured for
conventional commits. Public branches should be release-oriented and minimal. Research branches
may contain fuller experiment history, verification notes, and candidate configs.

Suggested branch roles:

- `main`: stable baseline branch.
- `cleanup/*`: public branch preparation and packaging cleanup.
- `feat/*`: public feature work intended to be reviewed and retained.
- `research/*`: detailed experiments, ablations, evidence, and exploratory configs.

## Git Workflow

Use focused commits with conventional commit messages. If Commitizen is available, prefer:

```bash
poetry run cz commit
```

Otherwise use conventional commit messages directly:

```bash
git commit -m "docs: rewrite public readme"
git commit -m "fix(token-prior): preserve causal mask shape"
git commit -m "chore(repo): clean public minimal branch"
```

Before committing, inspect the diff and check that no generated artefacts are staged:

```bash
git status --short
git diff --stat
git ls-files | grep -E '(^outputs/|^wandb/|^data/processed/|\.npy$|\.npz$|\.pt$|\.pkl$|\.pyc$|__pycache__)' || true
```

## Research Branch Policy

Research branches may keep detailed verification docs, ablation runners, optional configs, and
evidence summaries when they are useful for ongoing analysis. Do not remove research history from
research branches while preparing a public branch.

Public branches should keep only the source, notebooks, configs, scripts, and documentation needed
for the public baseline and any explicitly documented demo. RVQ q2, hidden128 variants, diffusion,
signature-kernel experiments, and other exploratory paths should remain on research branches
unless they are deliberately promoted.

## Future Directions

Future work should remain staged, evidence-backed, and separate from public defaults until it is
ready for review. Useful directions include:

- Per-experiment model selection and periodic registry refreshes, with clear metrics, caveats, and model cards.
- Pytest coverage: add unit and smoke tests for causal no-leakage checks, dataset tensor contracts, Ogata Hawkes simulation invariants, log-return-to-price conversion, model-registry selection, token-prior sampling shapes, and notebook-safe command generation. Integration tests should remain lightweight and should not require trained checkpoints.
- Stronger causal priors for hidden128 tokens, including more robust conv-transformer variants. Mamba or selective-SSM priors should wait for CUDA/package compatibility and strict stepwise-causality checks.
- Continuous-latent prior extensions such as LSGM, score-based, or flow-based alternatives to RealNVP, kept separate from the discrete branch.
- VQ-family tokenizer experiments such as GroupedResidualVQ and MGVQ, only after prior-calibration bottlenecks are controlled.
- Causal low/high-frequency decomposition inspired by TimeVQVAE, without introducing bidirectional priors.
- Optional signature and path-space diagnostics, including log-signature conditioning and signature-kernel metrics, without making them public defaults.
- Causal/adapted distances and downstream finance evaluations, including adapted Wasserstein, Deep Hedging, VaR/Expected Shortfall, and multistage portfolio or hedging stress tests.
- Public model releases where registry metadata stays in Git and weights or checkpoints are distributed through external release assets, not committed.
