# PyPI Release Audit

This document records a repository audit for the first TestPyPI/PyPI release of
`time-causal-vae`. It is an audit only: no package was published, no models were trained, and no
notebooks were run.

## Metadata Findings

- Distribution name: `time-causal-vae` in `pyproject.toml`. This matches the intended public
  package name and should remain unchanged for the release.
- Import package name: `time_causal_vae`, declared through
  `packages = [{ include = "time_causal_vae", from = "src" }]` and exported from
  `src/time_causal_vae/__init__.py`.
- Version: `0.1.0` appears in both `pyproject.toml` and `src/time_causal_vae/version.py`.
  Commitizen currently updates only `pyproject.toml`, so these two sources can drift.
- Licence: `GPL-3.0-only` is declared in `pyproject.toml`, and the repository includes a full
  `LICENSE` file for GNU GPL version 3.
- Authors: `Georgios Vakis <georgios.vakis@iacm.forth.gr>` is declared in `pyproject.toml`.
- Maintainers: no separate `maintainers` metadata is declared.
- Description: present as `Clean Python-package rewrite of Time-Causal VAE for financial
  time-series generation.`
- Python requirement: `>=3.11,<3.13` is declared. No Python version classifiers are declared.
- Project URLs: no `[project.urls]` entries are declared.
- Keywords: present as `time-series`, `vae`, `finance`, `generative-models`, and `pytorch`.
- Console scripts: eight `tcvae-*` entry points are declared and all respond to `--help`.
- README long description: `readme = "README.md"` is declared.
- Optional dependency groups: Poetry groups are defined for `dev`, `notebooks`, `tracking`, and
  `data`; only `data` is explicitly marked optional. These groups are not published as PEP 621
  extras, so `pip install time-causal-vae[data]` is not advertised by the current metadata.

## Missing Metadata

- Add Python trove classifiers, at least for Python 3.11 and Python 3.12, plus a licence
  classifier if desired.
- Add `[project.urls]`, for example `Homepage`, `Repository`, `Issues`, and `Documentation`.
- Consider adding `maintainers` if PyPI ownership or release responsibility should be explicit.
- Consider making the version single-source, or add `src/time_causal_vae/version.py` to
  Commitizen `version_files`.
- Consider moving `types-pyyaml` from runtime dependencies to the development group because it is
  a typing stub package, not a runtime requirement.

## README Findings

- Stable versus experimental status is stated clearly. The README keeps S&P500/VIX as the public
  default and labels Hawkes/SVMHJD and multidimensional benchmarks as optional or experimental.
- The README does not claim a selected multidimensional model. It explicitly states that no
  multidimensional generator is selected in `trained_models/model_registry.yaml`.
- Local-only empirical data policy is documented for S&P500/VIX and the S&P500 50-stock panel.
  Downloaded Yahoo-backed data, processed tensors, checkpoints, W&B runs, and local outputs are
  described as non-committed artefacts.
- Installation instructions are present for Poetry users. For PyPI users, add a direct
  `pip install time-causal-vae` instruction once the package is published.
- Quickstart commands are present, but most examples are repo-relative and use `poetry run`.
  A PyPI-installed user will not automatically have `configs/`, `scripts/`, or local data paths
  unless they also clone the repository or these resources are packaged.
- Upstream TC-VAE is cited and acknowledged in the README references, including the arXiv DOI and
  upstream code repository.
- The README uses repository-relative image paths:
  - `assets/figures/time_causal_vqvae_pipeline.svg`
  - `assets/figures/sp500_vix_best_research_paths.png`
  - `assets/figures/sp500_vix_hidden128_codebook_voronoi.png`
  - `assets/figures/hawkes_jump_ogata_jump_raster.png`
  - `assets/figures/hawkes_jump_model_metric_comparison.png`
  - `assets/figures/hawkes_jump_tail_jump_comparison.png`

For PyPI rendering, either keep these relative paths as GitHub-first documentation and accept a
text-first PyPI page, or switch the image links to absolute raw GitHub URLs before publishing.

## Packaging Risks

- The wheel is configured to include the Python package under `src/time_causal_vae`. There are no
  explicit include rules for `configs/`, `scripts/`, `trained_models/`, `docs/`, `notebooks/`, or
  `assets/figures/`. This is a risk because the README quickstarts reference those paths.
- No `MANIFEST.in`, `setup.py`, or `setup.cfg` was found. Packaging is controlled by
  `pyproject.toml` and Poetry Core.
- `poetry.lock` is tracked even though `.gitignore` says it is excluded to prevent version-locking
  conflicts in this environment. This is not automatically a PyPI blocker, but the policy should
  be made consistent before release.
- The README badge says Python 3.11+, while `pyproject.toml` caps supported Python below 3.13.
  The badge is broadly compatible but less precise than the package metadata.
- CLI `--help` imports Matplotlib for several commands and emits a temporary-cache warning when
  the default Matplotlib config directory is not writable. This did not fail any command, but it is
  noisy for first-run CLI help.

## Files To Exclude

The generated-output and binary/cache scan returned no tracked files:

```bash
git ls-files | grep -E '(^outputs/|^wandb/|^data/raw/|^data/processed/|\.npy$|\.npz$|\.pt$|\.pkl$|\.pyc$|__pycache__)' || true
```

The local/tooling scan returned one tracked file:

```text
poetry.lock
```

This came from:

```bash
git ls-files | grep -E '(^\.codex/|^\.agents/|^\.vscode/|^\.editorconfig$|^poetry.lock$)' || true
```

No `.codex/`, `.agents/`, `.vscode/`, or `.editorconfig` paths were tracked.

## Console Script Audit

All declared console scripts responded to `--help` with exit code 0:

- `tcvae-train`
- `tcvae-train-tokenizer`
- `tcvae-train-token-prior`
- `tcvae-evaluate`
- `tcvae-evaluate-tokenizer`
- `tcvae-evaluate-token-prior`
- `tcvae-inspect-checkpoint`
- `tcvae-select-model`

Several help commands printed a Matplotlib cache warning because
`/home/georgios-vourvachakis/.config/matplotlib` was not writable and Matplotlib created a
temporary cache under `/tmp`.

## Recommended Version

Use `0.1.0` for the first TestPyPI release and the first PyPI release if that version has not
already been uploaded to the target index. If `0.1.0` is uploaded to TestPyPI for rehearsal and
then needs another rehearsal, increment to a post-release or patch version before re-uploading to
the same index.

## Commands Run

```bash
poetry check
poetry run ruff format docs
poetry run ruff check docs --fix
```

Result: `poetry check` passed with `All set!`; Ruff found no Python files under `docs`, emitted
that warning for both commands, and `ruff check` reported `All checks passed!`

```bash
git ls-files | grep -E '(^outputs/|^wandb/|^data/raw/|^data/processed/|\.npy$|\.npz$|\.pt$|\.pkl$|\.pyc$|__pycache__)' || true
git ls-files | grep -E '(^\.codex/|^\.agents/|^\.vscode/|^\.editorconfig$|^poetry.lock$)' || true
```

Result: no generated-output or binary/cache matches; `poetry.lock` matched the local/tooling scan.

## Commands To Run Next

Before TestPyPI:

```bash
poetry check
poetry build
poetry run twine check dist/*
python -m venv /tmp/time-causal-vae-test
/tmp/time-causal-vae-test/bin/python -m pip install --upgrade pip
/tmp/time-causal-vae-test/bin/python -m pip install dist/*.whl
/tmp/time-causal-vae-test/bin/tcvae-train --help
/tmp/time-causal-vae-test/bin/tcvae-select-model --help
```

For TestPyPI upload rehearsal only:

```bash
poetry publish -r testpypi
```

For PyPI, publish only after the TestPyPI wheel installs in a clean environment and the README
rendering policy for images has been accepted or updated.
