# PyPI Release Plan

This plan describes the final PyPI release procedure for `time-causal-vae` version `0.1.0`. It
does not publish automatically and does not include tokens.

## Release Status

Version `0.1.0` is live on PyPI:

```text
https://pypi.org/project/time-causal-vae/
```

The post-upload smoke install from PyPI passed in a clean Python 3.12 virtual environment:

```bash
/tmp/tcvae-pypi/bin/python -m pip install time-causal-vae==0.1.0
/tmp/tcvae-pypi/bin/python -c "import time_causal_vae; print(time_causal_vae.__version__)"
```

The import check reported:

```text
0.1.0
```

## Post-Release Main State

The `v0.1.0` tag and PyPI distribution remain the immutable release artefacts for this version.
After the successful release, the source tree has moved to `0.1.1.dev0` for post-0.1.0
development. The post-release device hardening work is therefore expected to ship in a future patch
release rather than changing the already-published `0.1.0` package.

The PyPI long description for the published `0.1.0` distribution will update only when a future
package release is uploaded.

## Preconditions

- TestPyPI validation for the `0.1.0a1` candidate has passed.
- For the completed `v0.1.0` release tag, `pyproject.toml`,
  `src/time_causal_vae/version.py`, and package metadata all reported `0.1.0`.
- Release notes exist at `docs/release/0.1.0.md`.
- The working tree is clean except for intentionally ignored local build artefacts under `dist/`.
- No downloaded data, generated outputs, notebooks with outputs, weights, checkpoints, or virtual
  environments are staged.

## Build And Validate

Run from the repository root:

```bash
rm -rf dist build *.egg-info
poetry build
poetry run twine check dist/*
poetry check
poetry run ruff format pyproject.toml README.md docs
poetry run ruff check src scripts configs docs --fix
poetry run mypy src/time_causal_vae
```

Expected artefacts:

- `dist/time_causal_vae-0.1.0-py3-none-any.whl`
- `dist/time_causal_vae-0.1.0.tar.gz`

Do not commit `dist/`.

## Manual PyPI Publish

Manual publishing uses a locally configured PyPI identity. Do not commit `.pypirc`, tokens,
passwords, or environment files.

Final dry check:

```bash
poetry run twine check dist/*
```

Publish with Poetry, using a token supplied only for the current shell:

```bash
POETRY_PYPI_TOKEN_PYPI=<pypi-token> poetry publish
```

Alternatively, if credentials are already configured locally in Poetry or the system keyring:

```bash
poetry publish
```

Publish with Twine:

```bash
poetry run twine upload dist/*
```

If using a named repository alias with Twine, configure it outside the repository and then run:

```bash
poetry run twine upload --repository pypi dist/*
```

For a first upload of a new PyPI project, prefer an account-scoped API token. A project-scoped
token only works after the project already exists and the token has been granted access to that
project.

If PyPI returns `403 Invalid or non-existent authentication information`, check:

- the token was copied with its full `pypi-` prefix;
- the token is for PyPI, not TestPyPI;
- the publishing account has a verified email address;
- the token scope is valid for `time-causal-vae`;
- no stale `~/.pypirc`, Poetry credential, keyring entry, or `~/.netrc` entry is overriding the
  intended token.

To replace a stale local Poetry token, run this outside the repository with a fresh PyPI token:

```bash
poetry config pypi-token.pypi <pypi-token>
poetry publish
```

To avoid storing the token in Poetry configuration, prefer the one-command environment variable
form:

```bash
POETRY_PYPI_TOKEN_PYPI=<pypi-token> poetry publish
```

## Trusted Publishing Procedure

Trusted Publishing avoids repository-stored tokens. Configure the PyPI publisher manually in the
PyPI project settings before publishing.

Suggested PyPI publisher settings if a production workflow is added:

- Project: `time-causal-vae`
- Owner: `GVourvachakis`
- Repository: `TimeCausalVQVAE`
- Workflow filename: `publish-pypi.yml`
- Environment name: leave blank unless the workflow explicitly declares one.

The existing `.github/workflows/publish-testpypi.yml` workflow is for TestPyPI only because it
uses:

```text
repository-url: https://test.pypi.org/legacy/
```

Do not reuse that workflow for production PyPI unless the repository URL and Trusted Publishing
project configuration are deliberately changed for a production workflow.

If a production workflow is added later, it should:

- trigger manually or from a final tag such as `v0.1.0`;
- request `id-token: write`;
- run `poetry check`;
- run the release validation checks;
- build the package;
- run `poetry run twine check dist/*`;
- publish through `pypa/gh-action-pypi-publish` without any token or password.

## Post-Release Token Hygiene

After the first upload of the new project succeeds:

- revoke the temporary account-scoped PyPI token used for the initial release;
- create a new token scoped only to the `time-causal-vae` project;
- keep the project-scoped token outside the repository;
- prefer one-command environment variable use for manual future uploads.

## Post-Install Check

After the PyPI upload completes, test installation from a fresh venv outside the repository:

```bash
python -m venv /tmp/tcvae-pypi
/tmp/tcvae-pypi/bin/python -m pip install --upgrade pip
/tmp/tcvae-pypi/bin/python -m pip install time-causal-vae==0.1.0
/tmp/tcvae-pypi/bin/python -c "import time_causal_vae; print(time_causal_vae.__version__)"
```

Optional console smoke checks:

```bash
/tmp/tcvae-pypi/bin/tcvae-train --help
/tmp/tcvae-pypi/bin/tcvae-evaluate --help
/tmp/tcvae-pypi/bin/tcvae-train-tokenizer --help
/tmp/tcvae-pypi/bin/tcvae-train-token-prior --help
```

These checks must not require local data, checkpoints, generated outputs, or notebooks.

## GitHub Release Checklist

Before creating the GitHub release:

- Confirm the final commit includes `pyproject.toml`, `src/time_causal_vae/version.py`, README
  install wording, release notes, and this release plan.
- Confirm `git status --short` does not show staged generated artefacts.
- Confirm `git ls-files` does not include `dist/`, `outputs/`, `wandb/`, `data/raw/`,
  `data/processed/`, checkpoints, `.pt`, `.pkl`, `.npy`, or `.npz` artefacts.
- Create an annotated tag after the release commit is ready:

```bash
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0
```

- Create a GitHub release for `v0.1.0`.
- Use `docs/release/0.1.0.md` as the release-note source.
- Do not attach generated local outputs, trained weights, processed data, or notebooks with
  outputs.
- If distributing weights in the future, use explicit release assets or another external artefact
  channel and document their provenance separately.

## Stop Point

The `0.1.0` upload and import smoke have completed. Future uploads still require explicit human
approval and must not be run automatically.
