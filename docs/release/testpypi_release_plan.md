# TestPyPI Release Plan

This plan describes how to publish `time-causal-vae` version `0.1.0a1` to TestPyPI. It does not
include tokens and should not be used to store credentials in the repository.

## Pre-Release Checks

Start from a clean release branch and rebuild local artefacts:

```bash
rm -rf dist build *.egg-info
poetry check
poetry build
poetry run twine check dist/*
```

Expected artefacts:

- `dist/time_causal_vae-0.1.0a1-py3-none-any.whl`
- `dist/time_causal_vae-0.1.0a1.tar.gz`

Do not commit `dist/`.

## Option A: Manual TestPyPI Upload

Manual upload uses Twine and a locally configured TestPyPI identity. Do not commit `.pypirc`,
tokens, passwords, or environment files.

```bash
poetry build
poetry run twine check dist/*
poetry run twine upload --repository testpypi dist/*
```

If the `testpypi` repository alias is not configured locally, configure it outside the repository,
for example in user-level Twine or keyring configuration. Keep credentials out of Git.

## Option B: Trusted Publishing

Trusted Publishing avoids repository-stored tokens. Configure the TestPyPI publisher manually in
the TestPyPI project settings before using the workflow.

Suggested TestPyPI publisher settings:

- Project: `time-causal-vae`
- Owner: `GVourvachakis`
- Repository: `TimeCausalVQVAE`
- Workflow filename: `publish-testpypi.yml`
- Environment name: leave blank. The workflow does not declare a GitHub Actions environment, which
  avoids editor/schema validation issues around `jobs.<job_id>.environment`.

The workflow is:

```text
.github/workflows/publish-testpypi.yml
```

Triggers:

- manual `workflow_dispatch`;
- tag pushes matching `v0.1.0a*`, for example `v0.1.0a1`.

The workflow builds the package, runs `poetry check`, runs `twine check`, and publishes to
TestPyPI through `pypa/gh-action-pypi-publish` with:

```text
repository-url: https://test.pypi.org/legacy/
```

No token or password should be added to the workflow. The workflow requires GitHub Actions OIDC
permission:

```yaml
permissions:
  contents: read
  id-token: write
```

## Tag-Based Rehearsal

After the TestPyPI publisher is configured, create and push an alpha tag when ready:

```bash
git tag v0.1.0a1
git push origin v0.1.0a1
```

Alternatively, run the workflow manually from the GitHub Actions UI.

## Post-Upload Install Check

After the TestPyPI upload completes, test installation from a fresh venv outside the repository:

```bash
python -m venv /tmp/tcvae-testpypi
/tmp/tcvae-testpypi/bin/python -m pip install --upgrade pip
/tmp/tcvae-testpypi/bin/python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ time-causal-vae==0.1.0a1
/tmp/tcvae-testpypi/bin/python -c "import time_causal_vae; print('ok')"
```

Optional console smoke checks:

```bash
/tmp/tcvae-testpypi/bin/tcvae-train --help
/tmp/tcvae-testpypi/bin/tcvae-evaluate --help
/tmp/tcvae-testpypi/bin/tcvae-train-tokenizer --help
/tmp/tcvae-testpypi/bin/tcvae-train-token-prior --help
```

Do not require local market data, checkpoints, outputs, or notebooks for this smoke.

## Roll-Forward Notes

TestPyPI versions are immutable once uploaded. If `0.1.0a1` needs another rehearsal after upload,
increment to the next alpha version before rebuilding, for example `0.1.0a2`.

Do not publish to PyPI until the TestPyPI install smoke passes and the release notes have been
reviewed.
