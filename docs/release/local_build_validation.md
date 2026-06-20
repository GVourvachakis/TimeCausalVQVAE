# Local Build Validation

This document records the local package build and clean-wheel install smoke for
`time-causal-vae` version `0.1.0a1`. No package was published, no model was trained, and no
notebook was run.

## Build

Old build artefacts were removed with:

```bash
rm -rf dist build *.egg-info
```

The package was rebuilt with:

```bash
poetry build
```

Built artefacts:

- `dist/time_causal_vae-0.1.0a1-py3-none-any.whl`
- `dist/time_causal_vae-0.1.0a1.tar.gz`

The rebuilt files were approximately:

- wheel: `342K`
- source distribution: `257K`

## Package Contents

Package manifests were written to:

- `/tmp/time_causal_vae_sdist_files.txt`
- `/tmp/time_causal_vae_wheel_files.txt`

Manifest sizes:

- source distribution: `181` entries
- wheel: `183` entries, including the `WHEEL ...` header line in the captured manifest

Forbidden-file scan:

```bash
rg -n '(^|/)(outputs|wandb|data/raw|data/processed)(/|$)|\.(pt|pkl|npy|npz|pyc)$|__pycache__' \
  /tmp/time_causal_vae_sdist_files.txt \
  /tmp/time_causal_vae_wheel_files.txt || true
```

Result: no matches. The additional notebook scan for `.ipynb` files also returned no matches.

## Twine Check

`twine` was not initially available in the Poetry environment:

```text
Command not found: twine
```

Because Twine is release-validation tooling, it was added to the Poetry `dev` group and installed
into the Poetry environment. The subsequent check passed:

```bash
poetry run twine check dist/*
```

Result:

```text
Checking dist/time_causal_vae-0.1.0a1-py3-none-any.whl: PASSED
Checking dist/time_causal_vae-0.1.0a1.tar.gz: PASSED
```

## Clean Venv Install

A temporary venv outside the repository was created at:

```text
/tmp/tcvae-wheel-test
```

The initial sandboxed install could not resolve package dependencies from PyPI. After network
approval, the wheel and runtime dependencies installed successfully:

```bash
/tmp/tcvae-wheel-test/bin/python -m pip install dist/*.whl
```

The install selected `torch-2.12.1` for Python 3.12 on Linux, which pulled large GPU-enabled
PyTorch dependency wheels. This made the clean install slow but did not block validation.

## Import Smoke

The first clean import smoke exposed a missing public import name:

```text
ImportError: cannot import name 'load_model_registry' from 'time_causal_vae.experiments.model_registry'
```

The module already provided `load_registry`; a minimal compatibility alias named
`load_model_registry` was added and the package was rebuilt. The rebuilt wheel was reinstalled into
the temporary venv with:

```bash
/tmp/tcvae-wheel-test/bin/python -m pip install --force-reinstall --no-deps dist/*.whl
```

The requested import smoke then passed:

```bash
/tmp/tcvae-wheel-test/bin/python - <<'PY'
import time_causal_vae
from time_causal_vae.experiments.model_registry import load_model_registry
print("time_causal_vae import ok")
PY
```

Result:

```text
time_causal_vae import ok
```

## Console Script Smoke

The requested console entry points all responded to `--help` with exit code `0`:

- `/tmp/tcvae-wheel-test/bin/tcvae-train --help`
- `/tmp/tcvae-wheel-test/bin/tcvae-evaluate --help`
- `/tmp/tcvae-wheel-test/bin/tcvae-train-tokenizer --help`
- `/tmp/tcvae-wheel-test/bin/tcvae-train-token-prior --help`

Each command printed the expected argparse help. Commands that import Matplotlib also printed the
known warning that `/home/georgios-vourvachakis/.config/matplotlib` is not writable and that a
temporary cache directory was created under `/tmp`.

## Blockers

No release-blocking package-content issue remains from this local build validation.

Resolved during validation:

- Added `twine` to the Poetry `dev` group so `poetry run twine check dist/*` is available.
- Added `load_model_registry` as a compatibility alias for the clean install import smoke.

Follow-up candidate:

- Consider constraining or documenting CPU-only PyTorch install guidance for users who do not want
  pip to fetch GPU-enabled PyTorch dependency wheels during clean installs.
