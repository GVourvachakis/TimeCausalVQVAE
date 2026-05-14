# Signature Package Compatibility Results

Status: isolated compatibility probe. No dependencies were added to `pyproject.toml`, no package
source was modified, no models were trained, and no environment-specific outputs were committed.

## Environment

Repository branch:

- `research/signature-conditioning`

Repository dependency bounds from `pyproject.toml`:

- Python: `>=3.11,<3.13`
- PyTorch: `>=2.5,<3.0`

Current shell and Poetry environment:

```text
python --version
Python 3.12.3

poetry run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
2.11.0+cu130
False
```

The Poetry PyTorch probe also emitted a CUDA driver warning: the installed NVIDIA driver version
was reported as `12060`, while the installed PyTorch CUDA build expects a newer compatible driver.
All local GPU checks therefore fall back to CPU or fail at CUDA device discovery.

## Poetry environment import check

Command:

```bash
poetry run python - <<'PY'
packages = ["signatory", "iisignature", "sigkernel", "ksig", "pathsig"]
for pkg in packages:
    try:
        mod = __import__(pkg)
        print(pkg, "OK", getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        print(pkg, "MISSING_OR_FAILED", repr(exc))
PY
```

Result:

| Package | Poetry import status |
| --- | --- |
| `signatory` | `MISSING_OR_FAILED ModuleNotFoundError("No module named 'signatory'")` |
| `iisignature` | `MISSING_OR_FAILED ModuleNotFoundError("No module named 'iisignature'")` |
| `sigkernel` | `MISSING_OR_FAILED ModuleNotFoundError("No module named 'sigkernel'")` |
| `ksig` | `MISSING_OR_FAILED ModuleNotFoundError("No module named 'ksig'")` |
| `pathsig` | `MISSING_OR_FAILED ModuleNotFoundError("No module named 'pathsig'")` |

This confirms that the project environment remains free of signature-package dependencies.

## Isolated install probes

All install probes used temporary environments under:

```text
/tmp/timecausalvqvae-signature-probe/
```

The first networked `pip` command failed inside the sandbox with DNS resolution errors and was
rerun with approved network access. The reruns remained isolated to `/tmp`.

### `iisignature`

Environment:

- Path: `/tmp/timecausalvqvae-signature-probe/iisignature`
- Python: `3.12.3`
- NumPy installed by probe: `2.4.4`
- Package version: `iisignature 0.24`

Install status:

- `python -m pip install numpy iisignature` initially failed during source-build metadata
  generation because the isolated build environment could not import NumPy.
- Installing NumPy first and retrying with `--no-build-isolation` succeeded:

```bash
/tmp/timecausalvqvae-signature-probe/iisignature/bin/python -m pip install numpy
/tmp/timecausalvqvae-signature-probe/iisignature/bin/python -m pip install \
  iisignature \
  --no-build-isolation
```

Minimal computation:

```text
x sig_shape (6,) logsig_shape (3,) finite True
y sig_shape (6,) logsig_shape (3,) finite True
version 0.24
```

Interpretation:

- CPU signature and log-signature feature extraction works in an isolated Python 3.12 environment.
- The install path is not smooth because a source-build workaround is required.
- No GPU or PyTorch tensor support was observed or assumed.

Recommendation:

- Use `iisignature` as the leading candidate for offline CPU signature and log-signature feature
  extraction, subject to accepting the source-build workaround or finding a cleaner wheel path.
- Do not add it as a dependency until an implementation thread decides whether CPU-only feature
  extraction is acceptable.

### `signatory`

Environment:

- Path: `/tmp/timecausalvqvae-signature-probe/signatory`
- Python: `3.12.3`

Install status:

- `python -m pip install "torch==1.11.0"` failed because no Python 3.12-compatible wheel was
  available from the configured index.
- `python -m pip install "signatory==1.2.7.1.11.0"` failed because that exact pinned wheel was not
  available from the configured index.
- `python -m pip install signatory` selected `signatory-1.2.6.1.9.0.tar.gz`, but metadata
  generation failed because Signatory requires PyTorch to be installed before building.

Minimal computation:

- Not run. The package did not install.

Interpretation:

- The package is not compatible with the current Python 3.12 probe path without a legacy PyTorch
  environment or a more carefully matched custom build.
- The result is consistent with treating Signatory as inspection-only for this project phase.

Recommendation:

- Defer `signatory`.
- Do not downgrade Python or PyTorch for this project to satisfy Signatory.

### `sigkernel`

Environment:

- Path: `/tmp/timecausalvqvae-signature-probe/sigkernel`
- Python: `3.12.3`
- PyTorch installed by probe: `2.12.0+cu130`
- `torch.cuda.is_available()`: `False`
- Package version: `sigkernel 0.0.1`
- GitHub commit resolved by pip: `40a583155ea8d2194af0e90dddab37e2659cfcfd`

Install status:

- Installing `torch>=2.5,<3.0`, NumPy, and SciPy succeeded, but pulled a large CUDA-enabled
  PyTorch stack into the isolated `/tmp` environment.
- `python -m pip install "git+https://github.com/crispitagorico/sigkernel.git"` initially failed
  because setup imported Cython during metadata generation.
- Installing Cython and retrying with `--no-build-isolation` succeeded:

```bash
/tmp/timecausalvqvae-signature-probe/sigkernel/bin/python -m pip install Cython
/tmp/timecausalvqvae-signature-probe/sigkernel/bin/python -m pip install \
  "git+https://github.com/crispitagorico/sigkernel.git" \
  --no-build-isolation
```

Minimal computation:

```text
sigkernel_version unknown
torch 2.12.0+cu130 cuda False
api_has ['RBFKernel', 'SigKernel', 'LinearKernel']
gram [[5.9657258795219406, 4.0564340124271325], [4.0564340124271325, 6.7218390509526955]]
finite True
symmetric True
diag_positive True
```

Interpretation:

- CPU signature-kernel Gram computation worked on two toy paths.
- The result was finite, symmetric, and positive on the diagonal.
- Installation required a Cython and `--no-build-isolation` workaround.
- GPU execution was not verified because local CUDA device discovery failed.

Recommendation:

- Use `sigkernel` as the leading candidate for evaluation-only signature-kernel metrics.
- Keep it out of `pyproject.toml` until the install workaround, dependency footprint, and CPU/GPU
  expectations are accepted explicitly.

### `KSig`

Environment:

- Path: `/tmp/timecausalvqvae-signature-probe/ksig`
- Python: `3.12.3`
- NumPy installed before workaround: `2.4.4`
- Package version: `ksig 1.0.0` after no-dependencies workaround
- GitHub commit resolved by pip: `0700e8a3a7dc07333d70b281dcf688f4e3dd9503`
- CuPy installed during workaround: `cupy 14.0.1`

Install status:

- Normal installation from GitHub failed because `KSig` requested `numpy==1.24.4`, and that NumPy
  version failed to build under Python 3.12 in this environment.
- A no-dependencies package install succeeded:

```bash
/tmp/timecausalvqvae-signature-probe/ksig/bin/python -m pip install \
  "git+https://github.com/tgcsaba/ksig.git" \
  --no-deps
```

- Import then failed because `cupy` was missing.
- Installing `cupy-cuda12x` succeeded, but CUDA device discovery failed:

```text
cupy 14.0.1
cupy_device_count_failed CUDARuntimeError('cudaErrorNoDevice: no CUDA-capable device is detected')
```

- Installing `numba` resolved the next missing import, but the toy computation still failed at
  CUDA runtime discovery:

```text
ksig_version unknown
ksig_attrs ['static', 'kernels']
SMOKE_FAILED CUDARuntimeError('cudaErrorNoDevice: no CUDA-capable device is detected')
```

Minimal computation:

- Not successful. `KSig` imported only after no-dependencies and extra CuPy/Numba workarounds, and
  the Gram-matrix smoke test failed because no CUDA-capable device was detected.

Interpretation:

- The package currently looks GPU-oriented in practice.
- Normal dependency resolution is incompatible with this Python 3.12 probe because of the
  `numpy==1.24.4` requirement.
- The local CUDA situation prevents meaningful numerical comparison against `sigkernel`.

Recommendation:

- Defer `KSig` as a project dependency.
- Revisit only in a GPU-matched environment and preferably with a Python version compatible with
  its pinned NumPy dependency.

### `pathsig`

Install status:

- Not probed beyond the Poetry import check.

Interpretation:

- The package remains optional and experimental for this project phase.

Recommendation:

- Defer `pathsig` unless a later implementation thread identifies a concrete advantage over
  `iisignature` for feature extraction or `sigkernel` for evaluation.

## Signature-kernel numerical comparison

The planned rough comparison between `sigkernel` and `KSig` was not completed because `KSig` did
not produce a toy Gram matrix on this machine.

Observed `sigkernel` behaviour on the same two-path toy problem:

- finite output: yes;
- symmetric Gram matrix: yes;
- positive diagonal: yes.

Observed `KSig` behaviour:

- normal install: failed under Python 3.12 because of `numpy==1.24.4`;
- no-dependencies import path: required CuPy and Numba workarounds;
- toy Gram matrix: failed with `cudaErrorNoDevice`.

No exact or approximate numerical agreement claim is made.

## Summary table

| Package | Install status | Import status | Minimal computation | Recommendation |
| --- | --- | --- | --- | --- |
| `iisignature` | Succeeded with NumPy-first and `--no-build-isolation` workaround. | OK in isolated env. | Signature and log-signature finite on two toy paths. | Candidate for CPU offline feature extraction; do not add yet. |
| `signatory` | Failed on Python 3.12 legacy PyTorch and wheel constraints. | Not installed. | Not run. | Defer. |
| `sigkernel` | Succeeded with Cython and `--no-build-isolation` workaround. | OK in isolated env. | Signature-kernel Gram matrix finite, symmetric, positive diagonal. | Candidate for evaluation-only metric; do not add yet. |
| `KSig` | Normal install failed; no-deps workaround imported only after CuPy and Numba. | Partial import only. | Failed with `cudaErrorNoDevice`. | Defer until GPU-matched environment. |
| `pathsig` | Not probed beyond Poetry import. | Missing in Poetry env. | Not run. | Defer. |

## Final recommendation

For the next documentation or implementation phase:

1. Use `iisignature` first for CPU log-signature feature-extraction smoke tests if source builds
   are acceptable.
2. Use `sigkernel` first for evaluation-only signature-kernel distances, after documenting the
   Cython and `--no-build-isolation` install workaround.
3. Defer `signatory` because it does not fit the current Python 3.12 and PyTorch 2.x project path.
4. Defer `KSig` until a GPU-compatible machine and a Python/NumPy combination compatible with its
   dependency pins are available.
5. Defer `pathsig` until there is a specific reason to inspect it.

No signature package should be added to `pyproject.toml` from this probe alone.

## Final package decision

The project will prepare implementation around two optional packages only:

- `iisignature` for offline CPU truncated signature and log-signature feature extraction.
- `sigkernel` for evaluation-only signature-kernel metrics.

The project will not add `signatory` or `KSig` in the current phase. `signatory` remains
reference-only because the probe did not fit the current Python/PyTorch path. `KSig` remains
deferred until a GPU-compatible environment and compatible Python/NumPy setup are available.
`pathsig` remains untested and deferred.
