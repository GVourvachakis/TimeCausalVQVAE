# Signature Package Compatibility Plan

Status: verification checklist only. Do not run package installs in the project environment and do
not add dependencies before this checklist is executed in temporary environments.

## Baseline environment

The repository declares:

- Python requirement: `>=3.11,<3.13`.
- PyTorch requirement: `>=2.5,<3.0`.

The current local Poetry environment observed during this documentation pass was:

- Python version: `Python 3.12.3`.
- PyTorch version: `2.11.0+cu130`.
- `torch.version.cuda`: `13.0`.
- `torch.cuda.is_available()`: `False`.
- CUDA note: PyTorch reported that the local NVIDIA driver was too old for the installed CUDA
  runtime, so GPU checks must have a CPU fallback and a separate GPU-compatible rerun.

## Temporary environment convention

Use throwaway environments outside the project. Do not run these commands inside the Poetry
environment and do not edit `pyproject.toml` during compatibility checks.

```bash
python -m venv /tmp/tcvqvae-signatures
/tmp/tcvqvae-signatures/bin/python -m pip install --upgrade pip setuptools wheel
/tmp/tcvqvae-signatures/bin/python -m pip install "torch>=2.5,<3.0" numpy scipy scikit-learn
```

For GPU-specific checks, create a separate environment that matches the installed NVIDIA driver,
CUDA toolkit, and package-specific requirements. Record the driver, CUDA runtime, PyTorch build,
and package version in the compatibility log.

## Environment probes

Run these probes before installing each candidate package:

```bash
/tmp/tcvqvae-signatures/bin/python --version
/tmp/tcvqvae-signatures/bin/python - <<'PY'
import platform
import torch

print("platform:", platform.platform())
print("python:", platform.python_version())
print("torch:", torch.__version__)
print("torch_cuda_build:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_name:", torch.cuda.get_device_name(0))
PY
```

Expected outcome:

- CPU checks must run when `torch.cuda.is_available()` is `False`.
- GPU checks are optional unless the environment has a compatible CUDA driver and package build.
- A package that imports only on GPU is not acceptable as the first project dependency unless a
  documented CPU fallback is provided elsewhere.

## Package install commands

Run each package in a fresh temporary environment or reset the environment between checks.

### `signatory`

`signatory` has a published wheel scheme tied to specific PyTorch versions. Test it in a dedicated
legacy environment first rather than the current PyTorch 2.x environment.

```bash
python -m venv /tmp/tcvqvae-signatory
/tmp/tcvqvae-signatory/bin/python -m pip install --upgrade pip setuptools wheel
/tmp/tcvqvae-signatory/bin/python -m pip install "torch==1.11.0"
/tmp/tcvqvae-signatory/bin/python -m pip install \
  "signatory==1.2.7.1.11.0" \
  --no-cache-dir \
  --force-reinstall
```

Expected import check:

```bash
/tmp/tcvqvae-signatory/bin/python - <<'PY'
import signatory
import torch

x = torch.randn(4, 8, 3)
sig = signatory.signature(x, depth=3)
logsig = signatory.logsignature(x, depth=3)
print("signatory:", signatory.__version__)
print("signature_shape:", tuple(sig.shape))
print("logsignature_shape:", tuple(logsig.shape))
print("finite:", bool(torch.isfinite(sig).all() and torch.isfinite(logsig).all()))
PY
```

Compatibility decision:

- If the package cannot be installed on a supported Python and PyTorch combination for this
  project, keep it inspection-only.
- Do not downgrade the project to satisfy `signatory`.

### `iisignature`

```bash
python -m venv /tmp/tcvqvae-iisignature
/tmp/tcvqvae-iisignature/bin/python -m pip install --upgrade pip setuptools wheel
/tmp/tcvqvae-iisignature/bin/python -m pip install numpy iisignature
```

Expected import check:

```bash
/tmp/tcvqvae-iisignature/bin/python - <<'PY'
import iisignature
import numpy as np

x = np.random.default_rng(0).normal(size=(8, 3)).astype("float64")
sig = iisignature.sig(x, 3)
prep = iisignature.prepare(3, 3)
logsig = iisignature.logsig(x, prep)
print("iisignature:", getattr(iisignature, "__version__", "unknown"))
print("signature_shape:", sig.shape)
print("logsignature_shape:", logsig.shape)
print("finite:", bool(np.isfinite(sig).all() and np.isfinite(logsig).all()))
PY
```

Compatibility decision:

- Treat as an offline CPU feature-extraction candidate unless tensor integration and batching are
  explicitly validated.

### `sigkernel`

```bash
python -m venv /tmp/tcvqvae-sigkernel
/tmp/tcvqvae-sigkernel/bin/python -m pip install --upgrade pip setuptools wheel
/tmp/tcvqvae-sigkernel/bin/python -m pip install "torch>=2.5,<3.0" numpy scipy
/tmp/tcvqvae-sigkernel/bin/python -m pip install \
  "git+https://github.com/crispitagorico/sigkernel.git"
```

Expected CPU import and MMD check:

```bash
/tmp/tcvqvae-sigkernel/bin/python - <<'PY'
import sigkernel
import torch

device = "cpu"
dtype = torch.float64
x = torch.randn(6, 12, 2, dtype=dtype, device=device)
y = x.clone()
z = torch.randn(6, 12, 2, dtype=dtype, device=device)
static_kernel = sigkernel.RBFKernel(sigma=1.0)
kernel = sigkernel.SigKernel(static_kernel, dyadic_order=1)
mmd_same = kernel.compute_mmd(x, y, max_batch=4)
mmd_diff = kernel.compute_mmd(x, z, max_batch=4)
print("sigkernel:", getattr(sigkernel, "__version__", "unknown"))
print("mmd_same:", float(mmd_same.detach().cpu()))
print("mmd_diff:", float(mmd_diff.detach().cpu()))
print("finite:", bool(torch.isfinite(mmd_same) and torch.isfinite(mmd_diff)))
PY
```

Expected GPU fallback check:

```bash
/tmp/tcvqvae-sigkernel/bin/python - <<'PY'
import sigkernel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
x = torch.randn(4, 10, 2, dtype=dtype, device=device)
y = torch.randn(4, 10, 2, dtype=dtype, device=device)
kernel = sigkernel.SigKernel(sigkernel.RBFKernel(sigma=1.0), dyadic_order=1)
mmd = kernel.compute_mmd(x, y, max_batch=2)
print("device:", device)
print("mmd:", float(mmd.detach().cpu()))
PY
```

Compatibility decision:

- Promote to evaluation-candidate status only if CPU import, CPU MMD, and fallback behaviour pass.
- Record whether CUDA is usable separately from whether the package itself imports.

### `KSig`

```bash
python -m venv /tmp/tcvqvae-ksig
/tmp/tcvqvae-ksig/bin/python -m pip install --upgrade pip setuptools wheel
/tmp/tcvqvae-ksig/bin/python -m pip install numpy scipy scikit-learn
/tmp/tcvqvae-ksig/bin/python -m pip install "git+https://github.com/tgcsaba/ksig.git"
```

If installation requires CuPy or a CUDA-specific wheel, record the failure and repeat in a
GPU-matched environment:

```bash
/tmp/tcvqvae-ksig/bin/python -m pip install cupy-cuda12x
/tmp/tcvqvae-ksig/bin/python -m pip install "git+https://github.com/tgcsaba/ksig.git"
```

Expected import and Gram check:

```bash
/tmp/tcvqvae-ksig/bin/python - <<'PY'
import ksig
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=(6, 12, 2))
static_kernel = ksig.static.kernels.RBFKernel()
kernel = ksig.kernels.SignatureKernel(n_levels=3, static_kernel=static_kernel)
gram = kernel(x)
print("ksig:", getattr(ksig, "__version__", "unknown"))
print("gram_shape:", gram.shape)
print("symmetric:", bool(np.allclose(gram, gram.T, atol=1e-8)))
print("finite:", bool(np.isfinite(gram).all()))
PY
```

Compatibility decision:

- Treat as inspection-only until installation, CPU fallback, and GPU setup are understood.
- Use for metric cross-checking before considering it as a project dependency.

### `pathsig`

```bash
python -m venv /tmp/tcvqvae-pathsig
/tmp/tcvqvae-pathsig/bin/python -m pip install --upgrade pip setuptools wheel
/tmp/tcvqvae-pathsig/bin/python -m pip install "torch>=2.5,<3.0" numpy
/tmp/tcvqvae-pathsig/bin/python -m pip install pathsig
```

Expected import check:

```bash
/tmp/tcvqvae-pathsig/bin/python - <<'PY'
import importlib.metadata
import torch

import pathsig

print("pathsig:", importlib.metadata.version("pathsig"))
print("module:", pathsig)
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(4, 10, 2, device=device)
if hasattr(pathsig, "signature"):
    sig = pathsig.signature(x, truncation_level=3)
    print("signature_shape:", tuple(sig.shape))
    print("finite:", bool(torch.isfinite(sig).all()))
else:
    print("signature_api_missing")
PY
```

Compatibility decision:

- Keep optional and experimental unless documentation, CPU fallback, PyTorch behaviour, and API
  stability are clear.

## Numerical consistency checks

These checks compare qualitative behaviour between `sigkernel` and `KSig`. They are not expected
to produce identical numbers because the packages may use different static kernels, normalisation,
truncation, or PDE settings.

### Shared test data

Use the same deterministic paths:

```python
import numpy as np

rng = np.random.default_rng(123)
x = rng.normal(size=(8, 16, 2)).astype("float64")
y_same = x.copy()
y_shifted = x + 0.25
y_random = rng.normal(size=(8, 16, 2)).astype("float64")
```

### Expected consistency properties

- Gram matrices are finite.
- Self Gram matrices are symmetric up to numerical tolerance.
- MMD between `x` and `y_same` is near zero or below the package's documented numerical
  tolerance.
- MMD between `x` and `y_random` is larger than MMD between `x` and `y_same`.
- A small shift, `y_shifted`, gives a stable non-negative distance if the selected static kernel
  is shift-sensitive.
- Repeating the same computation with the same seed gives the same result.

### Cross-package record

Record these fields for both `sigkernel` and `KSig`:

- package name and version or commit hash;
- Python version;
- PyTorch version, if applicable;
- NumPy and CuPy versions, if applicable;
- CPU or GPU device;
- dtype;
- static kernel;
- signature level or dyadic order;
- normalisation settings;
- batch size;
- runtime;
- `same` distance;
- `shifted` distance;
- `random` distance;
- any warnings or fallback paths.

## Acceptance criteria

A package can become a dependency candidate only if:

- installation works in a clean temporary environment;
- import checks pass;
- CPU fallback is available or an explicit project decision accepts GPU-only use;
- small numerical checks are finite and reproducible;
- metric outputs are stable enough for model selection;
- package maintenance and licensing are acceptable;
- the implementation thread separately justifies adding the dependency to `pyproject.toml`.

Until those criteria are met, all packages remain inspection-only.
