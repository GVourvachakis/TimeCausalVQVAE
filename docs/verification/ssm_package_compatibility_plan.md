# SSM Package Compatibility Probe Plan

Status: verification plan only. Do not add packages to `pyproject.toml`, do not install globally,
and do not train models during this probe.

## Objective

The objective is to determine whether a selective state-space or Mamba-style prior is viable for
the hidden128 single-stream token prior without changing the tokenizer, condition interface,
sampling pipeline, or public defaults. The probe must test dependency installation, importability,
small forward passes, CPU fallback behaviour, and CUDA-only failure handling in an isolated
temporary environment.

## Baseline Environment Record

Before creating a temporary environment, record the active repo environment:

```bash
poetry run python -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
poetry check
```

At the time this plan was written, the local Poetry environment reported Python `3.12.3`, torch
`2.11.0+cu130`, CUDA build `13.0`, and no available CUDA device because the installed NVIDIA driver
was too old for that torch build. A failed CUDA import in this machine state should not be treated
as a project-level rejection until the same command is repeated in a driver-compatible environment.

## Isolated Environment Setup

Use a temporary virtual environment outside the repository dependency graph:

```bash
python3.12 -m venv /tmp/tcvae-ssm-probe
source /tmp/tcvae-ssm-probe/bin/activate
python -m pip install --upgrade pip wheel setuptools packaging ninja
```

Install a torch build explicitly before probing SSM packages. Select only one of the following
torch paths per probe run:

```bash
# CPU-only smoke path for pure-PyTorch alternatives.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA path, only when the NVIDIA driver supports the selected CUDA wheel.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Then record the environment:

```bash
python - <<'PY'
import sys
import torch

print("python", sys.version)
print("torch", torch.__version__)
print("torch_cuda_build", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
```

## Probe A: `mamba-ssm`

Run this only after a CUDA-enabled torch build is installed and `torch.cuda.is_available()` is true:

```bash
python -m pip install "mamba-ssm[causal-conv1d]" --no-build-isolation
python - <<'PY'
import torch
from mamba_ssm import Mamba

device = "cuda"
batch, length, dim = 2, 16, 32
x = torch.randn(batch, length, dim, device=device)
model = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2).to(device)
y = model(x)
print("shape", tuple(y.shape))
assert y.shape == x.shape
assert torch.isfinite(y).all()
PY
```

If the install fails because no matching prebuilt wheel exists, save the pip log and note whether
the failure occurred before or during source compilation. If import fails with
`selective_scan_cuda`, record the torch version, CUDA build, Python ABI, driver version, and package
versions. Do not retry inside the project Poetry environment.

## Probe B: `causal-conv1d`

Run this as a narrower CUDA-extension probe:

```bash
python -m pip install "causal-conv1d>=1.4.0" --no-build-isolation
python - <<'PY'
import torch
from causal_conv1d import causal_conv1d_fn

device = "cuda"
batch, dim, length, width = 2, 8, 16, 4
x = torch.randn(batch, dim, length, device=device)
weight = torch.randn(dim, width, device=device)
bias = torch.randn(dim, device=device)
y = causal_conv1d_fn(x, weight, bias, activation="silu")
print("shape", tuple(y.shape))
assert y.shape == x.shape
assert torch.isfinite(y).all()
PY
```

If this package fails, `mamba-ssm[causal-conv1d]` should be considered high risk on the same
machine. A manual native PyTorch convolution equivalent can still be used as a fallback for local
causal convolution, but that does not validate the compiled Mamba package chain.

## Probe C: Pure-PyTorch Mamba Alternative

Run this in the CPU-only torch environment first:

```bash
python -m pip install mambapy
python - <<'PY'
import torch
from mambapy.mamba import Mamba, MambaConfig

batch, length, dim = 2, 16, 32
x = torch.randn(batch, length, dim)
config = MambaConfig(d_model=dim, n_layers=1)
model = Mamba(config)
y = model(x)
print("shape", tuple(y.shape))
assert y.shape == x.shape
assert torch.isfinite(y).all()
PY
```

If this passes on CPU, repeat on CUDA only when `torch.cuda.is_available()` is true:

```bash
python - <<'PY'
import torch
from mambapy.mamba import Mamba, MambaConfig

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; CPU fallback already tested")

batch, length, dim = 2, 16, 32
device = "cuda"
x = torch.randn(batch, length, dim, device=device)
config = MambaConfig(d_model=dim, n_layers=1)
model = Mamba(config).to(device)
y = model(x)
print("shape", tuple(y.shape))
assert y.shape == x.shape
assert torch.isfinite(y).all()
PY
```

Passing this probe does not approve `mambapy` for implementation. It only establishes a portable
fallback candidate whose API, stepwise inference path, and maintenance status need separate review.

## Probe D: Native PyTorch Fallback

Use this as the minimum CPU-safe fallback check:

```bash
python - <<'PY'
import torch
import torch.nn as nn

batch, length, vocab, condition_dim, embed_dim = 2, 16, 64, 1, 32
tokens = torch.randint(vocab, (batch, length))
conditions = torch.randn(batch, length, condition_dim)

embedding = nn.Embedding(vocab, embed_dim)
cell = nn.GRU(input_size=embed_dim + condition_dim, hidden_size=embed_dim, batch_first=True)
head = nn.Linear(embed_dim, vocab)

shifted = torch.zeros_like(tokens)
shifted[:, 1:] = tokens[:, :-1]
x = torch.cat([embedding(shifted), conditions], dim=-1)
h, _ = cell(x)
logits = head(h)
print("shape", tuple(logits.shape))
assert logits.shape == (batch, length, vocab)
assert torch.isfinite(logits).all()
PY
```

This does not implement the final prior. It only checks that a portable left-to-right recurrent
baseline is available if compiled SSM packages fail.

## Causality And Sampling Acceptance Checks

Any candidate that passes import and forward probes must satisfy these follow-up checks before
implementation:

- the logits at time `t` must be unchanged when tokens after `t - 1` are perturbed;
- the training scan must match a stepwise left-to-right sampling loop within numerical tolerance;
- recurrent state caches must update exactly once per generated time step;
- batched evaluation must not use bidirectional scans, reverse scans, or centred temporal kernels;
- CPU-only alternatives must still pass the same no-future-leakage checks, even if they are slower;
- CUDA-only candidates must fail clearly on CPU-only systems and must not be silently selected by
  default configuration.

## Stop Criteria

Stop after documentation and isolated package probes. Do not train S&P500/VIX token priors, do not
change hidden128 configs, do not add dependencies to `pyproject.toml`, and do not change public
defaults. A later implementation branch may proceed only after the package probe records a
supported candidate and a separate architecture decision accepts the dependency and causality risk.

## Evidence Sources

- `mamba-ssm` PyPI: <https://pypi.org/project/mamba-ssm/>
- `state-spaces/mamba` repository: <https://github.com/state-spaces/mamba>
- `causal-conv1d` PyPI: <https://pypi.org/project/causal-conv1d/>
- `Dao-AILab/causal-conv1d` repository: <https://github.com/Dao-AILab/causal-conv1d>
- `mambapy` PyPI: <https://pypi.org/project/mambapy/>
