# SSM Package Compatibility Results

Status: isolated compatibility probe completed on 2026-05-16. No project dependency was added, no
source code was changed, no prior was implemented, no tokenizer code was touched, no model was
trained, and no public default was changed.

## Scope

The probe followed `docs/verification/ssm_package_compatibility_plan.md` for the hidden128
single-stream S&P500/VIX token-prior branch. The temporary environment was created outside the
repository at `/tmp/tcvae-ssm-probe`. Package installs were made only inside that environment.

## Active Poetry Environment

Command:

```bash
poetry run python -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
poetry check
```

Result:

```text
3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]
2.11.0+cu130
13.0
False
poetry check: All set!
```

Torch also emitted a CUDA initialisation warning: the installed NVIDIA driver is version `12060`,
which is too old for the active torch CUDA build. Therefore the active repository environment has
a CUDA-enabled torch wheel, but no driver-compatible CUDA runtime.

## Temporary Probe Environment

Commands:

```bash
python3.12 -m venv /tmp/tcvae-ssm-probe
source /tmp/tcvae-ssm-probe/bin/activate
python -m pip install --upgrade pip wheel setuptools packaging ninja
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The commands were executed through the venv Python executable. The environment was then
deactivated and the shell returned to the repository at:

```text
/home/georgios-vourvachakis/Desktop/TimeCausalVQVAE
```

Environment record:

```text
python 3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]
torch 2.12.0+cpu
torch_cuda_build None
cuda_available False
```

Installed probe-package versions:

| Package | Version | Notes |
| --- | --- | --- |
| `pip` | `26.1.1` | Temporary venv only. |
| `wheel` | `0.47.0` | Temporary venv only. |
| `setuptools` | `70.2.0` | Downgraded by the CPU torch install constraint `setuptools<82`. |
| `packaging` | `26.2` | Temporary venv only. |
| `ninja` | `1.13.0` | Temporary venv only. |
| `torch` | `2.12.0+cpu` | Installed from the PyTorch CPU wheel index. |

The minimal CPU torch import emitted a warning that NumPy was not installed in the temporary
environment. This did not affect the tensor-only forward probes.

## Package Install And Import Results

| Candidate | Local install result | Import or forward result | Interpretation |
| --- | --- | --- | --- |
| `mambapy` | Installed `mambapy==1.2.0` in `/tmp/tcvae-ssm-probe`. | Import and CPU forward pass succeeded. | Viable as a CPU-compatible pure-PyTorch reference candidate, subject to API, maintenance, and stepwise-inference review. |
| Native PyTorch recurrent fallback | No package install required beyond CPU torch. | CPU forward pass succeeded. | Viable as the most portable fallback family for an implementation branch. |
| `causal-conv1d` | Skipped. | Skipped. | The local machine had no driver-compatible CUDA runtime. The package remains a CUDA-extension risk to test in a CUDA-compatible environment. |
| `mamba-ssm` | Skipped. | Skipped. | The local machine had no driver-compatible CUDA runtime, and the reference package path depends on compiled CUDA kernels. |

## `mambapy` Probe

Command:

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

Result:

```text
Successfully installed mambapy-1.2.0
shape (2, 16, 32)
```

The probe passed on CPU. This establishes that `mambapy` can execute a minimal tensor forward pass
in an isolated CPU environment with Python 3.12 and torch `2.12.0+cpu`.

Limitations:

- this did not test a token-prior wrapper, condition injection, state caching, or sampling;
- this did not test no-future-leakage equivalence between full-sequence and stepwise execution;
- this did not evaluate performance or numerical parity with the reference `mamba-ssm` kernels.

## Native PyTorch Fallback Probe

Command:

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

Result:

```text
shape (2, 16, 64)
```

The probe passed on CPU. This confirms that a native PyTorch left-to-right recurrent fallback can
match the hidden128 token-prior tensor shape with a 64-code vocabulary and `condition_dim=1`.

Limitations:

- this is a shape and finiteness smoke test only;
- this is not a final architecture and was not added to the project;
- any implementation would still need the hidden128 no-future-leakage tests and sampling-cache
  checks.

## `causal-conv1d` Probe

The `causal-conv1d` package was not installed or executed locally. Both the active Poetry
environment and the temporary CPU probe environment reported `torch.cuda.is_available() == False`.
The active Poetry environment also reported a driver mismatch for its CUDA-enabled torch build.

Conclusion: the local result is `skipped`, not `failed`. A valid probe requires a CUDA-compatible
driver and torch build before the package's compiled extension risk can be assessed.

## `mamba-ssm` Probe

The `mamba-ssm` package was not installed or executed locally. The plan requires this probe only
after CUDA is available and, ideally, after `causal-conv1d` succeeds. Those preconditions were not
met on this machine.

Conclusion: the local result is `skipped`, not `failed`. The package remains high risk until tested
in a CUDA-compatible environment with matching Python, torch, CUDA, and driver versions.

## Recommendation

Use `mambapy`: acceptable as a CPU-compatible reference probe and possible research prototype, but
not yet recommended as a project dependency. Before adoption, audit its maintenance status,
inference-cache API, and strict stepwise causal behaviour.

Use native PyTorch fallback: recommended as the lowest-risk implementation direction if the next
branch needs a portable recurrent or state-space baseline. It inherits the existing project torch
dependency and avoids CUDA-extension packaging risk.

Wait for CUDA-compatible environment: required before making any decision on `causal-conv1d` or
`mamba-ssm`. The local machine cannot distinguish package incompatibility from driver-runtime
incompatibility for those candidates.

Reject SSM for now: not recommended. The CPU probes support continued SSM-style feasibility work,
but compiled Mamba compatibility remains unresolved.

## Next Acceptance Checks

Before any implementation branch, the selected candidate must pass:

- no-future-leakage checks where logits at time `t` do not change when tokens after `t - 1` are
  perturbed;
- equivalence checks between batched full-sequence evaluation and stepwise left-to-right sampling;
- recurrent-state cache checks that update state exactly once per generated time step;
- hidden128 sampling diagnostics against the existing additive and conv-transformer k3 baselines;
- a dependency decision that explicitly keeps `pyproject.toml` unchanged until the project accepts
  the chosen package or local implementation.
