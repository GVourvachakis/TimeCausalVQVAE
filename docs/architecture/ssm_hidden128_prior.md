# SSM Hidden128 Prior Plan

Status: planning note only. No source code was changed, no dependency was added to
`pyproject.toml`, and no models were trained for this note.

## Motivation

The selected hidden128 causal conv-transformer k3 prior improves local token dynamics relative to
the additive hidden128 prior, especially transition matching and run-length behaviour. Its seed
robustness remains imperfect: all tested seeds beat the hidden128 additive prior on the current
paper-style profile, but the profile range remains broad enough that the result should not become a
public default without further evidence.

The single-stream hidden128 tokenizer remains the best discrete interface for the active research
branch. It keeps one token per time step, preserves all 64-code usage, and avoids the sampled-token
collapse observed in the separate-frequency hierarchical prior. The current bottleneck is therefore
prior calibration and persistence, not tokenizer capacity.

A selective state-space or Mamba-style prior may be a viable next prior family because it can carry
persistent recurrent state while retaining linear-time left-to-right generation. The target
benefit is improved persistent state, more stable sampling calibration across seeds, and better
local-volatility dynamics without changing the tokenizer or the evaluation contract.

## Candidate Families

The preferred experimental family is a Mamba or selective state-space prior over the hidden128
token stream. This should be treated as a dependency-gated research candidate rather than an
immediate implementation target, because the reference implementation depends on compiled CUDA
extensions and has stricter environment constraints than the existing PyTorch-only
conv-transformer prior.

The fallback family is a simpler causal state-space or gated recurrent prior implemented with
native PyTorch operators. A diagonal state-space recurrence, GRU-style gated recurrence, or
minimal selective-SSM approximation would be acceptable fallback candidates if the Mamba package
chain fails the compatibility probe. The fallback should sacrifice exact Mamba fidelity before it
sacrifices the hidden128 interface or the left-to-right sampling contract.

## Causality Contract

Any SSM prior must satisfy the same autoregressive contract as the current token prior:

- logits at time `t` may depend only on tokens strictly before `t` and on the allowed condition;
- recurrent state must update left-to-right during sampling;
- no bidirectional scans, reverse scans, centred convolutions, or sequence-wide normalisation may
  enter the generation path;
- batched training scans must be equivalent to the prefix-visible recurrence used for sampling;
- no-future-leakage checks must include both full-sequence evaluation and stepwise sampling.

The condition remains the scalar VIX condition. The prior may use the condition at each generated
time step only through the same allowed conditioning path used by the current hidden128 priors.

## Target Interface

The target SSM prior keeps the existing hidden128 discrete interface:

- tokenizer config:
  `configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml`;
- token stream: one categorical token per time step over the same 64-code vocabulary;
- condition: scalar VIX with `condition_dim=1`;
- sampling and evaluation: same token-prior sampling, token diagnostics, decoded-path
  reconstruction, and paper-style market evaluation pipeline as the current hidden128 priors;
- comparison baselines: hidden128 additive prior, hidden128 conv-transformer k3, and the promoted
  public standard-VQ additive AR baseline.

No public default should change during package probing or initial feasibility work.

## Package Candidates

The following package facts were checked on 2026-05-16. They are documentation inputs only and do
not imply that any package should be added to `pyproject.toml`.

| Candidate | Python compatibility | PyTorch compatibility | CUDA requirement | CPU support | Licence | Probe install command | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mamba-ssm` | PyPI declares `Python >=3.9`; this covers the repo range `>=3.11,<3.13`. | Project docs state `PyTorch 1.12+`; the current repo pin `torch>=2.5,<3.0` is nominally within range. Current package metadata and setup code still key compiled wheels to the torch, CUDA, Python, platform, and C++ ABI combination, so torch `2.11` or CUDA `13.0` may require a source build if no matching wheel exists. | Reference path requires Linux, NVIDIA GPU, and CUDA `11.6+`; setup code also accepts CUDA major versions 11, 12, and 13 for wheel selection or source builds. | Treat as no CPU-supported package path for this project: the top-level selective-scan path imports compiled `selective_scan_cuda`. Reference functions exist in source but are not a stable CPU package contract. | Apache-2.0. | `pip install "mamba-ssm[causal-conv1d]" --no-build-isolation` after installing a CUDA-enabled torch build. | High. Build isolation, wheel availability, CUDA driver/toolkit mismatch, transitive package growth, and sampling-step API audit are all material risks. |
| `causal-conv1d` | PyPI declares `Python >=3.9`; this covers the repo range. | Source `install_requires` includes `torch`, `packaging`, and `ninja`; no narrow torch pin is declared, but binary wheels are keyed to torch major/minor and CUDA/HIP details. | Package is a CUDA depthwise convolution extension. Source setup rejects CUDA below `11.6`; it also has ROCm support with extra prerequisites. | Treat as no package-level CPU fallback: import routes through `causal_conv1d_cuda`. A manual native `torch.nn.functional.conv1d` equivalent can be used outside the package. | BSD licence, GitHub marks BSD-3-Clause. | `pip install "causal-conv1d>=1.4.0" --no-build-isolation` after installing the intended torch build. | Medium to high. It is narrower than `mamba-ssm`, but still depends on compiled extension compatibility and may fail before import on CPU-only or driver-mismatched systems. |
| `mambapy` | PyPI declares `Python >=3.6`; this covers the repo range. | Pure PyTorch package with torch as its effective runtime dependency; no precise torch version bound is advertised on PyPI. | No CUDA extension is required for the basic Mamba implementation. The package notes that its Mamba-2 adaptation currently requires CUDA, so the first probe should use the basic Mamba class only. | Yes for the basic Mamba path, because it is a pure PyTorch implementation and publishes a platform-independent wheel. | MIT. | `pip install mambapy`. | Medium. It is educational, last released in 2024, slower than the reference kernels, and its API and inference cache must be audited before use in a strict token-prior sampler. |
| Native PyTorch fallback | Inherits this repo's Python range `>=3.11,<3.13`. | Inherits this repo's `torch>=2.5,<3.0` dependency. | None. | Yes. | Inherits the repo licence if implemented locally. | No package install; prototype with existing PyTorch modules in an isolated branch only. | Medium. This is the most portable fallback, but it is not Mamba-equivalent and would need a careful recurrence, state-cache, and no-leakage test suite. |

Local environment note: the current Poetry environment reports Python `3.12.3`, torch
`2.11.0+cu130`, `torch.version.cuda == 13.0`, and `torch.cuda.is_available() == False` because the
installed NVIDIA driver is too old for that torch CUDA build. The compatibility probe should
therefore distinguish package import failures from local CUDA-driver failures.

## Non-Goals

This branch must not change the tokenizer or add new tokenizer families. It must not introduce
multi-code priors, GroupedRVQ, MGVQ, cross-attention, signatures, VQ diffusion, continuous
diffusion, or objective changes. It must not train models, change public defaults, or add SSM
packages to `pyproject.toml` before the isolated compatibility probe is reviewed.

## Evidence Sources

- `mamba-ssm` PyPI: <https://pypi.org/project/mamba-ssm/>
- `state-spaces/mamba` repository: <https://github.com/state-spaces/mamba>
- `causal-conv1d` PyPI: <https://pypi.org/project/causal-conv1d/>
- `Dao-AILab/causal-conv1d` repository: <https://github.com/Dao-AILab/causal-conv1d>
- `mambapy` PyPI: <https://pypi.org/project/mambapy/>
