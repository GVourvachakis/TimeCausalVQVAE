# Device Acceleration Audit

This note records the post-release device-handling audit and selective hardening pass for
`time-causal-vae`. It does not report trained-model results and does not require notebook
execution.

This work is post-`0.1.0` development. The published PyPI `0.1.0` package is unchanged, and these
device-handling improvements are intended for a future patch release.

## Summary

The project now supports selective single-device CPU/CUDA execution across the main public model
surfaces.

- Continuous TC-VAE training selects CUDA automatically when available, unless CUDA is disabled or
  an explicit device override is supplied.
- Continuous evaluation now accepts a device override and can move the loaded model and evaluation
  tensors to the selected device.
- Continuous generation now infers the active model device from parameters or buffers instead of
  relying only on mutable `model.device` state.
- Causal VQ tokenizer training/evaluation and token-prior training/evaluation accept `--device`,
  default to CUDA when available, move models to the selected device, and move batch tensors to the
  same device.
- `NeuralSDEDecoder` and `CRSigDecoder` fixed random matrices are registered as non-persistent
  buffers so they move with `model.to(device)` without changing checkpoint payloads.

The goal is single-GPU CUDA support when it is expected to improve training or sampling throughput.
This pass does not claim distributed training, MPS, or arbitrary accelerator support.

Local runtime probe:

```text
torch 2.12.1+cu130
cuda_available False
cuda_device_count 0
mps_available False
```

PyTorch emitted warnings that the installed NVIDIA driver is too old for the installed CUDA wheel.
Therefore this audit could verify CPU behaviour and static CUDA placement logic, but not a live CUDA
forward pass on this machine.

## Current Behaviour

### Continuous Training

The continuous trainer chooses CUDA by default when CUDA is available and `no_cuda` is false. An
explicit `--device` value overrides that selection.

The trainer now moves every tensor in each `DatasetOutput` to the selected device, rather than only
moving inputs when the device string contains `cuda`. Autocast also receives the selected
`torch.device(...).type`, which is safer for explicit devices such as `cuda:0`.

Relevant implementation:

```text
src/time_causal_vae/training/trainer.py
```

### Continuous Generation

Continuous VAE/CVAE generation now uses `infer_device()` to select a generation device from model
parameters or buffers. Callers may still pass an explicit `device=` argument. Conditional generation
moves tensor conditions to the selected generation device before decoding.

Relevant implementation:

```text
src/time_causal_vae/models/continuous/base.py
src/time_causal_vae/models/continuous/objectives/vae.py
```

### Continuous Evaluation

`tcvae-evaluate` now exposes:

```bash
tcvae-evaluate --device cpu
tcvae-evaluate --device cuda
```

When omitted, the CLI prefers CUDA if `torch.cuda.is_available()` is true and otherwise uses CPU.
The evaluator remains CPU-compatible by default for programmatic use when no device is supplied.

Relevant implementation:

```text
src/time_causal_vae/cli/evaluate.py
src/time_causal_vae/evaluation/checkpoints.py
```

### Discrete Tokenizer

Tokenizer training and evaluation already selected a requested device or CUDA by default. The model,
inputs, optional conditions, auxiliary-loss metadata, and checkpoint loading path all use the
selected device. CPU transfers are retained for diagnostics, plotting, and serialised payloads.

Relevant implementation:

```text
src/time_causal_vae/cli/train_tokenizer.py
src/time_causal_vae/cli/evaluate_tokenizer.py
src/time_causal_vae/evaluation/tokenizer.py
src/time_causal_vae/models/discrete/tokenizers/
```

### Discrete Token Prior

Token-prior training and evaluation already selected a requested device or CUDA by default. The
prior model, optional conditions, sampling tensors, masks, BOS tokens, position indices, and
checkpoint loading path use the selected device. CPU transfers are retained for token diagnostics
and persisted outputs.

Relevant implementation:

```text
src/time_causal_vae/cli/train_token_prior.py
src/time_causal_vae/cli/evaluate_token_prior.py
src/time_causal_vae/evaluation/token_prior.py
src/time_causal_vae/models/discrete/priors/
```

### NeuralSDE/CRSig Decoder Buffers

`NeuralSDEDecoder` now registers `B1`, `B2`, `lambda1`, and `lambda2` as non-persistent buffers.
This makes them move with `model.to(device)` while preserving the previous checkpoint contract,
where these random tensors were not saved as persistent model state.

Relevant implementation:

```text
src/time_causal_vae/models/continuous/decoders/neural_sde.py
```

## Remaining Boundaries

- CUDA support still depends on a compatible PyTorch wheel, NVIDIA driver, and local GPU runtime.
- This pass targets one selected device, not multi-GPU training.
- MPS and other non-CUDA accelerators are not advertised as supported release surfaces.
- Some report scripts and plotting-heavy utilities intentionally move tensors to CPU for metrics,
  serialisation, and Matplotlib output.
- GPU acceleration is most likely to improve performance for model training, token-prior sampling,
  tokenizer evaluation on larger batches, and continuous generation. Small smoke checks and
  plotting-heavy diagnostics may be faster or simpler on CPU.

## Documentation Guidance

The project documentation can safely state:

- single-device CPU/CUDA execution is supported for the main public training and evaluation CLIs;
- `--device cpu` and `--device cuda` are accepted where long-running model work is expected;
- CUDA is used only when PyTorch reports it is available, unless the user explicitly selects a
  device;
- local CPU execution remains the portable fallback.

The documentation should not state:

- multi-GPU support is release-ready;
- MPS support is release-ready;
- CUDA acceleration is guaranteed after installation, because driver and wheel compatibility still
  determine runtime availability.

## Suggested Future Work

- Add lightweight CPU forward/sampling smoke tests for continuous and discrete paths.
- Add CUDA smoke tests guarded by `torch.cuda.is_available()` in CI or local release validation.
- Consider documenting recommended CPU-only and CUDA-enabled PyTorch installation commands for
  users who want smaller CPU installs or explicit CUDA wheel selection.
