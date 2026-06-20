# Device Acceleration Audit

This note records a static audit of GPU and device handling after the `0.1.0` PyPI release. It
does not change model code and does not train models.

## Summary

The project admits GPU acceleration in the main training paths, but it is not uniformly device
agnostic across all model and evaluation surfaces.

- Continuous TC-VAE training selects CUDA automatically when available and no CUDA-disabling option
  is active.
- Causal VQ tokenizer training/evaluation and token-prior training/evaluation accept `--device`,
  default to CUDA when available, move the model to the selected device, and move input batches to
  that device.
- The discrete tokenizer and token-prior model internals are mostly device-local: tensors created
  during forward/sampling use the input or requested sample device, and CPU transfers are mainly for
  diagnostics, serialisation, and plotting.
- Continuous evaluation is CPU-oriented: `tcvae-evaluate` has no `--device` option, and continuous
  checkpoint loading maps weights to CPU.
- At least one continuous decoder implementation is not fully device agnostic under `model.to(...)`.

Local runtime probe:

```text
torch 2.12.1+cu130
cuda_available False
cuda_device_count 0
mps_available False
```

PyTorch emitted warnings that the installed NVIDIA driver is too old for the installed CUDA wheel.
Therefore this audit did not perform a live CUDA forward pass.

## Evidence

### Continuous Training

The continuous trainer chooses CUDA by default when CUDA is available and `no_cuda` is false:

```text
src/time_causal_vae/training/trainer.py:92-100
```

It then moves the model to the selected device and stores that device on the model:

```text
src/time_causal_vae/training/trainer.py:113-117
```

Training/evaluation inputs are moved only when the selected device string contains `cuda`:

```text
src/time_causal_vae/training/trainer.py:160-174
```

This is sufficient for the current CUDA/CPU continuous trainer path, but it is not a general
device-agnostic transfer rule for non-CUDA accelerators such as MPS or future backends.

### Discrete Tokenizer

Tokenizer training selects a requested device or CUDA by default:

```text
src/time_causal_vae/cli/train_tokenizer.py:95-96
src/time_causal_vae/cli/train_tokenizer.py:408-412
```

The training loop moves batch data and optional conditions to that device:

```text
src/time_causal_vae/cli/train_tokenizer.py:589-646
```

Tokenizer evaluation also accepts `--device`, selects CUDA by default, loads checkpoints with the
requested map location, and moves inputs to the selected device:

```text
src/time_causal_vae/cli/evaluate_tokenizer.py:47-76
src/time_causal_vae/cli/evaluate_tokenizer.py:179-183
src/time_causal_vae/evaluation/tokenizer.py:20-50
```

The tokenizer core creates auxiliary and diagnostic tensors on the reference/input device where
they participate in model computations. Explicit CPU transfers are used for code-usage counting
and persisted diagnostic payloads.

### Discrete Token Prior

Token-prior training selects a requested device or CUDA by default, moves the model to that device,
and moves token/condition batches in the train and eval loops:

```text
src/time_causal_vae/cli/train_token_prior.py:80-81
src/time_causal_vae/cli/train_token_prior.py:390-394
src/time_causal_vae/cli/train_token_prior.py:637-671
```

Token-prior evaluation accepts `--device`, loads the prior and tokenizer on that device, and samples
with the selected device:

```text
src/time_causal_vae/cli/evaluate_token_prior.py:57-94
src/time_causal_vae/cli/evaluate_token_prior.py:325-329
src/time_causal_vae/evaluation/token_prior.py:39-65
```

The token-prior model implementations allocate position indices, attention masks, BOS tokens, and
sampled-token tensors on the active input or sample device.

### Continuous Evaluation

`tcvae-evaluate` does not expose a `--device` option:

```text
src/time_causal_vae/cli/evaluate.py:21-52
```

The continuous evaluator builds the model and loads weights without moving the model or data to a
requested accelerator:

```text
src/time_causal_vae/cli/evaluate.py:205-217
src/time_causal_vae/evaluation/checkpoints.py:80-109
```

The underlying continuous model loader maps checkpoint weights to CPU:

```text
src/time_causal_vae/models/continuous/base.py:71
```

This makes the continuous evaluation path portable on CPU, but it should not be described as
GPU-aware.

## Non-Agnostic Or Fragile Areas

1. Continuous model generation relies on `model.device`.

   The trainer sets `self.model.device = device`, but a direct user who calls `model.to("cuda")`
   outside the trainer does not automatically update `model.device`. Continuous generation samples
   prior noise from `self.device`, so direct programmatic usage is only device-safe if the caller
   also sets `model.device` or uses the trainer-managed path.

2. Continuous input transfer is CUDA-specific.

   `_set_inputs_to_device` checks for `"cuda"` in the device string before moving tensors. This
   handles CUDA and CPU, but not arbitrary `torch.device` backends.

3. `NeuralSDEDecoder` and `CRSigDecoder` are not fully safe under `model.to(device)`.

   `B1`, `B2`, `lambda1`, and `lambda2` are plain tensor attributes, not registered parameters or
   buffers:

   ```text
   src/time_causal_vae/models/continuous/decoders/neural_sde.py:54-62
   ```

   They are used with tensors on the forward input device:

   ```text
   src/time_causal_vae/models/continuous/decoders/neural_sde.py:74-80
   src/time_causal_vae/models/continuous/decoders/neural_sde.py:100-112
   ```

   If the decoder is initialised on CPU and later moved with `model.to("cuda")`, these plain
   tensors would remain on CPU and can cause device mismatch errors. This decoder is therefore not
   device agnostic as implemented.

4. Continuous evaluation lacks GPU selection.

   Users can evaluate continuous checkpoints on CPU through `tcvae-evaluate`, but there is no
   documented or implemented CLI switch to place continuous evaluation on CUDA.

5. GPU dependency availability does not imply GPU usability.

   The local environment installed a CUDA-enabled PyTorch wheel, but CUDA could not initialise
   because the driver was too old for that wheel. Package installation can succeed while runtime GPU
   acceleration remains unavailable.

## Documentation Guidance

The project documentation can safely state:

- GPU acceleration is supported in the main CUDA training paths when a compatible CUDA-enabled
  PyTorch installation and driver are available.
- Discrete tokenizer and token-prior CLIs expose `--device` and default to CUDA when available.
- Continuous training defaults to CUDA when available.
- CPU remains the portable default/fallback.

The documentation should not state:

- all models are device agnostic;
- all evaluation paths are GPU-aware;
- MPS or non-CUDA accelerators are supported;
- every continuous decoder is safe under arbitrary `model.to(device)` calls.

## Suggested Future Work

These are implementation recommendations only and were not applied in this audit.

- Replace CUDA-string checks in the continuous trainer with unconditional tensor transfer to the
  selected `torch.device`.
- Register `NeuralSDEDecoder` fixed random tensors as buffers, or recreate them on the forward
  input device.
- Avoid relying on mutable `model.device` for continuous generation; infer the device from model
  parameters or accept an explicit generation device.
- Add `--device` to `tcvae-evaluate` and move the continuous evaluator model/data accordingly.
- Add lightweight CPU/CUDA smoke tests for selected continuous and discrete forward passes, guarded
  by `torch.cuda.is_available()`.
