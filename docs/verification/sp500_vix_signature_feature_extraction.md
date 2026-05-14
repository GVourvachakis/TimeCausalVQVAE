# S&P500/VIX Signature Feature Extraction

## Purpose

This note records the first real S&P500/VIX historical log-signature feature
extraction run for optional conditioning experiments. The run used the local
processed array at `data/processed/sp500vix/sp500vix_normalized.npy` and did not
train models or modify tokenizer/prior code.

## Environment

- Python environment: Poetry project virtual environment.
- `iisignature`: import succeeded, version `0.24`.
- `iisignature` binary:
  `/home/georgios-vourvachakis/Desktop/TimeCausalVQVAE/.venv/lib/python3.12/site-packages/iisignature.cpython-312-x86_64-linux-gnu.so`
- NumPy: `2.4.4`.
- PyTorch: `2.11.0+cu130`.
- CUDA availability: `False`.
- Compatibility note: `iisignature` imported and completed both CPU extraction
  runs. PyTorch emitted a CUDA driver warning during the environment probe, but
  these feature extraction runs are CPU-based and did not require CUDA.

## Commands

Depth 2:

```bash
poetry run python scripts/extract_signature_features.py \
  --dataset sp500_vix \
  --base-data-dir data/processed \
  --output-dir outputs/sp500_vix_discrete/signature_features/logsig_l2_ctx20 \
  --depth 2 \
  --context-length 20 \
  --use-lead-lag \
  --include-time \
  --include-vix \
  --seed 99
```

Depth 3:

```bash
poetry run python scripts/extract_signature_features.py \
  --dataset sp500_vix \
  --base-data-dir data/processed \
  --output-dir outputs/sp500_vix_discrete/signature_features/logsig_l3_ctx20 \
  --depth 3 \
  --context-length 20 \
  --use-lead-lag \
  --include-time \
  --include-vix \
  --seed 99
```

## Results

| Run | Feature dimension | Train feature shape | Eval feature shape | Labels shape | Finite check | Internal runtime | Wall runtime |
| --- | ---: | --- | --- | --- | --- | ---: | ---: |
| Depth 2 | 55 | `(2457, 55)` | `(2457, 55)` | `(2457, 1)` | Passed | 0.25 s | 0.80 s |
| Depth 3 | 385 | `(2457, 385)` | `(2457, 385)` | `(2457, 1)` | Passed | 1.24 s | 2.07 s |

The depth-2 output directory size was approximately `1.5M`. The depth-3 output
directory size was approximately `12M`.

## Output Files

Depth 2:

- `outputs/sp500_vix_discrete/signature_features/logsig_l2_ctx20/train_signature_features.npz`
- `outputs/sp500_vix_discrete/signature_features/logsig_l2_ctx20/eval_signature_features.npz`
- `outputs/sp500_vix_discrete/signature_features/logsig_l2_ctx20/signature_feature_summary.json`
- `outputs/sp500_vix_discrete/signature_features/logsig_l2_ctx20/signature_feature_summary.md`

Depth 3:

- `outputs/sp500_vix_discrete/signature_features/logsig_l3_ctx20/train_signature_features.npz`
- `outputs/sp500_vix_discrete/signature_features/logsig_l3_ctx20/eval_signature_features.npz`
- `outputs/sp500_vix_discrete/signature_features/logsig_l3_ctx20/signature_feature_summary.json`
- `outputs/sp500_vix_discrete/signature_features/logsig_l3_ctx20/signature_feature_summary.md`

The generated files are under `outputs/` and are not committed.

## Preprocessing Choices

- Dataset: `sp500_vix`.
- Source file: `data/processed/sp500vix/sp500vix_normalized.npy`.
- Context length: `20`.
- Target horizon used for alignment: existing script convention of `60` steps.
- Feature type: truncated log-signature.
- Channels before lead-lag expansion:
  - normalised price path;
  - one-step log-return path;
  - cumulative log-return path;
  - time channel;
  - VIX channel.
- Lead-lag transform: enabled.
- Channels after lead-lag expansion:
  - `normalised_price_lead`;
  - `log_return_lead`;
  - `cumulative_log_return_lead`;
  - `time_lead`;
  - `vix_lead`;
  - `normalised_price_lag`;
  - `log_return_lag`;
  - `cumulative_log_return_lag`;
  - `time_lag`;
  - `vix_lag`.

## Alignment Convention

For each target window, features are computed only from the historical context
ending before that target window starts. No target-window generated path values
are used as conditioning context.

Early windows with fewer than `20` historical observations are left-padded with
the first available historical value. At index `0`, the context uses
`series[0]`. This preserves one feature row per aligned target window and keeps
the sample count aligned with downstream token artefacts.

Both `train_signature_features.npz` and `eval_signature_features.npz` contain:

- `features`;
- `labels`;
- `sample_indices`;
- `metadata`.

The `sample_indices` arrays have shape `(2457,)` for both depths.

## Decision

Depth 3 is accepted for this offline feature-extraction stage. Its dimension
increases from `55` to `385`, but the CPU runtime and disk footprint remained
small for the current S&P500/VIX processed dataset. Depth 2 remains the lower
dimensional default candidate for the first conditioning ablation, while depth 3
is available for sensitivity analysis.
