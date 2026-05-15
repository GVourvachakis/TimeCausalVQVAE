# S&P500/VIX Frequency Tokenizer Smoke

Status: implementation smoke only. This run verifies the minimal joint low/high-frequency
tokenizer path with deterministic causal EMA decomposition. It does not promote a model, train a
prior, modify the promoted baseline configs, or add separate low/high tokenizers.

## Configuration

The smoke used:

```text
configs/experiments/sp500_vix_causal_vq_tokenizer_freq_ema_alpha02.yaml
```

The new data fields are:

```yaml
frequency_decomposition: ema
ema_alpha: 0.2
compose_output: true
```

The tokenizer keeps the standard vector VQ interface with one code per time step. The model and
data config use `data_dim: 2` so the tokenizer sees deterministic frequency channels rather than
the original scalar path.

## Decomposition

For an original path `x`, the preprocessing step applies the causal recurrence:

```text
low_0 = x_0
low_t = alpha * x_t + (1 - alpha) * low_{t-1}
high_t = x_t - low_t
```

The smoke uses `alpha = 0.2`. The transformed tokenizer input is:

```text
original path: [batch, 60, 1]
frequency path: [batch, 60, 2]
```

The two channels are `[low, high]`. Decoded frequency paths are composed as `low_hat + high_hat`
before report-facing scalar-path diagnostics when `compose_output: true`.

## No-Leakage Check

The tokenizer no-leakage smoke used the frequency config, batch size `8`, cutoff `29`, seed `99`,
and `data/processed` as the base data directory. It perturbed only original target values after
the cutoff and compared the prefix through the cutoff.

```text
original_shape=(8, 60, 1)
frequency_shape=(8, 60, 2)
conditions_shape=(8, 1)
alpha=0.2
cutoff=29
max_low_prefix_diff=0.00000000e+00
max_high_prefix_diff=0.00000000e+00
max_encoder_prefix_diff=0.00000000e+00
token_prefix_mismatch_count=0
max_reconstruction_prefix_diff=0.00000000e+00
```

The script warms up the untrained tokenizer once before comparing token indices and
reconstructions, because k-means codebook initialisation is data-dependent. The reported encoder
prefix comparison is the primary deterministic causal feature check.

## Smoke Training

The requested one-epoch command completed successfully:

```text
poetry run tcvae-train-tokenizer \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer_freq_ema_alpha02.yaml \
  --output-dir outputs/sp500_vix_discrete/frequency_tokenizer_smoke \
  --base-data-dir data/processed \
  --epochs 1 \
  --no-wandb
```

Final training output:

```text
epoch=1 loss=0.32982390 recon=0.32852209 active_codes=64
training_complete: outputs/sp500_vix_discrete/frequency_tokenizer_smoke/sp500_vix_causal_vq_tokenizer_freq_ema_alpha02_seed0
runtime_seconds: 1.581
final_loss: 0.32982390
```

## Tokenizer Evaluation Smoke

A small tokenizer evaluation with `n_sample_test = 64` completed successfully. The saved summary
reported:

```text
x_shape: [64, 60, 2]
recon_x_shape: [64, 60, 2]
original_x_shape: [64, 60, 1]
composed_recon_x_shape: [64, 60, 1]
indices_shape: [64, 60]
```

Native two-channel reconstruction metrics:

```text
reconstruction_l1: 0.23631792
reconstruction_l2: 0.29258209
```

Component reconstruction metrics:

```text
frequency_low_reconstruction_l1: 0.38919482
frequency_low_reconstruction_l2: 0.40486470
frequency_high_reconstruction_l1: 0.08344103
frequency_high_reconstruction_l2: 0.08539975
```

Composed original-path metrics:

```text
frequency_original_reconstruction_l1: 0.47263584
frequency_original_reconstruction_l2: 0.48556396
frequency_composed_terminal_return_error: 21.79524040
frequency_composed_volatility_reconstruction_error: 0.02233220
```

Code usage in the 64-sample evaluation was small, as expected for a one-epoch smoke:

```text
active_code_count: 6
active_code_ratio: 0.09375000
codebook_perplexity: 1.47834575
```

## Caveats

This is a wiring and causality smoke, not a competitive result. The tokenizer trained for one
epoch on CPU, no prior was trained, and no paper-style prior sampling run was used for model
selection. The native tokenizer summary still includes two-channel reconstruction diagnostics such
as `terminal_return_error`; report-facing financial diagnostics should use the composed scalar
path metrics instead.

The run emitted local environment warnings about the Matplotlib cache directory and CUDA driver
availability. They did not block the CPU smoke runs.
