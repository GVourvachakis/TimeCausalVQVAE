# S&P500/VIX Separate Frequency Tokenizer Smoke

Status: implementation smoke only. This verifies dataset plumbing for separate causal EMA low and
high tokenizers. It does not implement a hierarchical prior, train a non-smoke model, alter the
promoted joint tokenizer, or add GroupedRVQ, MGVQ, signatures, diffusion, or cross-attention.

## Configuration

The smoke used two standard vector VQ tokenizer configs:

```text
configs/experiments/sp500_vix_causal_vq_tokenizer_freq_low_alpha02.yaml
configs/experiments/sp500_vix_causal_vq_tokenizer_freq_high_alpha02.yaml
```

Both configs use the S&P500/VIX dataset with `condition_dim: 1`, `quantizer_type: vector`,
`data_dim: 1`, `frequency_decomposition: ema`, `ema_alpha: 0.2`, and `compose_output: false`.
The low config sets `frequency_component: low`; the high config sets
`frequency_component: high`.

When `frequency_component` is absent, the existing joint-tokenizer behaviour is unchanged and EMA
decomposition still returns `[low, high]` channels with shape `[batch, time, 2]`.

## Component Shapes

For original scalar paths:

```text
original path: [batch, 60, 1]
low component: [batch, 60, 1]
high component: [batch, 60, 1]
condition: [batch, 1]
```

The no-leakage smoke used batch size `8`, so the observed component shapes were:

```text
original_shape=(8, 60, 1)
conditions_shape=(8, 1)
low_shape=(8, 60, 1)
high_shape=(8, 60, 1)
```

## No-Leakage Check

Command:

```text
poetry run python scripts/check_separate_frequency_tokenizer_no_leakage.py \
  --alpha 0.2 \
  --batch-size 8 \
  --cutoff 29 \
  --seed 99
```

Result:

```text
PASS separate frequency tokenizer no-leakage check
alpha=0.2
cutoff=29
seed=99
max_low_component_prefix_diff=0.00000000e+00
max_low_encoder_prefix_diff=0.00000000e+00
low_deterministic_after_warmup=True
low_token_prefix_mismatch_count=0
max_low_reconstruction_prefix_diff=0.00000000e+00
max_high_component_prefix_diff=0.00000000e+00
max_high_encoder_prefix_diff=0.00000000e+00
high_deterministic_after_warmup=True
high_token_prefix_mismatch_count=0
max_high_reconstruction_prefix_diff=0.00000000e+00
```

The script perturbs only target values after the inclusive cutoff and compares prefixes through
the cutoff. Token-index checks run after a warm-up forward pass because k-means codebook
initialisation is data-dependent.

## One-Epoch Smokes

Low tokenizer:

```text
poetry run tcvae-train-tokenizer \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer_freq_low_alpha02.yaml \
  --output-dir outputs/sp500_vix_discrete/separate_frequency_tokenizer_smoke/low \
  --base-data-dir data/processed \
  --epochs 1 \
  --no-wandb
```

Final low metrics:

```text
training_complete: outputs/sp500_vix_discrete/separate_frequency_tokenizer_smoke/low/sp500_vix_causal_vq_tokenizer_freq_low_alpha02_seed0
mean_total_loss: 0.5582743939110841
mean_reconstruction_loss: 0.5565338510963846
active_code_count: 64
active_code_ratio: 1.0
perplexity: 3.620671510696411
token_count: 147420
```

High tokenizer:

```text
poetry run tcvae-train-tokenizer \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer_freq_high_alpha02.yaml \
  --output-dir outputs/sp500_vix_discrete/separate_frequency_tokenizer_smoke/high \
  --base-data-dir data/processed \
  --epochs 1 \
  --no-wandb
```

Final high metrics:

```text
training_complete: outputs/sp500_vix_discrete/separate_frequency_tokenizer_smoke/high/sp500_vix_causal_vq_tokenizer_freq_high_alpha02_seed0
mean_total_loss: 0.04732739870953133
mean_reconstruction_loss: 0.047267236418970536
active_code_count: 63
active_code_ratio: 0.984375
perplexity: 7.398025035858154
token_count: 147420
```

## Caveats

These are wiring and causality smokes only. The checkpoints are one-epoch artefacts, no
hierarchical prior was implemented, no token-prior data was used for generation, and no
paper-style sampling diagnostics were run. The low/high code streams are intentionally separate
and require a later prior interface before they can be sampled as a coherent generated path.

The runs emitted local environment warnings about the Matplotlib cache directory and CUDA driver
availability. They did not block the CPU smoke checks.
