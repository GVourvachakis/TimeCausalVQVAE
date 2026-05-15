# S&P500/VIX Frequency Alpha 0.1 Token Extraction Geometry

Status: completed. This document records token extraction and latent-geometry diagnostics for the
causal EMA frequency tokenizer selected by the tokenizer-only ablation. No token prior was
trained, and no new model architecture was added.

## Checkpoint

The alpha `0.1` tokenizer checkpoint was found at:

```text
outputs/sp500_vix_discrete/frequency_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_freq_ema_alpha01_seed0
```

The directory contains `tokenizer.pt`, `tokenizer_config.json`, and `training_config.json`. It
does not contain `exp_config.yaml`, so the checkpoint was identified by the expected run-name
convention and by the saved training config fields:

```text
frequency_decomposition: ema
ema_alpha: 0.1
```

The first extraction attempt failed because the legacy token extraction helper built the original
one-channel path dataset while the frequency tokenizer expects two channels. The extraction data
builder now applies the same optional deterministic EMA preprocessing used by tokenizer training
when `data.frequency_decomposition: ema` is present. Default one-channel tokenizer behaviour is
unchanged.

## Token Extraction

Command:

```text
poetry run python scripts/extract_token_indices.py \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer_freq_ema_alpha01.yaml \
  --tokenizer-dir outputs/sp500_vix_discrete/frequency_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_freq_ema_alpha01_seed0 \
  --output-dir outputs/sp500_vix_discrete/token_prior/freq_ema_alpha01_tokens \
  --base-data-dir data/processed \
  --seed 99
```

Token-data path:

```text
outputs/sp500_vix_discrete/token_prior/freq_ema_alpha01_tokens
```

Extracted artefacts:

| Split | Token shape | Label shape | Data shape |
| --- | ---: | ---: | ---: |
| train | `[2457, 60]` | `[2457, 1]` | `[2457, 60, 2]` |
| eval | `[2457, 60]` | `[2457, 1]` | `[2457, 60, 2]` |

The token dataset is a single-vector stream with one token per time step:

```text
quantizer_type: vector
codebook_size: 64
sequence_length: 60
combined active codes: 64 / 64
combined active-code ratio: 1.00000000
combined perplexity: 53.01815796
combined entropy: 3.97063446
combined token count: 294840
```

## VIX-Bucket Usage

The combined train/eval token dataset shows increasing code usage across VIX buckets.

| Bucket | Samples | VIX min | VIX max | Active codes | Perplexity | Entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| very_low | 983 | 0.11053332 | 0.15552060 | 60 | 37.38182449 | 3.62118459 |
| low | 983 | 0.15552060 | 0.18007015 | 60 | 43.79148865 | 3.77943945 |
| mid | 983 | 0.18031201 | 0.21755955 | 61 | 47.30006027 | 3.85651159 |
| high | 983 | 0.21755955 | 0.27512395 | 62 | 52.65108871 | 3.96368694 |
| very_high | 982 | 0.27536583 | 1.00000000 | 64 | 53.69179153 | 3.98326015 |

This preserves the desirable condition-sensitive geometry pattern: higher-VIX windows use a
wider and more evenly distributed part of the codebook.

## Latent Geometry

Command:

```text
poetry run python scripts/analyze_discrete_latent_geometry.py \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer_freq_ema_alpha01.yaml \
  --tokenizer-dir outputs/sp500_vix_discrete/frequency_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_freq_ema_alpha01_seed0 \
  --token-data-dir outputs/sp500_vix_discrete/token_prior/freq_ema_alpha01_tokens \
  --output-dir outputs/latent_geometry/sp500_vix_freq_ema_alpha01 \
  --base-data-dir data/processed \
  --plot-voronoi
```

Geometry output path:

```text
outputs/latent_geometry/sp500_vix_freq_ema_alpha01
```

The geometry summary reports:

```text
quantizer_type: vector
index shape: [4914, 60]
codebook embedding shape: [64, 16]
codebook source: direct_backend_state
projection method: sklearn_pca
active codes: 64 / 64
perplexity: 53.01815796
entropy: 3.97063446
```

Generated plot inventory:

- `codebook_projection.png`
- `code_usage_histogram.png`
- `codebook_usage_projection.png`
- `vix_bucket_code_usage.png`
- `token_trajectory_examples.png`
- `codebook_voronoi.png`

RVQ q0/q1 pair analysis was unavailable, as expected, because this is a standard vector tokenizer
with index shape `[batch, time]`, not a two-quantizer RVQ layout.

## Geometry Comparison

| Tokenizer | Combined active codes | Perplexity | very-low active/perplexity | very-high active/perplexity |
| --- | ---: | ---: | ---: | ---: |
| Promoted standard VQ | 63 / 64 | 39.05571747 | 53 / 28.4869 | 63 / 43.8219 |
| Hidden128 VQ | 64 / 64 | 50.35867310 | 61 / 40.7709 | 64 / 48.2237 |
| EMA alpha 0.1 frequency VQ | 64 / 64 | 53.01815796 | 60 / 37.3818 | 64 / 53.6918 |

Compared with the promoted standard VQ tokenizer, the alpha `0.1` frequency tokenizer uses the
full codebook and has substantially higher global perplexity. Compared with hidden128, it has
slightly higher global perplexity and stronger expansion into the very-high VIX bucket.

The promoted standard VQ tokenizer remains the validated public baseline because it already has a
complete prior-training and report path. The alpha `0.1` frequency tokenizer now clears the
geometry gate for a first prior-training experiment: it has the expected one-code interface,
complete token artefacts, full codebook usage, VIX-sensitive bucket structure, and generated
latent-geometry plots.

## Decision

Alpha `0.1` clears the tokenizer geometry gate for prior training. The next stage should train a
causal token prior on:

```text
outputs/sp500_vix_discrete/token_prior/freq_ema_alpha01_tokens
```

The prior stage should still be treated as experimental until paper-style generation diagnostics
confirm whether the frequency-tokenizer path improves volatility W1 and squared-return
autocorrelation without degrading the standard path metrics.
