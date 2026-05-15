# S&P500/VIX Frequency Alpha 0.2 Token Extraction Geometry

Status: completed. This document prepares the alpha `0.2` deterministic causal EMA frequency
tokenizer for prior training. No token prior was trained, and no new architecture was added.

## Checkpoint

The alpha `0.2` tokenizer checkpoint was found at:

```text
outputs/sp500_vix_discrete/frequency_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_freq_ema_alpha02_seed0
```

The directory contains `tokenizer.pt`, `tokenizer_config.json`, and `training_config.json`.
The saved training config confirms:

```text
frequency_decomposition: ema
ema_alpha: 0.2
compose_output: true
```

## Token Extraction

Command:

```text
poetry run python scripts/extract_token_indices.py \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer_freq_ema_alpha02.yaml \
  --tokenizer-dir outputs/sp500_vix_discrete/frequency_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_freq_ema_alpha02_seed0 \
  --output-dir outputs/sp500_vix_discrete/token_prior/freq_ema_alpha02_tokens \
  --base-data-dir data/processed \
  --seed 99
```

Token-data path:

```text
outputs/sp500_vix_discrete/token_prior/freq_ema_alpha02_tokens
```

Extracted artefacts:

| Split | Token shape | Label shape | Data shape |
|---|---:|---:|---:|
| train | `[2457, 60]` | `[2457, 1]` | `[2457, 60, 2]` |
| eval | `[2457, 60]` | `[2457, 1]` | `[2457, 60, 2]` |

The extracted token dataset keeps the one-code-per-time-step interface:

```text
quantizer_type: vector
codebook_size: 64
sequence_length: 60
combined active codes: 64 / 64
combined active-code ratio: 1.00000000
combined perplexity: 54.60705566
combined entropy: 4.00016308
combined token count: 294840
```

## VIX-Bucket Usage

The combined train/eval token dataset shows broad usage in every VIX bucket.

| Bucket | Samples | VIX min | VIX max | Active codes | Perplexity | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| very_low | 983 | 0.11053332 | 0.15552060 | 61 | 40.16682053 | 3.69304132 |
| low | 983 | 0.15552060 | 0.18007015 | 61 | 45.80641937 | 3.82442427 |
| mid | 983 | 0.18031201 | 0.21755955 | 62 | 49.50623703 | 3.90209866 |
| high | 983 | 0.21755955 | 0.27512395 | 62 | 54.54927826 | 3.99910450 |
| very_high | 982 | 0.27536583 | 1.00000000 | 64 | 54.76032639 | 4.00296593 |

The bucket structure remains condition-sensitive: high-VIX windows use the full codebook and have
higher perplexity than low-VIX windows.

## Latent Geometry

Command:

```text
poetry run python scripts/analyze_discrete_latent_geometry.py \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer_freq_ema_alpha02.yaml \
  --tokenizer-dir outputs/sp500_vix_discrete/frequency_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_freq_ema_alpha02_seed0 \
  --token-data-dir outputs/sp500_vix_discrete/token_prior/freq_ema_alpha02_tokens \
  --output-dir outputs/latent_geometry/sp500_vix_freq_ema_alpha02 \
  --base-data-dir data/processed \
  --plot-voronoi
```

Geometry output path:

```text
outputs/latent_geometry/sp500_vix_freq_ema_alpha02
```

The geometry run reports:

```text
quantizer_type: vector
index shape: [4914, 60]
codebook source: direct_backend_state
projection method: sklearn_pca
active codes: 64 / 64
perplexity: 54.60705566
entropy: 4.00016308
```

Generated plot inventory:

- `codebook_projection.png`
- `code_usage_histogram.png`
- `codebook_usage_projection.png`
- `vix_bucket_code_usage.png`
- `token_trajectory_examples.png`
- `codebook_voronoi.png`

As expected, RVQ pair diagnostics are unavailable because this is a standard vector tokenizer with
one token per time step.

## Alpha 0.1 Comparison

| Tokenizer | Combined active codes | Perplexity | Entropy | very-low active/perplexity | very-high active/perplexity |
|---|---:|---:|---:|---:|---:|
| EMA alpha 0.1 | 64 / 64 | 53.01815796 | 3.97063446 | 60 / 37.3818 | 64 / 53.6918 |
| EMA alpha 0.2 | 64 / 64 | 54.60705566 | 4.00016308 | 61 / 40.1668 | 64 / 54.7603 |

Alpha `0.2` has slightly stronger geometry than alpha `0.1` by code usage and perplexity. The
tokenizer-only ablation still showed worse composed original-path reconstruction for alpha `0.2`,
so this geometry result should be interpreted as prior-readiness rather than promotion evidence.

## Prior Config

The additive VIX-only causal AR prior config was created at:

```text
configs/experiments/sp500_vix_causal_token_prior_freq_ema_alpha02.yaml
```

It preserves the alpha `0.1` prior architecture and training schedule, with:

```text
experiment.name: sp500_vix_causal_token_prior_freq_ema_alpha02
data.tokenizer_dir: outputs/sp500_vix_discrete/frequency_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_freq_ema_alpha02_seed0
data.token_data_dir: outputs/sp500_vix_discrete/token_prior/freq_ema_alpha02_tokens
model.condition_dim: 1
model.condition_injection: additive
```

## Decision

Alpha `0.2` clears the geometry gate for prior training. It has complete token artefacts, the
expected `[2457, 60]` single-token train/eval streams, full codebook usage, high perplexity, and
VIX-sensitive bucket structure.

The prior-training stage should still treat alpha `0.2` as an experimental comparator, not a
promotion candidate. Its purpose is to test the decision note's open question: whether stronger
geometry and broader code usage can reduce the alpha `0.1` MMD/SWD and terminal-return weaknesses
while retaining the volatility and squared-return autocorrelation gains.
