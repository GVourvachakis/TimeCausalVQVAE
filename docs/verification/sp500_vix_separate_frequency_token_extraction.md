# S&P500/VIX Separate Frequency Token Extraction

Status: paired-token extraction for the trained separate causal EMA alpha 0.2 tokenizers. This
stage does not implement or train the hierarchical prior, and it does not add GroupedRVQ, MGVQ,
signatures, diffusion, or cross-attention.

## Inputs

Trained tokenizers:

```text
low:
outputs/sp500_vix_discrete/separate_frequency_tokenizer/low/sp500_vix_freq_low_alpha02_tokenizer

high:
outputs/sp500_vix_discrete/separate_frequency_tokenizer/high/sp500_vix_freq_high_alpha02_tokenizer
```

Configs:

```text
configs/experiments/sp500_vix_causal_vq_tokenizer_freq_low_alpha02.yaml
configs/experiments/sp500_vix_causal_vq_tokenizer_freq_high_alpha02.yaml
```

Both tokenizers use deterministic causal EMA decomposition with `alpha = 0.2`, one scalar
component input channel, standard vector VQ, and 64-code codebooks.

## Token Exports

Low-token extraction:

```text
poetry run python scripts/extract_token_indices.py \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer_freq_low_alpha02.yaml \
  --tokenizer-dir outputs/sp500_vix_discrete/separate_frequency_tokenizer/low/sp500_vix_freq_low_alpha02_tokenizer \
  --output-dir outputs/sp500_vix_discrete/token_prior/freq_low_alpha02_tokens \
  --base-data-dir data/processed \
  --seed 99
```

High-token extraction:

```text
poetry run python scripts/extract_token_indices.py \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer_freq_high_alpha02.yaml \
  --tokenizer-dir outputs/sp500_vix_discrete/separate_frequency_tokenizer/high/sp500_vix_freq_high_alpha02_tokenizer \
  --output-dir outputs/sp500_vix_discrete/token_prior/freq_high_alpha02_tokens \
  --base-data-dir data/processed \
  --seed 99
```

Export paths:

```text
outputs/sp500_vix_discrete/token_prior/freq_low_alpha02_tokens
outputs/sp500_vix_discrete/token_prior/freq_high_alpha02_tokens
```

The low and high exports each contain `train_tokens.pt`, `eval_tokens.pt`, and
`token_dataset_summary.json`. The payloads store token indices, component data, and VIX labels.

## Paired Dataset

Paired assembly command:

```text
poetry run python scripts/build_separate_frequency_token_dataset.py \
  --low-token-dir outputs/sp500_vix_discrete/token_prior/freq_low_alpha02_tokens \
  --high-token-dir outputs/sp500_vix_discrete/token_prior/freq_high_alpha02_tokens \
  --output-dir outputs/sp500_vix_discrete/token_prior/separate_freq_alpha02_tokens \
  --base-data-dir data/processed \
  --alpha 0.2
```

Paired output path:

```text
outputs/sp500_vix_discrete/token_prior/separate_freq_alpha02_tokens
```

Files:

```text
train_low_tokens.pt
train_high_tokens.pt
eval_low_tokens.pt
eval_high_tokens.pt
train_labels.pt
eval_labels.pt
train_data.pt
eval_data.pt
paired_token_dataset_summary.json
paired_token_dataset_summary.md
```

`train_data.pt` and `eval_data.pt` store the recomposed scalar path
`low_component + high_component`, which reconstructs the original scalar EMA input path.

## Shape Validation

| Tensor | Shape |
| --- | --- |
| train low tokens | `[2457, 60]` |
| train high tokens | `[2457, 60]` |
| eval low tokens | `[2457, 60]` |
| eval high tokens | `[2457, 60]` |
| train labels | `[2457, 1]` |
| eval labels | `[2457, 1]` |
| train low component data | `[2457, 60, 1]` |
| train high component data | `[2457, 60, 1]` |
| eval low component data | `[2457, 60, 1]` |
| eval high component data | `[2457, 60, 1]` |
| train recomposed scalar data | `[2457, 60, 1]` |
| eval recomposed scalar data | `[2457, 60, 1]` |

The paired builder validated that low/high token shapes match, component-data shapes match,
sample counts match, and low/high VIX labels match exactly for both train and eval splits.

## Code Usage

| Stream | Split | Active codes | Perplexity | Entropy | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Low | Train | 64 / 64 | 48.94255066 | 3.89064717 | 147420 |
| Low | Eval | 64 / 64 | 48.94255066 | 3.89064717 | 147420 |
| Low | Combined | 64 / 64 | 48.94255066 | 3.89064717 | 294840 |
| High | Train | 64 / 64 | 50.19335175 | 3.91588259 | 147420 |
| High | Eval | 64 / 64 | 50.19335175 | 3.91588259 | 147420 |
| High | Combined | 64 / 64 | 50.19335175 | 3.91588259 | 294840 |

Both streams retain complete codebook support after extraction.

## Same-Time Pair Support

| Split | Active pairs | Pair ratio | Pair perplexity | Pair entropy | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 3044 / 4096 | 0.74316406 | 1602.10266113 | 7.37907219 | 147420 |
| Eval | 3044 / 4096 | 0.74316406 | 1602.10266113 | 7.37907219 | 147420 |
| Combined | 3044 / 4096 | 0.74316406 | 1602.10266113 | 7.37907219 | 294840 |

The same-time support confirms that the low/high streams do not collapse to a small set of paired
states. The later hierarchical prior should model the pair compatibility causally by sampling
`low_t` first and then sampling `high_t` conditioned on the sampled current low token.

## VIX-Bucket Usage

Combined low-token usage by VIX bucket:

| Bucket | Samples | Active codes | Perplexity |
| --- | ---: | ---: | ---: |
| very_low | 983 | 55 / 64 | 37.73524475 |
| low | 983 | 57 / 64 | 41.86462784 |
| mid | 983 | 59 / 64 | 44.80484009 |
| high | 983 | 59 / 64 | 48.72618866 |
| very_high | 982 | 64 / 64 | 47.99995422 |

Combined high-token usage by VIX bucket:

| Bucket | Samples | Active codes | Perplexity |
| --- | ---: | ---: | ---: |
| very_low | 983 | 57 / 64 | 41.11725235 |
| low | 983 | 59 / 64 | 45.60589218 |
| mid | 983 | 59 / 64 | 46.17083740 |
| high | 983 | 60 / 64 | 48.56171036 |
| very_high | 982 | 63 / 64 | 48.60480118 |

The VIX-bucket summaries were carried through from the component token exports. Both streams keep
broad support across condition buckets, with the highest VIX bucket using nearly the full codebook
in both streams.

## Readiness

The paired alpha 0.2 token dataset is ready for the next hierarchical causal-prior implementation
stage. The required prior inputs are present as matched tensors:

```text
low_tokens: [batch, time]
high_tokens: [batch, time]
labels: [batch, 1]
```

The next stage should remain limited to the specified hierarchical causal factorisation:

```text
p(low_t | low_<t, high_<t, VIX)
p(high_t | low_<=t, high_<t, VIX)
```

No generation-quality conclusion is made here. This extraction only verifies that the trained
separate tokenizers can provide aligned low/high token streams and scalar reference paths for
future prior training and evaluation.

## Caveats

The train and eval token exports are deterministic for these configs and seed, so their summary
statistics match exactly. This is expected from the current S&P500/VIX data pipeline and does not
replace a later held-out prior evaluation.

The extraction commands emitted local warnings about the Matplotlib cache directory and CUDA
driver availability. They did not block CPU extraction or paired-dataset assembly.
