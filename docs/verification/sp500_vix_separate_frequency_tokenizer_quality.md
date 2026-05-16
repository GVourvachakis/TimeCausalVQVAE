# S&P500/VIX Separate Frequency Tokenizer Quality

Status: tokenizer-quality verification for separate causal EMA low/high tokenizers. This run does
not implement or train the hierarchical prior, and it does not add GroupedRVQ, MGVQ, signatures,
diffusion, or cross-attention.

## Inputs

Configs:

```text
configs/experiments/sp500_vix_causal_vq_tokenizer_freq_low_alpha02.yaml
configs/experiments/sp500_vix_causal_vq_tokenizer_freq_high_alpha02.yaml
```

Both tokenizers use deterministic causal EMA decomposition with `alpha = 0.2`, standard vector
VQ, 64 codes, `codebook_dim = 16`, `condition_dim = 1`, and one-channel component inputs.

Training outputs:

```text
outputs/sp500_vix_discrete/separate_frequency_tokenizer/low/sp500_vix_freq_low_alpha02_tokenizer
outputs/sp500_vix_discrete/separate_frequency_tokenizer/high/sp500_vix_freq_high_alpha02_tokenizer
```

Evaluation outputs, using `n_sample_test = 512` and seed `99`:

```text
outputs/sp500_vix_discrete/separate_frequency_tokenizer/evaluation_low
outputs/sp500_vix_discrete/separate_frequency_tokenizer/evaluation_high
```

## W&B

Both requested W&B-profile runs completed successfully, so no `--no-wandb` fallback was needed.

| Component | W&B run |
| --- | --- |
| Low | <https://wandb.ai/tc_vae/time-causal-vq-tokenizer/runs/bs4emkjo> |
| High | <https://wandb.ai/tc_vae/time-causal-vq-tokenizer/runs/hjdtzxes> |

The runs emitted local CUDA-driver warnings and used CPU. These warnings did not block training.

## Training Metrics

| Component | Epoch | Recon loss | Total loss | Active codes | Perplexity | Runtime seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | 100 | 0.01268005 | 0.01271411 | 64 / 64 | 47.63845444 | 90.893 |
| High | 100 | 0.00091424 | 0.00091696 | 64 / 64 | 50.25286484 | 96.742 |

Both full training runs ended with complete active-code coverage. The high component is much
easier to reconstruct in absolute scale because it is an EMA residual, not a path-level series.

## Evaluation Metrics

Evaluation used the trained checkpoints with `n_sample_test = 512` and seed `99`.

| Component | Input shape | Recon L1 | Recon L2 | Volatility error | Active codes | Perplexity | Entropy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | `[512, 60, 1]` | 0.00759831 | 0.00911571 | 0.00124286 | 55 / 64 | 41.87561798 | 3.73470378 |
| High | `[512, 60, 1]` | 0.00076673 | 0.00096738 | 0.00014078 | 61 / 64 | 46.23094559 | 3.83364940 |

The high-component evaluator also reports a very large terminal-return error. This is not a
component-quality gate, because that diagnostic divides by the initial value as if the input were
a positive scalar price path. The high component is a residual around zero, so reconstruction L1,
L2, volatility error, and code usage are the relevant tokenizer diagnostics here.

## VIX-Bucket Usage

### Low Tokenizer

| Bucket | Samples | Recon L1 | Volatility error | Active codes | Perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| very_low | 103 | 0.00882117 | 0.00130120 | 44 / 64 | 24.95156670 |
| low | 103 | 0.00799769 | 0.00106813 | 47 / 64 | 26.50380898 |
| mid | 102 | 0.00767099 | 0.00115063 | 51 / 64 | 34.65065002 |
| high | 102 | 0.00724474 | 0.00121008 | 53 / 64 | 39.92096329 |
| very_high | 102 | 0.00624104 | 0.00148540 | 55 / 64 | 39.95336533 |

Low-token code usage increases with VIX bucket, from 44 active codes in the very-low bucket to
55 active codes in the very-high bucket. Reconstruction L1 improves slightly as VIX rises.

### High Tokenizer

| Bucket | Samples | Recon L1 | Volatility error | Active codes | Perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| very_low | 103 | 0.00076931 | 0.00014328 | 52 / 64 | 40.34392548 |
| low | 103 | 0.00077237 | 0.00016835 | 54 / 64 | 43.08211899 |
| mid | 102 | 0.00076274 | 0.00013110 | 54 / 64 | 44.19786453 |
| high | 102 | 0.00076378 | 0.00010901 | 54 / 64 | 45.13250732 |
| very_high | 102 | 0.00076535 | 0.00015186 | 58 / 64 | 41.91359711 |

High-token reconstruction is stable across VIX buckets. Active-code coverage remains high in all
buckets and reaches 58 active codes in the very-high bucket.

## Joint Alpha 0.2 Comparison

The retained joint EMA alpha 0.2 tokenizer reported the following tokenizer-only metrics in the
joint frequency decision notes:

| Tokenizer | Native/component L1 | Original/composed L1 | Original/composed L2 | Volatility error | Active codes | Perplexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint EMA alpha 0.2 | 0.00883413 | 0.01165698 | 0.01289017 | 0.00075628 | 60 / 64 | 47.7753 |
| Separate low alpha 0.2 | 0.00759831 | n/a | n/a | 0.00124286 | 55 / 64 | 41.87561798 |
| Separate high alpha 0.2 | 0.00076673 | n/a | n/a | 0.00014078 | 61 / 64 | 46.23094559 |

This comparison is component-level, not a generation claim. The separate tokenizers cannot yet be
judged on composed generated paths because the hierarchical causal prior is intentionally not
implemented in this stage. The low tokenizer gives better component L1 than the joint tokenizer's
native two-channel average, while the high tokenizer is substantially sharper on the residual
component. The joint tokenizer remains the only model here with a composed-path tokenizer metric.

## Decision

Decision: both separate tokenizers clear the tokenizer-quality gate for the next stage.

Rationale:

- Both 100-epoch training runs completed with W&B tracking and full train-set active-code
  coverage.
- The 512-sample evaluation keeps broad code usage: 55 / 64 active low codes and 61 / 64 active
  high codes.
- Evaluation perplexity is healthy for both streams: 41.87561798 for low and 46.23094559 for
  high.
- Low reconstruction is acceptable for the smoother path component.
- High reconstruction is strong and stable across VIX buckets.
- VIX-bucket code usage is not collapsed in either stream.

The next implementation step should remain limited to token extraction and the hierarchical
causal prior interface. These tokenizer results do not justify adding GroupedRVQ, MGVQ,
signatures, diffusion, cross-attention, learned filters, or any non-causal prior.
