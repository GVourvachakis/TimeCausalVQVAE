# S&P500/VIX Frequency Tokenizer Ablation Results

Status: tokenizer-only ablation completed. No token prior was trained, and no architecture beyond
the existing joint causal EMA frequency-tokenizer path was implemented.

## Commands

The requested W&B run was attempted first with:

```text
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread \
poetry run python scripts/run_sp500_vix_tokenizer_ablation.py ...
```

It failed during `wandb.init()` with a `CommError` after a 90-second initialisation timeout. The
same three-config ablation was then rerun locally with `--no-wandb`, as requested.

The completed local run used:

```text
outputs/sp500_vix_discrete/frequency_tokenizer_ablation
```

All runs used standard vector VQ with `codebook_size = 64`, `codebook_dim = 16`, `data_dim = 2`,
`compose_output = true`, 100 epochs, evaluation seed `99`, and `n_sample_test = 512`.

## Latent-Geometry Type Check

The `latent_geometry.py` fix for the earlier `no-any-return` errors is correct in the downstream
type pass. After the change, `poetry run mypy src/time_causal_vae` completed successfully with no
issues across 98 source files.

## Alpha Comparison

Report-facing frequency metrics should use the composed scalar path
`low_hat + high_hat`. Native two-channel reconstruction metrics are retained as auxiliary
tokenizer diagnostics.

| EMA alpha | Native frequency L1 | Original-path L1 | Original-path L2 | Low L1 | High L1 | Composed volatility error | Active codes | Perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.00545110 | 0.00669039 | 0.00808084 | 0.00760220 | 0.00329999 | 0.00067805 | 59 | 47.1625 |
| 0.2 | 0.00883413 | 0.01165698 | 0.01289017 | 0.01414645 | 0.00352180 | 0.00075628 | 60 | 47.7753 |
| 0.5 | 0.00577282 | 0.00681011 | 0.00857277 | 0.00796046 | 0.00358519 | 0.00080509 | 26 | 18.0589 |

Alpha `0.1` is the best overall tokenizer candidate from this ablation. It has the lowest
composed original-path reconstruction error, the lowest composed volatility reconstruction error,
and broad code usage. Alpha `0.2` uses the codebook slightly more broadly but reconstructs the
composed scalar path materially worse. Alpha `0.5` reconstructs the scalar path almost as well as
alpha `0.1`, but it uses only 26 of 64 codes and has much lower perplexity.

## VIX-Bucket Code Usage

| EMA alpha | very-low active/perplexity | low active/perplexity | mid active/perplexity | high active/perplexity | very-high active/perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 47 / 32.53 | 51 / 34.79 | 53 / 39.66 | 56 / 44.44 | 59 / 43.01 |
| 0.2 | 50 / 31.15 | 52 / 34.18 | 55 / 39.85 | 58 / 45.83 | 59 / 44.74 |
| 0.5 | 23 / 12.68 | 23 / 13.38 | 24 / 16.09 | 26 / 17.42 | 26 / 17.85 |

Alpha `0.1` and alpha `0.2` both preserve VIX-sensitive code usage: active-code count expands
from low-volatility to high-volatility buckets, and bucket perplexity rises strongly. Alpha `0.5`
also shows a monotone trend, but from a much narrower effective vocabulary.

## Baseline Comparison

The comparison below uses the existing tokenizer evaluation summaries for the promoted standard
VQ baseline and the hidden128 tokenizer. These are tokenizer reconstruction diagnostics, not
generated-path prior diagnostics.

| Tokenizer | Original-path L1 | Original-path L2 | Volatility reconstruction error | Active codes | Perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Promoted standard VQ | 0.01120215 | 0.01255246 | 0.00080998 | 57 | 33.2962 |
| Hidden128 VQ | 0.00592622 | 0.00715618 | 0.00079694 | 58 | 44.7630 |
| EMA alpha 0.1 | 0.00669039 | 0.00808084 | 0.00067805 | 59 | 47.1625 |
| EMA alpha 0.2 | 0.01165698 | 0.01289017 | 0.00075628 | 60 | 47.7753 |
| EMA alpha 0.5 | 0.00681011 | 0.00857277 | 0.00080509 | 26 | 18.0589 |

Relative to the promoted standard VQ tokenizer, alpha `0.1` improves original-path
reconstruction, composed volatility reconstruction, active-code count, and perplexity. Relative to
hidden128, alpha `0.1` has slightly worse original-path reconstruction, but better composed
volatility reconstruction and broader code usage.

The promoted standard VQ geometry remains stronger as a validated public baseline because it has
already been checked through token extraction and latent-geometry diagnostics. The frequency
tokenizers are candidates for the next prior-training stage, not promoted replacements.

## Candidate For Prior Training

Use `configs/experiments/sp500_vix_causal_vq_tokenizer_freq_ema_alpha01.yaml` as the first
frequency-tokenizer prior-training candidate.

Reasons:

- It gives the best composed original-path reconstruction among the EMA ablations.
- It gives the best composed volatility reconstruction error among the EMA ablations.
- It uses 59 of 64 codes globally with perplexity `47.1625`.
- It keeps VIX-bucket code usage broad and condition-sensitive.

Keep alpha `0.2` as a secondary candidate only if the next stage prioritises maximum active-code
coverage over composed-path reconstruction. Do not use alpha `0.5` for the first prior run because
its effective vocabulary is too narrow.

## Caveats

These are tokenizer-only metrics. They do not measure sampled-path MMD, SWD, volatility W1,
squared-return autocorrelation, or token-prior compatibility. Those require extracting tokens for
the selected frequency tokenizer, training a causal token prior, and running paper-style
generation diagnostics.

The ablation ran on CPU because CUDA initialisation fell back in the local environment. The runs
also emitted Matplotlib cache-directory warnings; neither warning blocked training or evaluation.
