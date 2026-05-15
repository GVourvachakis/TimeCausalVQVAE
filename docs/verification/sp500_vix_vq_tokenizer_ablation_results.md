# S&P500/VIX VQ Tokenizer Ablation Results

Status: non-smoke standard-VQ tokenizer ablations completed. No token priors were trained, and no
RVQ, GroupedRVQ, MGVQ, diffusion, or signature-conditioning changes were introduced.

## W&B Status

The requested online W&B run was attempted first with:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread \
poetry run python scripts/run_sp500_vix_tokenizer_ablation.py ...
```

W&B found a configured API key, but `wandb.init()` entered repeated `ConnectionError` retries and
failed before training with:

```text
wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec.
```

An escalated retry for live W&B network/socket access was requested, but the approval reviewer
rejected it because online W&B would export run metadata to a third-party service. No W&B URLs are
therefore available for this run.

The same ablation command was rerun with `--no-wandb`. All training and evaluation artefacts were
written under:

```text
outputs/sp500_vix_discrete/vq_tokenizer_ablation/
```

The aggregate files are:

- `tokenizer_ablation_summary.json`;
- `tokenizer_ablation_summary.csv`.

The promoted baseline checkpoint was also evaluated on the same 512-path, seed-99 evaluation
slice for comparison only:

```text
outputs/sp500_vix_discrete/vq_tokenizer_ablation/promoted_baseline_seed0_evaluation/
```

## Aggregate Results

All rows use `n_sample_test=512` and `seed=99`. Runtime is the runner-level time per candidate,
including training and evaluation.

| Candidate | Runtime s | Recon L1 | Recon L2 | Terminal error | Volatility error | Active codes | Perplexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cb32_dim16` | 99.872 | 0.00751569 | 0.00920557 | 0.00595911 | 0.00178363 | 28/32 | 18.53537941 |
| `cb64_dim8` | 105.321 | 0.00461162 | 0.00632981 | 0.01054683 | 0.00154886 | 57/64 | 33.51201248 |
| `cb64_dim32` | 107.280 | 0.00750538 | 0.00885949 | 0.00547652 | 0.00125068 | 57/64 | 37.57756424 |
| `cb128_dim16` | 111.679 | 0.00880024 | 0.00992289 | 0.00543968 | 0.00148102 | 121/128 | 83.87483215 |
| `commitment005` | 99.844 | 0.01045772 | 0.01200777 | 0.00707890 | 0.00182669 | 58/64 | 33.00460434 |
| `commitment025` | 101.692 | 0.01275175 | 0.01412332 | 0.00880611 | 0.00178228 | 55/64 | 29.49415779 |
| `hidden128` | 203.546 | 0.00592622 | 0.00715618 | 0.00472229 | 0.00079694 | 58/64 | 44.76303864 |
| `dilations124816` | 120.412 | 0.03228592 | 0.03404456 | 0.01671461 | 0.00107547 | 56/64 | 34.20163727 |
| promoted baseline seed0 | n/a | 0.01120215 | 0.01255246 | 0.00556620 | 0.00080998 | 57/64 | 33.29620361 |

## VIX-Bucket Code Usage

Each cell reports active codes and codebook perplexity as `active / perplexity`.

| Candidate | Very low | Low | Mid | High | Very high |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cb32_dim16` | 22 / 11.01 | 23 / 11.51 | 26 / 15.10 | 27 / 18.02 | 28 / 20.46 |
| `cb64_dim8` | 41 / 19.17 | 44 / 21.60 | 50 / 28.21 | 52 / 32.00 | 56 / 32.77 |
| `cb64_dim32` | 43 / 21.59 | 44 / 23.16 | 48 / 30.60 | 52 / 36.51 | 55 / 34.92 |
| `cb128_dim16` | 91 / 50.45 | 99 / 56.91 | 104 / 71.37 | 112 / 73.91 | 119 / 58.64 |
| `commitment005` | 44 / 19.45 | 45 / 21.21 | 48 / 27.29 | 52 / 30.16 | 58 / 34.10 |
| `commitment025` | 37 / 16.19 | 40 / 16.76 | 47 / 22.31 | 50 / 29.73 | 55 / 28.43 |
| `hidden128` | 47 / 29.33 | 53 / 31.38 | 54 / 38.58 | 56 / 41.69 | 58 / 39.21 |
| `dilations124816` | 37 / 17.83 | 38 / 19.42 | 41 / 25.33 | 50 / 30.95 | 53 / 35.01 |
| promoted baseline seed0 | 37 / 18.88 | 41 / 20.33 | 45 / 27.23 | 52 / 33.11 | 55 / 35.59 |

Most viable candidates preserve the desirable VIX-sensitive pattern: active-code count and
perplexity generally increase from low-VIX to high-VIX buckets. The strongest version of this
pattern among 64-code candidates is `hidden128`, which combines high global perplexity with
controlled reconstruction errors.

## Interpretation

`hidden128` is the strongest tokenizer-side candidate. It improves reconstruction L1 relative to
the promoted baseline from `0.01120215` to `0.00592622`, improves terminal-return error from
`0.00556620` to `0.00472229`, and essentially matches the baseline volatility reconstruction
error. It also increases global codebook perplexity from `33.29620361` to `44.76303864` on the
same evaluation slice. The main cost is runtime: about `203.546` seconds for this run, roughly
twice the other 64-code variants.

`cb64_dim8` gives the best raw reconstruction L1 and L2, but its terminal-return error
(`0.01054683`) is substantially worse than both the promoted baseline and `hidden128`. It should
therefore not be the first prior-training candidate unless the next phase deliberately tests a
reconstruction-only tokenizer stress case.

`cb64_dim32` is a conservative second candidate. It improves reconstruction L1 relative to the
promoted baseline, keeps terminal-return error approximately baseline-level, and raises
perplexity to `37.57756424`. Its volatility reconstruction error is worse than `hidden128` and
the promoted baseline, but not as severe as the terminal-return regression seen in `cb64_dim8`.

`cb128_dim16` uses the larger vocabulary broadly, but the reconstruction improvement over the
promoted baseline is modest relative to the added codebook size. It would also require a larger
prior vocabulary, so it is not the next prior-training choice.

The commitment-weight variants do not improve the promoted baseline. Lower commitment
(`0.05`) increases active-code usage but weakens reconstruction and market reconstruction
metrics. Higher commitment (`0.25`) is worse across the main aggregate metrics.

The extended dilation variant should be rejected for now. Its final training loss destabilised
late in training, and its evaluation reconstruction L1 (`0.03228592`) is far worse than all
other candidates despite reasonable code usage.

## Latent Geometry Recommendation

Before training token priors, run latent-geometry diagnostics for:

1. `hidden128`, as the primary tokenizer candidate;
2. `cb64_dim32`, as the conservative codebook-dimension candidate;
3. `cb64_dim8`, only if a reconstruction-only ablation is kept for context.

The geometry check should confirm:

- broad code usage without a collapsed subset;
- VIX-bucket mass shifts similar to the promoted baseline;
- token trajectory interpretability;
- no pathological concentration in the high-VIX bucket.

The promoted baseline still has the strongest documented full-split geometry record:
`63/64` active codes and perplexity `39.05571746826172` over the combined train/eval token
artefacts in `docs/verification/sp500_vix_standard_vq_latent_geometry.md`. The new candidates
should not replace it in the promoted architecture until their full latent-geometry diagnostics
and downstream prior results are available.

## Prior-Training Candidates

Recommended top two for subsequent prior training:

1. `sp500_vix_causal_vq_tokenizer_hidden128`: best overall reconstruction and market-relevant
   reconstruction metrics, while preserving a simple 64-code prior interface.
2. `sp500_vix_causal_vq_tokenizer_cb64_dim32`: conservative second candidate with baseline-like
   terminal-return error, improved reconstruction L1, and the same 64-code prior vocabulary.

Do not advance `dilations124816` to prior training. Treat `cb64_dim8` as an auxiliary
reconstruction-stress ablation rather than a primary market-generation candidate because its
terminal-return reconstruction error is too high.

## Comparison To Promoted Baseline

The promoted baseline tokenizer remains the official default. It is still the only tokenizer in
this set with completed prior-facing promoted-method context and full latent-geometry
documentation.

On the same 512-path evaluation slice, `hidden128` is the only ablation that clearly improves the
baseline on reconstruction L1, terminal-return error, volatility reconstruction error, and
perplexity together. This makes it the best candidate for the next controlled prior-training
stage, but not yet a replacement for the promoted baseline.

`cb64_dim32` is the most reasonable second path because it tests whether a wider codebook
embedding improves the tokenizer without changing the prior vocabulary. It should be compared
against `hidden128` only after both have latent geometry and token-prior results.
