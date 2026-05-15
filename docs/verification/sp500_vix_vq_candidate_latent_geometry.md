# S&P500/VIX VQ Candidate Latent Geometry

Status: completed for the two standard-VQ tokenizer candidates selected after the tokenizer
ablation and candidate-prior runs. No models were trained, no tokenizer or prior code was
changed, and no RVQ, GroupedRVQ, MGVQ, diffusion, or signature-conditioning component was added.

## Inputs

The requested tokenizer checkpoints were present:

- `outputs/sp500_vix_discrete/vq_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_hidden128_seed99`
- `outputs/sp500_vix_discrete/vq_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_cb64_dim32_seed99`

The matching extracted token artefacts were already present from the candidate-prior stage, so
no token re-extraction was required:

- `outputs/sp500_vix_discrete/token_prior/sp500_vix_causal_vq_tokenizer_hidden128_tokens`
- `outputs/sp500_vix_discrete/token_prior/sp500_vix_causal_vq_tokenizer_cb64_dim32_tokens`

Each token directory contains `train_tokens.pt`, `eval_tokens.pt`, and
`token_dataset_summary.json` with train and eval token shapes `[2457, 60]`. The geometry script
therefore analysed combined index tensors with shape `[4914, 60]`.

## W&B Logging Attempt

The latent-geometry script supports optional W&B logging via `--wandb`. A live HTTPS attempt was
made first for `hidden128` with:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread \
WANDB_BASE_URL=https://api.wandb.ai WANDB_INIT_TIMEOUT=120 \
poetry run python scripts/analyze_discrete_latent_geometry.py ... --wandb
```

The local geometry artefacts were generated, but `wandb.init()` entered repeated
`ConnectionError` retries and failed with:

```text
wandb.errors.errors.CommError: Run initialization has timed out after 120.0 sec.
```

An elevated live HTTPS retry was then requested for W&B, explicitly to send summary metrics,
config metadata, and generated diagnostic plots. The execution policy rejected that request
because it would export local workspace data to W&B. No further W&B workaround was attempted.
The final successful geometry commands omitted `--wandb`. There is no separate `--no-wandb`
flag for this script.

No W&B URLs are available for these diagnostics.

## Commands Run

`hidden128`:

```bash
MPLBACKEND=Agg poetry run python scripts/analyze_discrete_latent_geometry.py \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml \
  --tokenizer-dir outputs/sp500_vix_discrete/vq_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_hidden128_seed99 \
  --token-data-dir outputs/sp500_vix_discrete/token_prior/sp500_vix_causal_vq_tokenizer_hidden128_tokens \
  --output-dir outputs/latent_geometry/sp500_vix_hidden128_vq \
  --base-data-dir data/processed \
  --plot-voronoi
```

`cb64_dim32`:

```bash
MPLBACKEND=Agg poetry run python scripts/analyze_discrete_latent_geometry.py \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer_cb64_dim32.yaml \
  --tokenizer-dir outputs/sp500_vix_discrete/vq_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_cb64_dim32_seed99 \
  --token-data-dir outputs/sp500_vix_discrete/token_prior/sp500_vix_causal_vq_tokenizer_cb64_dim32_tokens \
  --output-dir outputs/latent_geometry/sp500_vix_cb64_dim32_vq \
  --base-data-dir data/processed \
  --plot-voronoi
```

## Geometry Summary

| Tokenizer | Codebook shape | Active codes | Perplexity | Entropy | Pairwise distance mean | Pairwise distance min | Pairwise distance max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Promoted baseline | `[64, 16]` | 63/64 | 39.055717 | 3.664989 | n/a | n/a | n/a |
| `hidden128` | `[64, 16]` | 64/64 | 50.358673 | 3.919171 | 0.487139 | 0.022891 | 3.079115 |
| `cb64_dim32` | `[64, 32]` | 64/64 | 45.641132 | 3.820809 | 0.388488 | 0.022183 | 2.398700 |

Both candidates improve global code coverage relative to the promoted baseline. `hidden128`
has the strongest usage evidence: all codes are active and global perplexity is about 11.30
points higher than the promoted baseline. `cb64_dim32` also uses all codes, but its usage is less
broad than `hidden128` despite the wider `codebook_dim=32` embedding.

The PCA projection and exact Voronoi plot were generated for both candidates. The only unavailable
diagnostic was RVQ q0/q1 pair analysis, which is expected because these are single-vector
standard-VQ tokenizers with index shape `[batch, time]`.

## VIX-Bucket Usage

Each cell reports active codes and codebook perplexity as `active / perplexity`.

| Tokenizer | Very low | Low | Mid | High | Very high |
| --- | ---: | ---: | ---: | ---: | ---: |
| Promoted baseline | 53 / 28.49 | 56 / 32.34 | 58 / 35.60 | 60 / 40.41 | 63 / 43.82 |
| `hidden128` | 61 / 40.77 | 61 / 44.07 | 62 / 46.61 | 61 / 49.44 | 64 / 48.22 |
| `cb64_dim32` | 50 / 31.94 | 53 / 34.18 | 55 / 38.31 | 57 / 42.15 | 62 / 37.73 |

`hidden128` has the strongest VIX-bucket support. It uses at least 61 codes in every bucket and
reaches all 64 codes in the very-high VIX bucket. Its perplexity rises from very-low to high VIX,
then remains high in very-high VIX. The slight very-high perplexity decline is not a collapse,
because active support still expands to the full codebook.

`cb64_dim32` preserves the expected rise in active-code count from very-low to very-high VIX, but
its very-high perplexity falls below its high-bucket value. This suggests broader support in the
highest VIX bucket with more concentrated mass than `hidden128`.

## Plot Inventory

Both output directories contain:

- `codebook_projection.png`
- `code_usage_histogram.png`
- `codebook_usage_projection.png`
- `vix_bucket_code_usage.png`
- `token_trajectory_examples.png`
- `codebook_voronoi.png`
- `codebook_geometry_summary.json`
- `codebook_projection.csv`
- `latent_geometry_summary.md`

Output directories:

- `outputs/latent_geometry/sp500_vix_hidden128_vq`
- `outputs/latent_geometry/sp500_vix_cb64_dim32_vq`

## Comparison To Promoted Baseline

The promoted baseline remains the official architecture baseline because it already has a mature
geometry record and downstream promoted-method context. Its documented combined geometry was
63/64 active codes with perplexity `39.0557`.

Relative to that baseline, `hidden128` now has stronger geometry support:

- 64/64 active codes rather than 63/64;
- global perplexity `50.3587` rather than `39.0557`;
- at least 61 active codes in every VIX bucket;
- a generated-market prior profile that previously improved the promoted baseline
  (`0.266589` versus `0.298020`).

`cb64_dim32` also clears the basic geometry gate:

- 64/64 active codes;
- global perplexity `45.6411`;
- monotone active-code expansion across VIX buckets;
- token-prior likelihood close to the promoted baseline.

However, `cb64_dim32` remains weaker than `hidden128` as a generation candidate because its prior
evaluation regressed terminal and volatility Wasserstein metrics after decoding.

## Decision

`hidden128` has enough latent-geometry support to continue. It should remain the leading
standard-VQ tokenizer candidate for the next validation step, namely repeat-seed prior training
and paper-style evaluation before any promotion decision.

Do not replace the promoted baseline yet. The `hidden128` evidence is now positive on tokenizer
reconstruction, candidate-prior generated-market diagnostics, and latent geometry, but promotion
still needs repeatability across seeds and a final comparison under the exact promoted evaluation
profile.

`cb64_dim32` should remain secondary. It is a useful control for wider codebook embeddings and
token likelihood, but the current evidence says its improved prior likelihood does not translate
as cleanly into decoded market diagnostics.
