# S&P500/VIX Standard VQ Latent Geometry

Status: completed from regenerated public-repository artifacts.

The promoted standard S&P500/VIX VQ tokenizer was regenerated locally under ignored `outputs/`
paths in the public repository. No private `~/Desktop/time-causal-vae/outputs` artifacts were
used. W&B was requested for both tokenizer training and latent-geometry logging, but the local
W&B service failed to start because socket access was not permitted, so both runs were completed
without W&B.

## Commands Run

Tokenizer training was first attempted with W&B:

```bash
poetry run tcvae-train-tokenizer \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer.yaml \
  --output-dir outputs/sp500_vix_discrete/tokenizer \
  --base-data-dir data/processed \
  --wandb \
  --wandb-project time-causal-vq-tokenizer \
  --wandb-entity tc_vae \
  --wandb-run-name sp500_vix_causal_vq_tokenizer_codebook64dim16_seed0
```

It was rerun without W&B:

```bash
poetry run tcvae-train-tokenizer \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer.yaml \
  --output-dir outputs/sp500_vix_discrete/tokenizer \
  --base-data-dir data/processed \
  --no-wandb
```

Token extraction:

```bash
poetry run python scripts/extract_token_indices.py \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer.yaml \
  --tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --output-dir outputs/sp500_vix_discrete/token_prior/tokens_codebook64_codebookdim16 \
  --base-data-dir data/processed \
  --seed 99
```

Latent geometry was first attempted with W&B:

```bash
poetry run python scripts/analyze_discrete_latent_geometry.py \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer.yaml \
  --tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --token-data-dir outputs/sp500_vix_discrete/token_prior/tokens_codebook64_codebookdim16 \
  --output-dir outputs/latent_geometry/sp500_vix_standard_vq \
  --base-data-dir data/processed \
  --plot-voronoi \
  --wandb \
  --wandb-project time-causal-latent-diagnostics \
  --wandb-entity tc_vae \
  --run-name sp500_vix_standard_vq_geometry
```

It was rerun without W&B:

```bash
poetry run python scripts/analyze_discrete_latent_geometry.py \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer.yaml \
  --tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --token-data-dir outputs/sp500_vix_discrete/token_prior/tokens_codebook64_codebookdim16 \
  --output-dir outputs/latent_geometry/sp500_vix_standard_vq \
  --base-data-dir data/processed \
  --plot-voronoi
```

## Paths

- Tokenizer path:
  `outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0`
- Token data path:
  `outputs/sp500_vix_discrete/token_prior/tokens_codebook64_codebookdim16`
- Geometry output path:
  `outputs/latent_geometry/sp500_vix_standard_vq`
- W&B run URL: unavailable. W&B was skipped after local service/socket startup failed.

The actual tokenizer run directory differs from the earlier expected
`sp500_vix_causal_vq_tokenizer_codebook64dim16_seed0` name; the detected directory above was used.

## Token Artifacts

- `train_tokens.pt`: indices `[2457, 60]`, labels `[2457, 1]`
- `eval_tokens.pt`: indices `[2457, 60]`, labels `[2457, 1]`
- Combined geometry index shape: `[4914, 60]`

## Geometry Summary

- Quantizer type: `vector`
- Codebook size: `64`
- Codebook embedding shape: `[64, 16]`
- Active-code count: `63`
- Perplexity: `39.05571746826172`
- Entropy: `3.6649892330169678`
- Projection method: `sklearn_pca`
- Voronoi/fallback status: exact Voronoi succeeded and generated `codebook_voronoi.png`.

## VIX-Bucket Usage

| Bucket | Samples | VIX min | VIX max | Active codes | Perplexity | Entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| very_low | 983 | 0.11053332 | 0.15552060 | 53 | 28.48687363 | 3.34944344 |
| low | 983 | 0.15552060 | 0.18007015 | 56 | 32.33771515 | 3.47623420 |
| mid | 983 | 0.18031201 | 0.21755955 | 58 | 35.60297394 | 3.57242918 |
| high | 983 | 0.21755955 | 0.27512395 | 60 | 40.41440582 | 3.69918633 |
| very_high | 982 | 0.27536583 | 1.00000000 | 63 | 43.82186127 | 3.78013277 |

## Generated Figures

- `codebook_projection.png`
- `code_usage_histogram.png`
- `codebook_usage_projection.png`
- `vix_bucket_code_usage.png`
- `token_trajectory_examples.png`
- `codebook_voronoi.png`

Generated numeric summaries:

- `codebook_geometry_summary.json`
- `codebook_projection.csv`
- `latent_geometry_summary.md`

## Interpretation

Code usage is broad. The regenerated tokenizer uses 63 of 64 codes over the combined train and
eval token artifacts, and the perplexity of about 39 indicates that usage is distributed across a
substantial fraction of the codebook rather than collapsing to a small set of codes.

VIX buckets show condition-dependent usage. Active-code count and perplexity increase
monotonically from the lowest to highest VIX buckets, from 53 active codes and perplexity 28.49 in
the very-low VIX bucket to 63 active codes and perplexity 43.82 in the very-high VIX bucket. This
supports the interpretation that higher-volatility windows traverse a broader part of the
discrete latent space. The numeric summaries support differentiated bucket usage, while the
projection and heatmap figures should be used to inspect whether those differences appear as
distinct regions or as overlapping usage with changing mass.

The result supports keeping standard VQ as the promoted public baseline. The tokenizer has broad
code utilisation, VIX-sensitive usage structure, and a simple single-code interface compatible
with the promoted additive scalar-conditioned causal AR prior. No new grouped-tokenizer or RVQ
architecture is required by these diagnostics.
