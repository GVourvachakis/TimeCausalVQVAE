# S&P500/VIX Standard VQ Latent Geometry

Status: completed from regenerated public-repository artifacts.

The standard S&P500/VIX VQ tokenizer was available locally under ignored `outputs/` paths in the
public repository. Token indices and latent-geometry diagnostics were regenerated from that local
checkpoint. No private `~/Desktop/time-causal-vae/outputs` artifacts were used, and no token prior
was trained.

## Commands Run

Token extraction:

```bash
poetry run python scripts/extract_token_indices.py \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer.yaml \
  --tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --output-dir outputs/sp500_vix_discrete/token_prior/tokens_codebook64_codebookdim16 \
  --base-data-dir data/processed \
  --seed 99
```

Latent geometry:

```bash
poetry run python scripts/analyze_discrete_latent_geometry.py \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer.yaml \
  --tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --token-data-dir outputs/sp500_vix_discrete/token_prior/tokens_codebook64_codebookdim16 \
  --output-dir outputs/latent_geometry/sp500_vix_standard_vq \
  --base-data-dir data/processed \
  --plot-voronoi
```

An older empty directory named
`outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_codebook64dim16_seed0` was
ignored because it did not contain `tokenizer.pt`.

## Paths

- Tokenizer path:
  `outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0`
- Token-data path:
  `outputs/sp500_vix_discrete/token_prior/tokens_codebook64_codebookdim16`
- Geometry output path:
  `outputs/latent_geometry/sp500_vix_standard_vq`

## Token Artifacts

- `train_tokens.pt`: indices `[2457, 60]`, labels `[2457, 1]`
- `eval_tokens.pt`: indices `[2457, 60]`, labels `[2457, 1]`
- Combined geometry index shape: `[4914, 60]`

## Geometry Summary

- Quantizer type: `vector`
- Codebook size: `64`
- Codebook embedding shape: `[64, 16]`
- Codebook source: `direct_backend_state`
- Active-code count: `63`
- Active-code ratio: `0.984375`
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

The active-code count and perplexity rise monotonically across VIX buckets. This indicates that
higher-volatility windows use a broader part of the discrete codebook rather than only reweighting
the same small set of codes.

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

Standard VQ code usage is broad: 63 of 64 codes are active and the global perplexity is about 39
on a 64-code vocabulary. The PCA projection and usage overlay therefore describe a populated
latent map rather than a collapsed codebook.

The VIX-bucket diagnostics show condition-sensitive structure. Very-low VIX windows use 53 active
codes with lower perplexity, while very-high VIX windows use 63 active codes with substantially
higher perplexity. The geometry therefore supports the scalar-conditioned setup: VIX affects how
mass moves through the codebook without requiring a multi-code tokenizer.

This regenerated result supports keeping standard VQ as the promoted public baseline. It has
broad code utilisation, VIX-sensitive discrete structure, exact geometry diagnostics, and the
simplest one-code-per-time-step interface for the additive scalar-conditioned causal AR prior.
