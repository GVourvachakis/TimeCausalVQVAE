# S&P500/VIX RVQ q2 Latent Geometry

Status: completed from regenerated public-repository artifacts.

The S&P500/VIX RVQ q2 tokenizer was regenerated locally under ignored `outputs/` paths in the
public repository. No private `~/Desktop/time-causal-vae/outputs` artifacts were used. W&B was
requested for both tokenizer training and latent-geometry logging, but the local W&B service
failed to start because socket access was not permitted, so both runs were completed without W&B.

## Commands Run

Tokenizer training was first attempted with W&B:

```bash
poetry run tcvae-train-tokenizer \
  --config configs/experiments/sp500_vix_causal_rvq_tokenizer_q2.yaml \
  --output-dir outputs/sp500_vix_discrete/vq_family_tokenizer_ablation \
  --base-data-dir data/processed \
  --wandb \
  --wandb-project time-causal-vq-tokenizer \
  --wandb-entity tc_vae \
  --wandb-run-name sp500_vix_vq_family_tokenizer_sp500_vix_causal_rvq_tokenizer_q2_seed0
```

It was rerun without W&B:

```bash
poetry run tcvae-train-tokenizer \
  --config configs/experiments/sp500_vix_causal_rvq_tokenizer_q2.yaml \
  --output-dir outputs/sp500_vix_discrete/vq_family_tokenizer_ablation \
  --base-data-dir data/processed \
  --no-wandb
```

Token extraction:

```bash
poetry run python scripts/extract_token_indices.py \
  --config configs/experiments/sp500_vix_causal_rvq_tokenizer_q2.yaml \
  --tokenizer-dir outputs/sp500_vix_discrete/vq_family_tokenizer_ablation/sp500_vix_causal_rvq_tokenizer_q2_seed0 \
  --output-dir outputs/sp500_vix_discrete/token_prior/tokens_rvq_q2 \
  --base-data-dir data/processed \
  --seed 99
```

Latent geometry was first attempted with W&B:

```bash
poetry run python scripts/analyze_discrete_latent_geometry.py \
  --config configs/experiments/sp500_vix_causal_rvq_tokenizer_q2.yaml \
  --tokenizer-dir outputs/sp500_vix_discrete/vq_family_tokenizer_ablation/sp500_vix_causal_rvq_tokenizer_q2_seed0 \
  --token-data-dir outputs/sp500_vix_discrete/token_prior/tokens_rvq_q2 \
  --output-dir outputs/latent_geometry/sp500_vix_rvq_q2 \
  --base-data-dir data/processed \
  --plot-voronoi \
  --wandb \
  --wandb-project time-causal-latent-diagnostics \
  --wandb-entity tc_vae \
  --run-name sp500_vix_rvq_q2_geometry
```

It was rerun without W&B:

```bash
poetry run python scripts/analyze_discrete_latent_geometry.py \
  --config configs/experiments/sp500_vix_causal_rvq_tokenizer_q2.yaml \
  --tokenizer-dir outputs/sp500_vix_discrete/vq_family_tokenizer_ablation/sp500_vix_causal_rvq_tokenizer_q2_seed0 \
  --token-data-dir outputs/sp500_vix_discrete/token_prior/tokens_rvq_q2 \
  --output-dir outputs/latent_geometry/sp500_vix_rvq_q2 \
  --base-data-dir data/processed \
  --plot-voronoi
```

## Paths

- Tokenizer path:
  `outputs/sp500_vix_discrete/vq_family_tokenizer_ablation/sp500_vix_causal_rvq_tokenizer_q2_seed0`
- Token data path:
  `outputs/sp500_vix_discrete/token_prior/tokens_rvq_q2`
- Geometry output path:
  `outputs/latent_geometry/sp500_vix_rvq_q2`
- W&B run URL: unavailable. W&B was skipped after local service/socket startup failed.

The actual tokenizer run directory differs from the earlier expected
`sp500_vix_vq_family_tokenizer_sp500_vix_causal_rvq_tokenizer_q2_seed0` name; the detected
directory above was used.

## Token Artifacts

- `train_tokens.pt`: indices `[2457, 60, 2]`, labels `[2457, 1]`
- `eval_tokens.pt`: indices `[2457, 60, 2]`, labels `[2457, 1]`
- Combined geometry index shape: `[4914, 60, 2]`

## Geometry Summary

- Quantizer type: `residual_vq`
- Codebook size: `64`
- Quantizer count: `2`
- Codebook embedding shape: `[128, 16]`
- Codebook source: `direct_backend_state`
- Overall active-code count: `64`
- Overall perplexity: `23.462690353393555`
- Overall entropy: `3.1554114818573`
- Projection method: `sklearn_pca`
- Voronoi/fallback status: exact Voronoi succeeded and generated `codebook_voronoi.png`.

## q0/q1 Usage

| Component | Active codes | Perplexity | Entropy |
| --- | ---: | ---: | ---: |
| q0 | 6 | 4.36138487 | 1.47278965 |
| q1 | 64 | 39.99274445 | 3.68869805 |

Global top q0 codes were concentrated in six codes: `35`, `37`, `0`, `4`, `1`, and `33`.
Global q1 usage was much broader, with all 64 codes active.

## q0/q1 Same-Time Pair Summary

- Pair count: `294840`
- Active pair count: `231`
- Active pair ratio: `0.056396484375`
- Absent pair mass: `0.943603515625`
- Zero-count pairs: `3865` of `4096`
- Rare active pairs with count 1 to 5: `26`

Top observed q0/q1 pairs by count:

| q0 | q1 | Count |
| ---: | ---: | ---: |
| 35 | 42 | 9632 |
| 35 | 33 | 8906 |
| 37 | 43 | 8224 |
| 37 | 16 | 8134 |
| 35 | 57 | 7512 |
| 35 | 32 | 7336 |
| 35 | 45 | 7284 |
| 35 | 38 | 7164 |
| 35 | 43 | 7090 |
| 35 | 48 | 6992 |

## VIX-Bucket q0/q1 Usage

| Bucket | Samples | VIX min | VIX max | q0 active | q0 perplexity | q1 active | q1 perplexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| very_low | 983 | 0.11053332 | 0.15552060 | 6 | 3.84848523 | 57 | 31.12134361 |
| low | 983 | 0.15552060 | 0.18007015 | 6 | 4.05218744 | 57 | 33.47520065 |
| mid | 983 | 0.18031201 | 0.21755955 | 6 | 4.16901302 | 58 | 33.92953873 |
| high | 983 | 0.21755955 | 0.27512395 | 6 | 4.56048584 | 58 | 34.05591202 |
| very_high | 982 | 0.27536583 | 1.00000000 | 6 | 4.78456640 | 64 | 35.02769470 |

q0 remains a six-code component in every VIX bucket, but its perplexity increases with VIX. q1
is broad in every bucket and reaches all 64 active codes in the very-high VIX bucket.

## Generated Figures

- `codebook_projection.png`
- `code_usage_histogram.png`
- `codebook_usage_projection.png`
- `vix_bucket_code_usage.png`
- `token_trajectory_examples.png`
- `codebook_voronoi.png`
- `q0_q1_pair_heatmap.png`

Generated numeric summaries:

- `codebook_geometry_summary.json`
- `codebook_projection.csv`
- `latent_geometry_summary.md`
- `q0_q1_pair_summary.json`

## Interpretation

q0 behaves like a coarse component. It uses only six codes globally and in every VIX bucket, with
low perplexity relative to the 64-code codebook. Its bucket-level perplexity increases from 3.85
in very-low VIX to 4.78 in very-high VIX, so q0 is not a single static label, but it is clearly
much lower entropy than q1.

q1 behaves like a residual/detail component. It uses all 64 codes globally, has perplexity about
40, and expands from 57 active codes in the lowest VIX buckets to all 64 in the very-high VIX
bucket. This is consistent with q1 carrying within-regime detail and higher-volatility variation.

The pair geometry helps explain why RVQ q2 can reconstruct well while remaining difficult for
generation. Reconstruction can exploit the two-stage representation directly, but generation must
model a sparse and structured same-time joint distribution: only 231 of 4096 q0/q1 pairs are
observed, and 94.36 percent of pair mass is absent. A multi-code prior must preserve q0/q1
compatibility while also modelling time dynamics and VIX-dependent q1 detail. If it samples
plausible marginals but weak same-time compatibility, decoded paths can degrade even when
tokenizer reconstruction improves.

RVQ q2 should therefore remain an ablation rather than the promoted public baseline. The geometry
is interpretable, but it also shows a harder generative modelling problem than standard VQ. The
promoted standard VQ tokenizer keeps a simpler single-code interface with broad code utilisation
and condition-sensitive usage, which better matches the current public additive
scalar-conditioned causal AR prior.
