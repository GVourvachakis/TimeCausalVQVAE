# Final Deliverable Plan

Status: final course-project plan. This document defines the remaining report and notebook
deliverables for TimeCausalVQVAE. It does not introduce a new architecture and does not require
model training during plan preparation.

## 1. Final Project Thesis

The final project argues that financial time-series generation can be made more interpretable by
separating the problem into a causal discrete tokenizer and a causal token prior. For S&P500/VIX,
the promoted standard causal VQ tokenizer gives one code per time step, uses the codebook broadly,
and shows VIX-sensitive code usage. This makes the latent representation reportable: the model can
be evaluated not only by generated-path metrics, but also by codebook utilisation, volatility-
conditioned token usage, and token trajectories.

The final thesis is therefore:

> A standard causal VQ tokenizer, paired with an additive scalar-conditioned causal AR token prior,
> is a compact and interpretable public baseline for S&P500/VIX path generation. Its discrete
> latent geometry supports the use of VIX conditioning without requiring multi-code tokenizers or
> diffusion-style sampling.

The report should emphasise what is promoted and what is deliberately deferred. RVQ q2 is a useful
ablation because it reveals a coarse/detail split, but it also creates sparse same-time q0/q1
support. GroupedRVQ, MGVQ, diffusion, AdaLN, and cross-attention remain future-work ideas rather
than final-method components.

## 2. Promoted Architecture

The promoted architecture is:

```text
standard causal VQ tokenizer + additive scalar-conditioned causal AR token prior
```

Tokenizer configuration:

- Config: `configs/experiments/sp500_vix_causal_vq_tokenizer.yaml`.
- Dataset: S&P500/VIX.
- Sequence length: 60.
- Data dimension: 1.
- Condition dimension: 1.
- Condition feature: VIX.
- Embedding dimension: 64.
- Codebook size: 64.
- Codebook dimension: 16.
- Causal convolutional stack: 4 layers with dilations `[1, 2, 4, 8]`.
- Commitment weight: `0.1`.
- K-means initialisation: enabled with 10 iterations.

Token prior configuration:

- Config: `configs/experiments/sp500_vix_causal_token_prior_additive.yaml`.
- Family: `causal_token_prior`.
- Codebook size: 64.
- Sequence length: 60.
- Token embedding dimension: 128.
- Transformer depth: 4 layers, 4 heads, MLP hidden dimension 256.
- Dropout: `0.1`.
- Prediction convention: `bos_shifted_next_token`.
- BOS token id: 64.
- Condition dimension: 1.
- Condition injection: additive.

The final write-up should describe the architecture as a two-stage causal model:

1. The tokenizer maps each path to a sequence of discrete codes while preserving causal temporal
   structure.
2. The additive conditional AR prior samples code sequences using past tokens and the scalar VIX
   condition.
3. The frozen tokenizer decoder maps sampled token sequences back to path space.

## 3. Baselines

Use three baseline categories.

| Baseline | Role in final deliverable |
| --- | --- |
| Continuous TC-VAE S&P500/VIX baseline | Main continuous reference for generated-path metrics and source-paper comparison. |
| Black-Scholes, Heston, and PDV continuous configs | Compact sanity baselines and reproducibility anchors, not the main empirical claim. |
| RVQ q2 S&P500/VIX ablation | Discrete-tokenizer ablation that explains why standard VQ is promoted. |

The continuous S&P500/VIX baseline comes from `configs/experiments/sp500_vix_beta_cvae.yaml` and
the Time-Causal VAE reference protocol. It should be used only as a comparison point, not as the
promoted final method.

The RVQ q2 ablation should be presented as evidence about representation trade-offs:

- q0 uses 6 active codes with perplexity `4.361384868621826`.
- q1 uses all 64 codes with perplexity `39.99274444580078`.
- Only 231 of 4096 same-time q0/q1 pairs are active.
- The active pair ratio is `0.056396484375`.
- The absent pair mass is `0.943603515625`.

This supports the claim that multi-code tokenisation can be interpretable while making prior
sampling harder.

## 4. Datasets

Primary dataset:

- S&P500/VIX, stored locally as `data/processed/sp500vix/sp500vix_normalized.npy`.
- This local array is not committed.
- The S&P500/VIX tokenizer and token-prior configs use 2457 samples, sequence length 60, data
  dimension 1, and condition dimension 1.
- The VIX scalar is the conditioning feature.

Secondary datasets:

- Black-Scholes.
- Heston.
- Path-dependent volatility.

The secondary datasets should be described as compact benchmark or sanity-check settings inherited
from the continuous TC-VAE workflow. They are not the centre of the final empirical argument.

## 5. Evaluation Metrics

Use two complementary metric groups.

### Path-Level Metrics

The paper-style S&P500/VIX evaluation script reports:

- MMD.
- SWD.
- Market-style summary statistics.
- One-step log-return distribution.
- Terminal-return distribution.
- Path-volatility distribution.
- Maximum drawdown.
- Log-return autocorrelation.
- Squared-return autocorrelation.
- Skewness and kurtosis.
- Extreme-return and rolling-volatility tail diagnostics.
- VIX-bucket terminal-return and volatility comparisons.

The report should not promote the discrete method solely from MMD or SWD. The script explicitly
records that promotion should depend on stylised facts remaining competitive with the continuous
reference.

### Token and Geometry Metrics

The latent-geometry diagnostics report:

- Active-code count and active-code ratio.
- Codebook perplexity.
- Token entropy.
- PCA projection of codebook embeddings.
- Usage-weighted codebook projection.
- VIX-bucket code usage.
- Token trajectory examples.
- Exact Voronoi or nearest-region projection plots.
- q0/q1 pair support for RVQ q2 ablations.

Report-ready standard VQ claims:

- Combined geometry index shape: `[4914, 60]`.
- Quantizer type: `vector`.
- Codebook size: 64.
- Codebook embedding shape: `[64, 16]`.
- Active-code count: 63.
- Active-code ratio: `0.984375`.
- Perplexity: `39.05571746826172`.
- Entropy: `3.6649892330169678`.
- Projection method: `sklearn_pca`.

VIX-bucket standard VQ usage:

| Bucket | Samples | Active codes | Perplexity | Entropy |
| --- | ---: | ---: | ---: | ---: |
| very_low | 983 | 53 | 28.48687363 | 3.34944344 |
| low | 983 | 56 | 32.33771515 | 3.47623420 |
| mid | 983 | 58 | 35.60297394 | 3.57242918 |
| high | 983 | 60 | 40.41440582 | 3.69918633 |
| very_high | 982 | 63 | 43.82186127 | 3.78013277 |

## 6. Figures To Generate

Primary standard VQ geometry figures:

| Figure | Source path | Report use |
| --- | --- | --- |
| `codebook_projection.png` | `outputs/latent_geometry/sp500_vix_standard_vq/` | Shows the projected standard VQ codebook. |
| `codebook_usage_projection.png` | `outputs/latent_geometry/sp500_vix_standard_vq/` | Shows empirical code usage over the projection. |
| `vix_bucket_code_usage.png` | `outputs/latent_geometry/sp500_vix_standard_vq/` | Shows VIX-sensitive code usage. |
| `token_trajectory_examples.png` | `outputs/latent_geometry/sp500_vix_standard_vq/` | Shows example token paths through time. |
| `codebook_voronoi.png` | `outputs/latent_geometry/sp500_vix_standard_vq/` | Optional nearest-region visualisation. |

RVQ q2 ablation figure:

| Figure | Source path | Report use |
| --- | --- | --- |
| `q0_q1_pair_heatmap.png` | `outputs/latent_geometry/sp500_vix_rvq_q2/` | Shows sparse same-time q0/q1 support. |

Paper-style generated-path figures:

| Figure | Report use |
| --- | --- |
| `returns_distribution.png` | One-step return distribution comparison. |
| `terminal_return_distribution.png` | Terminal-return distribution comparison. |
| `volatility_distribution.png` | Path-volatility distribution comparison. |
| `maximum_drawdown_distribution.png` | Drawdown comparison. |
| `log_return_autocorrelation.png` | Within-path log-return autocorrelation. |
| `squared_return_autocorrelation.png` | Volatility-clustering diagnostic. |
| `skew_kurtosis.png` | Higher-moment comparison. |
| `extreme_discrete_paths.png` | Inspection of extreme generated paths. |
| `extreme_return_histogram.png` | Tail-return comparison against real thresholds. |
| `volatility_tail_comparison.png` | Rolling-volatility tail comparison. |
| `vix_bucket_paths.png` | Conditional path examples by VIX bucket. |
| `vix_bucket_terminal_returns.png` | VIX-bucket terminal-return comparison. |
| `vix_bucket_volatility.png` | VIX-bucket volatility comparison. |

## 7. Notebooks To Prepare

Required notebooks:

- `notebooks/sp500_vix.ipynb`.
- `notebooks/discrete_latent_geometry_demo.ipynb`.

Notebook roles:

- `sp500_vix.ipynb` should be the report-facing control notebook for the promoted S&P500/VIX
  workflow. It should show configuration checks, artefact inventory, latent-geometry summaries,
  figure manifests, and the paper-style evaluation command.
- `discrete_latent_geometry_demo.ipynb` should be the focused diagnostic notebook. Its default
  preset should be standard VQ. The `rvq_q2` preset should be clearly labelled as an ablation.

Notebook hygiene:

- Keep notebooks output-stripped in Git.
- Store executed notebooks, PNGs, tensors, checkpoints, and summaries under ignored `outputs/`
  paths.
- Do not embed local data or private checkpoint paths in committed notebooks.

## 8. Remaining Commands To Run

These commands are remaining deliverable commands. They should be run only when the required local
artefacts exist. They are not run as part of this plan creation.

### Inspect Selected Configs

```bash
poetry run python scripts/inspect_selected_configs.py
```

### Extract Standard VQ Tokens

```bash
poetry run python scripts/extract_token_indices.py \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer.yaml \
  --tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --output-dir outputs/sp500_vix_discrete/token_prior/tokens_codebook64_codebookdim16 \
  --base-data-dir data/processed \
  --seed 99
```

### Regenerate Standard VQ Latent Geometry

```bash
poetry run python scripts/analyze_discrete_latent_geometry.py \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer.yaml \
  --tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --token-data-dir outputs/sp500_vix_discrete/token_prior/tokens_codebook64_codebookdim16 \
  --output-dir outputs/latent_geometry/sp500_vix_standard_vq \
  --base-data-dir data/processed \
  --plot-voronoi
```

### Run Paper-Style Evaluation After Additive Prior Training

```bash
poetry run python scripts/evaluate_sp500_vix_paper_style.py \
  --discrete-config configs/experiments/sp500_vix_causal_token_prior_additive.yaml \
  --discrete-prior-dir <prior-dir> \
  --discrete-tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --continuous-config configs/experiments/sp500_vix_beta_cvae.yaml \
  --continuous-model-dir <continuous-final-model-dir> \
  --output-dir outputs/sp500_vix_discrete/paper_style \
  --base-data-dir data/processed \
  --n-sample 1000 \
  --seed 99 \
  --temperature 1.0 \
  --top-k 40
```

Before running the final paper-style evaluation, ensure that `<prior-dir>` points to the trained
additive prior directory and that the tokenizer path is the actual populated standard VQ tokenizer
directory.

### Execute Final Notebooks

```bash
poetry run jupyter nbconvert \
  --to notebook \
  --execute notebooks/sp500_vix.ipynb \
  --output-dir outputs/notebook_checks \
  --output sp500_vix.executed.ipynb

poetry run jupyter nbconvert \
  --to notebook \
  --execute notebooks/discrete_latent_geometry_demo.ipynb \
  --output-dir outputs/notebook_checks \
  --output discrete_latent_geometry_demo.executed.ipynb
```

### Final Documentation Checks

```bash
poetry run ruff format docs
poetry run ruff check docs --fix
poetry check
```

## 9. Limitations

- The final plan promotes a public-minimal method, not the largest architecture explored during
  development.
- The normalised S&P500/VIX array is local and not committed, so full reproduction requires local
  data placement under `data/processed/sp500vix/`.
- Latent-geometry evidence is strong for standard VQ code usage, but final generated-path claims
  require a completed additive-prior run and paper-style evaluation output.
- The paper-style evaluation is CPU-compatible but can be slow for large samples and optional
  continuous comparisons.
- VIX conditioning is scalar. The final method does not model richer market context, multi-asset
  conditioning, or exogenous event structure.
- The standard VQ tokenizer gives one code per time step. It is intentionally simpler than RVQ q2
  and may not capture all hierarchical latent factors.
- RVQ q2 improves interpretability of coarse/detail components but introduces sparse same-time
  joint-code support that is harder for generation.
- W&B logging is optional and may fail in restricted socket environments; local output files are
  the reproducibility source of record.

## 10. Future Work

- Revisit RVQ q2 only after the standard VQ additive-prior baseline has complete paper-style
  results.
- Study whether a multi-code prior can explicitly preserve sparse q0/q1 compatibility without
  harming path-level stylised facts.
- Treat GroupedRVQ as a controlled future ablation only if standard VQ shows a measured failure
  mode that grouped codes can address.
- Keep MGVQ as future conceptual work for grouped regime, volatility, and residual-movement
  factors.
- Explore causal diffusion only as future work after the AR-token baseline is stable.
- Add richer market conditions beyond scalar VIX, such as realised volatility features or
  multi-index context.
- Extend report diagnostics to robustness across seeds, market regimes, and rolling time splits.
- Package the final figures and summaries into a reproducible report appendix with exact command
  manifests and output hashes.
