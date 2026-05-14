# Discrete Latent Geometry Source Smoke

Status: public synthetic smoke run completed.

This verification records the source-level smoke path for the discrete-latent geometry
diagnostics. It does not use trained tokenizer checkpoints, token-prior checkpoints, private
artifacts, or private data. The reported values are therefore a public implementation smoke only,
not a trained-tokenizer result.

## Command Run

```bash
poetry run python scripts/analyze_discrete_latent_geometry.py \
  --synthetic \
  --output-dir outputs/latent_geometry_smoke \
  --plot-voronoi
```

## Source Files Added

- `src/time_causal_vae/evaluation/latent_geometry.py`
- `scripts/analyze_discrete_latent_geometry.py`

## Diagnostics Implemented

- Guarded codebook embedding extraction for vector VQ, ResidualVQ, and already supported
  GroupedResidualVQ layouts.
- Fallback codebook geometry from decoded or observed quantized embeddings when direct backend
  extraction is unavailable.
- PCA projection to two dimensions, using scikit-learn when available and NumPy SVD otherwise.
- Codebook projection CSV and PNG outputs.
- Exact 2D Voronoi plotting when SciPy succeeds, with a bounded nearest-region fallback.
- Marginal code usage, active-code count, entropy, usage probabilities, and perplexity.
- VIX/condition-bucket code-usage summaries when labels are present.
- RVQ per-quantizer and grouped per-group/per-quantizer usage summaries.
- Token trajectory examples through projected codebook space.
- RVQ q0/q1 pair summary and heatmap when indices have shape `[batch, time, 2]`.
- JSON and Markdown summaries for downstream inspection.

## Synthetic Output

- Synthetic index shape: `[128, 60]`
- Synthetic codebook size: `64`
- Active-code count: `64`
- Synthetic perplexity: `63.999969482421875`
- Voronoi mode: exact Voronoi succeeded.
- W&B default project: `time-causal-latent-diagnostics`
- W&B default entity: `tc_vae`

## Generated Artifacts

- `outputs/latent_geometry_smoke/codebook_geometry_summary.json`
- `outputs/latent_geometry_smoke/codebook_projection.csv`
- `outputs/latent_geometry_smoke/codebook_projection.png`
- `outputs/latent_geometry_smoke/code_usage_histogram.png`
- `outputs/latent_geometry_smoke/codebook_usage_projection.png`
- `outputs/latent_geometry_smoke/vix_bucket_code_usage.png`
- `outputs/latent_geometry_smoke/token_trajectory_examples.png`
- `outputs/latent_geometry_smoke/latent_geometry_summary.md`
- `outputs/latent_geometry_smoke/codebook_voronoi.png`

## Figure Format Note

PNG plots are intended for human and W&B inspection. Markdown, JSON, and CSV summaries are the
decision inputs for Codex and reports. SVG export can be added later if final-report figures need
vector output.
