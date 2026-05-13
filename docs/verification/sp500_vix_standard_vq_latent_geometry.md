# S&P500/VIX Standard VQ Latent Geometry

Status: not run.

The promoted private tokenizer and token-data directories were not available from this
workspace, so the standard VQ latent-geometry diagnostics were not executed. No W&B run,
projection files, trajectory figures, Voronoi plot, or geometry summaries were created for this
real trained-tokenizer analysis.

## Missing Paths

- Tokenizer path:
  `/home/georgios-vourvachakis/Desktop/time-causal-vae/outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_codebook64dim16_seed0`
- Token data path:
  `/home/georgios-vourvachakis/Desktop/time-causal-vae/outputs/sp500_vix_discrete/token_prior/tokens_codebook64_codebookdim16`

## Command Not Run

```bash
poetry run python scripts/analyze_discrete_latent_geometry.py \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer.yaml \
  --tokenizer-dir ~/Desktop/time-causal-vae/outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_codebook64dim16_seed0 \
  --token-data-dir ~/Desktop/time-causal-vae/outputs/sp500_vix_discrete/token_prior/tokens_codebook64_codebookdim16 \
  --output-dir outputs/latent_geometry/sp500_vix_standard_vq \
  --base-data-dir data/processed \
  --plot-voronoi \
  --wandb \
  --wandb-project time-causal-latent-diagnostics \
  --wandb-entity tc_vae \
  --run-name sp500_vix_standard_vq_geometry
```

## Result Fields

- W&B run URL: unavailable because the command was not run.
- Tokenizer path: missing, as listed above.
- Token data path: missing, as listed above.
- Quantizer type: unavailable.
- Index shape: unavailable.
- Codebook size: unavailable.
- Active-code count: unavailable.
- Perplexity: unavailable.
- VIX-bucket active-code/perplexity summary: unavailable.
- Codebook projection method: unavailable.
- Voronoi mode: unavailable.
- Token trajectory figures: unavailable.

## Interpretation

No trained-tokenizer geometry result can be interpreted from this workspace. In particular, this
missing-path check does not establish whether code usage is broad, whether VIX buckets occupy
different code regions, or whether the geometry supports keeping standard VQ as the promoted
public baseline.
