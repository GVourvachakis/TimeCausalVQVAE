# S&P500/VIX RVQ q2 Latent Geometry

Status: not run.

The requested private RVQ q2 tokenizer and token-data directories were not available from this
workspace, so the RVQ q2 latent-geometry diagnostics were not executed. No W&B run, projection
files, q0/q1 pair summary, trajectory figures, Voronoi plot, or geometry summaries were created
for this real trained-tokenizer analysis.

## Missing Paths

- Tokenizer path:
  `/home/georgios-vourvachakis/Desktop/time-causal-vae/outputs/sp500_vix_discrete/vq_family_tokenizer_ablation/sp500_vix_vq_family_tokenizer_sp500_vix_causal_rvq_tokenizer_q2_seed0`
- Token data path:
  `/home/georgios-vourvachakis/Desktop/time-causal-vae/outputs/sp500_vix_discrete/token_prior/tokens_rvq_q2`

## Command Not Run

```bash
poetry run python scripts/analyze_discrete_latent_geometry.py \
  --config configs/experiments/sp500_vix_causal_rvq_tokenizer_q2.yaml \
  --tokenizer-dir ~/Desktop/time-causal-vae/outputs/sp500_vix_discrete/vq_family_tokenizer_ablation/sp500_vix_vq_family_tokenizer_sp500_vix_causal_rvq_tokenizer_q2_seed0 \
  --token-data-dir ~/Desktop/time-causal-vae/outputs/sp500_vix_discrete/token_prior/tokens_rvq_q2 \
  --output-dir outputs/latent_geometry/sp500_vix_rvq_q2 \
  --base-data-dir data/processed \
  --plot-voronoi \
  --wandb \
  --wandb-project time-causal-latent-diagnostics \
  --wandb-entity tc_vae \
  --run-name sp500_vix_rvq_q2_geometry
```

## Result Fields

- W&B URL: unavailable because the command was not run.
- Tokenizer path: missing, as listed above.
- Token data path: missing, as listed above.
- Quantizer type: unavailable.
- Index shape: unavailable.
- q0 active codes and perplexity: unavailable.
- q1 active codes and perplexity: unavailable.
- q0/q1 same-time pair summary: unavailable.
- Absent or rare pair information: unavailable.
- VIX-bucket q0/q1 usage: unavailable.
- Codebook projection method: unavailable.
- Generated figures: unavailable.

## Interpretation

No trained RVQ q2 geometry result can be interpreted from this workspace. This missing-path check
does not establish whether q0 behaves as a coarse regime code, whether q1 behaves as a
residual/detail code, whether the geometry explains the private observation that RVQ q2
reconstruction improved while generation failed, or whether RVQ q2 should remain an ablation
rather than the promoted baseline.
