# S&P500/VIX Promoted Method Report Pack

Status: notebook and report-facing manifest for the final public method.

The promoted method is the standard causal VQ tokenizer with one discrete code per time step,
followed by an additive scalar-conditioned causal autoregressive token prior. The empirical
benchmark is S&P500/VIX. RVQ q2 is retained only as an ablation. Diffusion, AdaLN,
cross-attention, GroupedRVQ, and MGVQ are not part of the final method.

## Notebooks

- `notebooks/discrete/sp500_vix.ipynb`: final S&P500/VIX promoted-method notebook. It checks the
  tokenizer, token data, latent geometry, additive-prior, and paper-style output paths; displays
  report figure manifests; and prints the exact paper-style evaluation command.
- `notebooks/discrete/discrete_latent_geometry_demo.ipynb`: lightweight latent-geometry inspection
  notebook. The default preset is standard VQ; `rvq_q2` is available only for the ablation.

Both notebooks should remain output-stripped in Git. Generated PNGs, tensors, JSON summaries, and
executed notebooks belong under ignored `outputs/` paths.

## Primary Figure Set

Use the standard VQ figures from `outputs/latent_geometry/sp500_vix_standard_vq/` as the primary
latent-geometry evidence:

| Figure | Report role |
| --- | --- |
| `codebook_projection.png` | PCA projection of the promoted standard VQ codebook. |
| `codebook_usage_projection.png` | Codebook projection with empirical code-usage overlay. |
| `vix_bucket_code_usage.png` | VIX-bucket usage heatmap showing volatility-sensitive code use. |
| `token_trajectory_examples.png` | Example token trajectories through time. |
| `codebook_voronoi.png` | Optional nearest-region view for the two-dimensional projection. |

Use `outputs/latent_geometry/sp500_vix_rvq_q2/q0_q1_pair_heatmap.png` only in the RVQ q2
ablation discussion.

## Report-Ready Numeric Claims

The standard VQ geometry diagnostics support the promoted baseline:

- The combined S&P500/VIX token artefacts have index shape `[4914, 60]`.
- The tokenizer uses 63 of 64 codes, with active-code ratio `0.984375`.
- Global codebook perplexity is `39.05571746826172`.
- The VIX-bucket diagnostics show increasing active-code count and perplexity from the very-low
  VIX bucket to the very-high VIX bucket.
- Very-low VIX uses 53 active codes with perplexity `28.48687363`.
- Very-high VIX uses 63 active codes with perplexity `43.82186127`.

These claims come from `docs/verification/sp500_vix_standard_vq_latent_geometry.md` and the
generated JSON summary under `outputs/latent_geometry/sp500_vix_standard_vq/`.

## Ablation Summary

RVQ q2 is interpretable but should not replace the standard VQ baseline in the final method:

- q0 uses 6 active codes with perplexity `4.361384868621826`.
- q1 uses all 64 codes with perplexity `39.99274444580078`.
- Only 231 of 4096 same-time q0/q1 pairs are active.
- The active pair ratio is `0.056396484375`, with absent pair mass `0.943603515625`.

This supports the ablation narrative: RVQ q2 can expose a coarse/detail split, but it gives the
token prior a sparse same-time joint-support problem in addition to temporal and VIX-dependent
dynamics.

## Paper-Style Evaluation Outputs

After training the additive scalar-conditioned prior, run:

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

Do not report generation metrics until `outputs/sp500_vix_discrete/paper_style/paper_style_summary.json`
exists for the final additive-prior run.

Expected report figures from the paper-style script:

| Figure | Report role |
| --- | --- |
| `returns_distribution.png` | One-step return distribution comparison. |
| `terminal_return_distribution.png` | Terminal-return distribution comparison. |
| `volatility_distribution.png` | Path-volatility distribution comparison. |
| `maximum_drawdown_distribution.png` | Drawdown distribution comparison. |
| `log_return_autocorrelation.png` | Within-path return autocorrelation. |
| `squared_return_autocorrelation.png` | Volatility-clustering diagnostic. |
| `skew_kurtosis.png` | Higher-moment comparison. |
| `extreme_return_histogram.png` | Tail-return comparison against real thresholds. |
| `volatility_tail_comparison.png` | Rolling-volatility tail comparison. |
| `vix_bucket_paths.png` | Conditional path examples by VIX bucket. |
| `vix_bucket_terminal_returns.png` | Bucketed terminal-return comparison. |
| `vix_bucket_volatility.png` | Bucketed volatility comparison. |

## Suggested Report Wording

The standard causal VQ tokenizer provides a compact, volatility-sensitive discrete
representation for S&P500/VIX paths. Its codebook is broadly used, with 63 of 64 codes active,
and its VIX-bucket diagnostics show that higher-volatility windows use a wider subset of the
codebook. This supports pairing the tokenizer with a scalar-conditioned causal autoregressive
prior while retaining a simple one-code-per-time-step interface.

RVQ q2 is best interpreted as an ablation rather than as the promoted architecture. Its q0/q1
structure is meaningful, but the sparse same-time pair support makes generation harder: a prior
must preserve joint code compatibility while also modelling temporal and VIX-dependent dynamics.
The final report should therefore present RVQ q2 as evidence for the reconstruction-versus-
generation trade-off, not as a replacement for standard VQ.
