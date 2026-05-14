# Discrete Latent Geometry Decision

Status: decision recorded from regenerated local S&P500/VIX latent-geometry diagnostics.

The standard VQ and RVQ q2 tokenizers were regenerated locally under ignored `outputs/` paths in
the public repository. The decision below uses the numeric Markdown, JSON, and CSV summaries from
`outputs/latent_geometry/sp500_vix_standard_vq/` and
`outputs/latent_geometry/sp500_vix_rvq_q2/`, not private milestone outputs.

## Standard VQ Geometry Summary

The promoted S&P500/VIX standard VQ tokenizer uses one discrete code per time step, with a
64-entry codebook, 16-dimensional codebook embeddings, scalar VIX conditioning, and sequence
length 60.

- Code usage: broad. The combined train/eval token artifacts have index shape `[4914, 60]`, use
  63 of 64 codes, and have perplexity `39.05571746826172`.
- VIX-bucket usage: condition-sensitive. Active-code count and perplexity increase from
  `53` active codes and perplexity `28.48687363` in the very-low VIX bucket to `63` active codes
  and perplexity `43.82186127` in the very-high VIX bucket.
- Codebook projection interpretation: PCA was computed with `sklearn_pca` over a `[64, 16]`
  codebook. The projection and usage overlay are appropriate report figures for showing broad
  codebook coverage and VIX-dependent mass shifts.
- Promoted-baseline support: yes. The geometry supports the promoted standard VQ baseline because
  it combines broad code utilisation, condition-sensitive usage, and the simplest one-code
  interface for the additive scalar-conditioned causal AR prior.

## RVQ q2 Geometry Summary

The S&P500/VIX RVQ q2 tokenizer uses two residual code indices per time step, with
`quantizer_type: residual_vq`, `num_quantizers: 2`, `groups: 1`, and a 64-entry codebook per
level.

- q0 usage: coarse and low entropy. q0 uses 6 active codes with perplexity `4.361384868621826`
  and entropy `1.4727896451950073`.
- q1 usage: broad and detail-like. q1 uses all 64 codes with perplexity `39.99274444580078` and
  entropy `3.6886980533599854`.
- q0/q1 pair behaviour: sparse and structured. Across `294840` same-time pairs, only `231` of
  `4096` possible pairs are active, giving active pair ratio `0.056396484375` and absent pair
  mass `0.943603515625`.
- VIX-bucket usage: q0 remains a six-code component in every VIX bucket, while q0 perplexity
  rises from `3.84848523` to `4.78456640`. q1 is broad in every bucket and expands from 57
  active codes in the lowest VIX buckets to all 64 codes in the very-high VIX bucket.
- Failed-prior explanation: the geometry gives a plausible explanation for why RVQ q2 can improve
  reconstruction yet make generation harder. The representation is interpretable, but a prior has
  to learn sparse same-time q0/q1 compatibility, temporal dynamics, and VIX-dependent q1 detail.
  A prior that captures marginal q0 and q1 usage without respecting the sparse joint support can
  decode to poor market paths.

## Decision About GroupedRVQ

Decision: defer for this project phase.

GroupedRVQ should only proceed if geometry suggests meaningful group separation and the
multi-code prior interface is stable. The RVQ q2 result shows useful factorisation between a
coarse q0 component and a detailed q1 component, but it also shows that multi-code generation is
substantially harder because the valid same-time pair support is sparse. That is a warning signal
for adding grouped indices now.

GroupedRVQ should therefore remain a future controlled ablation, not the next implementation
step. Before revisiting it, the project should stabilise the multi-code prior interface and define
a concrete failure mode that standard VQ cannot address.

## Decision About MGVQ

Decision: reserve as future work.

MGVQ remains conceptually aligned with grouped financial factors such as regime, volatility, and
residual movement components. Direct implementation is premature, however, unless geometry and
benchmarks show that grouped sub-codebooks are needed. The current evidence does not show a
standard VQ failure that requires MGVQ. It instead shows that even the simpler RVQ q2 setting
creates a sparse joint-code modelling problem.

MGVQ should therefore be discussed as future work rather than implemented in the current public
phase.

## Recommended Next Step

Keep standard VQ as the public baseline. Use the latent-geometry plots in notebooks and the
report to support the baseline narrative: standard VQ has broad code usage, VIX-sensitive
discrete structure, and a simple prior-facing representation.

The immediate focus should be:

- finalise the standard VQ public baseline and paper-style S&P500/VIX evaluation;
- include the latent-geometry plots in notebooks and report figures;
- treat RVQ q2 as an ablation that explains the reconstruction-versus-generation trade-off;
- defer GroupedRVQ, MGVQ, and other new architectures unless a specific measured failure mode is
  identified.

## Plot Inventory

Generated standard VQ figures under `outputs/latent_geometry/sp500_vix_standard_vq/`:

- `codebook_projection.png`
- `codebook_usage_projection.png`
- `vix_bucket_code_usage.png`
- `token_trajectory_examples.png`
- `code_usage_histogram.png`
- `codebook_voronoi.png`

Generated RVQ q2 figures under `outputs/latent_geometry/sp500_vix_rvq_q2/`:

- `codebook_projection.png`
- `codebook_usage_projection.png`
- `vix_bucket_code_usage.png`
- `token_trajectory_examples.png`
- `q0_q1_pair_heatmap.png`
- `code_usage_histogram.png`
- `codebook_voronoi.png`

Recommended notebook figures:

- `codebook_projection.png`
- `codebook_usage_projection.png`
- `vix_bucket_code_usage.png`
- `token_trajectory_examples.png`
- `q0_q1_pair_heatmap.png` for RVQ q2

For the public report, the standard VQ projection, usage projection, VIX-bucket heatmap, and token
trajectory examples should be primary. The RVQ q2 q0/q1 heatmap is useful as ablation evidence
for why the multi-code representation is harder to sample from.
