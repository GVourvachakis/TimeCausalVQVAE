# Figure Assets

This directory contains a small curated set of locally generated TimeCausalVQVAE figures for the
public README. The source runs remain in local `outputs/` directories and are not required to use
the package.

No original TC-VAE, VQ-VAE, or VQ-VAE-2 paper figures are copied here. The architecture overview
SVG is a hand-written original diagram for this repository.

Current assets:

- `time_causal_vqvae_pipeline.svg`: original hand-written SVG overview of the default
  time-causal VQ-VAE path, causal token prior, continuous TC-VAE baseline, latent codebook sketch,
  and diagnostics workflow. Source: authored directly in this repository for README, PyPI, and
  report-facing use.
- `sp500_vix_best_research_paths.png`: hidden128 conv-transformer path diagnostic
  across VIX regimes. Source: local S&P500/VIX paper-style path output,
  matching `outputs/sp500_vix_discrete/paper_style_hidden128_conv_transformer_sampling_temp10_topk40/vix_bucket_paths.png`.
- `sp500_vix_best_generated_paths.png`: hidden128 conv-transformer decoded generated
  path examples. Source: local S&P500/VIX prior evaluation output,
  `outputs/sp500_vix_discrete/token_prior/hidden128_conv_transformer/evaluation_best/decoded_path_examples.png`.
- `sp500_vix_hidden128_codebook_voronoi.png`: hidden128 VQ projected Voronoi diagram
  of the discrete latent space. Source:
  `outputs/latent_geometry/sp500_vix_hidden128_vq/codebook_voronoi.png`.
- `sp500_vix_vix_bucket_code_usage.png`: hidden128 VQ VIX-bucket code usage.
  Source: `outputs/latent_geometry/sp500_vix_hidden128_vq/vix_bucket_code_usage.png`.
- `hawkes_jump_ogata_jump_raster.png`: Ogata-simulated Hawkes/SVMHJD jump
  indicator raster. Source:
  `outputs/hawkes_jump_plots/ogata/jump_indicator_raster.png`.
- `hawkes_jump_model_metric_comparison.png`: repaired continuous BetaCVAE,
  additive AR, and conv-transformer k3 metric comparison. Source: generated
  with a notebook-free matplotlib helper from
  `outputs/hawkes_jump_matched_continuous_comparison/aggregate_summary.json`.
- `hawkes_jump_tail_jump_comparison.png`: Hawkes/SVMHJD generated jump
  frequency and 1% VaR/ES comparison. Source: generated with the same local
  matplotlib helper and aggregate summary.
