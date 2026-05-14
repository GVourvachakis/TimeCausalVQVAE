# S&P500/VIX Histogram Bin Policy

## Purpose

This note records the paper-style plotting update for S&P500/VIX market
diagnostics. The change only affects histogram bin construction in plots. It
does not change metric formulas, generated samples, tokenizers, priors, or
numeric tail diagnostics.

## Previous Behaviour

The paper-style plots are generated in
`scripts/evaluate_sp500_vix_paper_style.py`. The metric functions in
`src/time_causal_vae/evaluation/market_diagnostics.py` did not previously
define plot-bin policy.

Before this update:

- `returns_distribution.png` used the generic `plot_tensor_histograms` helper.
- `terminal_return_distribution.png`, `volatility_distribution.png`,
  `maximum_drawdown_distribution.png`, and `volatility_tail_comparison.png`
  also used `plot_tensor_histograms`.
- `plot_tensor_histograms` computed up to 50 linearly spaced bins separately
  from each source tensor's own minimum and maximum.
- `extreme_return_histogram.png` called Matplotlib with `bins=80` separately
  for each source, so Matplotlib inferred source-specific ranges.
- The plots therefore used overlapping histograms with fixed or capped bin
  counts, but not shared bin edges across real, discrete, and continuous
  sources.

## New Shared-Bin Policy

The plotting path now uses `shared_histogram_bins` from
`time_causal_vae.evaluation.market_diagnostics`:

- finite values from real, discrete, and continuous sources are combined;
- the default bin policy is Freedman-Diaconis (`fd`);
- the range policy is the union of all finite source values;
- bin counts are clamped to `[30, 150]`;
- degenerate interquartile range cases fall back to 80 bins;
- display clipping is disabled by default (`none`);
- the same bin edges are passed to every source in each histogram.

The JSON and Markdown paper-style summaries record, per histogram:

- `bin_count`;
- `bin_policy`;
- `range_policy`;
- `display_clip_policy`;
- bin range endpoints in JSON.

## Smoke Command

```bash
env MPLBACKEND=Agg poetry run python scripts/evaluate_sp500_vix_paper_style.py \
  --discrete-config configs/experiments/sp500_vix_causal_token_prior_additive.yaml \
  --discrete-prior-dir outputs/sp500_vix_discrete/token_prior/additive_vix_only/sp500_vix_causal_token_prior_additive_seed0/best_model \
  --discrete-tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --continuous-config configs/experiments/sp500_vix_beta_cvae.yaml \
  --continuous-model-dir outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47/final_model \
  --output-dir outputs/sp500_vix_discrete/paper_style_plot_bins_smoke \
  --base-data-dir data/processed \
  --n-sample 128 \
  --seed 99 \
  --temperature 0.8 \
  --top-k 40
```

The smoke completed successfully. Matplotlib used a temporary cache directory
because the default user cache path was not writable in the sandbox, but this
did not affect plot generation.

## Smoke Outputs

The smoke generated the requested histogram plots:

| Plot | Bin count | Bin policy | Range policy | Display clipping |
| --- | ---: | --- | --- | --- |
| `returns_distribution.png` | 150 | `fd` | `union` | `none` |
| `terminal_return_distribution.png` | 33 | `fd` | `union` | `none` |
| `volatility_distribution.png` | 47 | `fd` | `union` | `none` |
| `maximum_drawdown_distribution.png` | 32 | `fd` | `union` | `none` |
| `extreme_return_histogram.png` | 150 | `fd` | `union` | `none` |
| `volatility_tail_comparison.png` | 57 | `fd` | `union` | `none` |

Additional non-histogram paper-style plots were also produced in the smoke
directory, including autocorrelation, skew/kurtosis, extreme path, and VIX
bucket plots.

## Caveats

The union range keeps all finite generated and real values visible. If a future
run has extreme outliers, the visible density may become compressed in the
centre of the plot. This is intentional for the default policy because numeric
tail diagnostics remain the authoritative source for tail exceedance and
outlier interpretation. Optional display clipping can be added later without
changing metric formulas.
