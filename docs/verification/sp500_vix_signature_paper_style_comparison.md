# S&P500/VIX Signature Paper-Style Comparison

## Purpose

This note records paper-style diagnostics for the VIX-only baseline and the two
depth-3 log-signature conditioning variants under identical sampling settings.
No model architecture, training code, signature code, tokeniser code, or new
metric implementation was changed.

## Settings

All runs used:

- `n_sample=1000`;
- `seed=99`;
- `temperature=0.8`;
- `top_k=40`;
- tokenizer:
  `outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0`;
- continuous checkpoint:
  `outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47/final_model`.

The continuous checkpoint was available and loaded successfully in all three
paper-style runs. Each run reported `continuous_paths: [1000, 60, 1]`.

`MPLBACKEND=Agg` was set for the paper-style commands to keep Matplotlib on a
headless backend while plots were generated.

## Commands

VIX-only baseline:

```bash
env MPLBACKEND=Agg poetry run python scripts/evaluate_sp500_vix_paper_style.py \
  --discrete-config configs/experiments/sp500_vix_causal_token_prior_additive.yaml \
  --discrete-prior-dir outputs/sp500_vix_discrete/token_prior/additive_vix_only/sp500_vix_causal_token_prior_additive_seed0/best_model \
  --discrete-tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --continuous-config configs/experiments/sp500_vix_beta_cvae.yaml \
  --continuous-model-dir outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47/final_model \
  --output-dir outputs/sp500_vix_discrete/paper_style_vix_only_temp08_topk40 \
  --base-data-dir data/processed \
  --n-sample 1000 \
  --seed 99 \
  --temperature 0.8 \
  --top-k 40
```

`logsig_l3_ctx10`:

```bash
env MPLBACKEND=Agg poetry run python scripts/evaluate_sp500_vix_paper_style.py \
  --discrete-config configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx10.yaml \
  --discrete-prior-dir outputs/sp500_vix_discrete/token_prior/signature_conditioning_ablation/sp500_vix_causal_token_prior_additive_logsig_l3_ctx10/train/sp500_vix_causal_token_prior_additive_logsig_l3_ctx10_seed0/best_model \
  --discrete-tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --continuous-config configs/experiments/sp500_vix_beta_cvae.yaml \
  --continuous-model-dir outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47/final_model \
  --output-dir outputs/sp500_vix_discrete/paper_style_logsig_l3_ctx10_temp08_topk40 \
  --base-data-dir data/processed \
  --n-sample 1000 \
  --seed 99 \
  --temperature 0.8 \
  --top-k 40
```

`logsig_l3_ctx20`:

```bash
env MPLBACKEND=Agg poetry run python scripts/evaluate_sp500_vix_paper_style.py \
  --discrete-config configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20.yaml \
  --discrete-prior-dir outputs/sp500_vix_discrete/token_prior/signature_conditioning_ablation/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20/train/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_seed0/best_model \
  --discrete-tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --continuous-config configs/experiments/sp500_vix_beta_cvae.yaml \
  --continuous-model-dir outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47/final_model \
  --output-dir outputs/sp500_vix_discrete/paper_style_logsig_l3_ctx20_temp08_topk40 \
  --base-data-dir data/processed \
  --n-sample 1000 \
  --seed 99 \
  --temperature 0.8 \
  --top-k 40
```

## Aggregate Metrics

Lower is better for all metrics in this table.

| Model | MMD | SWD | Returns W1 | Terminal W1 | Volatility W1 | Drawdown W1 | Return AC L1 | Squared-return AC L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VIX-only | 0.27934083 | 0.00767375 | 0.00124159 | 0.00981713 | 0.00118835 | 0.01050232 | 0.05174129 | 0.04129972 |
| `logsig_l3_ctx10` | 0.33967710 | 0.00991276 | 0.00111103 | 0.00461828 | 0.00103661 | 0.00802572 | 0.04819629 | 0.03690299 |
| `logsig_l3_ctx20` | 0.34163502 | 0.01082886 | 0.00099378 | 0.00451245 | 0.00080577 | 0.00549569 | 0.04789805 | 0.03506229 |
| Continuous BetaCVAE reference | 0.15442121 | 0.00878550 | 0.00060182 | 0.00905099 | 0.00063360 | 0.00766744 | 0.02597247 | 0.02946163 |

Flattened autocorrelation diagnostics:

| Model | Flattened return AC L1 | Flattened squared-return AC L1 |
| --- | ---: | ---: |
| VIX-only | 0.04207045 | 0.12882016 |
| `logsig_l3_ctx10` | 0.04387885 | 0.02932903 |
| `logsig_l3_ctx20` | 0.02918198 | 0.05403909 |
| Continuous BetaCVAE reference | 0.04292119 | 0.02914374 |

## Tail Diagnostics

The rates are one-step log-return exceedance rates relative to real-data tail
thresholds.

| Model | < real q001 | < real q01 | > real q99 | > real q999 |
| --- | ---: | ---: | ---: | ---: |
| VIX-only | 0.00096610 | 0.00572881 | 0.00869492 | 0.00291525 |
| `logsig_l3_ctx10` | 0.00123729 | 0.00633898 | 0.01042373 | 0.00337288 |
| `logsig_l3_ctx20` | 0.00154237 | 0.00801695 | 0.01157627 | 0.00396610 |
| Continuous BetaCVAE reference | 0.00252542 | 0.01205085 | 0.00954237 | 0.00352542 |

The depth-3 signature variants recover more upper-tail exceedances than
VIX-only. `logsig_l3_ctx20` also increases lower-tail exceedances, but all
three discrete models still underrepresent the 1 percent lower tail relative to
the real threshold convention.

## VIX-Bucket Diagnostics

Bucket diagnostics are reported from each saved `paper_style_summary.json`.
Lower values are better.

| Model | Bucket | MMD | SWD | Terminal W1 | Volatility W1 |
| --- | --- | ---: | ---: | ---: | ---: |
| VIX-only | very_low | 0.59723961 | 0.01430982 | 0.00710080 | 0.00152433 |
| VIX-only | low | 0.40146104 | 0.00903835 | 0.01461281 | 0.00098221 |
| VIX-only | mid | 0.39057434 | 0.01064593 | 0.02059070 | 0.00161874 |
| VIX-only | high | 0.42856687 | 0.01175834 | 0.02002885 | 0.00123207 |
| VIX-only | very_high | 0.28335714 | 0.01616320 | 0.01412871 | 0.00130342 |
| `logsig_l3_ctx10` | very_low | 0.74326366 | 0.01740434 | 0.01204278 | 0.00143999 |
| `logsig_l3_ctx10` | low | 0.32172361 | 0.00787911 | 0.00799153 | 0.00063165 |
| `logsig_l3_ctx10` | mid | 0.31444022 | 0.01142531 | 0.01642891 | 0.00140668 |
| `logsig_l3_ctx10` | high | 0.40787911 | 0.01242552 | 0.01602687 | 0.00098097 |
| `logsig_l3_ctx10` | very_high | 0.40735009 | 0.01782236 | 0.01076409 | 0.00166075 |
| `logsig_l3_ctx20` | very_low | 0.72897053 | 0.01684688 | 0.01024548 | 0.00144672 |
| `logsig_l3_ctx20` | low | 0.31984806 | 0.00843930 | 0.00788521 | 0.00056392 |
| `logsig_l3_ctx20` | mid | 0.30187473 | 0.01028724 | 0.01300928 | 0.00102189 |
| `logsig_l3_ctx20` | high | 0.37769336 | 0.01069114 | 0.01087887 | 0.00078570 |
| `logsig_l3_ctx20` | very_high | 0.40025932 | 0.02246254 | 0.01891296 | 0.00120390 |

The depth-3 signature variants improve most low, mid, and high VIX bucket
terminal and volatility distances. VIX-only remains strongest in the very-low
bucket by MMD and terminal W1, and in the very-high bucket by MMD.

## Best Variants

Best discrete variant by metric:

| Criterion | Best variant | Evidence |
| --- | --- | --- |
| MMD | VIX-only | `0.27934083`, lower than both depth-3 signature variants. |
| SWD | VIX-only | `0.00767375`, lower than both depth-3 signature variants. |
| Market-style score | `logsig_l3_ctx20` | Mean rank `1.0` across returns W1, terminal W1, volatility W1, drawdown W1, return AC L1, and squared-return AC L1. |
| Terminal/volatility W1 | `logsig_l3_ctx20` | Terminal W1 `0.00451245`; volatility W1 `0.00080577`. |
| Squared-return autocorrelation | `logsig_l3_ctx20` | Within-path squared-return AC L1 `0.03506229`. |
| Flattened squared-return autocorrelation | `logsig_l3_ctx10` | Flattened squared-return AC L1 `0.02932903`, nearly matching the continuous reference. |

The market-style score above is a documentation-only summary rank over six
already-computed paper-style diagnostics. It is not a new implemented metric and
does not replace the saved JSON metrics.

## Decision

Do not promote VIX+signature conditioning as the new default yet.

The evidence is mixed:

- VIX-only remains best on global MMD and SWD;
- `logsig_l3_ctx20` is best on returns W1, terminal W1, volatility W1,
  drawdown W1, return autocorrelation L1, and squared-return autocorrelation
  L1;
- `logsig_l3_ctx10` is strongest on flattened squared-return autocorrelation;
- the continuous BetaCVAE reference remains much stronger on MMD, returns W1,
  volatility W1, and within-path autocorrelation diagnostics.

Current recommendation: keep VIX-only as the promoted public default, run a seed
ablation for `logsig_l3_ctx20`, and add the planned evaluation-only
signature-kernel metric before revisiting promotion. Signatures should not be
deferred entirely, because depth-3 conditioning consistently improves several
path-space diagnostics that matter for market stylised facts.
