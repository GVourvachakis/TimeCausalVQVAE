# S&P500/VIX Signature-Kernel Metric Results

## 1. Environment

- Python version: `3.12.3`
- PyTorch version: `2.11.0+cu130`
- NumPy version: `1.26.4`
- `sigkernel` import status: success.
- `sigkernel` reported version: `unknown`
- `sigkernel` path:
  `.venv/lib/python3.12/site-packages/sigkernel/__init__.py`
- Confirmation command:

```bash
poetry run python - <<'PY'
import sigkernel
print("sigkernel", getattr(sigkernel, "__version__", "unknown"))
PY
```

Output:

```text
sigkernel unknown
```

The previous install probe built `sigkernel` from
`https://github.com/crispitagorico/sigkernel.git` at commit
`40a583155ea8d2194af0e90dddab37e2659cfcfd`. The earlier user-specified
float32 toy import smoke failed with a Cython buffer mismatch,
`expected 'double' but got 'float'`. The repository metric path converts path
batches to float64 before calling `sigkernel`, and all runs below completed
with finite Gram matrices.

`iisignature` fallback was not used. The current signature-kernel metric path
is `sigkernel`-only.

## 2. Commands

The no-lead-lag pass used `max_samples=256`, `dyadic_order=1`,
`include_time=true`, `use_lead_lag=false`, and internal float64 tensors.

VIX-only:

```bash
poetry run python scripts/evaluate_signature_kernel_metrics.py \
  --paper-style-batch outputs/sp500_vix_discrete/paper_style_vix_only_temp08_topk40/discrete_paper_style_batch.pt \
  --output-dir outputs/signature_kernel_results/vix_only_no_leadlag \
  --max-samples 256 \
  --dyadic-order 1 \
  --include-time
```

Raw `logsig_l3_ctx20`:

```bash
poetry run python scripts/evaluate_signature_kernel_metrics.py \
  --paper-style-batch outputs/sp500_vix_discrete/paper_style_logsig_l3_ctx20_temp08_topk40/discrete_paper_style_batch.pt \
  --output-dir outputs/signature_kernel_results/logsig_l3_ctx20_no_leadlag \
  --max-samples 256 \
  --dyadic-order 1 \
  --include-time
```

Standardised `logsig_l3_ctx20`:

```bash
poetry run python scripts/evaluate_signature_kernel_metrics.py \
  --paper-style-batch outputs/sp500_vix_discrete/paper_style_logsig_l3_ctx20_std_temp08_topk40/discrete_paper_style_batch.pt \
  --output-dir outputs/signature_kernel_results/logsig_l3_ctx20_std_no_leadlag \
  --max-samples 256 \
  --dyadic-order 1 \
  --include-time
```

Continuous BetaCVAE:

```bash
poetry run python scripts/evaluate_signature_kernel_metrics.py \
  --paper-style-batch outputs/sp500_vix_discrete/paper_style_vix_only_temp08_topk40/continuous_paper_style_batch.pt \
  --output-dir outputs/signature_kernel_results/continuous_betacvae_no_leadlag \
  --max-samples 256 \
  --dyadic-order 1 \
  --include-time
```

The lead-lag retry used `max_samples=64`, `dyadic_order=1`,
`include_time=true`, `use_lead_lag=true`, and internal float64 tensors.

VIX-only:

```bash
poetry run python scripts/evaluate_signature_kernel_metrics.py \
  --paper-style-batch outputs/sp500_vix_discrete/paper_style_vix_only_temp08_topk40/discrete_paper_style_batch.pt \
  --output-dir outputs/signature_kernel_results/vix_only_leadlag64 \
  --max-samples 64 \
  --dyadic-order 1 \
  --include-time \
  --use-lead-lag
```

Raw `logsig_l3_ctx20`:

```bash
poetry run python scripts/evaluate_signature_kernel_metrics.py \
  --paper-style-batch outputs/sp500_vix_discrete/paper_style_logsig_l3_ctx20_temp08_topk40/discrete_paper_style_batch.pt \
  --output-dir outputs/signature_kernel_results/logsig_l3_ctx20_leadlag64 \
  --max-samples 64 \
  --dyadic-order 1 \
  --include-time \
  --use-lead-lag
```

Standardised `logsig_l3_ctx20`:

```bash
poetry run python scripts/evaluate_signature_kernel_metrics.py \
  --paper-style-batch outputs/sp500_vix_discrete/paper_style_logsig_l3_ctx20_std_temp08_topk40/discrete_paper_style_batch.pt \
  --output-dir outputs/signature_kernel_results/logsig_l3_ctx20_std_leadlag64 \
  --max-samples 64 \
  --dyadic-order 1 \
  --include-time \
  --use-lead-lag
```

Continuous BetaCVAE:

```bash
poetry run python scripts/evaluate_signature_kernel_metrics.py \
  --paper-style-batch outputs/sp500_vix_discrete/paper_style_vix_only_temp08_topk40/continuous_paper_style_batch.pt \
  --output-dir outputs/signature_kernel_results/continuous_betacvae_leadlag64 \
  --max-samples 64 \
  --dyadic-order 1 \
  --include-time \
  --use-lead-lag
```

The script now supports both discrete paper-style batches with `decoded_paths`
and continuous paper-style batches with `fake_paths`.

## 3. Settings

| setting | no-lead-lag pass | lead-lag small pass |
| --- | --- | --- |
| `max_samples` | `256` | `64` |
| `dyadic_order` | `1` | `1` |
| `rbf_sigma` | `1.0` | `1.0` |
| `include_time` | `true` | `true` |
| `use_lead_lag` | `false` | `true` |
| dtype | float64 | float64 |

Preprocessing channels before lead-lag:

- normalised price;
- log return;
- cumulative log return;
- time.

The lead-lag pass doubles these channels into lead and lag copies.

## 4. Signature-Kernel MMD Table

Lower is better.

| Model | No-lead-lag MMD | No-lead-lag runtime s | Lead-lag MMD | Lead-lag runtime s |
| --- | ---: | ---: | ---: | ---: |
| Continuous BetaCVAE | 0.00087344 | 124.244 | 0.01637310 | 32.800 |
| Standardised `logsig_l3_ctx20` | 0.00224031 | 131.368 | 0.00623763 | 33.653 |
| Raw `logsig_l3_ctx20` | 0.00238908 | 130.870 | 0.01525021 | 35.350 |
| VIX-only | 0.00555772 | 129.755 | 0.00232711 | 36.374 |

No-lead-lag ranking:

1. Continuous BetaCVAE: `0.00087344`
2. Standardised `logsig_l3_ctx20`: `0.00224031`
3. Raw `logsig_l3_ctx20`: `0.00238908`
4. VIX-only: `0.00555772`

Lead-lag small-run ranking:

1. VIX-only: `0.00232711`
2. Standardised `logsig_l3_ctx20`: `0.00623763`
3. Raw `logsig_l3_ctx20`: `0.01525021`
4. Continuous BetaCVAE: `0.01637310`

The lead-lag ranking is based on only 64 samples and should be treated as a
runtime feasibility check rather than a promotion-grade comparison.

## 5. Numerical Checks

All completed runs passed the finite, symmetry, and positive-diagonal checks
implemented in `signature_kernel_metrics.py`.

| Model | Pass | Kxx finite/symmetric/diag+ | Kyy finite/symmetric/diag+ | Kxy finite |
| --- | --- | --- | --- | --- |
| VIX-only | no lead-lag | true / true / true | true / true / true | true |
| Raw `logsig_l3_ctx20` | no lead-lag | true / true / true | true / true / true | true |
| Standardised `logsig_l3_ctx20` | no lead-lag | true / true / true | true / true / true | true |
| Continuous BetaCVAE | no lead-lag | true / true / true | true / true / true | true |
| VIX-only | lead-lag 64 | true / true / true | true / true / true | true |
| Raw `logsig_l3_ctx20` | lead-lag 64 | true / true / true | true / true / true | true |
| Standardised `logsig_l3_ctx20` | lead-lag 64 | true / true / true | true / true / true | true |
| Continuous BetaCVAE | lead-lag 64 | true / true / true | true / true / true | true |

The Gram shapes were `[256, 256]` for the no-lead-lag pass and `[64, 64]` for
the lead-lag pass.

## 6. Comparison To Existing Metrics

The no-lead-lag signature-kernel ranking agrees with the continuous BetaCVAE's
strong global paper-style metrics: the continuous reference is already best in
the paper-style comparison by Gaussian MMD, returns W1, volatility W1, return
autocorrelation L1, and squared-return autocorrelation L1.

Among the discrete models, the no-lead-lag signature-kernel ranking reverses
the Gaussian MMD/SWD ordering:

- Gaussian MMD/SWD prefer VIX-only among the discrete models.
- Signature-kernel MMD prefers the two `logsig_l3_ctx20` variants over VIX-only.
- Standardised `logsig_l3_ctx20` is narrowly better than raw `logsig_l3_ctx20`
  by no-lead-lag signature-kernel MMD.

This supports the interpretation that signature-kernel MMD sees path-shape
agreement that is not captured by Gaussian MMD/SWD alone. It also matches the
research-branch reading that signature conditioning is valuable for
path-functional diagnostics, even though VIX-only remains the public default.

The comparison with the model-selection profiles remains mixed:

- distributional profile: VIX-only remains the strongest discrete model by
  Gaussian MMD/SWD, while continuous BetaCVAE is the strongest overall
  reference;
- tail-risk profile: raw `logsig_l3_ctx20` remains the most convincing
  discrete signature branch by drawdown and volatility-oriented path
  functionals, while standardised `logsig_l3_ctx20` is strongest on terminal W1
  in its own quality report;
- sequential-dependence profile: continuous BetaCVAE remains strongest on the
  recorded autocorrelation diagnostics, while raw `logsig_l3_ctx20` remains
  useful among discrete candidates;
- signature-kernel MMD: favours continuous BetaCVAE overall and favours
  signature-conditioned discrete models over VIX-only in the no-lead-lag pass.

The lead-lag 64-sample pass does not support the same ranking: it favours
VIX-only. Because it uses a smaller sample and much higher-dimensional
preprocessing, it is useful as a feasibility result but not yet a stable
selection metric.

## 7. Decision

Decision: exploratory metric only.

Signature-kernel MMD should not be used as the sole promotion gate. The
no-lead-lag result is informative and supports keeping signature-conditioned
models in the research branch: both raw and standardised `logsig_l3_ctx20`
improve the discrete signature-kernel MMD relative to VIX-only. This is also a
reasonable motivation for the planned FiLM/AdaLN signature-conditioning
ablation, because the fixed-vector additive path may be underusing information
that a path-level kernel finds relevant.

However, the metric remains optional and exploratory because:

- `sigkernel` is not a default project dependency;
- the backend requires float64 inputs to avoid the observed Cython dtype issue;
- the no-lead-lag and small lead-lag rankings disagree;
- Gaussian MMD/SWD, tail-risk diagnostics, sequential-dependence diagnostics,
  and VIX-bucket diagnostics must remain visible in model selection.

Use signature-kernel MMD as an additional path-space diagnostic when the time
budget allows. Do not promote or reject a model solely from this metric.
