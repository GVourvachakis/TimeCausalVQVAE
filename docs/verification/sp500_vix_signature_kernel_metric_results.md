# S&P500/VIX Signature-Kernel Metric Results

## 1. Environment

- Python version: `3.12.3`
- PyTorch version: `2.11.0+cu130`
- NumPy version after the requested install probe: `1.26.4`
- `sigkernel` install status: success. The package built from
  `https://github.com/crispitagorico/sigkernel.git` at commit
  `40a583155ea8d2194af0e90dddab37e2659cfcfd` and imported from
  `.venv/lib/python3.12/site-packages/sigkernel/__init__.py`.
- Exact import smoke-test status: failed for the user-specified float32 toy
  tensor. The Cython backend raised
  `ValueError: Buffer dtype mismatch, expected 'double' but got 'float'`.
  The repository evaluation path converts paths to float64 and therefore ran
  successfully.
- Install notes: the first non-escalated build-tools upgrade could not reach
  PyPI from the sandbox. The escalated rerun succeeded. `pip` temporarily
  installed `setuptools 82.0.1`, then the `sigkernel` install restored
  `setuptools 81.0.0`, satisfying Torch's `<82` bound.
- `iisignature` fallback used: no. The current
  `signature_kernel_metrics.py` implementation is `sigkernel`-only and has no
  iisignature CPU fallback path.

## 2. Synthetic smoke test

Command run:

```bash
poetry run python scripts/evaluate_signature_kernel_metrics.py \
    --synthetic \
    --output-dir outputs/signature_kernel_smoke \
    --max-samples 16 \
    --dyadic-order 1 \
    --include-time
```

- Exit status: `0`
- Wall-clock runtime: `3` seconds from the shell wrapper
- Script runtime: `0.922249` seconds
- MMD value: `0.0000399919`
- Finite checks: `Kxx=true`, `Kyy=true`, `Kxy=true`
- Symmetry checks: `Kxx=true`, `Kyy=true`, `Kxy=n/a`
- Positive-diagonal checks: `Kxx=true`, `Kyy=true`, `Kxy=n/a`

## 3. Metric settings

The current script does not expose `--real-dir`, `--fake-dir`, or `--lead-lag`.
It supports saved paper-style batches through `--paper-style-batch` and uses
`--use-lead-lag` for lead-lag preprocessing. The completed first pass therefore
used each existing `discrete_paper_style_batch.pt` through the supported
paper-style batch loader.

| setting | first pass | second pass |
| --- | --- | --- |
| max_samples | `256` | `128` |
| dyadic_order | `1` | `1` |
| include_time | `true` | `true` |
| lead_lag | `false` | `true`, but skipped after the interrupted/crashed run attempt |

## 4. Results per model

### VIX-only

- Output directory name:
  `outputs/signature_kernel_results/vix_only_no_leadlag`
- First-pass runtime: `132` seconds wall-clock, `129.755149` seconds from the
  script summary
- First-pass signature-kernel MMD: `0.00555772`
- First-pass checks:
  - `Kxx`: finite `true`, symmetric `true`, positive diagonal `true`
  - `Kyy`: finite `true`, symmetric `true`, positive diagonal `true`
  - `Kxy`: finite `true`, symmetry `n/a`, positive diagonal `n/a`
- Second-pass runtime and signature-kernel MMD: skipped after the lead-lag run
  attempt was interrupted/crashed and the user requested no retry.
- Second-pass checks: skipped.

### Raw `logsig_l3_ctx20`

- Output directory name:
  `outputs/signature_kernel_results/logsig_l3_ctx20_no_leadlag`
- First-pass runtime: `132` seconds wall-clock, `130.869959` seconds from the
  script summary
- First-pass signature-kernel MMD: `0.00238908`
- First-pass checks:
  - `Kxx`: finite `true`, symmetric `true`, positive diagonal `true`
  - `Kyy`: finite `true`, symmetric `true`, positive diagonal `true`
  - `Kxy`: finite `true`, symmetry `n/a`, positive diagonal `n/a`
- Second-pass runtime and signature-kernel MMD: skipped after the lead-lag run
  attempt was interrupted/crashed and the user requested no retry.
- Second-pass checks: skipped.

### Standardised `logsig_l3_ctx20`

- Output directory name:
  `outputs/signature_kernel_results/logsig_l3_ctx20_std_no_leadlag`
- First-pass runtime: `134` seconds wall-clock, `131.367742` seconds from the
  script summary
- First-pass signature-kernel MMD: `0.00224031`
- First-pass checks:
  - `Kxx`: finite `true`, symmetric `true`, positive diagonal `true`
  - `Kyy`: finite `true`, symmetric `true`, positive diagonal `true`
  - `Kxy`: finite `true`, symmetry `n/a`, positive diagonal `n/a`
- Second-pass runtime and signature-kernel MMD: skipped after the lead-lag run
  attempt was interrupted/crashed and the user requested no retry.
- Second-pass checks: skipped.

## 5. Comparison to existing metrics

The first-pass signature-kernel ranking over the evaluated discrete batches is:

1. standardised `logsig_l3_ctx20`: `0.00224031`
2. raw `logsig_l3_ctx20`: `0.00238908`
3. VIX-only: `0.00555772`

This reverses the global MMD/SWD ranking documented in
`sp500_vix_signature_paper_style_comparison.md`, where VIX-only was the best
discrete model for MMD and SWD. It agrees more closely with the path-functional
reading in `sp500_vix_logsig_l3_ctx20_robustness_results.md`, where signature
conditioning was retained as a promising research branch for tail-risk and
sequential-dependence diagnostics.

The standardised signature variant ranks best by signature-kernel MMD, even
though `sp500_vix_logsig_l3_ctx20_standardized_quality.md` concluded that raw
`logsig_l3_ctx20` remains stronger on the balanced discrete market profile.
That is a useful rank reversal: signature-kernel MMD appears sensitive to
path-shape similarity in a way that is not identical to the existing
market-style profile ranks. The continuous BetaCVAE was not included in this
signature-kernel pass because the requested model list covered the saved
discrete paper-style batches.

## 6. Decision

B. **Exploratory** — compute signature-kernel MMD when the time budget allows,
but do not block model selection on it yet.

The metric now runs successfully on synthetic and saved S&P500/VIX
paper-style batches, and its ranking is informative: both signature-conditioned
models beat VIX-only, while the standardised variant narrowly beats the raw
signature variant. However, the install remains non-trivial, the exact
user-provided float32 import smoke exposes a dtype caveat in the Cython
backend, the current implementation has no iisignature fallback, and the
lead-lag pass was interrupted/crashed before completion. These are enough
reasons to keep signature-kernel MMD as an exploratory evaluation metric rather
than a mandatory model-selection gate.
