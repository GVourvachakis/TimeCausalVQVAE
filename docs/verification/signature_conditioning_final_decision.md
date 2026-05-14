# Signature Conditioning Final Decision

## Purpose

This record consolidates the S&P500/VIX signature-conditioning evidence after
feature extraction, depth/context ablations, robustness runs, continuous
baseline reproduction, and the signature-kernel metric smoke. It is a
verification and merge-scope decision only. No source code was implemented and
no model was trained for this document.

## Baseline Status

The current promoted discrete baseline remains:

- standard causal VQ tokenizer;
- additive scalar VIX-only causal AR token prior;
- paper-style diagnostics at `temperature=0.8`, `top_k=40`, `n_sample=1000`,
  and `seed=99`;
- latent geometry and market-style token diagnostics as supporting evidence.

The VIX-only prior is still the best discrete model for the distributional
profile. In the same-setting paper-style comparison it achieved MMD
`0.27934083` and SWD `0.00767375`, lower than the depth-3 signature-conditioned
variants and all subsequent `logsig_l3_ctx20` robustness runs.

## Best Log-Signature Variant

The best original log-signature candidate remains `logsig_l3_ctx20`:

- historical context length: `20`;
- depth: `3`;
- lead-lag path transform;
- time and VIX channels included;
- `iisignature` truncated log-signature features;
- feature dimension: `385`;
- total condition dimension: `386`;
- same additive condition embedding as the VIX-only prior.

It is the strongest discrete signature-conditioned result for the
tail-risk/path-functional view in the original paper-style comparison:

| Metric | Original `logsig_l3_ctx20` |
| --- | ---: |
| Returns W1 | 0.00099378 |
| Terminal W1 | 0.00451245 |
| Volatility W1 | 0.00080577 |
| Drawdown W1 | 0.00549569 |
| Return AC L1 | 0.04789805 |
| Squared-return AC L1 | 0.03506229 |

However, the robustness grid did not reproduce this profile reliably. The
additional seeds weakened the path-functional gains, and the 200-epoch variant
improved token likelihood while worsening several paper-style diagnostics
relative to the original `logsig_l3_ctx20`.

## Continuous Reference

The reproduced continuous BetaCVAE checkpoint is available at:

```text
outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47/final_model
```

It remains a strong reference rather than a discrete-token baseline. In the
same-setting paper-style comparison it achieved:

| Metric | Continuous BetaCVAE |
| --- | ---: |
| MMD | 0.15442121 |
| SWD | 0.00878550 |
| Returns W1 | 0.00060182 |
| Terminal W1 | 0.00905099 |
| Volatility W1 | 0.00063360 |
| Drawdown W1 | 0.00766744 |
| Return AC L1 | 0.02597247 |
| Squared-return AC L1 | 0.02946163 |

The continuous model is strongest on MMD, returns W1, volatility W1, and
autocorrelation diagnostics, but it does not replace the discrete VQ-tokenizer
research objective.

## Metrics By Profile

### Distributional

Primary metrics: MMD, SWD, and returns W1.

Decision reading:

- VIX-only is the best discrete model for MMD and SWD.
- Continuous BetaCVAE is the strongest overall reference by MMD and returns W1.
- Signature conditioning does not earn default promotion under this profile.

### Tail-Risk

Primary metrics: terminal-return W1, volatility W1, drawdown W1, and tail
exceedance rates.

Decision reading:

- Original `logsig_l3_ctx20` is the strongest discrete signature run for
  volatility W1 and drawdown W1.
- The robustness `e200` and `lr1e4` variants improve some tail-risk ranks, but
  neither dominates the original `logsig_l3_ctx20` profile.
- The tail-risk signal is promising but not stable enough for default
  promotion.

### Sequential-Dependence

Primary metrics: return autocorrelation L1, squared-return autocorrelation L1,
and volatility-clustering diagnostics.

Decision reading:

- Original `logsig_l3_ctx20` improves squared-return autocorrelation relative to
  VIX-only.
- `logsig_l3_ctx20_e200` improves return autocorrelation within the robustness
  grid, but not enough to dominate the original signature run across sequential
  diagnostics.
- Continuous BetaCVAE remains the strongest reference on the recorded
  autocorrelation metrics.

### Balanced-Market

Primary metrics: SWD, returns W1, terminal W1, volatility W1, drawdown W1,
return AC L1, squared-return AC L1, and visible tail diagnostics.

Decision reading:

- Original `logsig_l3_ctx20` is the most convincing discrete
  path-functional ablation, but robustness did not confirm a stable promotion
  case.
- `logsig_l3_ctx20_e200` is best within the robustness-grid rank summary, but
  its global distributional metrics and several paper-style path functionals
  regress relative to both VIX-only and the original signature run.
- No signature-conditioned variant currently dominates enough profiles to
  replace the default VIX-only baseline.

## W&B And Local Tracking

Tracking was handled safely for this restricted environment. Live cloud W&B was
attempted using `MPLBACKEND=Agg`, `WANDB_DISABLE_SERVICE=true`, and
`WANDB_START_METHOD=thread`, but non-smoke robustness runs hit upstream W&B
`CommError` timeouts before training. Earlier live attempts also exposed local
socket restrictions and Tcl/Tk backend failures.

The accepted execution profile is therefore:

- use `--no-wandb` for non-smoke runs when cloud initialisation fails;
- keep metrics and summaries in local JSON/CSV artifacts under `outputs/`;
- use local file instrumentation and, where available, the local
  `poetry run wandb board wandb/latest-run` browser interface for chart
  inspection;
- document missing W&B URLs explicitly rather than treating telemetry absence as
  experiment failure.

This preserves sandbox safety and keeps the scientific comparison reproducible.

## Decision

Do not promote signature conditioning as the default public discrete method.

Do keep the VIX-only additive causal token prior as the public default and
retain signature conditioning as an optional research branch. Do not defer
signatures entirely: the original `logsig_l3_ctx20` run remains scientifically
useful because it improves several tail-risk and sequential path diagnostics
that are important for market generators.

The decision is therefore:

```text
keep VIX-only default; retain signatures as optional research branch
```

## Merge Scope For `feat/causal-vq-vae`

Merge only documentation and low-risk optional evaluation infrastructure that
does not change promoted defaults:

- architecture and verification notes documenting the signature-conditioning
  decision;
- optional log-signature feature extraction documentation and smoke reports;
- optional condition-feature loading support only if it is already default-null
  and does not affect VIX-only configs;
- model-selection profile documentation;
- telemetry execution notes for this restricted environment.

Do not merge trained robustness artifacts, generated output directories, W&B
state, or any dependency change that makes signature packages mandatory.

## Keep On `research/signature-conditioning`

Keep the following on the research branch unless a later promotion decision is
made:

- signature-conditioned experiment configs;
- robustness ablation configs and results;
- signature-feature extraction outputs;
- optional `iisignature` dependency experiments;
- optional `sigkernel` smoke code and dependency-review notes;
- any future signature-conditioned checkpoint selection work.

The promoted public workflow should remain VIX-only until robustness and
evaluation-only signature-kernel evidence justify changing it.

## Future Work

- Install and test `sigkernel` in an explicit optional environment, then rerun
  the synthetic signature-kernel smoke and apply it to saved paper-style
  batches.
- Do not implement Gumbel-Softmax until an objective-level signature-kernel loss
  is selected and its differentiability path is specified.
- Keep adapted/causal Wasserstein as a later heavy metric after final variant
  selection, not as a blocker for the current public baseline.
- If signatures are revisited, focus on a narrower checkpoint/epoch study around
  the original `logsig_l3_ctx20` and `e200` behaviour before running a broader
  grid.
