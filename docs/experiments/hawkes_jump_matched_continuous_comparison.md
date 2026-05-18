# Hawkes-Jump Matched Continuous Comparison

## Status

The repaired continuous log-return `BetaCVAE` baseline was trained and evaluated for
seeds `0`, `1`, and `2` using the identity-transform config family. All three runs
completed `50` epochs with W&B disabled and without NaN losses. Evaluation used
`scripts/evaluate_hawkes_jump_continuous.py`, which converts generated and real
log-return paths to normalised price paths before smooth and jump diagnostics.

The comparison below uses the existing discrete robustness outputs for:

- `cb64_additive_ar`;
- `cb64_conv_transformer_k3`.

No discrete prior was retrained.

## Aggregate Outputs

The reproducible aggregate was written by:

```bash
poetry run python scripts/aggregate_hawkes_jump_comparison.py \
  --output-dir outputs/hawkes_jump_matched_continuous_comparison
```

It reads:

- continuous summaries from
  `outputs/hawkes_jump_continuous_logreturn_identity/seed*/evaluation`;
- discrete summaries from
  `outputs/hawkes_jump_logreturn_robustness/evaluations`;
- runtime summaries from the matching training directories.

The generated aggregate files are:

- `outputs/hawkes_jump_matched_continuous_comparison/aggregate_summary.json`;
- `outputs/hawkes_jump_matched_continuous_comparison/aggregate_summary.csv`;
- `outputs/hawkes_jump_matched_continuous_comparison/aggregate_summary.md`.

## Mean/Std Metrics

| Model | MMD | SWD | Terminal W1 | Volatility W1 | Drawdown W1 |
|---|---:|---:|---:|---:|---:|
| Continuous log-return BetaCVAE | 1.3153 +/- 0.1405 | 0.1320 +/- 0.0231 | 0.1682 +/- 0.0316 | 0.0243 +/- 0.0003 | 0.1486 +/- 0.0754 |
| cb64 additive AR | 0.1567 +/- 0.0644 | 0.0238 +/- 0.0085 | 0.0320 +/- 0.0152 | 0.0011 +/- 0.0008 | 0.0106 +/- 0.0059 |
| cb64 conv-transformer k3 | 0.1141 +/- 0.0355 | 0.0186 +/- 0.0060 | 0.0217 +/- 0.0120 | 0.0010 +/- 0.0010 | 0.0111 +/- 0.0052 |

| Model | Jump-count W1 | Inter-arrival W1 | Jump-size W1 | Negative jump fraction |
|---|---:|---:|---:|---:|
| Continuous log-return BetaCVAE | 0.2236 +/- 0.0144 | 15.6138 +/- 2.9303 | 0.0844 +/- 0.0775 | 0.6667 +/- 0.5774 |
| cb64 additive AR | 0.0469 +/- 0.0319 | 6.3080 +/- 4.2239 | 0.0180 +/- 0.0101 | 0.9955 +/- 0.0078 |
| cb64 conv-transformer k3 | 0.0576 +/- 0.0324 | 8.1888 +/- 6.7270 | 0.0177 +/- 0.0101 | 0.9989 +/- 0.0019 |

| Model | VaR 1% | ES 1% | Active codes | Sampled code perplexity | Training runtime |
|---|---:|---:|---:|---:|---:|
| Continuous log-return BetaCVAE | -0.0061 +/- 0.0020 | -0.0113 +/- 0.0018 | n/a | n/a | 37.51s +/- 1.12s |
| cb64 additive AR | -0.0745 +/- 0.0028 | -0.1068 +/- 0.0080 | 64.00 +/- 0.00 | 44.36 +/- 0.70 | 93.19s +/- 0.26s |
| cb64 conv-transformer k3 | -0.0748 +/- 0.0026 | -0.1069 +/- 0.0080 | 64.00 +/- 0.00 | 44.42 +/- 0.80 | 109.69s +/- 0.20s |

## Interpretation

The repaired continuous baseline no longer fails numerically, but it does not form
a competitive matched baseline for this Hawkes/SVMHJD benchmark. Its generated
paths are too smooth in price space after log-return-to-price conversion:

- volatility W1 is about `0.0243`, versus about `0.0010` for the discrete priors;
- jump-count W1 is about `0.2236`, versus `0.0469` for additive AR and `0.0576`
  for the conv-transformer;
- one-percent VaR and ES are an order of magnitude too shallow, at about
  `-0.0061` and `-0.0113`, while the discrete priors remain close to the
  Hawkes evaluation tail around `-0.074` and `-0.107`.

The discrete rare-event advantage therefore survives the repaired matched
continuous comparison. The advantage is not only on token diagnostics. It appears
directly in price-space smooth metrics, jump-count and jump-size diagnostics,
and lower-tail VaR/ES.

Between the two discrete candidates, the conv-transformer k3 prior has the best
smooth profile, with lower MMD, SWD, and terminal W1. The additive AR prior has
a slightly better jump-count and inter-arrival profile. Both keep full sampled
code usage at `64/64`, with sampled codebook perplexity around `44`.

## Caveats

This comparison repairs only the transform mismatch in the continuous baseline.
It does not tune the continuous objective, variance decoder assumptions, latent
capacity, or training schedule. The continuous model may need standardised
log-return training, a different likelihood scale, or a diffusion/score-style
continuous branch before making a broader claim about continuous latent models.

The discrete priors were reused from the existing robustness run, so this note
does not introduce new discrete evidence. It strictly adds the matched repaired
continuous comparison requested after the NaN failure.

## Decision

For the current Hawkes/SVMHJD log-return benchmark, the discrete hidden128 cb64
priors remain the strongest candidates after a valid matched continuous
comparison. The registry should still remain unchanged until the simulator and
benchmark scope decision is separated from model-selection policy, but the
previous conclusion that the continuous baseline was unresolved due to NaNs is
now closed.

The next exact step is to update the final Hawkes decision memo to replace the
failed-continuous caveat with this repaired comparison, then decide whether the
Ogata simulator infrastructure should be merged independently of model registry
updates.
