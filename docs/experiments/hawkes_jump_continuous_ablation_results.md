# Hawkes-Jump Continuous Ablation Results

## Status

This note records the small continuous-latent ablation requested before Hawkes/SVMHJD
log-return registry promotion. It does not change Ogata simulator parameters,
discrete results, trained model registry entries, or model families.

The existing repaired `BetaCVAE` identity baseline was reused, and a matched
`InfoCVAE` identity baseline was trained and evaluated for seeds `0`, `1`, and
`2`. All continuous evaluations used
`scripts/evaluate_hawkes_jump_continuous.py` with `n_sample=1024`, which converts
generated and real log returns to normalised price paths before smooth,
jump-regime, and tail diagnostics.

## Continuous Configs Tested

| Objective | Configs | Seeds | Transform | Samples | Epochs |
|---|---|---:|---|---:|---:|
| `BetaCVAE` | `configs/experiments/hawkes_jump_beta_cvae_logreturn_identity*.yaml` | 0, 1, 2 | `id` / `id` | 1024 | 50 |
| `InfoCVAE` | `configs/experiments/hawkes_jump_info_cvae_logreturn_identity*.yaml` | 0, 1, 2 | `id` / `id` | 1024 | 50 |

The new `InfoCVAE` configs are based on the repaired BetaCVAE identity configs.
They keep the same Hawkes data convention, architecture, optimiser, batch sizes,
and RealNVP prior settings. The objective-specific values are `beta: 0.04` and
`alpha: 0.2`, matching the existing PDV InfoCVAE pairing where the beta weight is
also `0.04`.

The Ogata parameters remain unchanged: `simulation_scheme: ogata`,
`data_output: log_return`, `dt: 0.0166666667`, `brownian_volatility: 0.18`,
`baseline_intensity: 3.0`, `excitation: 2.0`, `decay: 12.0`,
`mark_excitation: 20.0`, `negative_jump_probability: 0.7`,
`severe_jump_probability: 0.08`, and `volatility_excitation: true`.

## Transform And Normalisation Status

The continuous transform registry currently supports only `id`, `log`, and
`exp`. The `log` / `exp` pair is invalid for signed log-return data because
negative returns produce NaNs before the loss is computed. The identity transform
is therefore the only signed-data-safe transform available without source-code
changes.

No standardised BetaCVAE or InfoCVAE configs were added. Signed log-return
standardisation remains future work because the current config schema has no
train-fitted standardisation transform with a matching inverse transform.

## Aggregate Outputs

The comparison aggregator was updated to include:

- repaired continuous log-return BetaCVAE identity seeds `0/1/2`;
- continuous log-return InfoCVAE identity seeds `0/1/2`;
- cb64 additive AR discrete seeds `0/1/2`;
- cb64 causal conv-transformer k3 discrete seeds `0/1/2`.

The aggregate was written by:

```bash
poetry run python scripts/aggregate_hawkes_jump_comparison.py \
  --output-dir outputs/hawkes_jump_continuous_ablation_comparison
```

Generated aggregate files:

- `outputs/hawkes_jump_continuous_ablation_comparison/aggregate_summary.json`;
- `outputs/hawkes_jump_continuous_ablation_comparison/aggregate_summary.csv`;
- `outputs/hawkes_jump_continuous_ablation_comparison/aggregate_summary.md`.

## Mean/Std Metrics

| Model | MMD | SWD | Terminal W1 | Volatility W1 | Drawdown W1 |
|---|---:|---:|---:|---:|---:|
| Continuous log-return BetaCVAE | 1.3153 / 0.1405 | 0.1320 / 0.0231 | 0.1682 / 0.0316 | 0.0243 / 0.0003 | 0.1486 / 0.0754 |
| Continuous log-return InfoCVAE | 1.2547 / 0.0678 | 0.1220 / 0.0130 | 0.1480 / 0.0143 | 0.0245 / 0.0001 | 0.1500 / 0.0604 |
| cb64 additive AR | 0.1567 / 0.0644 | 0.0238 / 0.0085 | 0.0320 / 0.0152 | 0.0011 / 0.0008 | 0.0106 / 0.0059 |
| cb64 conv-transformer k3 | 0.1141 / 0.0355 | 0.0186 / 0.0060 | 0.0217 / 0.0120 | 0.0010 / 0.0010 | 0.0111 / 0.0052 |

| Model | Jump-count W1 | Inter-arrival W1 | Jump-size W1 | Generated jump count | Negative jump fraction |
|---|---:|---:|---:|---:|---:|
| Continuous log-return BetaCVAE | 0.2236 / 0.0144 | 15.6138 / 2.9303 | 0.0844 / 0.0775 | 0.0026 / 0.0030 | 0.6667 / 0.5774 |
| Continuous log-return InfoCVAE | 0.2246 / 0.0138 | 15.6138 / 2.9303 | 0.2955 / 0.1241 | 0.0016 / 0.0006 | 0.6667 / 0.5774 |
| cb64 additive AR | 0.0469 / 0.0319 | 6.3080 / 4.2239 | 0.0180 / 0.0101 | 0.2314 / 0.0492 | 0.9955 / 0.0078 |
| cb64 conv-transformer k3 | 0.0576 / 0.0324 | 8.1888 / 6.7270 | 0.0177 / 0.0101 | 0.2357 / 0.0555 | 0.9989 / 0.0019 |

| Model | VaR 1% | ES 1% | Active codes | Sampled code perplexity | Runtime seconds |
|---|---:|---:|---:|---:|---:|
| Continuous log-return BetaCVAE | -0.0061 / 0.0020 | -0.0113 / 0.0018 | n/a | n/a | 37.51 / 1.12 |
| Continuous log-return InfoCVAE | -0.0060 / 0.0017 | -0.0105 / 0.0029 | n/a | n/a | 43.86 / 0.99 |
| cb64 additive AR | -0.0745 / 0.0028 | -0.1068 / 0.0080 | 64.00 / 0.00 | 44.36 / 0.70 | 93.19 / 0.26 |
| cb64 conv-transformer k3 | -0.0748 / 0.0026 | -0.1069 / 0.0080 | 64.00 / 0.00 | 44.42 / 0.80 | 109.69 / 0.20 |

## Interpretation

InfoCVAE is marginally better than BetaCVAE on several smooth metrics, including
MMD, SWD, and terminal W1, but it remains far weaker than both cb64 discrete
priors. Its volatility W1 remains around `0.0245`, compared with about `0.001`
for the discrete models.

The jump-regime result is also unchanged. Both continuous variants generate
almost no detected jumps after log-return-to-price conversion. The mean generated
jump count is `0.0026` for BetaCVAE and `0.0016` for InfoCVAE, while the real
evaluation paths average `0.2262` detected jumps and the discrete priors average
about `0.23`. InfoCVAE also worsens jump-size W1 relative to BetaCVAE.

The lower tail remains the clearest failure mode. InfoCVAE has one-percent VaR
`-0.0060` and ES `-0.0105`, which are effectively the same shallow-tail profile
as BetaCVAE. The cb64 additive and conv-transformer priors remain near
`-0.074` VaR and `-0.107` ES.

## Decision

No continuous variant tested here changes the current conclusion. The cb64
causal conv-transformer k3 remains the leading overall discrete registry
candidate, and cb64 additive AR remains the required jump-metric ablation.

Registry promotion can proceed as a separate follow-up because both repaired
continuous comparators are clearly weaker. This note intentionally stops after
documentation and does not update `trained_models/model_registry.yaml`.
