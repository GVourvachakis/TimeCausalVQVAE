# Hawkes-Jump Final Model Decision

## Status

This note supersedes the earlier model-selection caveat in
`hawkes_jump_final_decision.md`. At that point, the matched continuous
log-return baseline had failed with NaN loss. The repaired identity-transform
continuous baseline has now trained and evaluated for seeds `0/1/2`, so the
current Hawkes/SVMHJD model decision can be made against a valid matched
continuous comparator.

The current decision is:

- merge simulator infrastructure separately;
- do not update `trained_models/model_registry.yaml` yet;
- keep `hidden128_logreturn_cb64 + causal conv-transformer k3` as the leading
  discrete registry candidate;
- retain `hidden128_logreturn_cb64 + additive AR` as the required jump-metric
  ablation.

## Simulator Status

The Ogata backend is research-quality and ready for a simulator-only public
merge after review. It implements continuous-time Ogata modified thinning,
exponential Hawkes intensity decay, branching-ratio validation, asymmetric
folded-normal marks, mark-dependent excitation, jump-excited volatility, exact
Brownian variance integration inside event-free sub-intervals, and
O(n_events + n_timestep) projection onto the regular model grid.

The fixed-grid backend should remain available as the smoke and throughput
backend. It is useful for fast integration tests, but research comparisons
should default to `simulation_scheme: ogata`.

The simulator remains scenario-data infrastructure. It is not an arbitrage-free
risk-neutral pricing model, and no registry or model claim should imply
arbitrage-free generation.

## Continuous Log-Return Baseline Status

The repaired continuous baseline is valid but weak. The selected continuous
comparison is the identity-transform log-return `BetaCVAE` config family:

- `configs/experiments/hawkes_jump_beta_cvae_logreturn_identity.yaml`;
- `configs/experiments/hawkes_jump_beta_cvae_logreturn_identity_seed1.yaml`;
- `configs/experiments/hawkes_jump_beta_cvae_logreturn_identity_seed2.yaml`.

All three seeds trained for `50` epochs with W&B disabled and no NaN loss. The
Hawkes-specific continuous evaluator converted generated and real log returns
to normalised price paths before smooth and jump diagnostics.

Across seeds, the repaired continuous baseline still underfits the Hawkes jump
and tail structure:

| Metric | Mean | Std |
|---|---:|---:|
| MMD | 1.3153 | 0.1405 |
| SWD | 0.1320 | 0.0231 |
| Terminal W1 | 0.1682 | 0.0316 |
| Volatility W1 | 0.0243 | 0.0003 |
| Drawdown W1 | 0.1486 | 0.0754 |
| Jump-count W1 | 0.2236 | 0.0144 |
| Inter-arrival W1 | 15.6138 | 2.9303 |
| Jump-size W1 | 0.0844 | 0.0775 |
| VaR 1% | -0.0061 | 0.0020 |
| ES 1% | -0.0113 | 0.0018 |

The continuous baseline produces paths that are too smooth after
log-return-to-price conversion. Its one-percent VaR and ES are far too shallow
relative to the Ogata evaluation tail.

## Discrete Additive AR Status

The additive prior is the strongest jump-count and inter-arrival baseline among
the current discrete candidates. It is not the leading overall candidate because
the conv-transformer has the stronger smooth profile, but it remains essential
for registry evidence because it is simple, competitive, and slightly better on
some sparse-jump diagnostics.

| Metric | Mean | Std |
|---|---:|---:|
| MMD | 0.1567 | 0.0644 |
| SWD | 0.0238 | 0.0085 |
| Terminal W1 | 0.0320 | 0.0152 |
| Volatility W1 | 0.0011 | 0.0008 |
| Drawdown W1 | 0.0106 | 0.0059 |
| Jump-count W1 | 0.0469 | 0.0319 |
| Inter-arrival W1 | 6.3080 | 4.2239 |
| Jump-size W1 | 0.0180 | 0.0101 |
| VaR 1% | -0.0745 | 0.0028 |
| ES 1% | -0.1068 | 0.0080 |
| Sampled active codes | 64.00 | 0.00 |
| Sampled code perplexity | 44.36 | 0.70 |

## Discrete Conv-Transformer Status

The `hidden128_logreturn_cb64 + causal conv-transformer k3` prior is the leading
overall discrete candidate. It has the best smooth profile, similar jump-size
and tail metrics to additive AR, and full sampled code usage on every seed.

| Metric | Mean | Std |
|---|---:|---:|
| MMD | 0.1141 | 0.0355 |
| SWD | 0.0186 | 0.0060 |
| Terminal W1 | 0.0217 | 0.0120 |
| Volatility W1 | 0.0010 | 0.0010 |
| Drawdown W1 | 0.0111 | 0.0052 |
| Jump-count W1 | 0.0576 | 0.0324 |
| Inter-arrival W1 | 8.1888 | 6.7270 |
| Jump-size W1 | 0.0177 | 0.0101 |
| VaR 1% | -0.0748 | 0.0026 |
| ES 1% | -0.1069 | 0.0080 |
| Sampled active codes | 64.00 | 0.00 |
| Sampled code perplexity | 44.42 | 0.80 |

## Smooth Profile

The discrete priors dominate the repaired continuous baseline on all reported
smooth path metrics. The conv-transformer is the best smooth discrete model:

| Model | MMD | SWD | Terminal W1 | Volatility W1 | Drawdown W1 |
|---|---:|---:|---:|---:|---:|
| Continuous log-return BetaCVAE | 1.3153 / 0.1405 | 0.1320 / 0.0231 | 0.1682 / 0.0316 | 0.0243 / 0.0003 | 0.1486 / 0.0754 |
| cb64 additive AR | 0.1567 / 0.0644 | 0.0238 / 0.0085 | 0.0320 / 0.0152 | 0.0011 / 0.0008 | 0.0106 / 0.0059 |
| cb64 conv-transformer k3 | 0.1141 / 0.0355 | 0.0186 / 0.0060 | 0.0217 / 0.0120 | 0.0010 / 0.0010 | 0.0111 / 0.0052 |

This reverses the earlier diffusion-style expectation that the continuous model
might dominate smooth metrics. On the repaired Hawkes log-return benchmark, the
continuous model is not competitive.

## Jump-Regime Profile

The discrete priors also dominate the repaired continuous baseline on
jump-specific diagnostics:

| Model | Jump-count W1 | Inter-arrival W1 | Jump-size W1 | Negative jump fraction |
|---|---:|---:|---:|---:|
| Continuous log-return BetaCVAE | 0.2236 / 0.0144 | 15.6138 / 2.9303 | 0.0844 / 0.0775 | 0.6667 / 0.5774 |
| cb64 additive AR | 0.0469 / 0.0319 | 6.3080 / 4.2239 | 0.0180 / 0.0101 | 0.9955 / 0.0078 |
| cb64 conv-transformer k3 | 0.0576 / 0.0324 | 8.1888 / 6.7270 | 0.0177 / 0.0101 | 0.9989 / 0.0019 |

The additive AR prior is marginally stronger on jump-count and inter-arrival W1.
The conv-transformer remains close on these metrics while improving smooth-path
fit.

## VaR/ES

The discrete priors preserve the downside tail scale. The repaired continuous
baseline does not:

| Model | VaR 1% | ES 1% |
|---|---:|---:|
| Continuous log-return BetaCVAE | -0.0061 / 0.0020 | -0.0113 / 0.0018 |
| cb64 additive AR | -0.0745 / 0.0028 | -0.1068 / 0.0080 |
| cb64 conv-transformer k3 | -0.0748 / 0.0026 | -0.1069 / 0.0080 |

The seed-0 real Ogata reference reported VaR 1% `-0.0724` and ES 1% `-0.1064`.
Both discrete priors remain close to that tail scale. The continuous baseline is
too light-tailed by roughly an order of magnitude.

## Token Diagnostics

The log-return tokenizer-utilisation phase fixed the code-collapse blocker from
the first price-tokenizer comparison. Extracted active codes were `63/64`,
`64/64`, and `64/64` across seeds `0/1/2`; both priors sampled all `64/64`
codes on all seeds.

| Model | Sampled active codes | Sampled code perplexity | Transition L1 | Run-length W1 |
|---|---:|---:|---:|---:|
| cb64 additive AR | 64.00 / 0.00 | 44.36 / 0.70 | 0.4341 / 0.0153 | 0.0031 / 0.0033 |
| cb64 conv-transformer k3 | 64.00 / 0.00 | 44.42 / 0.80 | 0.4345 / 0.0177 | 0.0037 / 0.0022 |

Token diagnostics no longer explain away the discrete result as collapse. The
remaining transition L1 values are not perfect, but sampled usage and
run-lengths are stable enough for the current benchmark stage.

## Matched-Comparison Decision

The discrete advantage survives the matched continuous comparison. The repaired
continuous baseline is numerically valid, but it is worse on smooth metrics,
jump-regime metrics, and tail-risk metrics. The current result is therefore no
longer blocked by the continuous NaN failure.

The result should still be framed carefully: it compares one repaired
continuous `BetaCVAE` baseline against the improved log-return discrete stack. It
does not rule out stronger continuous objectives, standardised log-return
training, richer likelihoods, or a continuous diffusion/score branch.

## Registry Decision

Do not update `trained_models/model_registry.yaml` now.

The model evidence is positive, but the clean public action is a simulator-only
merge first. A registry update should wait until the simulator infrastructure is
reviewed independently and the model-selection package can be presented without
mixing infrastructure, generated outputs, and registry claims in one branch.

The selected future registry candidates are:

- continuous comparator:
  `configs/experiments/hawkes_jump_beta_cvae_logreturn_identity.yaml`;
- discrete tokenizer:
  `configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64.yaml`;
- additive prior ablation:
  `configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_additive.yaml`;
- leading discrete prior:
  `configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer.yaml`.

The registry sampling policy should be:

- `temperature=1.0`;
- `top_k=none`;
- Ogata simulator with unchanged benchmark parameters;
- evaluation sample count `1024`;
- seeds `0/1/2` reported as mean/std.

The registry metrics should include:

- MMD, SWD, terminal W1, volatility W1, drawdown W1;
- jump-count W1, inter-arrival W1, jump-size W1;
- negative jump fraction and paths-with-jumps fraction;
- VaR 1% and ES 1%;
- sampled active codes, sampled codebook perplexity, transition L1, and
  run-length W1.

## Exact Next Step

Create a simulator-only merge branch from `main`, for example
`feature/hawkes-jump-simulator`, and cherry-pick only infrastructure:

- Ogata and fixed-grid Hawkes dataset backends;
- jump diagnostics and Hawkes-specific smoke checks;
- plotting and no-leakage scripts;
- simulator and dataset documentation;
- smoke configs required to demonstrate integration.

Do not include trained outputs, model-comparison outputs, or registry changes in
that merge. After the simulator-only branch is reviewed, prepare a separate
model-selection branch for the cb64 log-return discrete results and the repaired
continuous comparator.
