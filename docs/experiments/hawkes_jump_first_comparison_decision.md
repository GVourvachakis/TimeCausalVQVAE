# Hawkes-Jump First Comparison Decision

## Status

The first Hawkes-jump model comparison should be treated as a diagnostic result,
not as a registry-quality benchmark win. The benchmark is integrated and useful,
but the discrete models have not yet shown a credible jump-regime advantage.

No model registry update should be made from this run.

## What Was Tested

The comparison used the research-quality Ogata SVMHJD backend with unchanged
simulator dynamics. Because the continuous trainer's evaluation path samples
1000 generated paths, the run used matched local `1024` configs under
`outputs/hawkes_jump_first_model_comparison/run_configs/` rather than changing
the public `512` sample configs.

Tested candidates:

- continuous `BetaCVAE`;
- standard VQ tokenizer plus additive autoregressive token prior;
- hidden128 VQ tokenizer plus causal conv-transformer k3 token prior.

Run settings:

| Field | Value |
| --- | ---: |
| Simulation scheme | `ogata` |
| Train and eval samples | 1024 |
| Timesteps | 60 |
| Seed | 0 |
| Epochs | 2 |
| Evaluation sample count | 1024 |
| W&B | disabled |
| Device | CPU |

The hidden128 VQ plus additive AR candidate was not run because no Hawkes config
currently exists for that exact combination.

## What Passed

The benchmark infrastructure passed the required readiness gates:

- dataset smoke checks passed for positive finite paths, expected tensor shapes,
  finite intensities, finite volatilities, non-trivial jumps, and downside jump
  asymmetry;
- visual diagnostics passed for Ogata and fixed-grid comparison plots and
  summary statistics;
- dataset no-leakage checks passed, with oracle jump labels and simulator
  metadata kept out of model-visible tensors;
- tokenizer and token-prior no-leakage checks passed for the Hawkes configs;
- pipeline integration passed for continuous and tokenizer training entry
  points;
- the first comparison completed model training, token extraction, prior
  training, path evaluation, jump diagnostics, and token diagnostics.

## Main Result

The continuous `BetaCVAE` remains strongest on the main smooth distributional
metrics from the first comparison:

- MMD: `0.9638`, versus `1.8068` for standard VQ + additive AR and `1.6380`
  for hidden128 VQ + conv-transformer k3;
- SWD: `0.1133`, versus `0.1806` and `0.1446`;
- terminal W1: `0.2263`, versus `1.9214` and `1.1902`.

Among the discrete candidates, the hidden128 VQ plus conv-transformer k3 prior
is better than the standard VQ plus additive AR prior on MMD, SWD, terminal W1,
volatility W1, drawdown W1, tokenizer reconstruction error, marginal token
usage distance, and transition-matrix distance.

However, the run does not show a credible discrete jump-regime advantage yet.
The discrete decoded paths still place detected jumps on every generated path,
understate left-tail VaR and ES, and fail to reproduce the strongly negative
detected jump sign profile of the Ogata evaluation paths.

## Main Blocker

Tokenizer utilisation collapsed before prior training:

| Tokenizer | Extracted active codes | Extracted perplexity |
| --- | ---: | ---: |
| Standard VQ | 6 / 64 | 2.8179 |
| Hidden128 VQ | 4 / 64 | 2.6268 |

This means the prior comparison is operating on a very small effective token
alphabet. The hidden128 conv-transformer prior can improve transition metrics
within that collapsed alphabet, but the tokens are not yet rich enough to
represent rare-event regimes, downside jump signs, or tail-shape variation.

The most likely bottleneck is the current price-level tokenisation target:
`data_output: price`. For Hawkes-jump paths, price levels can make jump
information too diffuse for the tokenizer, especially over a short two-epoch
run.

## Decision

Do not update `trained_models/model_registry.yaml`.

Do not abandon the Hawkes-jump benchmark. The dataset, diagnostics, and pipeline
are ready enough to expose meaningful failure modes.

Do not change the token-prior family yet. The next bottleneck is upstream of the
prior: the tokenizer needs healthier code utilisation and more jump-sensitive
representations before longer or more expressive priors can be evaluated fairly.

Run Hawkes tokenizer-utilisation ablations next.

## Next Ablations

The next phase should focus on tokenizer inputs and codebook utilisation before
training full priors:

- switch the tokenizer target from `data_output: price` to
  `data_output: log_return`;
- test a two-channel price/return representation only if it is already
  supported by the dataset and tokenizer pipeline;
- ablate `codebook_size` over `16`, `32`, and `64`;
- ablate `codebook_dim` around the current value of `16`;
- ablate `commitment_weight` around the current value of `0.1`;
- review existing `kmeans_init` and `kmeans_iters` settings, currently `true`
  and `10`, and any available dead-code or code-refresh settings before adding
  new mechanisms;
- train tokenizers longer than two epochs while monitoring active codes,
  perplexity, reconstruction error, volatility reconstruction error, and
  jump-diagnostic reconstruction behaviour.

Only after token utilisation is healthy should the additive AR and hidden128
conv-transformer k3 priors be rerun for a fair jump-regime comparison.
