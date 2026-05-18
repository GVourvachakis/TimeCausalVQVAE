# Hawkes-Jump Log-Return Robustness Setup

## Status

Seed-robustness setup is complete for the Hawkes/SVMHJD log-return benchmark.
No non-smoke training was run, simulator parameters were unchanged, no registry
files were updated, and no new model families were added.

## Configs Created

Continuous log-return baseline:

- `configs/experiments/hawkes_jump_beta_cvae_logreturn.yaml`
- `configs/experiments/hawkes_jump_beta_cvae_logreturn_seed1.yaml`
- `configs/experiments/hawkes_jump_beta_cvae_logreturn_seed2.yaml`

Hidden128 cb64 log-return tokenizer seeds:

- `configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64_seed1.yaml`
- `configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64_seed2.yaml`

Hidden128 cb64 additive AR prior seeds:

- `configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_additive_seed1.yaml`
- `configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_additive_seed2.yaml`

Hidden128 cb64 causal conv-transformer k3 prior seeds:

- `configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer_seed1.yaml`
- `configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer_seed2.yaml`

The seed configs keep the base experiment names stable and set the seed field to
`1` or `2`. This matches the existing training CLIs, which append `_seedN` to
run directories from the config seed. The prior seed configs point to
`outputs/hawkes_jump_logreturn_robustness/tokenizers/..._seedN` and
`outputs/hawkes_jump_logreturn_robustness/tokens/..._seedN`.

## Continuous Baseline Convention

`hawkes_jump_beta_cvae_logreturn.yaml` is based on
`hawkes_jump_beta_cvae.yaml`. The only data-convention change is:

```yaml
data:
  params:
    simulation_scheme: ogata
    data_output: log_return
```

The simulator parameters remain unchanged. Generated continuous log-return paths
must be converted to normalised prices before market and jump diagnostics. The
generic `tcvae-evaluate` path is not used for this conversion in the robustness
runner; a Hawkes-specific continuous evaluator should be added in a later prompt
before continuous log-return evaluation is reported.

## Runner

`scripts/run_hawkes_logreturn_robustness.py` now builds the seed robustness
workflow for seeds `0`, `1`, and `2`.

Supported stages:

- tokenizer training;
- token extraction;
- additive AR prior training;
- causal conv-transformer k3 prior training;
- continuous BetaCVAE log-return training;
- Hawkes jump-aware token-prior evaluation through
  `scripts/evaluate_hawkes_jump_token_prior.py`;
- aggregate JSON, CSV, and Markdown command-plan outputs.

Supported controls:

- `--dry-run`;
- `--epochs-tokenizer`;
- `--epochs-prior`;
- `--epochs-continuous`;
- `--n-sample`;
- `--no-wandb`;
- `--temperature`;
- `--top-k`;
- `--device`;
- `--base-data-dir`.

For each seed, the runner writes seed-adjusted runtime configs under
`<output-dir>/run_configs/`. This lets the same script override sample count and
output-local tokenizer/token-data paths without changing the checked-in
experiment configs.

## Dry-Run Status

The requested dry-run completed:

```bash
poetry run python scripts/run_hawkes_logreturn_robustness.py \
  --seeds 0 1 2 \
  --output-dir outputs/hawkes_jump_logreturn_robustness_dry \
  --epochs-tokenizer 1 \
  --epochs-prior 1 \
  --epochs-continuous 1 \
  --n-sample 128 \
  --dry-run \
  --no-wandb
```

It wrote:

- `outputs/hawkes_jump_logreturn_robustness_dry/aggregate_summary.json`;
- `outputs/hawkes_jump_logreturn_robustness_dry/aggregate_summary.csv`;
- `outputs/hawkes_jump_logreturn_robustness_dry/command_plan.md`;
- seed-adjusted runtime configs under
  `outputs/hawkes_jump_logreturn_robustness_dry/run_configs/`.

No training or evaluation subprocesses were executed in dry-run mode. The
continuous log-return evaluation stage is marked as pending because it requires
a Hawkes-specific continuous evaluator that converts generated log returns to
normalised prices before diagnostics.

## Non-Smoke Command

Use the following command for the first full robustness pass:

```bash
poetry run python scripts/run_hawkes_logreturn_robustness.py \
  --seeds 0 1 2 \
  --output-dir outputs/hawkes_jump_logreturn_robustness \
  --epochs-tokenizer 50 \
  --epochs-prior 50 \
  --epochs-continuous 50 \
  --n-sample 1024 \
  --no-wandb
```

After the Hawkes-specific continuous evaluator exists, rerun or extend the same
output tree with continuous log-return diagnostics using the same seeds and
sample count.

## Metrics To Report

For tokenizers:

- active codes;
- codebook perplexity;
- reconstruction L1/L2;
- volatility reconstruction error;
- jump/non-jump code usage;
- rare-code lift near jumps.

For token priors:

- prior token cross-entropy, accuracy, and perplexity;
- sampled active codes and sampled token perplexity;
- marginal token L1;
- transition matrix L1;
- run-length W1 and run-length distance.

For generated paths:

- MMD;
- SWD;
- terminal W1;
- volatility W1;
- drawdown W1;
- jump-count W1;
- inter-arrival W1;
- jump-size W1;
- paths-with-jumps fraction;
- negative jump fraction;
- VaR/ES.

For the continuous log-return baseline, report the same path and jump metrics
only after decoded/generated log returns are converted to normalised price paths.
