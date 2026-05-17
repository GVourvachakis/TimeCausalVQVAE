# Per-Experiment Model-Selection Results

## Scope

This document records the non-smoke per-experiment discrete candidate run executed on
2026-05-17 on branch `research/per-experiment-model-selection`. It records generated outputs only
by path. Checkpoints, token tensors, CSV files, JSON summaries, and other generated artefacts
remain under ignored `outputs/` paths and must not be committed.

The runner command was:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread \
poetry run python scripts/run_per_experiment_model_selection.py \
  --experiments black_scholes heston pdv sp500_vix \
  --output-dir outputs/per_experiment_selection \
  --epochs full \
  --n-sample 1000 \
  --wandb \
  --wandb-entity tc_vae
```

The W&B run failed during the first tokenizer initialisation with a 90 second run-initialisation
timeout and no usable run URL was produced. The same command was rerun with `--no-wandb`. During
that rerun, a runner path bug was found for legacy public tokenizer directories; the runner was
fixed to derive tokenizer run directories from the tokenizer config experiment name and seed. The
successful fallback command was:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread \
poetry run python scripts/run_per_experiment_model_selection.py \
  --experiments black_scholes heston pdv sp500_vix \
  --output-dir outputs/per_experiment_selection \
  --epochs full \
  --n-sample 1000 \
  --no-wandb \
  --wandb-entity tc_vae
```

Aggregate outputs:

- `outputs/per_experiment_selection/selection_results.json`;
- `outputs/per_experiment_selection/selection_results.csv`.

## Run Status

| Experiment | Candidates completed | W&B status | No-leakage status | Notebook reproduction status |
| --- | ---: | --- | --- | --- |
| Black-Scholes | 3/3 | failed, then `--no-wandb` fallback | not rerun in this pass | not rerun in this pass |
| Heston | 3/3 | failed, then `--no-wandb` fallback | not rerun in this pass | not rerun in this pass |
| PDV4 | 3/3 | failed, then `--no-wandb` fallback | not rerun in this pass | not rerun in this pass |
| S&P500/VIX | 2/2 | failed, then `--no-wandb` fallback | not rerun in this pass | not rerun in this pass |

The run validates candidate config paths and trains tokenizers, extracts token indices, and trains
token priors. It does not run the dedicated no-future-leakage scripts, paper-style path evaluation,
or notebooks. Consequently, the selections below are token-run selections, not final public
registry promotions. Before updating `trained_models/model_registry.yaml`, rerun the relevant
no-leakage checks and notebook or paper-style reproduction workflow.

Path metrics from the model-selection plan, including MMD, SWD, terminal W1, volatility W1,
drawdown W1, return AC L1, and squared-return AC L1, were not produced by this runner and are
therefore unavailable in this result file.

## Selected Candidates

The primary available selection metric is prior validation cross-entropy, with token usage kept
visible. Lower cross-entropy and perplexity are better; higher accuracy and healthier token usage
are better.

| Experiment | Selected token-run candidate | Reason |
| --- | --- | --- |
| Black-Scholes | `hidden128_conv_transformer_k3` | Lowest available prior eval cross-entropy, `2.419572`, with full 64-code usage. The margin over standard VQ is small, so path diagnostics are still required. |
| Heston | `standard_vq_additive_ar` | Best prior eval cross-entropy, `2.652545`, and full 64-code usage. Hidden128 variants regressed token likelihood. |
| PDV4 | `conditional_standard_vq_additive_ar` | Best prior eval cross-entropy, `1.104750`. Hidden128 variants regressed token likelihood in this pass. |
| S&P500/VIX | `conditional_hidden128_conv_transformer_k3` | Best prior eval cross-entropy, `0.883431`, and better codebook perplexity than the public standard baseline. Public promotion still requires path metrics and reproducibility checks. |

## Black-Scholes

| Candidate | Status | Tokenizer config | Prior config | Runtime, seconds | Best epoch | Best eval CE | Best eval perplexity | Best eval accuracy | Token perplexity | Active codes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `standard_vq_additive_ar` | ok | `configs/experiments/black_scholes_causal_vq_tokenizer_codebook64_codebookdim16.yaml` | `configs/experiments/black_scholes_causal_token_prior_additive.yaml` | 589.6 | 70 | 2.421391 | 11.261929 | 0.178267 | 52.321541 | 64 |
| `hidden128_additive_ar` | ok | `configs/experiments/black_scholes_causal_vq_tokenizer_hidden128.yaml` | `configs/experiments/black_scholes_causal_token_prior_hidden128_additive.yaml` | 710.0 | 72 | 2.446540 | 11.548759 | 0.179450 | 54.278244 | 64 |
| `hidden128_conv_transformer_k3` | selected | `configs/experiments/black_scholes_causal_vq_tokenizer_hidden128.yaml` | `configs/experiments/black_scholes_causal_token_prior_hidden128_conv_transformer.yaml` | 777.5 | 51 | 2.419572 | 11.241549 | 0.181267 | 54.278244 | 64 |

## Heston

| Candidate | Status | Tokenizer config | Prior config | Runtime, seconds | Best epoch | Best eval CE | Best eval perplexity | Best eval accuracy | Token perplexity | Active codes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `standard_vq_additive_ar` | selected | `configs/experiments/heston_causal_vq_tokenizer.yaml` | `configs/experiments/heston_causal_token_prior_additive.yaml` | 611.8 | 67 | 2.652545 | 14.195220 | 0.167333 | 50.723011 | 64 |
| `hidden128_additive_ar` | ok | `configs/experiments/heston_causal_vq_tokenizer_hidden128.yaml` | `configs/experiments/heston_causal_token_prior_hidden128_additive.yaml` | 716.5 | 65 | 2.746341 | 15.590053 | 0.163433 | 58.029953 | 64 |
| `hidden128_conv_transformer_k3` | ok | `configs/experiments/heston_causal_vq_tokenizer_hidden128.yaml` | `configs/experiments/heston_causal_token_prior_hidden128_conv_transformer.yaml` | 749.2 | 50 | 2.748385 | 15.622853 | 0.164600 | 58.029953 | 64 |

## PDV4

| Candidate | Status | Tokenizer config | Prior config | Runtime, seconds | Best epoch | Best eval CE | Best eval perplexity | Best eval accuracy | Token perplexity | Active codes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `conditional_standard_vq_additive_ar` | selected | `configs/experiments/pdv_causal_vq_tokenizer_codebook64_codebookdim16.yaml` | `configs/experiments/pdv_causal_token_prior_additive_seed1.yaml` | 511.0 | 83 | 1.104750 | 3.019962 | 0.566133 | 51.275726 | 57 |
| `conditional_hidden128_additive_ar` | ok | `configs/experiments/pdv_causal_vq_tokenizer_hidden128.yaml` | `configs/experiments/pdv_causal_token_prior_hidden128_additive.yaml` | 726.9 | 93 | 1.254469 | 3.514406 | 0.521417 | 52.524532 | 64 |
| `conditional_hidden128_conv_transformer_k3` | ok | `configs/experiments/pdv_causal_vq_tokenizer_hidden128.yaml` | `configs/experiments/pdv_causal_token_prior_hidden128_conv_transformer.yaml` | 756.9 | 70 | 1.262250 | 3.540622 | 0.521167 | 52.524532 | 64 |

The first PDV4 attempt failed at token extraction before the runner path fix because the prior
config pointed to a stale prefixed tokenizer directory. After the runner fix, token extraction and
all PDV4 prior stages completed successfully.

## S&P500/VIX

| Candidate | Status | Tokenizer config | Prior config | Runtime, seconds | Best epoch | Best eval CE | Best eval perplexity | Best eval accuracy | Token perplexity | Active codes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `conditional_standard_vq_additive_ar` | ok | `configs/experiments/sp500_vix_causal_vq_tokenizer.yaml` | `configs/experiments/sp500_vix_causal_token_prior_additive.yaml` | 1207.4 | 100 | 0.914807 | 2.507927 | 0.647449 | 29.406109 | 63 |
| `conditional_hidden128_conv_transformer_k3` | selected | `configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml` | `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml` | 748.6 | 100 | 0.883431 | 2.426584 | 0.659917 | 51.726196 | 60 |

The public baseline reused existing ignored tokenizer and token-index artefacts under
`outputs/sp500_vix_discrete/`; its prior was trained in this run. The hidden128 conv-transformer
candidate trained tokenizer, token extraction, and prior stages under
`outputs/per_experiment_selection/sp500_vix/conditional_hidden128_conv_transformer_k3/`.

## Registry Metadata

Copy the following metadata into `trained_models/model_registry.yaml` only after the missing
no-leakage and reproduction checks pass. Do not copy weight paths into the registry and do not
commit generated outputs.

```yaml
per_experiment_selection:
  generated_outputs_root: outputs/per_experiment_selection
  weights_committed: false
  wandb:
    requested: true
    entity: tc_vae
    status: "initialisation timeout; successful fallback used --no-wandb"
    urls: []
  required_before_public_promotion:
    no_leakage_checks: pending
    notebook_reproduction: pending
    path_metrics: pending
  selected:
    black_scholes:
      discrete_candidate: hidden128_conv_transformer_k3
      tokenizer_config: configs/experiments/black_scholes_causal_vq_tokenizer_hidden128.yaml
      prior_config: configs/experiments/black_scholes_causal_token_prior_hidden128_conv_transformer.yaml
      condition_policy: unconditioned_discrete_prior
      selection_metric:
        best_eval_cross_entropy: 2.419572
        best_eval_perplexity: 11.241549
        best_eval_accuracy: 0.181267
        active_codes: 64
    heston:
      discrete_candidate: standard_vq_additive_ar
      tokenizer_config: configs/experiments/heston_causal_vq_tokenizer.yaml
      prior_config: configs/experiments/heston_causal_token_prior_additive.yaml
      condition_policy: unconditioned_discrete_prior
      selection_metric:
        best_eval_cross_entropy: 2.652545
        best_eval_perplexity: 14.195220
        best_eval_accuracy: 0.167333
        active_codes: 64
    pdv:
      discrete_candidate: conditional_standard_vq_additive_ar
      tokenizer_config: configs/experiments/pdv_causal_vq_tokenizer_codebook64_codebookdim16.yaml
      prior_config: configs/experiments/pdv_causal_token_prior_additive_seed1.yaml
      condition_policy: r2_volatility_feature_additive
      selection_metric:
        best_eval_cross_entropy: 1.104750
        best_eval_perplexity: 3.019962
        best_eval_accuracy: 0.566133
        active_codes: 57
    sp500_vix:
      discrete_candidate: conditional_hidden128_conv_transformer_k3
      tokenizer_config: configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml
      prior_config: configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml
      condition_policy: vix_additive
      selection_metric:
        best_eval_cross_entropy: 0.883431
        best_eval_perplexity: 2.426584
        best_eval_accuracy: 0.659917
        active_codes: 60
```

