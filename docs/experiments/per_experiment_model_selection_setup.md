# Per-Experiment Model-Selection Setup

## Scope

This setup adds candidate standard-VQ and hidden128 discrete configs plus a dry-run runner for
per-experiment model selection. It does not train non-smoke models, commit generated outputs, add
excluded variants, or change the continuous TC-VAE selected configs.

## Configs Created

Black-Scholes:

- `configs/experiments/black_scholes_causal_vq_tokenizer_codebook64_codebookdim16.yaml`;
- `configs/experiments/black_scholes_causal_token_prior_additive.yaml`;
- `configs/experiments/black_scholes_causal_vq_tokenizer_hidden128.yaml`;
- `configs/experiments/black_scholes_causal_token_prior_hidden128_additive.yaml`;
- `configs/experiments/black_scholes_causal_token_prior_hidden128_conv_transformer.yaml`.

Heston:

- `configs/experiments/heston_causal_vq_tokenizer.yaml`;
- `configs/experiments/heston_causal_token_prior_additive.yaml`;
- `configs/experiments/heston_causal_vq_tokenizer_hidden128.yaml`;
- `configs/experiments/heston_causal_token_prior_hidden128_additive.yaml`;
- `configs/experiments/heston_causal_token_prior_hidden128_conv_transformer.yaml`.

PDV4:

- existing standard conditional tokenizer:
  `configs/experiments/pdv_causal_vq_tokenizer_codebook64_codebookdim16.yaml`;
- existing standard additive conditional prior:
  `configs/experiments/pdv_causal_token_prior_additive_seed1.yaml`;
- `configs/experiments/pdv_causal_vq_tokenizer_hidden128.yaml`;
- `configs/experiments/pdv_causal_token_prior_hidden128_additive.yaml`;
- `configs/experiments/pdv_causal_token_prior_hidden128_conv_transformer.yaml`.

S&P500/VIX:

- existing public baseline tokenizer:
  `configs/experiments/sp500_vix_causal_vq_tokenizer.yaml`;
- existing public baseline prior:
  `configs/experiments/sp500_vix_causal_token_prior_additive.yaml`;
- `configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml`;
- `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml`.

The new hidden128 conv-transformer configs use `prior_type: causal_conv_transformer` with two
causal residual convolution layers, kernel size `3`, dilations `[1, 2]`, and the existing
single-code next-token objective. PDV4 and S&P500/VIX use additive scalar conditioning. The
Black-Scholes and Heston candidates remain unconditioned because their synthetic labels are
constant structural labels rather than experiment-specific market conditions.

## Excluded Variants

The setup deliberately does not add:

- RVQ q2;
- GroupedRVQ;
- MGVQ;
- VQ-Diffusion or diffusion priors;
- frequency tokenizers;
- native recurrent GRU priors;
- signature-conditioned configs.

## Runner

The runner is:

```bash
poetry run python scripts/run_per_experiment_model_selection.py --help
```

It validates required config paths, lists candidate tokenizer and prior pairs, writes aggregate
JSON and CSV command plans, and builds dry-run tokenizer and token-prior commands without training.
The aggregate files are written under ignored `outputs/` paths.

## Dry-Run Result

Executed command:

```bash
poetry run python scripts/run_per_experiment_model_selection.py \
  --experiments black_scholes heston pdv sp500_vix \
  --output-dir outputs/per_experiment_selection_dry \
  --dry-run \
  --epochs 1 \
  --n-sample 128 \
  --no-wandb
```

Result:

- status: passed;
- candidates planned: 11;
- output files:
  `outputs/per_experiment_selection_dry/selection_dry_run_plan.json` and
  `outputs/per_experiment_selection_dry/selection_dry_run_plan.csv`;
- smoke execution: not run;
- generated dry-run outputs remain ignored and must not be committed.

## Next Non-Smoke Commands

For each candidate, run the three-stage local workflow from the aggregate CSV or JSON after
choosing an experiment and candidate:

1. Train the tokenizer intentionally, removing `--dry-run` from the planned tokenizer command.
2. Extract frozen token indices with the planned `scripts/extract_token_indices.py` command.
3. Train the token prior intentionally, removing `--dry-run` from the planned prior command.

Use the W&B profile from the plan:

```bash
--wandb --wandb-project time-causal-vq-tokenizer --wandb-entity tc_vae
--wandb --wandb-project time-causal-token-prior --wandb-entity tc_vae
```

If W&B initialisation times out, rerun the same command with `--no-wandb` and record that fallback
in the local candidate summary.

A representative non-smoke sequence for one candidate is:

```bash
poetry run tcvae-train-tokenizer \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml \
  --output-dir outputs/per_experiment_selection/sp500_vix/conditional_hidden128_conv_transformer_k3/tokenizer \
  --base-data-dir data/processed \
  --wandb \
  --wandb-project time-causal-vq-tokenizer \
  --wandb-entity tc_vae

poetry run python scripts/extract_token_indices.py \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml \
  --tokenizer-dir outputs/per_experiment_selection/sp500_vix/conditional_hidden128_conv_transformer_k3/tokenizer/sp500_vix_causal_vq_tokenizer_hidden128_seed0 \
  --output-dir outputs/per_experiment_selection/sp500_vix/conditional_hidden128_conv_transformer_k3/tokens \
  --base-data-dir data/processed

poetry run tcvae-train-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml \
  --output-dir outputs/per_experiment_selection/sp500_vix/conditional_hidden128_conv_transformer_k3/prior \
  --wandb \
  --wandb-project time-causal-token-prior \
  --wandb-entity tc_vae
```

After local training and evaluation, keep all generated files under `outputs/`, compare all visible
metrics from the model-selection plan, and update `trained_models/model_registry.yaml` with
selected metadata only. Do not commit weights or generated evaluation artefacts.
