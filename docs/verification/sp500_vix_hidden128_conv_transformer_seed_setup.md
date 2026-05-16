# Hidden128 Conv-Transformer Seed Robustness Setup

Status: setup completed. No non-smoke models were trained, no tokenizer code was changed, and no
new architecture family or objective was added.

## Seed Configs

The selected hidden128 causal conv-transformer k3 prior now has two seed robustness configs:

- `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer_seed1.yaml`
- `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer_seed2.yaml`

Both configs are based on
`configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml` and change only
the experiment name and experiment seed. They keep:

- hidden128 tokenizer path:
  `outputs/sp500_vix_discrete/vq_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_hidden128_seed99`;
- hidden128 token-data path:
  `outputs/sp500_vix_discrete/token_prior/sp500_vix_causal_vq_tokenizer_hidden128_tokens`;
- `prior_type: causal_conv_transformer`;
- convolution kernel size `3`;
- convolution dilations `[1, 2]`;
- scalar VIX condition with `condition_dim=1`;
- additive condition injection.

## Runner Support

`scripts/run_sp500_vix_token_prior_candidate_ablation.py` supports the seed robustness workflow:

- validates controlled S&P500/VIX token-prior configs, including `causal_conv_transformer`;
- trains each config through `time_causal_vae.cli.train_token_prior`;
- evaluates each best checkpoint through `time_causal_vae.cli.evaluate_token_prior`;
- can run paper-style diagnostics through `scripts/evaluate_sp500_vix_paper_style.py`;
- parses unrestricted sampling as `--top-k none`;
- writes aggregate JSON and CSV summaries;
- uses the W&B-compatible subprocess environment:

```text
WANDB_MODE=unset
MPLBACKEND=Agg
WANDB_DISABLE_SERVICE=true
WANDB_START_METHOD=thread
```

If a W&B-enabled training subprocess fails, the runner retries the same training command with
`--no-wandb` and records `wandb_fallback_used` and `wandb_failure` in the aggregate outputs.

## Dry-Run Command

Command:

```bash
MPLBACKEND=Agg poetry run python scripts/run_sp500_vix_token_prior_candidate_ablation.py \
  --configs \
    configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer_seed1.yaml \
    configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer_seed2.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/hidden128_conv_transformer_seed_ablation_dry \
  --base-data-dir data/processed \
  --epochs 1 \
  --n-sample 128 \
  --seed 99 \
  --temperature 1.0 \
  --top-k none \
  --dry-run \
  --no-wandb
```

Result: passed for both seed configs.

The dry-run verified:

- train-token shape: `(2457, 60)`;
- train-condition shape: `(2457, 1)`;
- eval-token shape: `(2457, 60)`;
- eval-condition shape: `(2457, 1)`;
- parameter count: `653504`;
- device resolution: `cpu`;
- epoch override: `1`;
- W&B disabled for the dry-run;
- no training was started by the training CLI.

Aggregate dry-run outputs were written to:

- `outputs/sp500_vix_discrete/token_prior/hidden128_conv_transformer_seed_ablation_dry/token_prior_candidate_ablation_summary.json`
- `outputs/sp500_vix_discrete/token_prior/hidden128_conv_transformer_seed_ablation_dry/token_prior_candidate_ablation_summary.csv`

Both rows have `train_status=passed`, `best_eval_status=skipped_dry_run`,
`paper_style_status=skipped_dry_run`, `temperature=1.0`, and `top_k=none`.

## Selected Evaluation Settings

Primary evaluation setting:

```text
temperature=1.0
top_k=none
```

This is the selected paper-style profile setting for the hidden128 conv-transformer k3 prior.

Secondary optional setting:

```text
temperature=1.0
top_k=20
```

This setting is reserved for comparisons where volatility W1, transition L1, or squared-return
autocorrelation is prioritised over the primary profile.

## W&B Profile

Full seed robustness runs should use the established execution profile:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread
```

The runner already applies the equivalent environment to subprocesses. When running online W&B
seed robustness, use `--wandb` with the project `time-causal-token-prior` and entity `tc_vae`.
If W&B initialisation fails, the runner will retry that config with `--no-wandb` and continue the
aggregate.
