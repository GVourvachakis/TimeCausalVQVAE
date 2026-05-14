# S&P500/VIX Log-Signature Condition Smoke

## Purpose

This note records the smoke test for concatenating optional precomputed
log-signature features with the existing scalar VIX labels in the additive
causal token prior. The tokenizer and prior architectures were not changed.

## Configuration

- Config:
  `configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l2.yaml`.
- Base config:
  `configs/experiments/sp500_vix_causal_token_prior_additive.yaml`.
- Condition feature directory:
  `outputs/sp500_vix_discrete/signature_features/logsig_l2_ctx20`.
- Train feature file: `train_signature_features.npz`.
- Eval feature file: `eval_signature_features.npz`.
- Feature key: `features`.
- Base scalar label dimension: `1`.
- Signature feature dimension: `55`.
- Total token-prior condition dimension: `56`.
- Condition injection: additive.
- W&B: disabled with `--no-wandb`.

The default VIX-only config remains behaviourally unchanged: its
`condition_feature_dir` is `null`, `model.condition_dim` remains `1`, and a dry
run loaded train/eval condition tensors with shape `(2457, 1)`.

## Alignment Checks

The optional condition-feature loader validates:

- feature row count against the token-artifact label row count;
- finite feature values;
- non-empty 2-D feature matrices;
- optional `sample_indices` length and uniqueness;
- optional NPZ `labels` against token-artifact `labels`.

For this smoke run, train and eval token artefacts each contained `2457` rows.
The depth-2 signature features also contained `2457` rows, giving concatenated
condition tensors of shape `(2457, 56)`.

During evaluation, the token prior sampled with the full `(128, 56)`
condition tensor. The frozen tokenizer decoder was trained with scalar VIX
conditioning, so decoded-path evaluation used the original scalar labels as
decoder conditions with shape `(128, 1)`.

## Commands

Dry-run loader check for the log-signature config:

```bash
poetry run tcvae-train-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l2.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/additive_logsig_l2_dry_run \
  --epochs 1 \
  --no-wandb \
  --dry-run
```

Smoke training:

```bash
poetry run tcvae-train-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l2.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/additive_logsig_l2_smoke \
  --epochs 1 \
  --no-wandb
```

Smoke evaluation:

```bash
poetry run tcvae-evaluate-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l2.yaml \
  --prior-dir outputs/sp500_vix_discrete/token_prior/additive_logsig_l2_smoke/sp500_vix_causal_token_prior_additive_logsig_l2_seed0 \
  --tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --output-dir outputs/sp500_vix_discrete/token_prior/additive_logsig_l2_smoke/evaluation \
  --base-data-dir data/processed \
  --n-sample 128 \
  --seed 99
```

VIX-only dry-run regression check:

```bash
poetry run tcvae-train-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_additive.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/additive_vix_only_dry_run \
  --epochs 1 \
  --no-wandb \
  --dry-run
```

## Smoke Training Metrics

The one-epoch smoke run completed on CPU in `13.138` seconds.

| Metric | Value |
| --- | ---: |
| Train cross-entropy | 4.12432081 |
| Train accuracy | 0.08386922 |
| Train perplexity | 67.46810927 |
| Eval cross-entropy | 3.32675386 |
| Eval accuracy | 0.22748610 |
| Eval perplexity | 29.09005935 |
| Best epoch | 1 |

The training run wrote:

- `token_prior.pt`;
- `token_prior_config.json`;
- `training_config.json`;
- `runtime_summary.json`;
- `best_checkpoint_summary.json`;
- `best_model/`.

These artefacts are under ignored `outputs/` and are not committed.

## Decoded Smoke Metrics

The evaluation run sampled `128` paths with shape `(128, 60, 1)`.

| Metric | Value |
| --- | ---: |
| Sampled active code count | 64 |
| Sampled active code ratio | 1.00000000 |
| Sampled token perplexity | 51.60400009 |
| Sampled token entropy | 3.94359922 |
| MMD | 1.15142870 |
| SWD | 0.05935442 |
| Terminal return mean error | 0.06226841 |
| Terminal return Wasserstein | 0.09696801 |
| Volatility mean error | 0.10334975 |
| Volatility Wasserstein | 0.10334975 |
| Marginal code L1 | 0.58463544 |
| Transition matrix L1 | 1.55821645 |
| Run-length distance | 0.99636078 |
| Real token perplexity | 30.50984955 |
| Sampled token perplexity | 51.60400009 |

Saved evaluation tensor shapes:

- sampled tokens: `(128, 60)`;
- decoded paths: `(128, 60, 1)`;
- real paths: `(128, 60, 1)`;
- prior conditions: `(128, 56)`;
- decoder conditions: `(128, 1)`;
- quantized embeddings: `(128, 60, 64)`.

The evaluation wrote:

- `token_prior_summary.json`;
- `sampled_tokens.pt`;
- `decoded_paths.pt`;
- `sampled_code_usage.png`;
- `decoded_path_examples.png`;
- `real_vs_sampled_code_usage.png`;
- `transition_matrix_real.png`;
- `transition_matrix_sampled.png`.

These artefacts are under ignored `outputs/` and are not committed.

## Decision

The smoke path is accepted. Optional depth-2 log-signature features can be
concatenated with scalar VIX labels for additive token-prior training and paired
evaluation. This remains a conditioning-input change only: there is no
cross-attention, no tokenizer change, no prior architecture change, and no
non-smoke model training in this step.
