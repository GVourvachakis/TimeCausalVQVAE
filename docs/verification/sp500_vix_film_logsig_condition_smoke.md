# S&P500/VIX FiLM/AdaLN Log-Signature Condition Smoke

## Purpose

This note records the first smoke run for FiLM/AdaLN-style conditioning of the
single-code causal token prior on scalar VIX plus standardised
`logsig_l3_ctx20` features.

No tokenizer code was modified. No cross-attention, Gumbel-Softmax,
signature-kernel loss, or model-selection run was added.

## Conditioning Mode

The repository already supports the FiLM/AdaLN-style mode as:

```yaml
model:
  condition_injection: adaln_lite
  condition_dim: 386
  adaln_hidden_dim: 128
```

The naming convention used for this ablation is therefore `adaln_lite`, not a
new `film` alias.

Implementation status:

- `src/time_causal_vae/token_prior/config.py` already validates
  `condition_injection: adaln_lite` and `adaln_hidden_dim`;
- `src/time_causal_vae/token_prior/causal_transformer.py` already routes
  single-code priors through `AdaLNCausalTransformerBlock`;
- the AdaLN-lite block emits per-token attention and MLP scale/shift tensors
  from the full condition vector;
- the final modulation linear layer is zero-initialised, so the path starts as
  a stable pre-norm transformer;
- causal masking is the same additive causal attention mask used by the
  additive prior.

No token-prior source changes were required for this smoke. The new experiment
config is:

```text
configs/experiments/sp500_vix_causal_token_prior_film_logsig_l3_ctx20_std.yaml
```

## Config

The FiLM/AdaLN config is based on the additive standardised log-signature
config and changes only:

- experiment name:
  `sp500_vix_causal_token_prior_film_logsig_l3_ctx20_std`;
- condition injection mode: `additive` -> `adaln_lite`;
- AdaLN modulation hidden dimension: `adaln_hidden_dim: 128`.

The feature directory remains:

```text
outputs/sp500_vix_discrete/signature_features/logsig_l3_ctx20_std
```

The condition vector is scalar VIX plus 385 standardised log-signature
features, giving `condition_dim=386`.

The VIX-only additive config and the additive standardised log-signature config
remain unchanged.

## Causality Check

Command:

```bash
poetry run python scripts/check_conditional_token_prior_no_leakage.py \
  --config configs/experiments/sp500_vix_causal_token_prior_film_logsig_l3_ctx20_std.yaml \
  --batch-size 8 \
  --cutoff 29 \
  --seed 99 \
  --device cpu
```

Result:

| Field | Value |
| --- | --- |
| Status | PASS |
| Token shape | `(8, 60)` |
| Condition shape | `(8, 386)` |
| Condition injection | `adaln_lite` |
| Cutoff | `29` |
| Logits shape | `(8, 60, 64)` |
| Sample shape | `(8, 60)` |
| Max prefix difference | `0.00000000e+00` |

The check perturbed future tokens after the cutoff while holding the global
condition vector fixed. Prefix logits through the cutoff were unchanged. The
condition was not perturbed because it represents pre-window historical context
and is global for the generated window.

## Smoke Training

Command:

```bash
poetry run tcvae-train-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_film_logsig_l3_ctx20_std.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/film_logsig_l3_ctx20_std_smoke \
  --epochs 1 \
  --no-wandb
```

Output directory:

```text
outputs/sp500_vix_discrete/token_prior/film_logsig_l3_ctx20_std_smoke/sp500_vix_causal_token_prior_film_logsig_l3_ctx20_std_seed0
```

Training metrics:

| Metric | Value |
| --- | ---: |
| Runtime seconds | 16.940549 |
| Best epoch | 1 |
| Train CE | 3.80078071 |
| Train accuracy | 0.12623118 |
| Train perplexity | 48.37856678 |
| Eval CE | 3.06474395 |
| Eval accuracy | 0.30199430 |
| Eval perplexity | 22.06419393 |

W&B was disabled for the smoke run.

## Smoke Evaluation

Command:

```bash
poetry run tcvae-evaluate-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_film_logsig_l3_ctx20_std.yaml \
  --prior-dir outputs/sp500_vix_discrete/token_prior/film_logsig_l3_ctx20_std_smoke/sp500_vix_causal_token_prior_film_logsig_l3_ctx20_std_seed0 \
  --tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --output-dir outputs/sp500_vix_discrete/token_prior/film_logsig_l3_ctx20_std_smoke/evaluation \
  --base-data-dir data/processed \
  --n-sample 128 \
  --seed 99
```

Tensor shapes:

| Tensor | Shape |
| --- | --- |
| Conditions | `[128, 386]` |
| Decoder conditions | `[128, 1]` |
| Sampled tokens | `[128, 60]` |
| Decoded paths | `[128, 60, 1]` |
| Real paths | `[128, 60, 1]` |

Decoded smoke metrics:

| Metric | Value |
| --- | ---: |
| MMD | 1.12596810 |
| SWD | 0.05756735 |
| Terminal-return W1 | 0.17777662 |
| Volatility W1 | 0.09996173 |
| Sampled token perplexity | 50.19320679 |
| Active sampled codes | 64 / 64 |
| Marginal code L1 | 0.60624993 |
| Transition matrix L1 | 1.54279733 |
| Run-length distance | 0.99636078 |

These decoded metrics are not quality evidence because the prior was trained
for only one epoch. They only confirm that the `adaln_lite` condition path can
train, sample, concatenate standardised log-signature features, and decode
without shape or finite-value failures.

## Decision

The FiLM/AdaLN-style condition path is smoke-validated for the
standardised `logsig_l3_ctx20` condition vector.

The next non-smoke comparison should keep the VIX-only additive baseline
unchanged and evaluate whether `adaln_lite` improves the tail-risk or
balanced-market profiles without severe MMD/SWD regression.

## Check Status

Completed checks:

- `poetry run ruff format src scripts docs configs`;
- `poetry run ruff check src scripts docs configs --fix`;
- `poetry check`.

`poetry run mypy src/time_causal_vae` still fails on two existing
`latent_geometry.py` return-type issues that are outside the token-prior
FiLM/AdaLN smoke scope:

```text
src/time_causal_vae/evaluation/latent_geometry.py:1139: error: Returning Any from function declared to return "ndarray[Any, Any] | None"  [no-any-return]
src/time_causal_vae/evaluation/latent_geometry.py:1161: error: Returning Any from function declared to return "ndarray[Any, Any] | None"  [no-any-return]
```
