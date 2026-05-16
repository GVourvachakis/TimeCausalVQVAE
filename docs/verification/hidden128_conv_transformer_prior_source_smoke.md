# Hidden128 Causal Conv-Transformer Prior Source Smoke

Status: source smoke completed. No tokenizer code was modified, no promoted public baseline config
was changed, and no non-smoke model was trained.

## Source Files Changed

- `src/time_causal_vae/token_prior/causal_transformer.py`
- `src/time_causal_vae/token_prior/config.py`
- `src/time_causal_vae/token_prior/__init__.py`
- `src/time_causal_vae/cli/train_token_prior.py`
- `src/time_causal_vae/evaluation/token_prior.py`
- `scripts/check_causal_conv_transformer_prior_no_leakage.py`
- `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml`

## Prior Type And Config

The new prior type is:

```yaml
model:
  prior_type: causal_conv_transformer
```

The smoke config is
`configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml`. It keeps the
hidden128 tokenizer path, hidden128 token-data path, scalar VIX condition with `condition_dim=1`,
additive conditioning, sequence length `60`, and codebook size `64`.

The convolutional front-end settings are:

```yaml
model:
  conv_num_layers: 2
  conv_kernel_size: 3
  conv_dilations: [1, 2]
  conv_dropout: 0.1
```

## Convolution Causality Convention

The `CausalConvTransformerPrior` uses the same single-stream BOS-shifted input convention as the
existing prior:

```text
input tokens:  [BOS, k_0, k_1, ..., k_{T-2}]
target tokens: [k_0, k_1, k_2, ..., k_{T-1}]
```

Token embeddings, position embeddings, and additive VIX condition embeddings are built before the
convolutional front-end. The front-end is a residual `Conv1d` stack over hidden states with explicit
left-only padding of `(kernel_size - 1) * dilation`, followed by the existing causal transformer
encoder. The convolution output keeps the same sequence length, and no right padding or
bidirectional operation is used.

## No-Leakage Result

Command:

```bash
poetry run python scripts/check_causal_conv_transformer_prior_no_leakage.py \
  --config configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml \
  --batch-size 8 \
  --cutoff 29 \
  --seed 99
```

Result:

```text
PASS causal conv-transformer prior no-leakage check
tokens=(8, 60)
conditions=(8, 1)
logits=(8, 60, 64)
cutoff=29
conv_num_layers=2
conv_kernel_size=3
conv_dilations=[1, 2]
conv_dropout=0.1
cross_entropy=4.78304291
accuracy=0.01875000
perplexity=119.46732330
max_prefix_diff=0.00000000e+00
```

The scalar VIX condition was held fixed while future tokens after the cutoff were perturbed. Prefix
logits through the cutoff were unchanged.

## Smoke Training Result

Command:

```bash
poetry run tcvae-train-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/hidden128_conv_transformer_smoke \
  --epochs 1 \
  --no-wandb
```

Result:

```text
epoch=1 train_ce=4.14245146 train_acc=0.06813187 eval_ce=3.51265518 eval_acc=0.17184235
training_complete: outputs/sp500_vix_discrete/token_prior/hidden128_conv_transformer_smoke/sp500_vix_causal_token_prior_hidden128_conv_transformer_seed99
runtime_seconds: 13.413
final_eval_cross_entropy: 3.51265518
final_eval_accuracy: 0.17184235
final_eval_perplexity: 34.14811119
best_epoch: 1
best_eval_cross_entropy: 3.51265518
best_model_dir: outputs/sp500_vix_discrete/token_prior/hidden128_conv_transformer_smoke/sp500_vix_causal_token_prior_hidden128_conv_transformer_seed99/best_model
```

The smoke run used CPU after PyTorch reported that the local CUDA driver was too old for the
installed CUDA runtime.

## Backward Compatibility

The default `prior_type` remains `single_code`. Existing single-code configs continue to build
`CausalTokenTransformerPrior` without a convolutional preprocessor. The convolution fields default
to inactive values and are only required when `prior_type` is `causal_conv_transformer`.
