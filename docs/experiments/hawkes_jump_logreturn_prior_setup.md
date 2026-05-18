# Hawkes-Jump Log-Return Prior Setup

## Status

This setup prepares token-prior training and evaluation for the Hawkes-jump
log-return tokenizers selected by the tokenizer-utilisation ablation. It does not
train non-smoke priors, change prior architecture, update the registry, or merge
anything to `main`.

The tokenizer artifacts from the non-smoke ablation were confirmed at:

- `outputs/hawkes_jump_tokenizer_ablation/tokenizers/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64_seed0`
- `outputs/hawkes_jump_tokenizer_ablation/tokens/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64`
- `outputs/hawkes_jump_tokenizer_ablation/tokenizers/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb32_seed0`
- `outputs/hawkes_jump_tokenizer_ablation/tokens/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb32`

## Prior Configs

Four token-prior configs were added:

- `configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_additive.yaml`
- `configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_conv_transformer.yaml`
- `configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb32_additive.yaml`
- `configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb32_conv_transformer.yaml`

The additive configs use the existing single-code additive conditioning prior. The
conv configs use the existing causal conv-transformer prior with kernel size `3`,
`conv_num_layers: 2`, and dilations `[1, 2]`. All configs preserve
`condition_dim: 1` and the scalar constant labels from the extracted tokenizer token
datasets.

The cb64 configs use `codebook_size: 64`, `bos_token_id: 64`, and sequence length
`60`. The cb32 configs use `codebook_size: 32`, `bos_token_id: 32`, and sequence
length `60`.

Each config includes `data_output: log_return` in the `data` section as metadata for
the Hawkes-specific evaluator.

## Log-Return Conversion

Log-return tokenizers decode paths in return space, not price space. The helper
`log_returns_to_normalized_prices(log_returns, initial_price=1.0)` converts decoded
log returns to normalised positive price paths before market and jump diagnostics.

The conversion requires input shape `[batch, time, 1]`, checks finite inputs, takes a
cumulative sum along time, and returns:

```text
price_t = initial_price * exp(cumulative_log_return_t)
```

Price-output behaviour is unchanged. The conversion is used only when the config
metadata says `data_output: log_return`.

## Evaluator

The new evaluator is `scripts/evaluate_hawkes_jump_token_prior.py`. It supports:

- loading a prior config, token-prior checkpoint, and tokenizer checkpoint;
- paired conditional sampling with the scalar labels stored in the token artifact;
- token sequence sampling with `--temperature` and `--top-k`;
- tokenizer decoding;
- log-return-to-price conversion for log-return configs;
- smooth path metrics including MMD, SWD, terminal W1, and volatility W1;
- market-style summaries and tail exceedance using real evaluation paths as reference;
- jump diagnostics with a common robust jump threshold fitted on real Ogata
  evaluation paths;
- token diagnostics including active codes, perplexity, marginal code L1, transition
  matrix L1, and run-length distance;
- `evaluation_summary.json`, `evaluation_summary.md`, and `evaluation_batch.pt`.

## Smoke Status

A one-epoch additive prior smoke was run for the primary cb64 log-return tokenizer:

```bash
poetry run tcvae-train-token-prior \
  --config configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_additive.yaml \
  --output-dir outputs/hawkes_jump_prior_smoke/hidden128_logreturn_cb64_additive \
  --epochs 1 \
  --no-wandb
```

The run completed successfully and wrote:

```text
outputs/hawkes_jump_prior_smoke/hidden128_logreturn_cb64_additive/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_additive_seed0/best_model
```

The one-epoch smoke had eval cross-entropy `4.01749572`, eval accuracy
`0.05699870`, and eval perplexity `55.56616116`.

The Hawkes evaluator smoke was then run with 128 samples, seed `99`, temperature
`1.0`, and unrestricted top-k:

```bash
poetry run python scripts/evaluate_hawkes_jump_token_prior.py \
  --config configs/experiments/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_additive.yaml \
  --prior-dir outputs/hawkes_jump_prior_smoke/hidden128_logreturn_cb64_additive/hawkes_jump_causal_token_prior_hidden128_logreturn_cb64_additive_seed0/best_model \
  --tokenizer-dir outputs/hawkes_jump_tokenizer_ablation/tokenizers/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64_seed0 \
  --output-dir outputs/hawkes_jump_prior_smoke/hidden128_logreturn_cb64_additive/evaluation \
  --n-sample 128 \
  --seed 99 \
  --temperature 1.0 \
  --top-k none
```

The evaluator completed successfully. Smoke metrics were:

- sampled active codes: `64`;
- sampled token perplexity: `45.32234192`;
- MMD: `0.31624496`;
- SWD: `0.03800552`;
- terminal W1: `0.04990859`;
- volatility W1: `0.00194993`;
- detected jump-count W1: `0.20312500`;
- detected jump-size W1: `0.03009466`.

These smoke numbers are not a model-selection result because the prior was trained
for one epoch only. They confirm that the prepared prior configs and evaluator run
end to end.

## Non-Smoke Evaluation Metrics

The non-smoke prior comparison should report:

- smooth path metrics: MMD, SWD, terminal W1, volatility W1, drawdown W1, return
  autocorrelation, and squared-return autocorrelation;
- jump metrics: detected jump count distribution, detected jump-size distribution,
  inter-arrival distribution, clustering, tail exceedance, VaR, and ES;
- token metrics: active codes, sampled perplexity, marginal code L1, transition
  matrix L1, run-length distance, and rare-event transition behaviour;
- representation check: decoded log returns converted to normalised prices before
  any price-path metric is interpreted.

The next non-smoke run should train and compare at least the cb64 additive prior and
cb64 causal conv-transformer prior. The cb32 versions remain useful as backup
experiments if cb64 sampling is too diffuse or rare-code dynamics dominate the jump
metrics.
