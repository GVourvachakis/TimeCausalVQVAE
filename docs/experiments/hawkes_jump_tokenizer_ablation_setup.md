# Hawkes-Jump Tokenizer Ablation Setup

## Status

This note records the tokenizer-utilisation phase for the Hawkes-jump benchmark. The
phase targets the code-collapse blocker observed in the first diagnostic model
comparison, where the standard tokenizer used `6/64` active codes and the hidden128
tokenizer used `4/64` active codes before prior training.

No token priors are trained in this setup step. No prior architecture, model
registry entry, or simulator parameter is changed.

## Configs Created

The ablation grid keeps the Ogata Hawkes-jump simulator parameters fixed and varies
only the visible data representation, encoder and decoder width, and codebook size:

- `configs/experiments/hawkes_jump_causal_vq_tokenizer_logreturn_cb32.yaml`
- `configs/experiments/hawkes_jump_causal_vq_tokenizer_logreturn_cb64.yaml`
- `configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb32.yaml`
- `configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64.yaml`
- `configs/experiments/hawkes_jump_causal_vq_tokenizer_price_cb32.yaml`
- `configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_price_cb32.yaml`

All six configs use `n_samples: 1024`, `n_timesteps: 60`, `data_dim: 1`,
`condition_dim: 1`, and `simulation_scheme: ogata`. The log-return configs set
`data_output: log_return`; the price configs set `data_output: price`.

## Runner

The runner is `scripts/run_hawkes_tokenizer_ablation.py`. It invokes the existing
tokenizer training CLI and, for non-dry runs, follows with tokenizer evaluation and
frozen token-index extraction. It then writes aggregate JSON and CSV summaries under
the requested output directory.

Supported utilisation diagnostics are:

- active code count, active-code ratio, entropy, and codebook perplexity;
- tokenizer reconstruction metrics from the existing evaluator;
- oracle jump-window versus non-jump-window code usage;
- rare-code activation in jump and non-jump windows;
- token-change rate around jump windows;
- empirical L1 distance between jump-window and non-jump code distributions.

Oracle jump indicators are used only for post-hoc diagnostics. They are not added to
the model-visible data tensor and are not used to train token priors.

## Dry-Run Status

The requested smoke command was run successfully:

```bash
poetry run python scripts/run_hawkes_tokenizer_ablation.py \
  --configs \
    configs/experiments/hawkes_jump_causal_vq_tokenizer_logreturn_cb32.yaml \
    configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb32.yaml \
  --output-dir outputs/hawkes_jump_tokenizer_ablation_dry \
  --epochs 1 \
  --dry-run \
  --no-wandb
```

Both configs completed with `status: dry_run_complete`. The dry run built 1024 train
paths and 1024 eval paths with shape `[1024, 60, 1]` for each config. It did not train
models and did not write tokenizer checkpoints.

The command wrote:

- `outputs/hawkes_jump_tokenizer_ablation_dry/aggregate_summary.json`
- `outputs/hawkes_jump_tokenizer_ablation_dry/aggregate_summary.csv`

## Expected Non-Smoke Command

The first non-smoke utilisation pass should train tokenizers only, then evaluate code
usage and jump alignment:

```bash
poetry run python scripts/run_hawkes_tokenizer_ablation.py \
  --configs \
    configs/experiments/hawkes_jump_causal_vq_tokenizer_logreturn_cb32.yaml \
    configs/experiments/hawkes_jump_causal_vq_tokenizer_logreturn_cb64.yaml \
    configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb32.yaml \
    configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64.yaml \
    configs/experiments/hawkes_jump_causal_vq_tokenizer_price_cb32.yaml \
    configs/experiments/hawkes_jump_causal_vq_tokenizer_hidden128_price_cb32.yaml \
  --output-dir outputs/hawkes_jump_tokenizer_ablation \
  --epochs 10 \
  --no-wandb
```

This command deliberately excludes token-prior training. If W&B is working, the
training invocation can omit `--no-wandb` and pass the usual W&B environment settings.

## Success Criteria

The ablation phase should be considered successful only if the tokenizer improves the
input representation before any prior is trained:

- active codes are significantly above the first comparison baselines of `6/64` and
  `4/64`;
- no selected tokenizer collapses to fewer than 10 active codes;
- code usage differs between jump and non-jump windows;
- rare-code activation or token-change rates are measurably elevated near jumps;
- reconstruction remains acceptable relative to the price-level first comparison.

If these criteria fail, the next changes should remain tokenizer-local, for example
codebook dimension, commitment weight, k-means settings, dead-code handling, or a
supported two-channel price-return representation. Prior-family changes should remain
frozen until tokenizer utilisation is no longer the dominant blocker.
