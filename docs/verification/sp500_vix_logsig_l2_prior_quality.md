# S&P500/VIX Log-Signature L2 Prior Quality

## Purpose

This note records the first non-smoke S&P500/VIX additive token-prior ablation
with depth-2 log-signature conditioning features. The model architecture was not
changed: the prior used the existing additive condition embedding, and the
frozen tokenizer decoded with the original scalar VIX condition.

## W&B Status

Requested W&B run:

```bash
poetry run tcvae-train-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l2.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/additive_logsig_l2 \
  --wandb \
  --wandb-project time-causal-token-prior \
  --wandb-entity tc_vae \
  --wandb-run-name sp500_vix_additive_logsig_l2_seed0
```

W&B did not start in this local sandbox. The run failed before training with:

```text
failed to start server, exiting error="listen tcp 127.0.0.1:0: socket: operation not permitted"
```

The full run was therefore repeated with `--no-wandb`. No W&B URL is available.

## Log-Signature Training

Command:

```bash
poetry run tcvae-train-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l2.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/additive_logsig_l2 \
  --no-wandb
```

Output directory:
`outputs/sp500_vix_discrete/token_prior/additive_logsig_l2/sp500_vix_causal_token_prior_additive_logsig_l2_seed0`.

| Metric | Value |
| --- | ---: |
| Runtime | 1241.232 s |
| Device | CPU |
| W&B enabled | false |
| Best epoch | 100 |
| Train cross-entropy | 0.94003383 |
| Train accuracy | 0.64101207 |
| Eval cross-entropy | 0.90663672 |
| Eval accuracy | 0.65104463 |
| Eval perplexity | 2.48757760 |

Condition shapes:

- token-prior conditions: `(2457, 56)`;
- scalar VIX decoder conditions: `(2457, 1)`.

## Best-Checkpoint Evaluation

Command:

```bash
poetry run tcvae-evaluate-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l2.yaml \
  --prior-dir outputs/sp500_vix_discrete/token_prior/additive_logsig_l2/sp500_vix_causal_token_prior_additive_logsig_l2_seed0/best_model \
  --tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --output-dir outputs/sp500_vix_discrete/token_prior/additive_logsig_l2/evaluation_best \
  --base-data-dir data/processed \
  --n-sample 1000 \
  --seed 99 \
  --temperature 0.8 \
  --top-k 40
```

Sampling convention:

- `n_sample`: 1000;
- temperature: `0.8`;
- top-k: `40`;
- prior conditions: `(1000, 56)`;
- decoder conditions: `(1000, 1)`;
- decoded paths: `(1000, 60, 1)`.

| Metric | Value |
| --- | ---: |
| MMD | 0.31721902 |
| SWD | 0.00859528 |
| Terminal-return Wasserstein | 0.00708540 |
| Volatility Wasserstein | 0.00134660 |
| Sampled active codes | 60 / 64 |
| Sampled token perplexity | 28.70219612 |
| Sampled token entropy | 3.35697365 |
| Marginal code L1 | 0.25096664 |
| Transition matrix L1 | 0.31828666 |
| Run-length distance | 0.92794842 |

## VIX-Bucket Diagnostics

These buckets use the scalar VIX decoder condition, not the full 56-dimensional
prior condition vector.

| Bucket | Samples | MMD | SWD | Volatility W1 | Terminal W1 | Active codes | Token perplexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| very_low | 200 | 0.33442381 | 0.00940963 | 0.00109261 | 0.00743208 | 58 | 29.71372795 |
| low | 200 | 0.42025021 | 0.01069876 | 0.00111946 | 0.00660674 | 57 | 26.91120338 |
| mid | 200 | 0.38073134 | 0.00852215 | 0.00122573 | 0.00941124 | 54 | 25.32784271 |
| high | 200 | 0.29021367 | 0.00944742 | 0.00133609 | 0.00554842 | 54 | 27.66407585 |
| very_high | 200 | 0.39032674 | 0.01386700 | 0.00203369 | 0.01245451 | 58 | 32.64564896 |

## Paper-Style Diagnostics

Command:

```bash
poetry run python scripts/evaluate_sp500_vix_paper_style.py \
  --discrete-config configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l2.yaml \
  --discrete-prior-dir outputs/sp500_vix_discrete/token_prior/additive_logsig_l2/sp500_vix_causal_token_prior_additive_logsig_l2_seed0/best_model \
  --discrete-tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --continuous-config configs/experiments/sp500_vix_beta_cvae.yaml \
  --continuous-model-dir outputs/sp500_vix_continuous/final_model_unavailable \
  --output-dir outputs/sp500_vix_discrete/paper_style_logsig_l2 \
  --base-data-dir data/processed \
  --n-sample 1000 \
  --seed 99 \
  --temperature 0.8 \
  --top-k 40
```

The continuous checkpoint was unavailable, so the continuous comparison was
recorded as skipped:

```text
continuous checkpoint not found: outputs/sp500_vix_continuous/final_model_unavailable
```

Paper-style discrete metrics:

| Metric | Value |
| --- | ---: |
| MMD | 0.31115741 |
| SWD | 0.01007967 |
| One-step return Wasserstein | 0.00127050 |
| Terminal-return Wasserstein | 0.00699868 |
| Volatility Wasserstein | 0.00134263 |
| Maximum-drawdown Wasserstein | 0.01103095 |
| Return autocorrelation L1 | 0.05549955 |
| Squared-return autocorrelation L1 | 0.04449294 |
| Flattened squared-return autocorrelation L1 | 0.14269510 |
| Terminal-return mean error | 0.00463955 |
| Volatility mean error | 0.00068269 |

Tail exceedance rates against real thresholds:

| Threshold event | Count | Fraction |
| --- | ---: | ---: |
| Below real q001 | 83 | 0.00140678 |
| Below real q01 | 313 | 0.00530508 |
| Above real q99 | 514 | 0.00871186 |
| Above real q999 | 154 | 0.00261017 |

Paper-style VIX buckets:

| Bucket | VIX min | VIX max | MMD | SWD | Volatility W1 | Terminal W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| very_low | 0.11053332 | 0.13907364 | 0.56091678 | 0.01790283 | 0.00168521 | 0.00661597 |
| low | 0.13907364 | 0.15672995 | 0.43397298 | 0.00996641 | 0.00102533 | 0.01142675 |
| mid | 0.15709275 | 0.17728867 | 0.30686641 | 0.01137645 | 0.00177846 | 0.01700475 |
| high | 0.17740960 | 0.20909421 | 0.41048455 | 0.01168733 | 0.00131228 | 0.01564763 |
| very_high | 0.20933606 | 0.49268353 | 0.41132587 | 0.01550077 | 0.00172759 | 0.01084654 |

## VIX-Only Same-Setting Baseline

No trained VIX-only prior checkpoint was present locally, so the standard
VIX-only additive prior was trained with the existing promoted config and W&B
disabled:

```bash
poetry run tcvae-train-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_additive.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/additive_vix_only \
  --no-wandb
```

Training summary:

| Metric | VIX-only | VIX + logsig L2 |
| --- | ---: | ---: |
| Runtime | 1264.156 s | 1241.232 s |
| Best epoch | 100 | 100 |
| Eval cross-entropy | 0.91480655 | 0.90663672 |
| Eval accuracy | 0.64744947 | 0.65104463 |
| Eval perplexity | 2.50792742 | 2.48757760 |

The VIX-only model was then evaluated with the same sampling setting:

```bash
poetry run tcvae-evaluate-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_additive.yaml \
  --prior-dir outputs/sp500_vix_discrete/token_prior/additive_vix_only/sp500_vix_causal_token_prior_additive_seed0/best_model \
  --tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --output-dir outputs/sp500_vix_discrete/token_prior/additive_vix_only/evaluation_best_temp08_topk40 \
  --base-data-dir data/processed \
  --n-sample 1000 \
  --seed 99 \
  --temperature 0.8 \
  --top-k 40
```

Best-checkpoint decoded comparison:

| Metric | VIX-only | VIX + logsig L2 | Direction |
| --- | ---: | ---: | --- |
| MMD | 0.28880176 | 0.31721902 | VIX-only lower |
| SWD | 0.00889640 | 0.00859528 | Logsig lower |
| Terminal-return W1 | 0.01076925 | 0.00708540 | Logsig lower |
| Volatility W1 | 0.00132492 | 0.00134660 | VIX-only lower |
| Sampled active codes | 62 | 60 | VIX-only broader |
| Sampled token perplexity | 30.11016655 | 28.70219612 | VIX-only broader |
| Marginal code L1 | 0.25939998 | 0.25096664 | Logsig lower |
| Transition matrix L1 | 0.34322238 | 0.31828666 | Logsig lower |
| Run-length distance | 0.98253310 | 0.92794842 | Logsig lower |

The comparison is therefore mixed. Log-signature conditioning improves the token
training objective, terminal-return Wasserstein, SWD, marginal-code distance,
transition distance, and run-length distance. The VIX-only baseline is better on
decoded MMD, slightly better on volatility Wasserstein, and keeps broader sampled
code support.

## Decision

Decision: needs depth/context ablation.

The first log-signature ablation is promising but not yet a clean replacement
for the VIX-only prior. It improves several sequence-level and token-level
diagnostics, but it does not dominate the promoted scalar baseline on decoded
path metrics. The next ablation should vary signature depth, context length, and
possibly feature normalisation before treating log-signature conditioning as a
promoted default.
