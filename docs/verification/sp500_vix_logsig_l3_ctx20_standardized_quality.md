# S&P500/VIX Standardised `logsig_l3_ctx20` Quality

## Purpose

This note records the optional standardised log-signature conditioning ablation
for the S&P500/VIX additive causal token prior. The run did not change the
tokenizer, prior architecture, condition injection mechanism, objective, or
sampling policy.

## Feature Standardisation

Feature extraction command:

```bash
poetry run python scripts/extract_signature_features.py \
  --dataset sp500_vix \
  --base-data-dir data/processed \
  --output-dir outputs/sp500_vix_discrete/signature_features/logsig_l3_ctx20_std \
  --depth 3 \
  --context-length 20 \
  --use-lead-lag \
  --include-time \
  --include-vix \
  --standardize \
  --seed 99
```

The extractor fitted mean and standard deviation on the train feature matrix
only, guarded standard deviations with `epsilon=1e-8`, and applied the same
transform to train and eval feature files. Scalar VIX labels were not
standardised by this extractor.

Feature outputs:

- feature directory:
  `outputs/sp500_vix_discrete/signature_features/logsig_l3_ctx20_std`;
- feature shape: `[2457, 385]`;
- total prior condition dimension: `386`, equal to scalar VIX plus 385
  log-signature features;
- stats file: `signature_feature_standardization.npz`;
- finite feature check: `true`;
- extraction runtime: `1.30` seconds.

Standardisation summary:

| Quantity | Value |
| --- | ---: |
| Mean shape | 385 |
| Std shape | 385 |
| Maximum absolute fitted mean | 1.00000000 |
| Minimum guarded fitted std | 0.00000049 |
| Maximum guarded fitted std | 1.00000000 |
| Maximum absolute post-transform mean | 0.00000000 |
| Minimum post-transform std | 0.00000000 |
| Maximum post-transform std | 1.00000000 |

The zero post-transform minimum standard deviation indicates at least one
constant or effectively constant feature column after the guarded transform.
This is acceptable for the smoke and ablation because the values are finite and
the condition dimension remains aligned.

## Config

Config created:

`configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_std.yaml`

The config preserves the additive causal token-prior architecture and changes
only the condition feature directory and experiment name relative to the raw
`logsig_l3_ctx20` setting.

## Smoke

Smoke command:

```bash
poetry run tcvae-train-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_std.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/additive_logsig_l3_ctx20_std_smoke \
  --epochs 1 \
  --no-wandb
```

Smoke result:

| Metric | Value |
| --- | ---: |
| Train CE | 4.06852044 |
| Train accuracy | 0.10733957 |
| Eval CE | 3.28650260 |
| Eval accuracy | 0.26111111 |
| Eval perplexity | 28.49268803 |

The smoke confirmed that the standardised feature matrix loads and concatenates
with scalar VIX labels under `condition_dim=386`.

## Non-Smoke Training

The requested W&B-enabled command was attempted:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread poetry run tcvae-train-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_std.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/additive_logsig_l3_ctx20_std \
  --wandb \
  --wandb-project time-causal-token-prior \
  --wandb-entity tc_vae \
  --wandb-run-name sp500_vix_additive_logsig_l3_ctx20_std_seed0
```

W&B did not initialise in the sandboxed environment. The run failed before
training with repeated `ConnectionError` retries and:

```text
wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec.
```

The same non-smoke training was rerun with `--no-wandb`, keeping the same config
and output directory. No W&B cloud URL is available.

Training output:

- run directory:
  `outputs/sp500_vix_discrete/token_prior/additive_logsig_l3_ctx20_std/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_std_seed0`;
- best model:
  `outputs/sp500_vix_discrete/token_prior/additive_logsig_l3_ctx20_std/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_std_seed0/best_model`;
- device: CPU;
- runtime: `1385.80` seconds;
- best epoch: `100`.

| Metric | Value |
| --- | ---: |
| Train CE | 0.93054476 |
| Train accuracy | 0.64375933 |
| Train perplexity | 2.53614213 |
| Eval CE | 0.89357658 |
| Eval accuracy | 0.65556235 |
| Eval perplexity | 2.45419447 |

## Decoded Evaluation

Evaluation command:

```bash
poetry run tcvae-evaluate-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_std.yaml \
  --prior-dir outputs/sp500_vix_discrete/token_prior/additive_logsig_l3_ctx20_std/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_std_seed0/best_model \
  --tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --output-dir outputs/sp500_vix_discrete/token_prior/additive_logsig_l3_ctx20_std/evaluation_best \
  --base-data-dir data/processed \
  --n-sample 1000 \
  --seed 99 \
  --temperature 0.8 \
  --top-k 40
```

Decoded metrics:

| Metric | Value |
| --- | ---: |
| MMD | 0.35280779 |
| SWD | 0.01013594 |
| Terminal-return W1 | 0.00370423 |
| Volatility W1 | 0.00107429 |
| Sampled token perplexity | 30.08028984 |
| Active sampled codes | 59 / 64 |
| Marginal code L1 | 0.13616668 |
| Transition matrix L1 | 0.27316475 |
| Run-length distance | 0.87393731 |

Decoded VIX-bucket diagnostics:

| Bucket | MMD | SWD | Terminal W1 | Volatility W1 | Token ppl |
| --- | ---: | ---: | ---: | ---: | ---: |
| very_low | 0.28808013 | 0.00952558 | 0.00675053 | 0.00063123 | 31.46559525 |
| low | 0.44784069 | 0.01178017 | 0.00418206 | 0.00114920 | 27.51410866 |
| mid | 0.44578981 | 0.01049495 | 0.00424003 | 0.00085433 | 27.05276871 |
| high | 0.35258910 | 0.00956678 | 0.00618930 | 0.00126682 | 27.79064369 |
| very_high | 0.38678747 | 0.01489455 | 0.00621060 | 0.00156743 | 34.67290497 |

## Paper-Style Evaluation

Paper-style command:

```bash
env MPLBACKEND=Agg poetry run python scripts/evaluate_sp500_vix_paper_style.py \
  --discrete-config configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_std.yaml \
  --discrete-prior-dir outputs/sp500_vix_discrete/token_prior/additive_logsig_l3_ctx20_std/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_std_seed0/best_model \
  --discrete-tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --continuous-config configs/experiments/sp500_vix_beta_cvae.yaml \
  --continuous-model-dir outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47/final_model \
  --output-dir outputs/sp500_vix_discrete/paper_style_logsig_l3_ctx20_std_temp08_topk40 \
  --base-data-dir data/processed \
  --n-sample 1000 \
  --seed 99 \
  --temperature 0.8 \
  --top-k 40
```

The continuous BetaCVAE checkpoint loaded successfully.

Paper-style metrics:

| Model | MMD | SWD | Returns W1 | Terminal W1 | Volatility W1 | Drawdown W1 | Return AC L1 | Sq-return AC L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standardised `logsig_l3_ctx20` | 0.37982193 | 0.01154852 | 0.00108519 | 0.00272364 | 0.00111281 | 0.00570303 | 0.05253890 | 0.03923871 |
| continuous BetaCVAE | 0.15442121 | 0.00878550 | 0.00060182 | 0.00905099 | 0.00063360 | 0.00766744 | 0.02597247 | 0.02946163 |

Tail exceedance rates for standardised `logsig_l3_ctx20`, relative to real
return thresholds:

| Tail rate | Value |
| --- | ---: |
| Below real q001 | 0.00111864 |
| Below real q01 | 0.00588136 |
| Above real q99 | 0.00983051 |
| Above real q999 | 0.00318644 |

The paper-style summary also recorded the shared histogram-bin policy from the
plot-bin update. Returns and volatility-related histograms used shared union
ranges with Freedman-Diaconis bins capped at 150.

## Model-Selection Profile Scores

Profile scores below are simple lower-is-better average ranks across the
already reported paper-style metrics in
`docs/architecture/model_selection_profiles.md`. They are documentation-only
summaries and do not replace the component metrics above.

Compared models:

- VIX-only;
- raw `logsig_l3_ctx20`;
- `logsig_l3_ctx20_e200`;
- standardised `logsig_l3_ctx20`;
- continuous BetaCVAE.

| Model | Distributional | Tail-risk | Sequential-dependence | Balanced-market |
| --- | ---: | ---: | ---: | ---: |
| VIX-only | 2.667 | 5.000 | 4.000 | 4.143 |
| raw `logsig_l3_ctx20` | 3.000 | 2.000 | 2.500 | 2.429 |
| `logsig_l3_ctx20_e200` | 3.333 | 3.000 | 3.500 | 3.143 |
| standardised `logsig_l3_ctx20` | 4.667 | 2.333 | 4.000 | 3.429 |
| continuous BetaCVAE | 1.333 | 2.667 | 1.000 | 1.857 |

Best metric readings in this comparison:

- MMD: continuous BetaCVAE;
- SWD: VIX-only;
- terminal-return W1: standardised `logsig_l3_ctx20`;
- drawdown W1: raw `logsig_l3_ctx20`;
- volatility W1: continuous BetaCVAE;
- sequential autocorrelation metrics: continuous BetaCVAE.

## Comparison To Existing Discrete References

Same-setting references from earlier verification:

| Model | MMD | SWD | Returns W1 | Terminal W1 | Volatility W1 | Drawdown W1 | Return AC L1 | Sq-return AC L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VIX-only | 0.27934083 | 0.00767375 | 0.00124159 | 0.00981713 | 0.00118835 | 0.01050232 | 0.05174129 | 0.04129972 |
| raw `logsig_l3_ctx20` | 0.34163502 | 0.01082886 | 0.00099378 | 0.00451245 | 0.00080577 | 0.00549569 | 0.04789805 | 0.03506229 |
| `logsig_l3_ctx20_e200` | 0.35770059 | 0.00996951 | 0.00106480 | 0.00445077 | 0.00105878 | 0.00803712 | 0.04379739 | 0.04184117 |
| standardised `logsig_l3_ctx20` | 0.37982193 | 0.01154852 | 0.00108519 | 0.00272364 | 0.00111281 | 0.00570303 | 0.05253890 | 0.03923871 |

The standardised variant improves terminal-return W1 relative to all listed
discrete references. It does not improve the distributional profile, and it
does not beat the raw signature run on returns W1, volatility W1, drawdown W1,
or autocorrelation.

## Decision

Keep VIX-only as the public default and keep the raw `logsig_l3_ctx20` result
as the stronger signature-conditioning research reference.

Standardised log-signature conditioning is useful to retain as an optional
ablation because it improves terminal-return W1, confirms that standardised
features load correctly, and may help future optimisation studies. It should
not replace the raw `logsig_l3_ctx20` branch yet because:

- distributional MMD/SWD worsened relative to both VIX-only and raw
  `logsig_l3_ctx20`;
- raw `logsig_l3_ctx20` remains better on volatility, drawdown, returns W1, and
  squared-return autocorrelation;
- the balanced-market profile still favours the raw signature result among
  the discrete signature variants.

Next useful follow-up: compare raw versus standardised features under a smaller
learning-rate or checkpoint-selection sweep only after the signature-kernel
evaluation metric is available.
