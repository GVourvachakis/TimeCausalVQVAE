# S&P500/VIX FiLM/AdaLN `logsig_l3_ctx20_std` Quality

## Purpose

This note records the first non-smoke FiLM/AdaLN-style signature-conditioning
experiment for the S&P500/VIX discrete causal token prior.

The run used the existing `adaln_lite` conditioning path with scalar VIX plus
standardised depth-3 context-20 log-signature features. No tokenizer code,
cross-attention, Gumbel-Softmax path, signature-kernel loss, or model
architecture outside the configured condition-injection mode was changed during
this prompt.

## Configuration

Config:

```text
configs/experiments/sp500_vix_causal_token_prior_film_logsig_l3_ctx20_std.yaml
```

Conditioning:

- condition injection: `adaln_lite`;
- condition vector: scalar VIX plus standardised `logsig_l3_ctx20` features;
- condition shape: `[batch, 386]`;
- AdaLN hidden dimension: `128`;
- token-prior architecture: same causal single-code transformer dimensions as
  the additive S&P500/VIX prior;
- sampling for evaluation: `temperature=0.8`, `top_k=40`, `n_sample=1000`,
  `seed=99`.

## W&B Status

The requested live W&B command was attempted with the documented execution
profile:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread poetry run tcvae-train-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_film_logsig_l3_ctx20_std.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/film_logsig_l3_ctx20_std \
  --wandb \
  --wandb-project time-causal-token-prior \
  --wandb-entity tc_vae \
  --wandb-run-name sp500_vix_film_logsig_l3_ctx20_std_seed0
```

W&B did not initialise in the sandboxed environment. The run failed before
training with repeated `ConnectionError` retries and:

```text
wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec.
```

The same training was rerun with `--no-wandb`, preserving the config and output
directory. No W&B cloud URL is available for this run.

## Training

Fallback command:

```bash
env MPLBACKEND=Agg poetry run tcvae-train-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_film_logsig_l3_ctx20_std.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/film_logsig_l3_ctx20_std \
  --no-wandb
```

Output:

- run directory:
  `outputs/sp500_vix_discrete/token_prior/film_logsig_l3_ctx20_std/sp500_vix_causal_token_prior_film_logsig_l3_ctx20_std_seed0`;
- best model:
  `outputs/sp500_vix_discrete/token_prior/film_logsig_l3_ctx20_std/sp500_vix_causal_token_prior_film_logsig_l3_ctx20_std_seed0/best_model`;
- runtime: `1644.260` seconds;
- device: CPU;
- best epoch: `100`;
- W&B enabled: `false`.

Token-prior metrics:

| Metric | Value |
| --- | ---: |
| Train CE | 0.69486250 |
| Train accuracy | 0.73076924 |
| Train perplexity | 2.00356812 |
| Eval CE | 0.56481865 |
| Eval accuracy | 0.78709130 |
| Eval perplexity | 1.76141225 |

The FiLM/AdaLN condition path substantially improves teacher-forced token
metrics relative to the additive standardised log-signature run, which recorded
eval CE `0.89357658` and eval perplexity `2.45419447`.

## Decoded Evaluation

Command:

```bash
poetry run tcvae-evaluate-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_film_logsig_l3_ctx20_std.yaml \
  --prior-dir outputs/sp500_vix_discrete/token_prior/film_logsig_l3_ctx20_std/sp500_vix_causal_token_prior_film_logsig_l3_ctx20_std_seed0/best_model \
  --tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --output-dir outputs/sp500_vix_discrete/token_prior/film_logsig_l3_ctx20_std/evaluation_best \
  --base-data-dir data/processed \
  --n-sample 1000 \
  --seed 99 \
  --temperature 0.8 \
  --top-k 40
```

Decoded metrics:

| Metric | Value |
| --- | ---: |
| MMD | 0.40207493 |
| SWD | 0.01012650 |
| Terminal-return W1 | 0.00654597 |
| Volatility W1 | 0.00114467 |
| Sampled token perplexity | 26.93493462 |
| Active sampled codes | 59 / 64 |
| Marginal code L1 | 0.20490003 |
| Transition matrix L1 | 0.25882697 |
| Run-length distance | 1.08743310 |

Decoded VIX-bucket diagnostics:

| Bucket | MMD | SWD | Terminal W1 | Volatility W1 | Token ppl |
| --- | ---: | ---: | ---: | ---: | ---: |
| very_low | 0.36768940 | 0.01128804 | 0.01051386 | 0.00113587 | 27.24514008 |
| low | 0.50297517 | 0.01165249 | 0.00697506 | 0.00138482 | 23.08637047 |
| mid | 0.44929358 | 0.01023964 | 0.00984708 | 0.00117259 | 23.94597626 |
| high | 0.41276637 | 0.01054763 | 0.00886110 | 0.00132263 | 25.50958252 |
| very_high | 0.42531762 | 0.01321921 | 0.00341917 | 0.00099738 | 32.75095367 |

## Paper-Style Evaluation

Command:

```bash
env MPLBACKEND=Agg poetry run python scripts/evaluate_sp500_vix_paper_style.py \
  --discrete-config configs/experiments/sp500_vix_causal_token_prior_film_logsig_l3_ctx20_std.yaml \
  --discrete-prior-dir outputs/sp500_vix_discrete/token_prior/film_logsig_l3_ctx20_std/sp500_vix_causal_token_prior_film_logsig_l3_ctx20_std_seed0/best_model \
  --discrete-tokenizer-dir outputs/sp500_vix_discrete/tokenizer/sp500_vix_causal_vq_tokenizer_seed0 \
  --continuous-config configs/experiments/sp500_vix_beta_cvae.yaml \
  --continuous-model-dir outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47/final_model \
  --output-dir outputs/sp500_vix_discrete/paper_style_film_logsig_l3_ctx20_std_temp08_topk40 \
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
| FiLM/AdaLN `logsig_l3_ctx20_std` | 0.36974820 | 0.01028688 | 0.00126779 | 0.00780297 | 0.00119087 | 0.01071865 | 0.05361288 | 0.04005349 |
| Continuous BetaCVAE | 0.15442121 | 0.00878550 | 0.00060182 | 0.00905099 | 0.00063360 | 0.00766744 | 0.02597247 | 0.02946163 |

Tail exceedance rates for FiLM/AdaLN `logsig_l3_ctx20_std`, relative to real
return thresholds:

| Tail rate | Value |
| --- | ---: |
| Below real q001 | 0.00118644 |
| Below real q01 | 0.00554237 |
| Above real q99 | 0.00879661 |
| Above real q999 | 0.00300000 |

## Signature-Kernel MMD

`sigkernel` was installed and imported successfully:

```bash
poetry run python -c "import sigkernel; print('sigkernel', getattr(sigkernel, '__version__', 'unknown'))"
```

Output:

```text
sigkernel unknown
```

No-lead-lag command:

```bash
poetry run python scripts/evaluate_signature_kernel_metrics.py \
  --paper-style-batch outputs/sp500_vix_discrete/paper_style_film_logsig_l3_ctx20_std_temp08_topk40/discrete_paper_style_batch.pt \
  --output-dir outputs/signature_kernel_results/film_logsig_l3_ctx20_std_no_leadlag \
  --max-samples 256 \
  --dyadic-order 1 \
  --include-time
```

Lead-lag command:

```bash
poetry run python scripts/evaluate_signature_kernel_metrics.py \
  --paper-style-batch outputs/sp500_vix_discrete/paper_style_film_logsig_l3_ctx20_std_temp08_topk40/discrete_paper_style_batch.pt \
  --output-dir outputs/signature_kernel_results/film_logsig_l3_ctx20_std_leadlag64 \
  --max-samples 64 \
  --dyadic-order 1 \
  --include-time \
  --use-lead-lag
```

Signature-kernel results:

| Pass | Samples | Runtime s | Signature-kernel MMD | Finite | Symmetric | Positive diagonal |
| --- | ---: | ---: | ---: | --- | --- | --- |
| no lead-lag | 256 | 126.388 | 0.00650810 | true | true | true |
| lead-lag | 64 | 35.022 | 0.00540422 | true | true | true |

The no-lead-lag signature-kernel MMD is worse than the earlier VIX-only,
raw-signature, standardised-additive signature, and continuous reference
results. The smaller lead-lag result is better than raw-signature and
continuous on the prior 64-sample comparison, but still worse than VIX-only and
standardised additive. As before, the lead-lag result is exploratory because it
uses only 64 samples.

## Comparison To Existing References

Same-setting paper-style references:

| Model | MMD | SWD | Returns W1 | Terminal W1 | Volatility W1 | Drawdown W1 | Return AC L1 | Sq-return AC L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VIX-only additive | 0.27934083 | 0.00767375 | 0.00124159 | 0.00981713 | 0.00118835 | 0.01050232 | 0.05174129 | 0.04129972 |
| Raw `logsig_l3_ctx20` additive | 0.34163502 | 0.01082886 | 0.00099378 | 0.00451245 | 0.00080577 | 0.00549569 | 0.04789805 | 0.03506229 |
| Standardised `logsig_l3_ctx20` additive | 0.37982193 | 0.01154852 | 0.00108519 | 0.00272364 | 0.00111281 | 0.00570303 | 0.05253890 | 0.03923871 |
| FiLM/AdaLN `logsig_l3_ctx20_std` | 0.36974820 | 0.01028688 | 0.00126779 | 0.00780297 | 0.00119087 | 0.01071865 | 0.05361288 | 0.04005349 |
| Continuous BetaCVAE | 0.15442121 | 0.00878550 | 0.00060182 | 0.00905099 | 0.00063360 | 0.00766744 | 0.02597247 | 0.02946163 |

Signature-kernel no-lead-lag references:

| Model | Signature-kernel MMD |
| --- | ---: |
| Continuous BetaCVAE | 0.00087344 |
| Standardised `logsig_l3_ctx20` additive | 0.00224031 |
| Raw `logsig_l3_ctx20` additive | 0.00238908 |
| VIX-only additive | 0.00555772 |
| FiLM/AdaLN `logsig_l3_ctx20_std` | 0.00650810 |

Profile scores are lower-is-better average ranks across the visible component
metrics in `docs/architecture/model_selection_profiles.md`.

| Model | Distributional | Tail-risk | Sequential-dependence | Balanced-market |
| --- | ---: | ---: | ---: | ---: |
| VIX-only additive | 2.333 | 4.333 | 4.000 | 3.714 |
| Raw `logsig_l3_ctx20` additive | 3.000 | 1.667 | 2.000 | 2.143 |
| Standardised `logsig_l3_ctx20` additive | 4.333 | 2.000 | 3.500 | 3.000 |
| FiLM/AdaLN `logsig_l3_ctx20_std` | 4.000 | 4.333 | 4.500 | 4.286 |
| Continuous BetaCVAE | 1.333 | 2.667 | 1.000 | 1.857 |

## Interpretation

FiLM/AdaLN modulation uses the high-dimensional standardised log-signature
condition vector much more aggressively than additive conditioning in
teacher-forced token space. The token CE and token accuracy therefore improve
substantially.

That improvement does not transfer to the market-generation diagnostics under
the current sampling policy. FiLM/AdaLN is worse than raw additive
`logsig_l3_ctx20` on tail-risk, sequential-dependence, balanced-market profile,
and no-lead-lag signature-kernel MMD. It is also not competitive with the
VIX-only additive model on the distributional guardrail metrics.

The result suggests that stronger condition modulation can improve token
prediction while overfitting or sharpening token transitions in a way that does
not preserve decoded path functionals.

## Decision

Decision: keep the VIX-only additive prior as the public default and keep raw
additive `logsig_l3_ctx20` as the stronger signature-conditioning research
candidate. Do not promote the FiLM/AdaLN variant from this run.

The FiLM/AdaLN path should not receive a seed ablation yet. It may be revisited
only after a narrower regularisation or sampling-temperature question is posed,
because the first non-smoke run improved token likelihood but regressed the
market-style and signature-kernel profiles that matter for this branch.

## Check Status

Completed checks:

- `poetry run ruff format docs`;
- `poetry run ruff check docs --fix`;
- `poetry check`.
