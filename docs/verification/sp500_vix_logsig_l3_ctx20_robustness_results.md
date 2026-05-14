# S&P500/VIX `logsig_l3_ctx20` Robustness Results

## Purpose

This note records the non-smoke robustness ablations for the depth-3,
context-20 log-signature conditioned S&P500/VIX token prior. The runs preserved
the existing tokenizer, additive token-prior architecture, signature feature
directory, and sampling settings. No Gumbel-Softmax, new objective, tokenizer
change, or signature-kernel objective was added.

## Command And Telemetry

The requested live W&B command was attempted first:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread poetry run python scripts/run_sp500_vix_signature_conditioning_ablation.py \
  --configs \
    configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_seed1.yaml \
    configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_seed2.yaml \
    configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_lr1e4.yaml \
    configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_lr5e4.yaml \
    configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20_e200.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/signature_conditioning_robustness \
  --base-data-dir data/processed \
  --n-sample 1000 \
  --seed 99 \
  --temperature 0.8 \
  --top-k 40 \
  --run-paper-style \
  --continuous-model-dir outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47/final_model \
  --wandb \
  --wandb-project time-causal-token-prior \
  --wandb-entity tc_vae
```

W&B did not initialise. The run failed before training with repeated
`ConnectionError` retries followed by:

```text
wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec.
```

The robustness grid was then rerun with the requested fallback, replacing
`--wandb` with `--no-wandb` while keeping the same configs, output directory,
sampling settings, and continuous checkpoint. No W&B cloud run URLs are
available; the aggregate rows therefore have `wandb_run_url: null`.

Outputs:

- aggregate JSON:
  `outputs/sp500_vix_discrete/token_prior/signature_conditioning_robustness/ablation_results.json`;
- aggregate CSV:
  `outputs/sp500_vix_discrete/token_prior/signature_conditioning_robustness/ablation_results.csv`;
- continuous checkpoint:
  `outputs/sp500_vix_continuous/beta_cvae/BetaCVAE_training_2026-05-14_18-13-47/final_model`.

All runs used `n_sample=1000`, `seed=99`, `temperature=0.8`, and `top_k=40`.
Paper-style diagnostics loaded the continuous BetaCVAE checkpoint successfully
for every robustness variant.

## Token Metrics

Lower is better for cross-entropy and perplexity. Higher is better for
accuracy.

| Model | Runtime s | Best epoch | Train CE | Eval CE | Eval acc | Eval ppl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `logsig_l3_ctx20_seed1` | 1298.7 | 100 | 0.934305 | 0.899914 | 0.653710 | 2.470195 |
| `logsig_l3_ctx20_seed2` | 1258.7 | 100 | 0.935285 | 0.900910 | 0.653337 | 2.472544 |
| `logsig_l3_ctx20_lr1e4` | 1259.8 | 100 | 1.040608 | 1.011755 | 0.628137 | 2.769453 |
| `logsig_l3_ctx20_lr5e4` | 1246.4 | 100 | 0.859088 | 0.800069 | 0.687329 | 2.231858 |
| `logsig_l3_ctx20_e200` | 2641.5 | 200 | 0.783583 | 0.699797 | 0.728300 | 2.016635 |

The 200-epoch variant is the clear token-likelihood winner. The higher
learning-rate variant also improves token CE substantially relative to the
100-epoch seed variants. The lower learning-rate variant underfits by token
metrics.

## Decoded Metrics

These are from `tcvae-evaluate-token-prior` on the best checkpoint for each
variant.

| Model | MMD | SWD | Terminal W1 | Volatility W1 | Sampled token ppl |
| --- | ---: | ---: | ---: | ---: | ---: |
| `logsig_l3_ctx20_seed1` | 0.31727087 | 0.00877613 | 0.00633235 | 0.00126968 | 29.31511497 |
| `logsig_l3_ctx20_seed2` | 0.34162942 | 0.01079321 | 0.00906648 | 0.00124223 | 30.01376534 |
| `logsig_l3_ctx20_lr1e4` | 0.34352624 | 0.01142613 | 0.00426132 | 0.00086689 | 31.11126900 |
| `logsig_l3_ctx20_lr5e4` | 0.38435179 | 0.00998254 | 0.00444546 | 0.00116353 | 28.89450073 |
| `logsig_l3_ctx20_e200` | 0.38044661 | 0.01176067 | 0.00448689 | 0.00112286 | 29.09955788 |

`seed1` is best in this table for decoded MMD and SWD. The lower learning-rate
variant is best for decoded terminal and volatility Wasserstein, despite weaker
token likelihood.

## Paper-Style Metrics

These metrics come from each `paper_style_summary.json`. Lower is better.

| Model | MMD | SWD | Returns W1 | Terminal W1 | Volatility W1 | Drawdown W1 | Return AC L1 | Sq-return AC L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `logsig_l3_ctx20_seed1` | 0.31833130 | 0.00852141 | 0.00126174 | 0.00608329 | 0.00126076 | 0.00968228 | 0.05363005 | 0.04498956 |
| `logsig_l3_ctx20_seed2` | 0.35144007 | 0.00974451 | 0.00124176 | 0.00686464 | 0.00131644 | 0.01159743 | 0.04761778 | 0.04560776 |
| `logsig_l3_ctx20_lr1e4` | 0.32993868 | 0.01159214 | 0.00106899 | 0.00469961 | 0.00106873 | 0.00678544 | 0.05180580 | 0.03324079 |
| `logsig_l3_ctx20_lr5e4` | 0.34506252 | 0.01008294 | 0.00123485 | 0.00431034 | 0.00130034 | 0.01000923 | 0.05028445 | 0.04594274 |
| `logsig_l3_ctx20_e200` | 0.35770059 | 0.00996951 | 0.00106480 | 0.00445077 | 0.00105878 | 0.00803712 | 0.04379739 | 0.04184117 |

Tail exceedance rates, relative to real-data return thresholds:

| Model | < real q001 | < real q01 | > real q99 | > real q999 |
| --- | ---: | ---: | ---: | ---: |
| `logsig_l3_ctx20_seed1` | 0.00094915 | 0.00455932 | 0.00783051 | 0.00208475 |
| `logsig_l3_ctx20_seed2` | 0.00120339 | 0.00547458 | 0.00903390 | 0.00277966 |
| `logsig_l3_ctx20_lr1e4` | 0.00181356 | 0.00722034 | 0.01200000 | 0.00386441 |
| `logsig_l3_ctx20_lr5e4` | 0.00106780 | 0.00454237 | 0.00811864 | 0.00235593 |
| `logsig_l3_ctx20_e200` | 0.00105085 | 0.00569492 | 0.00925424 | 0.00271186 |

The lower learning-rate variant gives the heaviest tail rates among the
robustness variants. The 200-epoch variant gives the best paper-style return
autocorrelation and the strongest balanced-market rank among this grid, but it
does not recover the original `logsig_l3_ctx20` volatility or drawdown results.

## Model-Selection Profile Scores

The profile scores are the runner's documentation-only average-rank summaries
over already-reported paper-style metrics. Lower is better. They do not replace
the component metric tables above.

| Model | Distributional | Tail-risk | Sequential-dependence | Balanced-market |
| --- | ---: | ---: | ---: | ---: |
| `logsig_l3_ctx20_seed1` | 2.333 | 3.333 | 4.000 | 3.429 |
| `logsig_l3_ctx20_seed2` | 3.333 | 5.000 | 3.000 | 3.857 |
| `logsig_l3_ctx20_lr1e4` | 3.000 | 2.000 | 2.500 | 2.571 |
| `logsig_l3_ctx20_lr5e4` | 3.333 | 3.000 | 4.000 | 3.429 |
| `logsig_l3_ctx20_e200` | 3.000 | 1.667 | 1.500 | 1.714 |

Profile reading:

- distributional: `seed1` is best within the robustness grid;
- tail-risk: `e200` is best, with `lr1e4` second;
- sequential-dependence: `e200` is best;
- balanced-market: `e200` is best.

## Comparison To Current References

Same-setting paper-style references:

| Model | MMD | SWD | Returns W1 | Terminal W1 | Volatility W1 | Drawdown W1 | Return AC L1 | Sq-return AC L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VIX-only | 0.27934083 | 0.00767375 | 0.00124159 | 0.00981713 | 0.00118835 | 0.01050232 | 0.05174129 | 0.04129972 |
| Original `logsig_l3_ctx20` | 0.34163502 | 0.01082886 | 0.00099378 | 0.00451245 | 0.00080577 | 0.00549569 | 0.04789805 | 0.03506229 |
| Continuous BetaCVAE | 0.15442121 | 0.00878550 | 0.00060182 | 0.00905099 | 0.00063360 | 0.00766744 | 0.02597247 | 0.02946163 |

Main comparisons:

- VIX-only remains the best discrete model for global MMD and SWD.
- The original `logsig_l3_ctx20` remains the best discrete signature result for
  returns W1, volatility W1, drawdown W1, and squared-return autocorrelation
  among the reference and robustness rows.
- `logsig_l3_ctx20_e200` improves token CE strongly and is competitive on
  terminal W1 and return autocorrelation, but its paper-style MMD, SWD,
  volatility W1, drawdown W1, and squared-return autocorrelation do not dominate
  the original `logsig_l3_ctx20`.
- `logsig_l3_ctx20_seed1` improves paper-style MMD and SWD relative to the
  original `logsig_l3_ctx20`, but it loses most path-functional advantages.
- `logsig_l3_ctx20_seed2` is weaker than the original signature run on most
  paper-style path metrics.
- Continuous BetaCVAE remains the strongest reference for MMD, returns W1,
  volatility W1, and autocorrelation diagnostics, but it is not the promoted
  discrete-token baseline.

## Decision

Do not promote signature conditioning as the public default yet.

The robustness grid confirms that log-signature conditioning can improve
important path-functional diagnostics, but the gains are not stable enough for
promotion:

- the two additional seeds do not reproduce the original `logsig_l3_ctx20`
  path-functional profile;
- the 200-epoch run improves token likelihood and model-selection rank within
  this robustness grid, but it worsens several paper-style metrics relative to
  the original signature run;
- VIX-only remains the cleanest default for distributional MMD/SWD;
- the continuous baseline remains stronger on several high-level paper-style
  diagnostics.

Current status:

- keep VIX-only as the default public discrete prior;
- retain `logsig_l3_ctx20` as a promising research branch for tail-risk and
  sequential-dependence diagnostics;
- run more seeds only for a narrowed target, likely `e200` or a checkpoint/epoch
  selection study around the original seed-0 training path;
- add the planned evaluation-only signature-kernel metric before reconsidering
  promotion.
