# S&P500/VIX Signature-Conditioning Ablation Results

## Purpose

This note records the first non-smoke S&P500/VIX log-signature conditioning
ablation grid. The runs used the existing additive token-prior conditioning
path. No model architecture, tokeniser code, cross-attention, AdaLN, or
signature-kernel metric was added.

The first config path in the requested command used
`configs/sp500/experiments_vix_causal_token_prior_additive_logsig_l2_ctx10.yaml`,
which is not present. The run used the committed setup path
`configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l2_ctx10.yaml`.

## Command

```bash
WANDB_DISABLE_SERVICE=true WANDB_MODE=offline poetry run python scripts/run_sp500_vix_signature_conditioning_ablation.py \
  --configs \
    configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l2_ctx10.yaml \
    configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l2_ctx20.yaml \
    configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx10.yaml \
    configs/experiments/sp500_vix_causal_token_prior_additive_logsig_l3_ctx20.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/signature_conditioning_ablation \
  --base-data-dir data/processed \
  --epochs 100 \
  --n-sample 1000 \
  --seed 99 \
  --wandb \
  --wandb-project time-causal-token-prior \
  --wandb-entity tc_vae
```

W&B did not produce cloud URLs because the local socket-restricted environment
requires offline mode with service disablement. The offline W&B run directories
are:

| Experiment | Offline W&B run |
| --- | --- |
| `logsig_l2_ctx10` | `wandb/offline-run-20260514_161402-fqxgdrqv` |
| `logsig_l2_ctx20` | `wandb/offline-run-20260514_163652-c7nz8ig3` |
| `logsig_l3_ctx10` | `wandb/offline-run-20260514_165859-0lvqzwe2` |
| `logsig_l3_ctx20` | `wandb/offline-run-20260514_172058-bpban6nq` |

Sync command pattern:

```bash
poetry run wandb sync wandb/offline-run-20260514_161402-fqxgdrqv
```

## Feature Grid

All signature features used lead-lag paths with time and VIX channels.

| Experiment | Depth | Context | Feature dim | Total condition dim |
| --- | ---: | ---: | ---: | ---: |
| `logsig_l2_ctx10` | 2 | 10 | 55 | 56 |
| `logsig_l2_ctx20` | 2 | 20 | 55 | 56 |
| `logsig_l3_ctx10` | 3 | 10 | 385 | 386 |
| `logsig_l3_ctx20` | 3 | 20 | 385 | 386 |

## Token Metrics

All runs used CPU, 100 epochs, seed `0` in the config, and best checkpoint
selection by evaluation cross-entropy.

| Model | Runtime s | Train CE | Train acc | Eval CE | Eval acc | Eval ppl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VIX-only baseline | 1264.156 | 0.94478149 | 0.63869896 | 0.91480655 | 0.64744947 | 2.50792742 |
| `logsig_l2_ctx10` | 1290.263 | 0.94007571 | 0.64116131 | 0.90660984 | 0.65081401 | 2.48749447 |
| `logsig_l2_ctx20` | 1251.489 | 0.94003383 | 0.64101207 | 0.90663672 | 0.65104463 | 2.48757760 |
| `logsig_l3_ctx10` | 1242.184 | 0.93966978 | 0.64013703 | 0.90385556 | 0.65223850 | 2.48001624 |
| `logsig_l3_ctx20` | 1305.036 | 0.93910854 | 0.64032695 | 0.90376776 | 0.65244200 | 2.47960005 |

Depth 3 improved the token-prior likelihood and accuracy relative to both
depth 2 and the VIX-only baseline.

## Decoded Metrics

Sampling used `n_sample=1000`, temperature `0.8`, and top-k `40`.

| Model | MMD | SWD | Terminal W1 | Volatility W1 | Token ppl |
| --- | ---: | ---: | ---: | ---: | ---: |
| VIX-only baseline | 0.28880176 | 0.00889640 | 0.01076925 | 0.00132492 | 30.11016655 |
| `logsig_l2_ctx10` | 0.31092322 | 0.00906386 | 0.00744769 | 0.00136060 | 29.08641434 |
| `logsig_l2_ctx20` | 0.31721902 | 0.00859528 | 0.00708540 | 0.00134660 | 28.70219612 |
| `logsig_l3_ctx10` | 0.31865206 | 0.01090581 | 0.00425231 | 0.00103497 | 31.25273705 |
| `logsig_l3_ctx20` | 0.32337123 | 0.01074053 | 0.00472903 | 0.00091678 | 31.62069702 |

The VIX-only baseline retained the best MMD. `logsig_l2_ctx20` produced the
best SWD. Depth-3 signature conditioning produced the best terminal-return and
volatility Wasserstein distances.

## Market-Style Token Diagnostics

| Model | Marginal code L1 | Transition L1 | Run-length W1 | Active codes |
| --- | ---: | ---: | ---: | ---: |
| VIX-only baseline | 0.25939998 | 0.34322238 | 0.98253310 | 59 |
| `logsig_l2_ctx10` | 0.24506667 | 0.30939198 | 0.93564314 | 59 |
| `logsig_l2_ctx20` | 0.25096664 | 0.31828666 | 0.92794842 | 60 |
| `logsig_l3_ctx10` | 0.15570000 | 0.31605566 | 0.85673350 | 60 |
| `logsig_l3_ctx20` | 0.15313333 | 0.28188375 | 0.81454802 | 59 |

The depth-3 variants substantially improved marginal code usage, transition
matrix distance, and run-length distance. `logsig_l3_ctx20` was strongest on
these token-space diagnostics.

## VIX-Bucket Diagnostics

The evaluation buckets are formed from the condition vector used by the prior.
For signature-conditioned runs this is the concatenated condition vector rather
than the scalar VIX coordinate alone, so the bucket intervals are not directly
interpretable as scalar VIX ranges.

VIX-only baseline:

| Bucket | MMD | SWD | Terminal W1 | Volatility W1 |
| --- | ---: | ---: | ---: | ---: |
| very_low | 0.60906404 | 0.01298606 | 0.00784337 | 0.00136446 |
| low | 0.41611275 | 0.01190404 | 0.01448328 | 0.00112104 |
| mid | 0.38877460 | 0.01004604 | 0.02129920 | 0.00154563 |
| high | 0.35574767 | 0.01195208 | 0.02149734 | 0.00131412 |
| very_high | 0.34523797 | 0.01770910 | 0.01418822 | 0.00173272 |

Best token-space signature variant, `logsig_l3_ctx20`:

| Bucket | MMD | SWD | Terminal W1 | Volatility W1 |
| --- | ---: | ---: | ---: | ---: |
| very_low | 0.41033438 | 0.01152913 | 0.00762285 | 0.00058543 |
| low | 0.36109909 | 0.01358004 | 0.00726316 | 0.00115342 |
| mid | 0.38002902 | 0.01226918 | 0.00375455 | 0.00084995 |
| high | 0.25568834 | 0.01040529 | 0.00738958 | 0.00095081 |
| very_high | 0.41453210 | 0.01463821 | 0.00895366 | 0.00123934 |

`logsig_l3_ctx20` improved most bucket-level terminal and volatility distances
relative to the VIX-only baseline. The MMD improvement was not uniform; the
very-high bucket was worse than VIX-only.

## Paper-Style Diagnostics

The ablation runner was executed exactly as requested and did not include
`--run-paper-style`, so paper-style autocorrelation, drawdown, tail plots, and
continuous-model comparison were not rerun for all four grid points.

The previously recorded `logsig_l2_ctx20` paper-style run remains available at
`outputs/sp500_vix_discrete/paper_style_logsig_l2` and reported:

| Metric | `logsig_l2_ctx20` |
| --- | ---: |
| MMD | 0.31115741 |
| SWD | 0.01007967 |
| One-step return W1 | 0.00127050 |
| Terminal-return W1 | 0.00699868 |
| Volatility W1 | 0.00134263 |
| Maximum-drawdown W1 | 0.01103095 |
| Return autocorrelation L1 | 0.05549955 |
| Squared-return autocorrelation L1 | 0.04449294 |
| Flattened squared-return autocorrelation L1 | 0.14269510 |

The continuous checkpoint was unavailable in that prior paper-style run, so the
continuous comparison was skipped.

## Decision

Do not promote VIX+signature conditioning as the new default yet.

The evidence supports running further ablations:

- depth-3 log-signatures improved token likelihood, terminal-return W1,
  volatility W1, marginal code usage, transition distance, and run-length
  distance;
- VIX-only remained better on aggregate decoded MMD;
- `logsig_l2_ctx20` was best on SWD among this grid;
- depth-3 variants need paper-style diagnostics before any promotion decision;
- at least one additional seed is needed before treating the gain as robust.

Recommended next step: keep the VIX-only prior as the promoted public default
for now, and run paper-style diagnostics plus seed/sampling ablations for
`logsig_l3_ctx10`, `logsig_l3_ctx20`, and the VIX-only baseline under identical
settings.
