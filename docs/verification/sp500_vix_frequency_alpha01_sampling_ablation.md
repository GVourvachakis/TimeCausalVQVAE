# S&P500/VIX Frequency Alpha 0.1 Sampling Ablation

## Scope

This note records a sampling-only ablation for the additive VIX-only causal AR prior trained on the
alpha 0.1 causal EMA frequency tokenizer. No models were trained and no source code was changed.

Inputs:

- Prior checkpoint:
  `outputs/sp500_vix_discrete/token_prior/freq_ema_alpha01/sp500_vix_causal_token_prior_freq_ema_alpha01_seed0/best_model`
- Tokenizer checkpoint:
  `outputs/sp500_vix_discrete/frequency_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_freq_ema_alpha01_seed0`
- Token-prior sampling ablation:
  `outputs/sp500_vix_discrete/token_prior/freq_ema_alpha01/sampling_ablation`
- Paper-style ablation outputs:
  `outputs/sp500_vix_discrete/paper_style_freq_ema_alpha01_temp*_topk*`

Grid:

- Temperatures: `0.6`, `0.8`, `1.0`
- Top-k: unrestricted, `20`, `40`
- `n_sample=1000`
- Seed: `99`

## Decoded Token-Prior Profile

The `run_token_prior_sampling_ablation.py` helper was run across the full grid. Because that helper
does not compose frequency-tokenizer `[low, high]` outputs internally, the decoded path metrics
below are native two-channel frequency-space diagnostics. They are useful for token/code behaviour,
but the terminal-return scale is not used for scalar-path model selection.

| Temp | Top-k | MMD | SWD | Volatility W1 | Active codes | Token perplexity | Marginal L1 | Transition L1 | Run length |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6 | none | 0.58215129 | 0.01103355 | 0.00162294 | 59 | 32.98244476 | 0.32696664 | 0.32781351 | 1.25224447 |
| 0.6 | 20 | 0.60067564 | 0.01103090 | 0.00162093 | 60 | 32.31309128 | 0.32866666 | 0.34080625 | 1.20386374 |
| 0.6 | 40 | 0.57108200 | 0.01061516 | 0.00160692 | 59 | 33.39359665 | 0.31200001 | 0.33832172 | 1.24004614 |
| 0.8 | none | 0.51503813 | 0.00912419 | 0.00131069 | 63 | 39.92897797 | 0.21143331 | 0.29427016 | 0.63582957 |
| 0.8 | 20 | 0.51280248 | 0.00891456 | 0.00131223 | 62 | 39.98262787 | 0.19813332 | 0.28231648 | 0.67020166 |
| 0.8 | 40 | 0.49195930 | 0.00914592 | 0.00130730 | 63 | 40.88571167 | 0.19380000 | 0.32527560 | 0.65693432 |
| 1.0 | none | 0.44800889 | 0.00817246 | 0.00144765 | 63 | 45.50671387 | 0.21276666 | 0.29434511 | 0.03009012 |
| 1.0 | 20 | 0.46217775 | 0.00847911 | 0.00136620 | 63 | 45.49062729 | 0.19076666 | 0.29549724 | 0.07052936 |
| 1.0 | 40 | 0.45186573 | 0.00777408 | 0.00137778 | 63 | 45.01213837 | 0.18733333 | 0.29180014 | 0.02382771 |

The native decoded profile prefers warmer sampling by MMD/SWD and code perplexity. The helper's
native score selected `temperature=1.0, top_k=20`, but the score is not used as the final decision
criterion because it includes uncomposed frequency-channel terminal-return values.

## Paper-Style Profile

Paper-style diagnostics compose decoded frequency channels back to scalar paths before computing
market diagnostics. This is the decision table for sampling.

| Temp | Top-k | Guardrail score | MMD | SWD | Volatility W1 | Sq-return AC L1 | Flat sq-return AC L1 | Drawdown W1 | Terminal W1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6 | none | 0.49988284 | 0.46798772 | 0.01577175 | 0.00220311 | 0.03977151 | 0.05431695 | 0.01857388 | 0.01392026 |
| 0.6 | 20 | 0.51915601 | 0.48730984 | 0.01575460 | 0.00216434 | 0.04087482 | 0.07266629 | 0.01860889 | 0.01392722 |
| 0.6 | 40 | 0.49305029 | 0.46125007 | 0.01515895 | 0.00214772 | 0.04189021 | 0.04526834 | 0.01869622 | 0.01449355 |
| 0.8 | none | 0.43159628 | 0.40561655 | 0.01276135 | 0.00100521 | 0.03417068 | 0.08714594 | 0.00929958 | 0.01221317 |
| 0.8 | 20 | 0.43953074 | 0.41227192 | 0.01241671 | 0.00107656 | 0.03628839 | 0.11011288 | 0.01014716 | 0.01376555 |
| 0.8 | 40 | 0.41109909 | 0.38490504 | 0.01312830 | 0.00105531 | 0.03702611 | 0.05022892 | 0.00993632 | 0.01201044 |
| 1.0 | none | 0.36820728 | 0.34736192 | 0.01155593 | 0.00123859 | 0.03023081 | 0.06069550 | 0.00815493 | 0.00805083 |
| 1.0 | 20 | 0.38504723 | 0.36112088 | 0.01229755 | 0.00102655 | 0.02993220 | 0.05171447 | 0.00740420 | 0.01060225 |
| 1.0 | 40 | 0.37180876 | 0.35045806 | 0.01099494 | 0.00102956 | 0.03425314 | 0.06868259 | 0.00646785 | 0.00932620 |

Guardrail score is `MMD + SWD + volatility W1 + terminal W1`, matching the prior sampling-ablation
spirit but using composed scalar paths. The best scalar guardrail setting is
`temperature=1.0, top_k=none`. The best volatility W1 is `temperature=0.8, top_k=none`, and the
best within-path squared-return autocorrelation is `temperature=1.0, top_k=20`.

## Comparison

| Model | MMD | SWD | Volatility W1 | Sq-return AC L1 | Flat sq-return AC L1 | Drawdown W1 | Terminal W1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Alpha01 selected, temp 1.0 top-k none | 0.34736192 | 0.01155593 | 0.00123859 | 0.03023081 | 0.06069550 | 0.00815493 | 0.00805083 |
| Alpha01 prior-quality, temp 0.8 top-k 40 | 0.38490504 | 0.01312830 | 0.00105531 | 0.03702611 | 0.05022892 | 0.00993632 | 0.01201044 |
| Promoted baseline, temp 0.8 top-k 40 | 0.27934083 | 0.00767375 | 0.00118835 | 0.04129972 | 0.12882016 | 0.01050232 | 0.00981713 |
| Hidden128, temp 0.8 top-k 20 | 0.22907834 | 0.00717505 | 0.00125777 | 0.06088475 | 0.07886228 | 0.00891025 | 0.00450928 |

The selected alpha01 setting improves over the previous alpha01 `temp=0.8, top_k=40` profile on
MMD, SWD, within-path squared-return autocorrelation, drawdown W1, and terminal-return W1. It gives
up some volatility W1 and flattened squared-return autocorrelation.

Relative to the promoted baseline, alpha01 at `temperature=1.0, top_k=none` remains worse on MMD and
SWD, but improves squared-return autocorrelation, drawdown W1, terminal-return W1, and is close on
volatility W1. Relative to hidden128 top-k20, it improves squared-return autocorrelation and
drawdown W1, is close on volatility W1, but remains worse on MMD, SWD, terminal-return W1, and
flattened squared-return autocorrelation.

## Decision

Selected sampling setting for alpha01: `temperature=1.0`, unrestricted top-k.

Alpha01 remains viable only as a diagnostic frequency-tokenizer branch. The sampling ablation
removes part of the earlier terminal-return and drawdown weakness, and it gives strong
within-path squared-return autocorrelation. However, it still fails the broader replacement gate
because MMD and SWD remain materially worse than both the promoted baseline and hidden128. Continue
with alpha 0.2 prior comparison before investing in grouped tokenizers or a wider alpha grid.
