# S&P500/VIX Frequency Alpha 0.1 Prior Quality

## Scope

This note records the additive VIX-only causal autoregressive prior trained on tokens from the
alpha 0.1 causal EMA frequency tokenizer. The prior architecture and objective were unchanged
from the promoted VIX-only additive prior.

Non-goals remained in force: no token-prior architecture changes, no signatures, no GroupedRVQ,
no MGVQ, no diffusion, no cross-attention, and no new objectives.

## Inputs

- Prior config:
  `configs/experiments/sp500_vix_causal_token_prior_freq_ema_alpha01.yaml`
- Tokenizer checkpoint:
  `outputs/sp500_vix_discrete/frequency_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_freq_ema_alpha01_seed0`
- Token data:
  `outputs/sp500_vix_discrete/token_prior/freq_ema_alpha01_tokens`
- Prior output:
  `outputs/sp500_vix_discrete/token_prior/freq_ema_alpha01/sp500_vix_causal_token_prior_freq_ema_alpha01_seed0`
- Best prior checkpoint:
  `outputs/sp500_vix_discrete/token_prior/freq_ema_alpha01/sp500_vix_causal_token_prior_freq_ema_alpha01_seed0/best_model`
- Best-checkpoint evaluation:
  `outputs/sp500_vix_discrete/token_prior/freq_ema_alpha01/evaluation_best`
- Paper-style evaluation:
  `outputs/sp500_vix_discrete/paper_style_freq_ema_alpha01_temp08_topk40`

The decoded prior evaluation composed the tokenizer's `[low, high]` outputs back to scalar paths
before computing path diagnostics. The reported decoded path shape is `[1000, 60, 1]`.

## W&B

The required W&B-profile run failed during initialisation with a network `CommError` timeout after
90 seconds. The same training command was rerun with `--no-wandb`, as requested. No W&B URL is
available for this run.

## Token Likelihood

| Metric | Value |
|---| --- :|
| Best epoch | 100 |
| Best eval cross-entropy | 1.05693870 |
| Best eval accuracy | 0.59681862 |
| Best eval perplexity | 2.88918605 |
| Final eval cross-entropy | 1.05693870 |
| Runtime seconds | 1213.060 |

The best checkpoint is the final epoch, so the 100-epoch schedule had not visibly plateaued by the
selection metric.

## Decoded Token-Prior Evaluation

Sampling used `n_sample=1000`, seed 99, temperature 0.8, and top-k 40.

| Metric | Value |
|---| --- :|
| Sampled active codes | 61 / 64 |
| Sampled token perplexity | 38.84451675 |
| Real token perplexity | 42.43110275 |
| Marginal code L1 | 0.21696667 |
| Transition matrix L1 | 0.27819496 |
| Run-length distance | 0.65960556 |
| MMD | 0.42002434 |
| SWD | 0.01306464 |
| Terminal-return W1 | 0.01346600 |
| Volatility W1 | 0.00112879 |

## VIX-Bucket Diagnostics

| Bucket | n | MMD | Volatility W1 | Terminal W1 | Active codes | Token perplexity |
|---| --- :| --- :| --- :| --- :| --- :| --- :|
| very_low | 200 | 0.40261537 | 0.00104584 | 0.00458284 | 56 | 25.90332031 |
| low | 200 | 0.61337960 | 0.00083256 | 0.01669003 | 57 | 31.05168915 |
| mid | 200 | 0.61059898 | 0.00141071 | 0.02029739 | 58 | 36.74652863 |
| high | 200 | 0.51764745 | 0.00155320 | 0.02608061 | 59 | 38.60998917 |
| very_high | 200 | 0.26551816 | 0.00122461 | 0.01062380 | 61 | 48.40855789 |

The prior keeps broad code usage in every VIX bucket, but the low, mid, and high buckets carry the
largest path-distribution errors.

## Paper-Style Metrics

| Model | MMD | SWD | Volatility W1 | Sq-return AC L1 | Flattened sq-return AC L1 | Drawdown W1 | Terminal W1 |
|---| --- :| --- :| --- :| --- :| --- :| --- :| --- :|
| Frequency alpha 0.1 discrete | 0.38490504 | 0.01312830 | 0.00105531 | 0.03702611 | 0.05022892 | 0.00993632 | 0.01201044 |
| Promoted baseline discrete | 0.27934083 | 0.00767375 | 0.00118835 | 0.04129972 | 0.12882016 | 0.01050232 | 0.00981713 |
| Hidden128 top-k20 discrete | 0.22907834 | 0.00717505 | 0.00125777 | 0.06088475 | 0.07886228 | 0.00891025 | 0.00450928 |
| Hidden128 top-k40 candidate | 0.25358254 | 0.00721005 | 0.00130064 | 0.06439025 | 0.12390518 | 0.01013566 | 0.00449585 |
| Continuous BetaCVAE | 0.15442121 | 0.00878550 | 0.00063360 | 0.02946163 | 0.02914374 | 0.00766744 | 0.00905099 |

The frequency tokenizer improves the promoted baseline on volatility W1 and both squared-return
autocorrelation metrics, especially the flattened squared-return autocorrelation. It also slightly
improves drawdown W1 relative to the promoted baseline. However, it materially worsens MMD, SWD,
and terminal-return W1. Against hidden128, the frequency prior improves squared-return
autocorrelation and volatility W1, but loses clearly on MMD, SWD, terminal-return W1, and usually
drawdown.

The continuous BetaCVAE remains ahead on MMD, volatility W1, squared-return autocorrelation,
drawdown W1, and returns W1. Its terminal-return W1 is also better than the frequency prior in this
paper-style run.

## Paper-Style VIX Buckets

| Bucket | Discrete MMD | Discrete SWD | Volatility W1 | Sq-return AC L1 | Drawdown W1 | Terminal W1 |
|---| --- :| --- :| --- :| --- :| --- :| --- :|
| very_low | 0.29783794 | 0.01566142 | 0.00120966 | 0.03115140 | 0.01381801 | 0.00829731 |
| low | 0.62183762 | 0.01569005 | 0.00069234 | 0.06620464 | 0.00776177 | 0.01535573 |
| mid | 0.63282794 | 0.02026321 | 0.00143847 | 0.06575284 | 0.01521172 | 0.01910943 |
| high | 0.44651634 | 0.02007722 | 0.00154318 | 0.05566245 | 0.01702195 | 0.01970866 |
| very_high | 0.23522997 | 0.00870749 | 0.00103304 | 0.05324206 | 0.00900120 | 0.00911046 |

The bucket view shows the same mixed result. Volatility matching is strongest in the low-VIX
bucket, but the low-to-high VIX middle range has large MMD and terminal-return errors.

## Decision

Decision: run the alpha 0.2 frequency-tokenizer prior for comparison before rejecting the frequency
path. Alpha 0.1 demonstrates the intended improvement on volatility W1 and squared-return
autocorrelation, so the causal frequency decomposition is not a dead end. It does not clear the
overall prior-quality gate because MMD, SWD, and terminal-return guardrails regress too much.

Do not move to grouped tokenizers yet. The next minimal comparison should be alpha 0.2 with the
same additive VIX-only prior and the same sampling settings. If alpha 0.2 keeps the
volatility/autocorrelation gains while reducing MMD and terminal-return drift, continue the
frequency tokenizer. If not, reject the current joint EMA frequency-tokenizer path or run a small
alpha grid before considering grouped tokenizers.
