# S&P500/VIX Frequency Alpha 0.2 Prior Quality

## Scope

This note records the additive VIX-only causal autoregressive prior trained on tokens from the
alpha 0.2 causal EMA frequency tokenizer. The prior architecture, objective, token interface, and
sampling conventions were kept fixed relative to the promoted additive prior.

Non-goals remained in force: no model-code changes, no signatures, no GroupedRVQ, no MGVQ, no
diffusion, no cross-attention, and no new objectives.

## Inputs

- Prior config:
  `configs/experiments/sp500_vix_causal_token_prior_freq_ema_alpha02.yaml`
- Tokenizer checkpoint:
  `outputs/sp500_vix_discrete/frequency_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_freq_ema_alpha02_seed0`
- Token data:
  `outputs/sp500_vix_discrete/token_prior/freq_ema_alpha02_tokens`
- Prior output:
  `outputs/sp500_vix_discrete/token_prior/freq_ema_alpha02/sp500_vix_causal_token_prior_freq_ema_alpha02_seed0`
- Best prior checkpoint:
  `outputs/sp500_vix_discrete/token_prior/freq_ema_alpha02/sp500_vix_causal_token_prior_freq_ema_alpha02_seed0/best_model`
- Best-checkpoint evaluation:
  `outputs/sp500_vix_discrete/token_prior/freq_ema_alpha02/evaluation_best`
- Paper-style default evaluation:
  `outputs/sp500_vix_discrete/paper_style_freq_ema_alpha02_temp08_topk40`
- Paper-style sampling grid:
  `outputs/sp500_vix_discrete/paper_style_freq_ema_alpha02_temp*_topk*`

The decoded prior evaluation composed the tokenizer's `[low, high]` outputs back to scalar paths
before computing path diagnostics. The reported decoded path shape is `[1000, 60, 1]`.

## W&B

The required W&B-profile run failed during run initialisation with a network `CommError` timeout
after 90 seconds. The same training command was rerun with `--no-wandb`, as requested. No W&B URL
is available for this run.

## Token Likelihood

| Metric | Value |
|---|---:|
| Best epoch | 100 |
| Best eval cross-entropy | 1.19572794 |
| Best eval accuracy | 0.53785104 |
| Best eval perplexity | 3.32426902 |
| Final eval cross-entropy | 1.19572794 |
| Runtime seconds | 1285.088 |

The best checkpoint is the final epoch. Token likelihood is weaker than the alpha 0.1 prior
(`eval_cross_entropy=1.05693870`, `eval_perplexity=2.88918605`), even though the downstream
composed-path diagnostics improve.

## Decoded Token-Prior Evaluation

Sampling used `n_sample=1000`, seed 99, temperature 0.8, and top-k 40.

| Metric | Value |
|---|---:|
| Sampled active codes | 64 / 64 |
| Sampled token perplexity | 42.20855331 |
| Real token perplexity | 44.06784439 |
| Marginal code L1 | 0.16560000 |
| Transition matrix L1 | 0.29789194 |
| Run-length distance | 0.55757147 |
| MMD | 0.28922948 |
| SWD | 0.00982244 |
| Terminal-return W1 | 0.00667941 |
| Volatility W1 | 0.00122682 |

Alpha 0.2 improves decoded MMD, active-code coverage, marginal-code L1, run-length distance, and
terminal-return W1 relative to the alpha 0.1 default evaluation. Transition L1 is slightly worse
than alpha 0.1, but not enough to dominate the path-level result.

## VIX-Bucket Diagnostics

These are token-prior decoded diagnostics at temperature 0.8 and top-k 40.

| Bucket | n | MMD | Volatility W1 | Terminal W1 | Active codes | Token perplexity |
|---|---:|---:|---:|---:|---:|---:|
| very_low | 200 | 0.65219975 | 0.00138577 | 0.01765098 | 59 | 30.76552391 |
| low | 200 | 0.28952196 | 0.00091836 | 0.00740334 | 58 | 34.52880096 |
| mid | 200 | 0.18336831 | 0.00175372 | 0.00710226 | 61 | 37.87649536 |
| high | 200 | 0.42648956 | 0.00119347 | 0.01052874 | 61 | 43.32655334 |
| very_high | 200 | 0.34948796 | 0.00136194 | 0.00506318 | 64 | 51.01077271 |

The very-low VIX bucket remains the weakest MMD bucket. Code usage rises with VIX level, and the
very-high bucket keeps full codebook coverage.

## Paper-Style Default Metrics

Default frequency sampling used temperature 0.8 and top-k 40. Diagnostics are computed after
composing decoded `[low, high]` channels back to scalar paths.

| Model | MMD | SWD | Volatility W1 | Sq-return AC L1 | Flat sq-return AC L1 | Drawdown W1 | Terminal W1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frequency alpha 0.2, temp 0.8 top-k 40 | 0.29730234 | 0.00940700 | 0.00113100 | 0.03540378 | 0.02522561 | 0.00707990 | 0.00666096 |
| Frequency alpha 0.1, temp 0.8 top-k 40 | 0.38490504 | 0.01312830 | 0.00105531 | 0.03702611 | 0.05022892 | 0.00993632 | 0.01201044 |
| Promoted baseline, temp 0.8 top-k 40 | 0.27934083 | 0.00767375 | 0.00118835 | 0.04129972 | 0.12882016 | 0.01050232 | 0.00981713 |
| Hidden128, temp 0.8 top-k 20 | 0.22907834 | 0.00717505 | 0.00125777 | 0.06088475 | 0.07886228 | 0.00891025 | 0.00450928 |
| Continuous BetaCVAE | 0.15442121 | 0.00878550 | 0.00063360 | 0.02946163 | 0.02914374 | 0.00766744 | 0.00905099 |

Alpha 0.2 is a clear improvement over alpha 0.1 on MMD, SWD, flattened squared-return
autocorrelation, drawdown W1, and terminal-return W1. Against the promoted baseline, it is weaker
on MMD and SWD but stronger on volatility W1, squared-return autocorrelation, drawdown W1, and
terminal-return W1. Against hidden128, it is weaker on MMD, SWD, and terminal W1, but materially
stronger on the targeted squared-return autocorrelation metrics and slightly stronger on volatility
W1 and drawdown W1.

## Paper-Style Sampling Ablation

Guardrail score is `MMD + SWD + volatility W1 + terminal W1`, computed on composed scalar paths.

| Temp | Top-k | Guardrail score | MMD | SWD | Volatility W1 | Sq-return AC L1 | Flat sq-return AC L1 | Drawdown W1 | Terminal W1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6 | none | 0.45340659 | 0.43103704 | 0.01090122 | 0.00208134 | 0.03342787 | 0.06026680 | 0.01468031 | 0.00938699 |
| 0.6 | 20 | 0.45237799 | 0.42856660 | 0.01121485 | 0.00211427 | 0.03293128 | 0.05757030 | 0.01498734 | 0.01048227 |
| 0.6 | 40 | 0.41923460 | 0.39838305 | 0.01027701 | 0.00207679 | 0.03543849 | 0.07578422 | 0.01480055 | 0.00849775 |
| 0.8 | none | 0.31656290 | 0.29887241 | 0.00982954 | 0.00118184 | 0.03223158 | 0.05420407 | 0.00664260 | 0.00667911 |
| 0.8 | 20 | 0.32010960 | 0.30190790 | 0.00978038 | 0.00110980 | 0.03228358 | 0.02689342 | 0.00598835 | 0.00731153 |
| 0.8 | 40 | 0.31450129 | 0.29730234 | 0.00940700 | 0.00113100 | 0.03540378 | 0.02522561 | 0.00707990 | 0.00666096 |
| 1.0 | none | 0.32533289 | 0.29724580 | 0.01305861 | 0.00109388 | 0.03339342 | 0.05478247 | 0.01035609 | 0.01393459 |
| 1.0 | 20 | 0.31739378 | 0.29125270 | 0.01301961 | 0.00120758 | 0.03082476 | 0.08207801 | 0.01096551 | 0.01191389 |
| 1.0 | 40 | 0.33761020 | 0.31043744 | 0.01303795 | 0.00109758 | 0.03043181 | 0.08161082 | 0.01058256 | 0.01303722 |

The best scalar-path guardrail score is temperature 0.8 with top-k 40. Temperature 1.0 with top-k
20 gives the best MMD and strongest within-path squared-return autocorrelation, but it worsens SWD,
flattened squared-return autocorrelation, drawdown W1, and terminal-return W1. The selected
alpha 0.2 setting is therefore the default frequency setting: `temperature=0.8`, `top_k=40`.

## Paper-Style VIX Buckets

These bucket diagnostics use the selected `temperature=0.8`, `top_k=40` setting.

| Bucket | Discrete MMD | Discrete SWD | Volatility W1 | Sq-return AC L1 | Drawdown W1 | Terminal W1 |
|---|---:|---:|---:|---:|---:|---:|
| very_low | 0.64249372 | 0.01646713 | 0.00131597 | 0.03867964 | 0.01206428 | 0.01690392 |
| low | 0.28781700 | 0.00833730 | 0.00110960 | 0.05672179 | 0.00847216 | 0.00581335 |
| mid | 0.18789494 | 0.00666971 | 0.00160647 | 0.06933151 | 0.01331414 | 0.00118935 |
| high | 0.43620110 | 0.01411938 | 0.00147163 | 0.05557476 | 0.01120624 | 0.01365788 |
| very_high | 0.36143661 | 0.01360902 | 0.00074089 | 0.05372540 | 0.00629242 | 0.00983839 |

Bucket quality is uneven. The middle bucket has the best MMD and terminal-return W1, while the
very-low and high buckets dominate the remaining MMD weakness.

## Comparison Summary

| Model / setting | MMD | SWD | Volatility W1 | Sq-return AC L1 | Flat sq-return AC L1 | Drawdown W1 | Terminal W1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Alpha02 selected, temp 0.8 top-k 40 | 0.29730234 | 0.00940700 | 0.00113100 | 0.03540378 | 0.02522561 | 0.00707990 | 0.00666096 |
| Alpha02 best-MMD, temp 1.0 top-k 20 | 0.29125270 | 0.01301961 | 0.00120758 | 0.03082476 | 0.08207801 | 0.01096551 | 0.01191389 |
| Alpha01 selected, temp 1.0 top-k none | 0.34736192 | 0.01155593 | 0.00123859 | 0.03023081 | 0.06069550 | 0.00815493 | 0.00805083 |
| Promoted baseline, temp 0.8 top-k 40 | 0.27934083 | 0.00767375 | 0.00118835 | 0.04129972 | 0.12882016 | 0.01050232 | 0.00981713 |
| Hidden128, temp 0.8 top-k 20 | 0.22907834 | 0.00717505 | 0.00125777 | 0.06088475 | 0.07886228 | 0.00891025 | 0.00450928 |
| Continuous BetaCVAE | 0.15442121 | 0.00878550 | 0.00063360 | 0.02946163 | 0.02914374 | 0.00766744 | 0.00905099 |

Alpha 0.2 is the strongest joint EMA frequency result so far. It materially improves the alpha 0.1
branch and addresses the hidden128 squared-return autocorrelation regression. It does not beat the
hidden128 prior on broad distribution metrics, and it remains behind the continuous BetaCVAE on the
main market-diagnostic guardrails.

## Decision

Decision: continue the frequency branch as a research candidate, with alpha 0.2 as the current
joint-EMA setting. It should not replace hidden128 as the broad promoted prior candidate because
hidden128 remains substantially stronger on MMD and SWD. It should, however, remain active because
it improves the targeted volatility-clustering proxy: squared-return autocorrelation drops from
0.06088475 for hidden128 to 0.03540378 at the selected alpha 0.2 setting, while volatility W1 and
drawdown W1 also improve slightly.

Recommended next step: run a small joint EMA alpha grid around alpha 0.2 only if the goal is to
recover MMD/SWD while preserving the squared-return autocorrelation gain. If that grid does not
close the broad-distribution gap, move to separate low/high tokenizers with a causal hierarchical
prior. Do not jump directly to MGVQ. GroupedResidualVQ should remain a later branch and should only
be attempted with strict multi-code diagnostics.
