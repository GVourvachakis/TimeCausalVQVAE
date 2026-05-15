# S&P500/VIX Joint Frequency Tokenizer Final Decision

Status: final decision note for the joint one-code causal EMA frequency-tokenizer path. This
document does not implement code, train models, or alter promoted baseline configs.

## Motivation

The frequency-tokenizer branch was opened to address a focused residual weakness in the
S&P500/VIX discrete path. Hidden128 improves many broad path metrics, but its generated samples
still regress the volatility-clustering diagnostics, especially volatility W1 and squared-return
autocorrelation. Those metrics depend on local shocks and residual dynamics rather than only on
path level or terminal return.

The working hypothesis was that a standard one-stream VQ tokenizer may entangle slower trend and
local high-frequency shocks in the same token. A causal low/high split could make residual
variation explicit while preserving the TC-VAE no-anticipation constraint.

## Method

The tested method used deterministic causal EMA decomposition:

```text
low_0 = x_0
low_t = alpha * x_t + (1 - alpha) * low_{t-1}
high_t = x_t - low_t
```

The tokenizer input was the joint two-channel sequence `[low, high]`. One standard vector VQ
tokenizer was trained on that two-channel input, and it emitted one code per time step:

```text
x:          [batch, length, 1]
[low, high]: [batch, length, 2]
tokens:     [batch, length]
```

Generation used the existing additive VIX-only causal AR prior:

```text
p(z_t | z_<t, VIX)
```

Decoded frequency channels were composed back to scalar S&P500 paths by `low_hat + high_hat`
before paper-style market diagnostics. The design intentionally did not add separate tokenizers,
a hierarchical prior, GroupedRVQ, MGVQ, signatures, diffusion, cross-attention, new objectives,
or bidirectional filtering.

The deterministic decomposition no-leakage smoke passed for both `[batch, time]` and
`[batch, time, 1]` tensors. Future perturbations after cutoff left low/high prefixes unchanged,
and `low + high` reconstructed the original path under the script tolerance.

## Alpha Comparison

### Tokenizer Metrics

The tokenizer-only ablation compared EMA `alpha` values `0.1`, `0.2`, and `0.5`, all with the
joint one-code interface.

| EMA alpha | Original-path L1 | Original-path L2 | Volatility error | Active codes | Perplexity |
|--- :| --- :| --- :| --- :| --- :| --- :|
| 0.1 | 0.00669039 | 0.00808084 | 0.00067805 | 59 | 47.1625 |
| 0.2 | 0.01165698 | 0.01289017 | 0.00075628 | 60 | 47.7753 |
| 0.5 | 0.00681011 | 0.00857277 | 0.00080509 | 26 | 18.0589 |

Alpha `0.1` was the best tokenizer-only candidate by composed reconstruction and volatility
reconstruction. Alpha `0.2` had slightly broader active-code count but worse scalar
reconstruction. Alpha `0.5` had weak code usage and did not advance.

### Geometry Metrics

Token extraction for both alpha `0.1` and alpha `0.2` preserved the required one-token stream:
train and eval indices had shape `[2457, 60]`, labels had shape `[2457, 1]`, and transformed data
had shape `[2457, 60, 2]`.

| Tokenizer | Active codes | Perplexity | Entropy | Very-low bucket | Very-high bucket |
|--- | --- :| --- :| --- :| --- :| --- :|
| EMA alpha 0.1 | 64 / 64 | 53.01815796 | 3.97063446 | 60 / 37.3818 | 64 / 53.6918 |
| EMA alpha 0.2 | 64 / 64 | 54.60705566 | 4.00016308 | 61 / 40.1668 | 64 / 54.7603 |

Both candidates cleared the geometry gate. Alpha `0.2` had slightly stronger global and
VIX-bucket code usage, which justified training the alpha `0.2` prior despite its weaker
tokenizer-only reconstruction.

### Prior Metrics

Both priors used the same additive VIX-only causal AR architecture and 100-epoch schedule.

| Alpha | Best eval CE | Best eval accuracy | Best eval perplexity | Runtime seconds |
|--- :| --- :| --- :| --- :| --- :|
| 0.1 | 1.05693870 | 0.59681862 | 2.88918605 | 1213.060 |
| 0.2 | 1.19572794 | 0.53785104 | 3.32426902 | 1285.088 |

Alpha `0.1` was easier for the token prior to model. Alpha `0.2` nevertheless gave better
composed-path generation quality, so token likelihood alone was not a sufficient selector.

Default decoded prior evaluation used temperature `0.8`, top-k `40`, `n_sample=1000`, and seed
`99`.

| Alpha | Active sampled codes | Sampled perplexity | MMD | SWD | Volatility W1 | Terminal W1 |
|--- :| --- :| --- :| --- :| --- :|--- :| --- :|
| 0.1 | 61 / 64 | 38.84451675 | 0.42002434 | 0.01306464 | 0.00112879 | 0.01346600 |
| 0.2 | 64 / 64 | 42.20855331 | 0.28922948 | 0.00982244 | 0.00122682 | 0.00667941 |

Alpha `0.2` is clearly stronger after prior sampling and decoding: it improves active sampled
codes, sampled perplexity, MMD, SWD, and terminal-return W1, with only a small volatility-W1
regression in this decoded-prior table.

### Sampling Ablation

Paper-style sampling ablations composed decoded frequency channels back to scalar paths before
market diagnostics. The grid used temperatures `0.6`, `0.8`, and `1.0`, with unrestricted
top-k, top-k `20`, and top-k `40`.

| Candidate | Selected setting | Guardrail score | MMD | SWD | Volatility W1 | Sq-return AC L1 | Flat sq-return AC L1 | Drawdown W1 | Terminal W1 |
|--- | --- | --- :| --- :| --- :| --- :| --- :| --- :| --- :| --- :|
| Alpha 0.1 | temp 1.0, top-k none | 0.36820728 | 0.34736192 | 0.01155593 | 0.00123859 | 0.03023081 | 0.06069550 | 0.00815493 | 0.00805083 |
| Alpha 0.2 | temp 0.8, top-k 40 | 0.31450129 | 0.29730234 | 0.00940700 | 0.00113100 | 0.03540378 | 0.02522561 | 0.00707990 | 0.00666096 |

The guardrail score is `MMD + SWD + volatility W1 + terminal W1`. Alpha `0.2` is the better
joint EMA candidate by the scalar-path guardrail score and by most broad metrics. Alpha `0.1`
has the best within-path squared-return autocorrelation among the selected settings, but it loses
too much on MMD, SWD, flattened squared-return autocorrelation, and terminal-return W1.

## Baseline Comparison

| Model / setting | MMD | SWD | Volatility W1 | Sq-return AC L1 | Flat sq-return AC L1 | Drawdown W1 | Terminal W1 |
|--- | --- :| --- :| --- :| --- :| --- :| --- :| --- :|
| Joint EMA alpha 0.2, temp 0.8 top-k 40 | 0.29730234 | 0.00940700 | 0.00113100 | 0.03540378 | 0.02522561 | 0.00707990 | 0.00666096 |
| Joint EMA alpha 0.1, temp 1.0 top-k none | 0.34736192 | 0.01155593 | 0.00123859 | 0.03023081 | 0.06069550 | 0.00815493 | 0.00805083 |
| Promoted baseline, temp 0.8 top-k 40 | 0.27934083 | 0.00767375 | 0.00118835 | 0.04129972 | 0.12882016 | 0.01050232 | 0.00981713 |
| Hidden128, temp 0.8 top-k 20 | 0.22907834 | 0.00717505 | 0.00125777 | 0.06088475 | 0.07886228 | 0.00891025 | 0.00450928 |
| Continuous BetaCVAE | 0.15442121 | 0.00878550 | 0.00063360 | 0.02946163 | 0.02914374 | 0.00766744 | 0.00905099 |

Against the promoted baseline, alpha `0.2` improves volatility W1, squared-return
autocorrelation, flattened squared-return autocorrelation, drawdown W1, and terminal-return W1.
It still regresses MMD and SWD.

Against hidden128, alpha `0.2` addresses the targeted residual weakness: squared-return
autocorrelation improves from `0.06088475` to `0.03540378`, flattened squared-return
autocorrelation improves from `0.07886228` to `0.02522561`, volatility W1 improves slightly, and
drawdown W1 improves. However, hidden128 remains much stronger on MMD, SWD, and terminal-return
W1.

Against the continuous BetaCVAE reference, alpha `0.2` remains behind on MMD, SWD, volatility W1,
within-path squared-return autocorrelation, and drawdown W1. It is better on terminal-return W1
and flattened squared-return autocorrelation in this comparison, but not enough to change the
discrete-path decision.

The requested `docs/verification/sp500_vix_standard_vq_tuning_final_decision.md` file was not
present in this checkout, so the promoted-baseline comparison above uses the promoted-baseline
metrics already recorded in the frequency decision and prior-quality notes.

## Decision

Final decision: do not promote the joint one-code causal EMA frequency tokenizer as the main
S&P500/VIX discrete replacement. Promote alpha `0.2` only as the research candidate for this
branch, with selected sampling `temperature=0.8` and `top_k=40`.

Detailed decisions:

- Promote alpha `0.1`: no. It has strong tokenizer reconstruction and prior likelihood, but it
  remains too weak on MMD, SWD, and terminal-return guardrails.
- Promote alpha `0.2` as a research candidate: yes. It is the best joint EMA result and directly
  improves the residual volatility-clustering diagnostics.
- Promote alpha `0.2` as the broad discrete replacement: no. Hidden128 remains stronger on MMD,
  SWD, and terminal-return W1.
- Continue broad alpha tuning for the joint one-code EMA path: no. The branch has established the
  useful signal and the one-code limitation; further scalar alpha tuning is unlikely to solve the
  broad MMD/SWD gap by itself.
- Move to separate low/high tokenizers: yes, as the next decomposition branch if this line of
  research continues.
- Move to GroupedResidualVQ: only if the next priority is a multi-code interface rather than a
  clean TimeVQVAE-style decomposition test.

## Rejection Rationale

The joint EMA branch is not rejected because it failed entirely. It is rejected as the main
promotion path because the one-code joint representation trades one failure mode for another.
It improves the intended volatility-clustering proxies, especially squared-return
autocorrelation, but it cannot match hidden128 on broad distributional quality.

The likely limitation is interface capacity and factorisation. A single token must encode both
low-frequency path state and high-frequency residual shock at each time step. That preserves the
prior contract, but it may still entangle the components the decomposition was meant to separate.
The alpha `0.2` result is therefore evidence for the decomposition idea, not evidence that a
joint one-code interface is sufficient.

## Recommended Next Branch

The next branch should be separate low/high tokenizers with a hierarchical causal prior:

```text
p(low_t | low_<t, high_<t, VIX)
p(high_t | low_<=t, high_<t, VIX)
```

This directly tests whether low and residual paths need separate codebooks and an explicit
same-time causal factorisation. It should keep the deterministic causal EMA transform first, then
only later consider learned causal filters if the fixed split proves too rigid.

GroupedResidualVQ is the alternative if the priority is a multi-code interface inside one
encoder. That path should require strict multi-code diagnostics before any promotion claim:
per-code active usage, per-code perplexity, same-time code compatibility, transition structure,
run-length diagnostics, VIX-bucket usage, and composed-path paper-style metrics.

Do not jump directly to MGVQ. Do not add signatures, diffusion, cross-attention, bidirectional
filters, bidirectional priors, or new objectives for this decision stage.

## Selected Research Candidate

For future comparisons, the retained joint EMA research candidate is:

```text
tokenizer: outputs/sp500_vix_discrete/frequency_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_freq_ema_alpha02_seed0
token data: outputs/sp500_vix_discrete/token_prior/freq_ema_alpha02_tokens
prior: outputs/sp500_vix_discrete/token_prior/freq_ema_alpha02/sp500_vix_causal_token_prior_freq_ema_alpha02_seed0/best_model
sampling: temperature=0.8, top_k=40, n_sample=1000, seed=99
```

This candidate should be cited as a residual-diagnostics research baseline, not as a promoted
S&P500/VIX discrete model.
