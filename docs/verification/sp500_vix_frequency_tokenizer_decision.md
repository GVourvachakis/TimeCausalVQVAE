# S&P500/VIX Frequency Tokenizer Decision

Status: decision note only. This document does not implement code, train models, or change the
promoted baseline.

## Motivation

The causal EMA frequency-tokenizer branch was opened to address a narrow residual failure mode in
the S&P500/VIX discrete pipeline. The hidden128 VQ variant improves many path-level metrics, but
it still leaves weaknesses in volatility W1 and squared-return autocorrelation. Those diagnostics
are high-frequency or residual-path properties: they depend on local shocks, absolute returns, and
volatility clustering rather than only on level or terminal-return behaviour.

The architectural hypothesis was therefore that a one-stream VQ tokenizer may entangle slower path
movement with local shocks. A causal low/high split could expose residual variation to the
tokenizer while preserving the TC-VAE no-anticipation rule.

## Evidence

### Decomposition No-Leakage

The deterministic EMA decomposition smoke check passed for both `[batch, time]` and
`[batch, time, 1]` tensors. With `alpha=0.2`, cutoff `29`, batch size `8`, and length `60`,
future perturbations after the cutoff produced zero prefix difference in both low and high
components. Reconstruction by `low + high` also matched the original path with zero maximum
difference under the script tolerance.

This validates the deterministic decomposition utility. It does not by itself validate the
tokenizer or prior, but it clears the first no-leakage gate.

### Tokenizer Alpha Grid

The tokenizer-only ablation compared EMA `alpha` values `0.1`, `0.2`, and `0.5`, all using the
joint low/high one-code interface with standard VQ.

| EMA alpha | Original-path L1 | Original-path L2 | Volatility error | Active codes | Perplexity |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.00669039 | 0.00808084 | 0.00067805 | 59 | 47.1625 |
| 0.2 | 0.01165698 | 0.01289017 | 0.00075628 | 60 | 47.7753 |
| 0.5 | 0.00681011 | 0.00857277 | 0.00080509 | 26 | 18.0589 |

Alpha `0.1` was the strongest tokenizer candidate: it gave the lowest composed original-path
reconstruction error, the lowest composed volatility reconstruction error, and broad code usage.
Alpha `0.2` had slightly broader active-code count, but materially worse scalar reconstruction.
Alpha `0.5` had narrow effective vocabulary and was not suitable for the first prior run.

Relative to tokenizer-only baselines, alpha `0.1` improved over the promoted standard VQ on
reconstruction, volatility error, active-code count, and perplexity. Relative to hidden128, it had
slightly worse original-path reconstruction but better volatility reconstruction and broader code
usage.

### Alpha01 Geometry

Alpha `0.1` token extraction produced a standard one-token stream:

| Split | Token shape | Label shape | Data shape |
|---|---:|---:|---:|
| train | `[2457, 60]` | `[2457, 1]` | `[2457, 60, 2]` |
| eval | `[2457, 60]` | `[2457, 1]` | `[2457, 60, 2]` |

Combined geometry diagnostics reported full codebook usage: `64 / 64` active codes, perplexity
`53.01815796`, and entropy `3.97063446`. VIX-bucket usage increased from 60 active codes in the
very-low bucket to 64 in the very-high bucket, with very-high perplexity `53.69179153`.

This clears the geometry gate. The alpha `0.1` frequency tokenizer has a usable one-code
interface, strong code usage, and VIX-sensitive latent structure.

### Alpha01 Prior Quality

The additive VIX-only causal AR prior was trained on alpha `0.1` tokens without architecture or
objective changes. The best checkpoint was epoch 100 with eval cross-entropy `1.05693870`, eval
accuracy `0.59681862`, and eval perplexity `2.88918605`.

The initial paper-style run used temperature `0.8` and top-k `40`:

| Model | MMD | SWD | Volatility W1 | Sq-return AC L1 | Flat sq-return AC L1 | Drawdown W1 | Terminal W1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Alpha01, temp 0.8 top-k 40 | 0.38490504 | 0.01312830 | 0.00105531 | 0.03702611 | 0.05022892 | 0.00993632 | 0.01201044 |
| Promoted baseline | 0.27934083 | 0.00767375 | 0.00118835 | 0.04129972 | 0.12882016 | 0.01050232 | 0.00981713 |
| Hidden128 top-k20 | 0.22907834 | 0.00717505 | 0.00125777 | 0.06088475 | 0.07886228 | 0.00891025 | 0.00450928 |
| Continuous BetaCVAE | 0.15442121 | 0.00878550 | 0.00063360 | 0.02946163 | 0.02914374 | 0.00766744 | 0.00905099 |

This run confirmed the motivation: alpha `0.1` improved volatility W1 and squared-return
autocorrelation relative to the promoted baseline, and it improved those residual metrics relative
to hidden128 in several cases. It did not clear the replacement gate because MMD, SWD, and
terminal-return W1 regressed materially.

### Alpha01 Sampling Ablation

The alpha `0.1` sampling ablation compared temperatures `0.6`, `0.8`, and `1.0` with unrestricted
top-k, top-k `20`, and top-k `40`. The scalar paper-style decision setting was
`temperature=1.0`, unrestricted top-k.

| Model | MMD | SWD | Volatility W1 | Sq-return AC L1 | Flat sq-return AC L1 | Drawdown W1 | Terminal W1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Alpha01 selected, temp 1.0 top-k none | 0.34736192 | 0.01155593 | 0.00123859 | 0.03023081 | 0.06069550 | 0.00815493 | 0.00805083 |
| Alpha01 prior-quality, temp 0.8 top-k 40 | 0.38490504 | 0.01312830 | 0.00105531 | 0.03702611 | 0.05022892 | 0.00993632 | 0.01201044 |
| Promoted baseline | 0.27934083 | 0.00767375 | 0.00118835 | 0.04129972 | 0.12882016 | 0.01050232 | 0.00981713 |
| Hidden128 top-k20 | 0.22907834 | 0.00717505 | 0.00125777 | 0.06088475 | 0.07886228 | 0.00891025 | 0.00450928 |

Sampling improved alpha `0.1` on MMD, SWD, within-path squared-return autocorrelation, drawdown
W1, and terminal-return W1. It gave up some volatility W1 and flattened squared-return
autocorrelation relative to the `0.8`/top-k `40` run. The selected setting is still worse than
the promoted baseline and hidden128 on MMD and SWD, and worse than hidden128 on terminal-return
W1. It is close to the promoted baseline on volatility W1 and better on within-path
squared-return autocorrelation, drawdown W1, and terminal-return W1.

## Comparison Summary

The promoted standard VQ remains the public baseline because it has broad geometry, simple
one-code causal prior compatibility, and stronger aggregate MMD/SWD than alpha `0.1`.

Hidden128 remains the stronger discrete candidate for broad paper-style quality. It is clearly
better than alpha `0.1` on MMD, SWD, and terminal-return W1, although alpha `0.1` improves the
specific residual metrics that motivated this branch.

The continuous BetaCVAE remains ahead of the frequency prior on MMD, volatility W1, squared-return
autocorrelation, drawdown W1, and returns W1 in the recorded paper-style comparison. Its role is
still a continuous reference rather than a replacement for the discrete tokenizer path.

The causal EMA frequency tokenizer therefore has a real signal but not a promotion result. It
improves the intended residual diagnostics, but the joint one-code alpha `0.1` setup degrades too
many broad distribution guardrails to replace the promoted baseline or hidden128.

## Decision

Decision: continue alpha tuning narrowly by training the alpha `0.2` prior, then stop the joint
EMA branch unless alpha `0.2` materially improves the broad guardrails.

Detailed decisions:

- Continue alpha tuning: yes, but only for the already-trained alpha `0.2` tokenizer candidate.
- Train alpha02 prior: yes. Use the same additive VIX-only causal AR prior and the same
  paper-style sampling grid.
- Promote alpha01 as research candidate: no. Alpha `0.1` is a diagnostic candidate, not a
  promotion candidate.
- Reject frequency tokenizer now: no. The residual-metric gains justify one alpha `0.2` prior
  comparison.
- Move to GroupedRVQ now: no.
- Move to TimeVQVAE-style two-token decomposition now: no.

The alpha `0.2` prior should be evaluated against the selected alpha `0.1` sampling setting,
the promoted baseline, hidden128 top-k20, and the continuous BetaCVAE reference. The acceptance
bar is strict: it must retain residual improvements while reducing MMD/SWD and terminal-return
drift. If alpha `0.2` does not do that, reject the joint one-code EMA frequency tokenizer.

## If The Frequency Tokenizer Fails

If alpha `0.2` fails the gate, do not jump directly to MGVQ. MGVQ is too large a step from the
current evidence and would add unnecessary modelling degrees of freedom before the decomposition
question is settled.

The next branch should be one of:

1. Separate low/high tokenizers with a hierarchical causal prior:
   `p(low_t | past, condition)` followed by `p(high_t | past, low_t, condition)`.
   This is the more TimeVQVAE-style route and directly tests whether low and residual information
   need distinct token streams.
2. GroupedResidualVQ with strict multi-code diagnostics. This should be used only if the goal is
   to keep a single encoder while allowing a coarse/detail code structure, and it must include
   multi-code usage, transition, run-length, and prior-sampling diagnostics before any promotion
   claim.

Both branches must preserve causal filtering and causal prior factorisation. Bidirectional
filtering, bidirectional priors, cross-attention, signatures, diffusion, and MGVQ remain
non-goals for this decision stage.
