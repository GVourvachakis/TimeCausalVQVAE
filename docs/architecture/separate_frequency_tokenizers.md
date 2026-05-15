# Separate Frequency Tokenizers

Status: architecture plan only. This document does not implement code, train models, or change
the promoted S&P500/VIX baseline.

## Motivation

The joint causal EMA frequency tokenizer established that deterministic low/high decomposition is
useful for the residual failure mode. With `alpha = 0.2`, it improved volatility W1,
squared-return autocorrelation, flattened squared-return autocorrelation, drawdown W1, and
terminal-return W1 relative to the promoted standard baseline. It also addressed the hidden128
variant's targeted squared-return autocorrelation weakness.

The same result exposed a limitation. The joint EMA tokenizer still regressed MMD and SWD
relative to the promoted baseline and hidden128. A single token per time step may still entangle
the low-frequency path level with the high-frequency residual shock. This preserves the old prior
interface, but it may force one codebook to represent two different objects: smooth level state
and local volatility burst.

Separate low/high tokenizers test the next narrow hypothesis: the low and high EMA components may
need separate codebooks before the prior can model them cleanly. This branch should therefore
separate tokenisation first, then add only the causal prior structure required to sample the two
streams coherently.

## Deterministic Decomposition

Reuse the causal EMA decomposition from the joint frequency branch:

```text
low_0 = x_0
low_t = alpha * x_t + (1 - alpha) * low_{t-1}
high_t = x_t - low_t
```

The decomposition is deterministic and causal: `low_t` and `high_t` depend only on `x_<=t`.
This rules out centred filters, full-window Fourier transforms, bidirectional smoothing, and
learned filters at this stage.

Start with `alpha = 0.2` because it is the strongest joint EMA research candidate after prior
sampling and composed-path evaluation. Keep `alpha = 0.1` only as a secondary reference, since it
had better tokenizer-only reconstruction and prior likelihood but weaker broad decoded-path
guardrails.

## Tokenizer Design

Train two independent standard VQ tokenizers:

```text
low tokenizer input:  [batch, time, 1]
high tokenizer input: [batch, time, 1]
low indices:          [batch, time]
high indices:         [batch, time]
```

The low tokenizer models the causal EMA level component. The high tokenizer models the residual
component. Each tokenizer has its own encoder, decoder, and codebook. The initial capacity should
match the joint EMA tokenizer unless a specific underfitting diagnostic requires more capacity:

- standard vector VQ;
- 64 codes for the low tokenizer;
- 64 codes for the high tokenizer;
- `codebook_dim = 16`;
- the same causal convolutional encoder/decoder shape used by the standard tokenizer;
- `condition_dim = 1` for the VIX condition where the existing tokenizer path uses it.

Use hidden128 only as a capacity follow-up if the baseline-capacity separate tokenizers underfit
one or both components. Hidden128 should not be the first separate-tokenizer run, because the
initial question is whether separate codebooks help before capacity is increased.

## Prior Design

Separate token streams require a causal hierarchical prior:

```text
p(low_t | low_<t, high_<t, VIX)
p(high_t | low_<=t, high_<t, VIX)
```

The low head samples the low-frequency token for time `t` using only past low tokens, past high
tokens, and VIX. The high head then samples the residual token for the same time step using past
low tokens, past high tokens, VIX, and the just-sampled `low_t`.

This factorisation is causal because `high_t` may condition on current `low_t` only after
`low_t` has been sampled. Neither head may condition on future low tokens, future high tokens, or
future target values. The prior must remain autoregressive and left-to-right; no bidirectional
prior, bidirectional filter, or future-aware teacher signal is allowed.

The prior can be implemented as a shared causal trunk with two output heads or as two explicitly
staged causal modules. In either case, the masking contract must make the same-time dependency
asymmetric:

```text
low_t  sees low_<t and high_<t
high_t sees low_<=t and high_<t
```

## Sampling

Sampling proceeds time step by time step:

1. Sample `low_t` from the low head conditioned on `low_<t`, `high_<t`, and VIX.
2. Sample `high_t` from the high head conditioned on `low_<=t`, `high_<t`, and VIX.
3. Append both sampled tokens to their streams.
4. After the full sequence is sampled, decode low tokens through the low tokenizer.
5. Decode high tokens through the high tokenizer.
6. Compose the report-facing scalar path as `low_hat + high_hat`.

Decoded low and high paths should also be retained for component diagnostics. Paper-style market
metrics should use the composed scalar path.

## Metrics

Tokenizer metrics should report the component and composed errors:

- low reconstruction error against the EMA low component;
- high reconstruction error against the EMA high component;
- composed reconstruction error for `low_hat + high_hat` against the original scalar path;
- token usage, active-code count, active-code ratio, entropy, and perplexity for both streams.

Generated-path metrics should keep the standard S&P500/VIX diagnostics:

- volatility W1;
- squared-return autocorrelation;
- MMD and SWD;
- terminal-return W1;
- drawdown W1;
- log-return autocorrelation;
- VIX-bucket terminal-return and volatility comparisons.

The separate-tokenizer branch also needs same-time low/high compatibility diagnostics:

- empirical compatibility of sampled `(low_t, high_t)` pairs against real extracted pairs;
- same-time pair frequency or distance by VIX bucket;
- conditional high-token entropy given `low_t`;
- transition diagnostics for each stream and for paired stream states;
- reconstruction and generated-path failures split by low/high component.

## Causality Checks

Run no-leakage checks before any model comparison:

- decomposition no-leakage: perturb `x_>t` and verify low/high prefixes through `t` are unchanged;
- low tokenizer no-leakage: perturb future low-component inputs and verify prefix encoder
  states, tokens, and reconstructions are unchanged;
- high tokenizer no-leakage: perturb future high-component inputs and verify prefix encoder
  states, tokens, and reconstructions are unchanged;
- hierarchical prior no-leakage: perturb future low/high tokens and verify prefix logits are
  unchanged.

For the hierarchical prior, the high-head logits at time `t` may change when `low_t` changes,
because `low_t` is part of the allowed same-time conditioning set. They must not change when only
future low or future high tokens are perturbed. The low-head logits at time `t` must not depend on
`low_t`, `high_t`, or any future tokens.

## Non-Goals

- No GroupedRVQ.
- No MGVQ.
- No signatures.
- No diffusion.
- No cross-attention.
- No learned filters yet.
- No bidirectional filtering or bidirectional prior.
- No new objectives before the deterministic separate-tokenizer baseline is measured.

## Stage Gate

Proceed beyond the separate-tokenizer branch only if it improves the residual metrics without the
large MMD/SWD regression observed in the joint EMA candidate. The minimum promotion signal is:

- residual metrics improve or remain competitive, especially volatility W1 and squared-return
  autocorrelation;
- MMD and SWD recover materially relative to joint EMA alpha `0.2`;
- terminal-return and drawdown W1 do not materially regress;
- both token streams show healthy usage and perplexity;
- same-time low/high compatibility diagnostics do not indicate incoherent sampled pairs;
- all decomposition, tokenizer, and hierarchical-prior no-leakage checks pass.

If this stage gate fails, the result should be treated as evidence about factorisation rather
than a prompt to add unrelated modelling machinery. Learned causal filters may be considered only
after the deterministic EMA split has been fully diagnosed.
