# Causal Frequency Tokenizer Plan

Status: architecture plan only. This document does not implement code, train models, or change the
promoted S&P500/VIX baseline.

## Motivation

The promoted S&P500/VIX discrete baseline remains the standard causal VQ tokenizer with one code
per time step and an additive scalar-conditioned causal autoregressive token prior. Its latent
geometry is reportable: the standard tokenizer uses 63 of 64 codes, has global perplexity around
39, and shows increasing active-code count and perplexity across VIX buckets.

The current tuning evidence also identifies a narrower remaining failure mode. The hidden128
variant improves many path-level metrics, but it still regresses squared-return autocorrelation
and volatility W1. This pattern suggests that the remaining gap is not primarily codebook collapse
or a lack of global condition sensitivity. It is more likely a residual-dynamics issue.

Volatility clustering is a high-frequency or residual path property: local shocks, absolute
returns, and squared returns need persistent structure even when the lower-frequency path level is
reasonable. A standard one-stream VQ tokenizer may entangle slower movement and local shocks in a
single code. The prior then has to model trend, level adjustment, and volatility bursts through
one token stream, which can preserve broad geometry while still weakening residual volatility
statistics.

The proposed extension is therefore a narrow causal low/high-frequency tokenizer. Its purpose is
to make local residual variation explicit while preserving the no-anticipation design of the
existing TC-VAE workflow.

## Literature Grounding

TimeVQVAE motivates the decomposition idea. It uses VQ modelling for time-series generation and
separates time-frequency content into low-frequency and high-frequency parts. This project should
borrow only the decomposition principle: the representation may separate smoother path content
from sharper residual content before quantisation.

The TimeVQVAE prior is not adopted. Its published prior is bidirectional, whereas this project's
S&P500/VIX prior must remain causal. The extension must therefore preserve the TC-VAE
no-anticipation requirement: every low component, high component, token, prior logit, and decoded
sample at time `t` may depend on `x_<=t`, past tokens, current allowed conditions, and no future
target values.

This also rules out centred filters, full-window FFT/STFT decompositions, symmetric smoothing
kernels, or any learned module whose padding or attention mask can access `x_>t`.

## Causal Decomposition Options

### A. Causal Exponential Moving Average

Define a deterministic causal smoother:

```text
low_t = EMA(x_<=t)
high_t = x_t - low_t
```

A concrete recurrence can initialise `low_0 = x_0` and use
`low_t = alpha * x_t + (1 - alpha) * low_{t-1}` for `t > 0`, with `alpha` fixed by configuration.
The high component is then an additive residual at the same timestamp.

Advantages:

- It is deterministic, auditable, and easy to test for no leakage.
- It introduces no new trainable filter that can silently learn future-dependent behaviour.
- It gives a direct residual channel for volatility clustering diagnostics.

Risks:

- The smoothing strength may be too rigid for all VIX regimes.
- EMA lag can move part of a sharp but persistent regime change into the high component.
- The best `alpha` may be data-scale dependent and should be selected by a small validation grid,
  not by test-set inspection.

### B. Causal Convolutional Low-Pass Filter

Define a causal low-pass filter:

```text
low_t = ConvCausal(x_<=t)
high_t = residual
```

The filter may be fixed or learned, but it must use left-only padding and a finite causal kernel.
If the filter is learned, normalised non-negative weights or another explicit low-pass constraint
should be considered so that the low branch remains interpretable rather than becoming an
unconstrained second encoder.

Advantages:

- It can represent a richer causal low-pass family than a single EMA timescale.
- It aligns with the existing causal convolutional encoder style.

Risks:

- Padding, dilation, or implementation details can introduce subtle look-ahead.
- A learned filter can collapse into an unconstrained feature extractor unless the objective and
  diagnostics enforce separation.

### C. Learned Two-Head Causal Encoder

Use a shared causal encoder followed by two causal heads:

```text
shared causal encoder -> low head
                     -> high head
```

The heads may reconstruct low-like and high-like components directly or provide component-specific
latents before quantisation. This option is the most expressive, but it is also the least
transparent.

Advantages:

- It can learn market-specific separation rather than imposing a fixed smoother.
- It may handle regime changes better than a fixed EMA.

Risks:

- It requires careful no-leakage tests for the shared encoder and both heads.
- Without explicit constraints, the heads may entangle again or allocate information
  arbitrarily.
- It expands the architecture before the deterministic decomposition has established that the
  frequency split is useful.

## First Implementation Recommendation

Start with deterministic causal EMA decomposition.

The first implementation should add only the data transform:

```text
x_t -> [low_t, high_t]
```

After that transform is verified, compare two minimal tokenisation layouts:

1. One standard VQ tokenizer over concatenated low/high channels, preserving one joint code per
   time step.
2. Two standard VQ tokenizers, one for the low component and one for the high component.

The first layout should be preferred initially if reconstruction and generation metrics are
competitive, because it keeps the existing one-code-per-time-step interface. The second layout is
the fallback if the joint code still entangles components or if latent-geometry diagnostics show
that low and high usage have meaningfully different structure.

RVQ q2 is not the template for this phase. The existing RVQ q2 geometry is useful evidence that a
coarse/detail split can be interpretable, but it also shows that multi-code same-time support is
sparse and harder for generation. The EMA experiment should therefore avoid adding another
multi-code prior burden unless the one-code joint representation fails a specific measured
diagnostic.

## Prior Options

### Single AR Prior Over Joint Low/High Token

If the concatenated low/high tokenizer emits one code `z_t`, keep the current prior shape:

```text
p(z_t | z_<t, condition)
```

This is the lowest-risk prior option. The decomposition is represented inside the tokenizer
inputs and decoder, while the prior still sees a single causal token stream.

### Two-Stage Causal Prior

If low and high tokenizers emit separate tokens, factorise each time step causally:

```text
p(low_t | past, condition)
p(high_t | past, low_t, condition)
```

Here, `past` may include low and high tokens from times `< t`, but not future tokens. The second
stage may condition on the current low token because it is sampled first at the same time step.
It must not condition on future low tokens, future high tokens, or future target values.

### Explicit Exclusions

Do not use bidirectional priors, masked-token reconstruction objectives that inspect both sides of
the target time, future-aware teacher signals, diffusion sampling, cross-attention to future
paths, or any filtering step that requires future observations.

## Causality Checks

Future perturbation checks should be mandatory before any metric comparison.

For decomposition, choose a cutoff `tau`, copy a batch, perturb target values only at times
`> tau`, and recompute `low` and `high`. The components through `tau` must be unchanged up to
floating-point tolerance:

```text
low_perturbed_<=tau == low_original_<=tau
high_perturbed_<=tau == high_original_<=tau
```

For tokenizer no-leakage, repeat the same perturbation at the model input and compare encoder
outputs, quantised embeddings, token indices, and reconstructions through the cutoff. Tokens and
pre-quantisation latents through `tau` must be unchanged. Decoder reconstructions should be
checked according to the model's declared causal reconstruction convention.

For prior no-leakage, perturb future tokens or future target-derived conditions and verify that
logits through `tau` are unchanged. The test should cover both the single joint-token prior and
the two-stage factorisation. In the two-stage case, the high-token logits at `t` may change when
`low_t` changes, but not when only future low or high tokens change.

## Evaluation

Use the standard paper-style S&P500/VIX metrics before making any promotion claim:

- MMD and SWD.
- Return and terminal-return distributions.
- Path-volatility distribution.
- Maximum drawdown.
- Log-return autocorrelation.
- Squared-return autocorrelation.
- Skewness and kurtosis.
- Extreme-return and rolling-volatility tail diagnostics.
- VIX-bucket terminal-return and volatility comparisons.

Add the residual-focused diagnostics that motivate this extension:

- Volatility W1, globally and by VIX bucket where sample size permits.
- Squared-return autocorrelation, with special attention to short lags.
- High-component return and squared-return autocorrelation.
- Reconstruction error split by low and high component.

Run latent-geometry diagnostics per component:

- Active-code count, active-code ratio, perplexity, and entropy for the joint tokenizer or for
  each component tokenizer.
- PCA or other projection diagnostics for low and high codebooks separately when separate
  tokenizers are used.
- Token trajectories for low/high streams or annotated joint tokens.
- Code usage by VIX bucket, with the expectation that high/residual codes should show stronger
  volatility sensitivity if the split is doing useful work.

The comparison set is:

- Promoted standard causal VQ baseline.
- Hidden128 tokenizer/prior variant.
- Deterministic EMA frequency-tokenizer candidate.

## Non-Goals

- No MGVQ in this phase.
- No GroupedRVQ yet.
- No cross-attention.
- No signatures.
- No adapted Wasserstein in this phase.
- No diffusion prior or diffusion-style sampler.
- No bidirectional filtering or bidirectional token prior.

## Stage Gate

Stage 1 implements only deterministic causal EMA decomposition. It should compare the resulting
candidate against the promoted standard VQ baseline and the hidden128 variant.

The candidate may advance only if it passes the decomposition, tokenizer, and prior no-leakage
checks and shows a measured improvement on the residual failure mode, especially volatility W1
and squared-return autocorrelation, without materially degrading the standard paper-style metrics.

If EMA does not improve the residual diagnostics, do not proceed directly to GroupedRVQ, MGVQ,
signatures, cross-attention, diffusion, or adapted Wasserstein. Instead, either tune the fixed EMA
timescale within a predeclared validation grid or stop and record that deterministic causal
frequency separation did not address the observed S&P500/VIX volatility gap.
