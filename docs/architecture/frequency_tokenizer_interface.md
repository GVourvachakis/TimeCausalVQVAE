# Frequency Tokenizer Interface Decision

Status: design decision only. This document does not implement code, train models, or modify the
promoted S&P500/VIX baseline.

## Decision

Use Option A first: a joint low/high tokenizer built from deterministic causal EMA decomposition.

The minimal first interface should transform each scalar path `x` into a two-channel sequence
`[low, high]`, train one standard vector VQ tokenizer on that two-channel sequence, keep one code
per time step, decode to `[low_hat, high_hat]`, and compose the report-facing path as
`low_hat + high_hat`.

This preserves the current prior-facing contract. The existing vector tokenizer already accepts
`[batch, length, data_dim]`, returns `recon_x` with the same shape, and emits one vector index per
time step when `quantizer_type = vector`. Setting `data_dim = 2` after decomposition is therefore
the smallest interface change that can test whether a deterministic frequency split helps
volatility W1 and squared-return autocorrelation.

Option B, with separate low and high tokenizers, should be deferred until Option A has a measured
failure mode.

## Current Interface Constraints

The current tokenizer and evaluation path assume:

- tokenizer input shape: `[batch, length, data_dim]`;
- tokenizer reconstruction shape: `[batch, length, data_dim]`;
- vector-VQ token shape: `[batch, length]`;
- prior sample shape for the promoted baseline: `[batch, length]`;
- prior decoding path: sampled token indices -> quantized embeddings -> tokenizer decoder;
- paper-style S&P500/VIX metrics compare decoded paths against real paths with matching shape.

The current paper-style evaluation expects final financial paths, not frequency components. A
frequency tokenizer must therefore either compose decoded low/high channels before paper-style
metrics or expose a wrapper that returns composed paths whenever it is used as a generator.

## Option A: Joint Low/High Tokenizer

### Interface

Use a deterministic preprocessing step:

```text
x:    [batch, length, 1]
low:  [batch, length, 1]
high: [batch, length, 1]
x_freq = concat([low, high], dim=-1): [batch, length, 2]
```

Train one standard vector VQ tokenizer on `x_freq`:

```text
tokenizer input:  [batch, length, 2]
tokenizer output: [batch, length, 2]
indices:          [batch, length]
```

Decode sampled tokens through the same tokenizer:

```text
decoded_freq = [low_hat, high_hat]: [batch, length, 2]
decoded_path = low_hat + high_hat:  [batch, length, 1]
```

### Config Fields

The minimal new configuration surface should be:

```yaml
frequency_decomposition: ema
ema_alpha: 0.2
compose_output: true
```

Recommended placement is in the data or preprocessing section of the tokenizer and evaluation
configs. The model-facing `data_dim` should be `2` for the frequency-tokenizer run, while the
original market path dimension remains `1`.

Field meanings:

- `frequency_decomposition: ema` enables deterministic causal EMA decomposition.
- `ema_alpha` sets the EMA smoothing parameter and must satisfy `0 < alpha <= 1`.
- `compose_output: true` means decoded frequency channels are composed back to the original path
  space before paper-style market diagnostics, generated-path persistence, and comparison against
  real S&P500/VIX paths.

### Decoder Output Handling

The tokenizer decoder should continue to return its native reconstruction:

```text
recon_x_freq: [batch, length, 2]
```

The composition step should be explicit and downstream:

```text
low_hat = recon_x_freq[..., 0:1]
high_hat = recon_x_freq[..., 1:2]
recon_path = low_hat + high_hat
```

This keeps tokenizer internals simple and allows component-specific diagnostics. Training can use
the two-channel reconstruction loss against `[low, high]`, while evaluation records both
component reconstruction and composed-path reconstruction.

### Prior Interface

The prior remains the existing single autoregressive token stream:

```text
p(z_t | z_<t, condition)
```

No hierarchical prior, multi-code token tensor, or same-time compatibility model is required for
the first experiment. This is the main advantage of Option A.

### Paper-Style Evaluation Handling

Paper-style S&P500/VIX evaluation should receive composed scalar paths:

```text
real_paths:    [batch, length, 1]
decoded_path:  [batch, length, 1]
```

Auxiliary artefacts may store `decoded_freq`, `low_hat`, and `high_hat`, but MMD, SWD,
volatility W1, squared-return autocorrelation, terminal return, drawdown, and VIX-bucket
diagnostics should be computed on `decoded_path`.

For tokenizer-only evaluation, report both:

- native two-channel reconstruction metrics against `[low, high]`;
- composed-path metrics against the original path `x`.

## Option B: Separate Low and High Tokenizers

### Interface

Use the same deterministic decomposition but train two independent standard tokenizers:

```text
low tokenizer input:  [batch, length, 1]
high tokenizer input: [batch, length, 1]
low indices:          [batch, length]
high indices:         [batch, length]
```

Generation then requires two decoded streams:

```text
low_hat  = decode_low(low_tokens)
high_hat = decode_high(high_tokens)
decoded_path = low_hat + high_hat
```

### Prior Requirement

Option B cannot use the promoted prior unchanged unless the two token streams are flattened into
a larger single vocabulary or otherwise packed. The natural causal factorisation is hierarchical:

```text
p(low_t | low_<t, high_<t, condition)
p(high_t | low_<=t, high_<t, condition)
```

This is more expressive, but it introduces a new prior interface and a same-time compatibility
problem between low and high tokens.

### Advantages

- Low and high codebooks can specialise.
- Component-specific latent geometry is easier to interpret.
- The high tokenizer can allocate capacity directly to residual shocks and volatility bursts.

### Costs

- Requires two token datasets, two tokenizer checkpoints, and more evaluation bookkeeping.
- Requires a two-token or hierarchical causal prior before generation is meaningful.
- Increases the risk of sampling incompatible low/high token pairs.
- Moves the project towards a multi-code interface before the simplest joint representation has
  been tested.

## Comparison

| Criterion | Option A: joint tokenizer | Option B: separate tokenizers |
| --- | --- | --- |
| Token shape | `[batch, length]` | two `[batch, length]` streams |
| Prior change | none for the first experiment | required |
| Decoder output | `[batch, length, 2]` | two `[batch, length, 1]` outputs |
| Paper-style path | compose low/high | compose low/high |
| Component diagnostics | available from channels | strongest separation |
| Implementation surface | small | moderate to large |
| Main risk | joint code may still entangle components | prior complexity and token compatibility |

Option A is the recommended first implementation because it directly tests the decomposition
hypothesis while preserving the current one-code-per-time-step contract. Option B should be kept
as a second-stage design if Option A passes no-leakage checks but fails to improve residual
volatility diagnostics.

## No-Leakage Checks

The deterministic decomposition already passed a source-level smoke check for synthetic positive
paths with `alpha = 0.2`, cutoff `29`, and shapes `(8, 60)` and `(8, 60, 1)`. The maximum low
prefix difference, high prefix difference, and reconstruction difference were all
`0.00000000e+00`.

Before training a frequency tokenizer, the implementation should add checks for:

- decomposition prefix invariance after perturbing target values only after the cutoff;
- transformed data prefix invariance for `[low, high]`;
- tokenizer no-leakage on the two-channel input, including encoder latents, indices, and
  reconstruction prefix;
- decode-and-compose prefix invariance for `low_hat + high_hat`;
- prior no-leakage unchanged from the one-code prior in Option A.

For Option B, add separate no-leakage checks for each tokenizer and for the hierarchical prior
factorisation before any generation experiment.

## Metrics

Tokenizer metrics should include:

- original path reconstruction error after composition:
  `compose_low_high(low_hat, high_hat)` against `x`;
- low reconstruction error: `low_hat` against `low`;
- high reconstruction error: `high_hat` against `high`;
- native two-channel reconstruction L1 and L2;
- active-code count, perplexity, entropy, and VIX-bucket code usage.

Generated-path metrics should include the existing paper-style metrics plus the residual-focused
promotion checks:

- volatility W1;
- squared-return autocorrelation, especially short lags;
- log-return autocorrelation;
- terminal-return and return-distribution distances;
- VIX-bucket terminal-return and volatility comparisons.

Promotion should depend on the composed path improving the residual failure mode without
materially degrading the standard S&P500/VIX diagnostics.

## Non-Goals

- No GroupedRVQ.
- No MGVQ.
- No signatures.
- No diffusion.
- No cross-attention.
- No bidirectional filtering or future-aware decomposition.
- No adapted Wasserstein in this phase.

## Implementation Boundary

The next implementation step should be limited to Option A plumbing:

1. Apply deterministic causal EMA decomposition in the data path.
2. Expose the three config fields: `frequency_decomposition`, `ema_alpha`, and `compose_output`.
3. Train/evaluate one standard vector VQ tokenizer with `data_dim = 2`.
4. Compose decoder output only at evaluation and sampling boundaries.

Do not implement Option B until Option A has been evaluated against the promoted baseline and the
hidden128 variant.
