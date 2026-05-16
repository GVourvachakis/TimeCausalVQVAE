# Separate Frequency Hierarchical Prior

Status: architecture specification only. This document does not implement code, train a prior, or
change the promoted S&P500/VIX baseline.

## Scope

The separate frequency tokenizer branch now has two trained alpha `0.2` tokenizers:

```text
low tokenizer:  outputs/sp500_vix_discrete/separate_frequency_tokenizer/low/sp500_vix_freq_low_alpha02_tokenizer
high tokenizer: outputs/sp500_vix_discrete/separate_frequency_tokenizer/high/sp500_vix_freq_high_alpha02_tokenizer
```

Both clear the tokenizer-quality gate: the low stream keeps 55 / 64 active codes with evaluation
perplexity 41.87561798, and the high stream keeps 61 / 64 active codes with evaluation
perplexity 46.23094559. The next step should specify the causal prior required to sample the two
streams coherently. No GroupedRVQ, MGVQ, signatures, diffusion, or cross-attention are part of
this specification.

## Token Shapes

The prior consumes two token tensors extracted from the separate tokenizers:

```text
low_tokens:  [batch, time]
high_tokens: [batch, time]
```

For the current S&P500/VIX runs, `time = 60` and both codebooks have 64 codes. A combined training
batch may store the streams either as two named tensors or as a final component axis:

```text
paired_tokens[..., 0] = low_tokens
paired_tokens[..., 1] = high_tokens
paired_tokens: [batch, time, 2]
```

The named-tensor contract is preferred at data boundaries because it avoids implying that the
two tokens came from one multi-code tokenizer. The packed `[batch, time, 2]` layout may be used
inside the model if the component order is explicit and stable: component `0` is low, component
`1` is high.

## Factorisation

Use the hierarchical same-time factorisation:

```text
p(low_t | low_<t, high_<t, VIX)
p(high_t | low_<=t, high_<t, VIX)
```

The low head predicts the low-frequency token for time `t` from past low tokens, past high
tokens, and VIX. The high head predicts the residual token for the same time after `low_t` is
available. This expresses the intended generation order: level first, residual shock second.

The joint likelihood for one sequence is:

```text
sum_t log p(low_t | low_<t, high_<t, VIX)
    + log p(high_t | low_<=t, high_<t, VIX)
```

## Implementation Options

### A. Shared Transformer Trunk and Two Heads

Build a shared causal calendar-time trunk over previous low/high token blocks. At position `t`,
the trunk input is the shifted previous-time block:

```text
input_t = [low_{t-1}, high_{t-1}]
input_0 = [BOS_low, BOS_high]
```

The trunk hidden state `h_t` therefore sees only `low_<t`, `high_<t`, and the allowed condition.
Two heads sit on top:

```text
low_logits_t = low_head(h_t)
high_logits_t = high_head(h_t + embed_current_low(low_t))
```

During training, `embed_current_low(low_t)` uses the true teacher-forced low token. During
sampling, it uses the sampled low token. This matches the required factorisation without giving
the low head access to same-time high information.

This option matches the spirit of the existing hierarchical q2 prior, but the public interface
must name low/high streams rather than RVQ levels. The prior should expose stream-specific logits
and losses:

```text
low_logits:  [batch, time, low_codebook_size]
high_logits: [batch, time, high_codebook_size]
```

### B. Low Prior Plus Conditional High Prior

Train two modules:

```text
low prior:  p(low_t | low_<t, high_<t, VIX)
high prior: p(high_t | low_<=t, high_<t, VIX)
```

The low module and high module may use separate causal trunks. The high module consumes a
same-time low-token embedding in addition to shifted past tokens. This is easier to reason about
locally, but it duplicates temporal modelling capacity and may make sampling slower. It also
risks the two priors learning inconsistent hidden representations of the same past.

This option is a fallback if the shared trunk underfits or if diagnostics show that low and high
streams require substantially different temporal receptive fields.

### C. Flattened Joint Token Vocabulary

Flatten each same-time pair into one token:

```text
joint_token_t = low_t * high_codebook_size + high_t
joint_token_t: [batch, time]
joint_vocab_size = low_codebook_size * high_codebook_size
```

For the current 64-code low and high tokenizers, this gives a 4096-token vocabulary. It recovers
the existing single-stream prior interface but discards the hierarchical structure and makes rare
pair modelling harder. It also hides separate low/high losses and weakens compatibility
diagnostics.

Use this only as a rejected baseline or sanity reference, not as the first implementation.

## Recommendation

Start with Option A: shared transformer trunk plus sequential same-time high head.

Reasons:

- It implements the desired factorisation directly.
- It reuses the current causal transformer prior style: shifted previous-time inputs, causal mask,
  VIX conditioning, and token cross-entropy.
- It keeps one shared temporal representation for past low/high context.
- It exposes separate low and high logits, losses, accuracies, and perplexities.
- It samples in the same order required by the generative story.

The first implementation should use additive VIX conditioning to match the existing additive
token-prior experiments unless a later comparison explicitly tests AdaLN-lite.

## Causality Contract

The prior must satisfy:

```text
low_logits_t  may depend on low_<t, high_<t, VIX
high_logits_t may depend on low_<=t, high_<t, VIX
```

The high head may use the current low token because generation samples `low_t` before `high_t`.
The high head must not use `high_t` as input when predicting `high_t`. Neither head may use
future low tokens, future high tokens, future decoded paths, or future target-derived
conditions.

The masking convention should remain calendar-time causal. A standard shifted input block gives
the trunk no access to same-time targets:

```text
trunk_input_t = [BOS_low, BOS_high]           for t = 0
trunk_input_t = [low_{t-1}, high_{t-1}]       for t > 0
```

The only same-time edge is the explicit low-to-high edge:

```text
low_t -> high_logits_t
```

There is no high-to-low same-time edge.

## Training

Use teacher forcing. For each batch:

```text
low_tokens:  [batch, time]
high_tokens: [batch, time]
conditions:  [batch, 1]
```

Build shifted past blocks for the shared trunk and predict both streams at every time step. The
high head receives the true `low_t` embedding during training:

```text
low_logits_t = low_head(h_t)
high_logits_t = high_head(h_t + low_current_embedding(true_low_t))
```

The training objective is:

```text
loss = CE_low + CE_high
```

The implementation may later expose non-negative component loss weights, but the first run should
use equal weights. Report stream-specific and aggregate metrics:

```text
CE_low, CE_high, CE_total
accuracy_low, accuracy_high
perplexity_low, perplexity_high
same_time_pair_perplexity
```

Padding is not required for the current fixed-length S&P500/VIX token tensors. If padding is
introduced later, it must mask both stream losses consistently.

## Sampling

Sample left to right:

```text
for t in 0 .. time - 1:
    h_t = trunk(BOS_or_past_low_high_blocks)_t
    low_t = sample(low_head(h_t))
    high_t = sample(high_head(h_t + embed_current_low(low_t)))
    append low_t and high_t
```

After sampling:

```text
low_hat = low_tokenizer.decode_indices(low_tokens, VIX)
high_hat = high_tokenizer.decode_indices(high_tokens, VIX)
decoded_path = low_hat + high_hat
```

The composed `decoded_path` is the report-facing S&P500 path. Retain `low_hat`, `high_hat`,
`low_tokens`, and `high_tokens` for diagnostics.

Sampling controls should mirror the existing token prior where possible:

- temperature;
- optional top-k;
- seed;
- VIX condition source and shape.

Top-k may be shared between streams for the first run. Stream-specific top-k is a later ablation,
not part of the first implementation.

## Evaluation

Evaluate token quality per stream:

- low cross-entropy, accuracy, and perplexity;
- high cross-entropy, accuracy, and perplexity;
- active codes, entropy, marginal-code L1, transition L1, and run-length distance per stream;
- VIX-bucket code usage per stream.

Evaluate same-time low/high compatibility:

- empirical real pair distribution versus sampled pair distribution;
- same-time pair perplexity;
- conditional high-token entropy given low token;
- pair-frequency L1 globally and by VIX bucket;
- examples of frequent incompatible sampled pairs if diagnostics indicate mismatch.

Evaluate composed decoded paths with the standard paper-style metrics:

- MMD and SWD;
- volatility W1;
- squared-return autocorrelation;
- flattened squared-return autocorrelation;
- terminal-return W1;
- drawdown W1;
- VIX-bucket terminal-return and volatility diagnostics.

Component reconstructions and sampled components should be saved, but promotion decisions should
use the composed scalar path for market diagnostics.

## No-Leakage Checks

Before training or sampling comparisons, add a hierarchical prior no-leakage check:

1. Build a batch of `low_tokens` and `high_tokens`.
2. Perturb only future low/high tokens after an inclusive cutoff.
3. Verify low-head logits through the cutoff are unchanged.
4. Verify high-head logits through the cutoff are unchanged when only future tokens are changed.
5. Verify high-head logits at time `t` may change when `low_t` changes, and document this as the
   allowed same-time edge.

For temporal conditions, perturb only future condition values after the cutoff and verify prefix
logits are unchanged. For scalar VIX conditions `[batch, 1]`, no future temporal condition axis
exists.

## Rejected Baseline

The flattened 4096-token vocabulary is rejected as the starting point. It may be useful as a
small sanity experiment only if the hierarchical prior is hard to debug. It should not be the
main branch because it:

- loses stream-specific training metrics;
- makes pair sparsity worse;
- obscures the low-to-high causal ordering;
- turns same-time compatibility into a large categorical modelling problem;
- removes the main reason for training separate tokenizers.

## Non-Goals

- No GroupedRVQ.
- No MGVQ.
- No diffusion.
- No signatures.
- No cross-attention.
- No learned filters in the first prior implementation.
- No bidirectional prior or bidirectional filtering.
