# VQ Tokenizer Tuning Roadmap

Status: architecture roadmap after closing the signature-conditioning branch. This document does
not introduce source-code changes, training runs, configuration edits, GroupedRVQ, or MGVQ.

## 1. Current Promoted Architecture

The promoted public path remains:

```text
standard causal VQ tokenizer + additive VIX-only causal AR prior
```

The current benchmark is S&P500/VIX. The tokenizer maps each length-60 path to one discrete code
per time step, and the additive causal autoregressive prior models the resulting token sequence
using past tokens and the scalar VIX condition. Generated paths are decoded through the frozen
tokenizer decoder.

The promoted evidence package combines:

- paper-style S&P500/VIX diagnostics, including MMD, SWD, distributional summaries, drawdown,
  volatility, autocorrelation, tail, and VIX-bucket plots;
- latent-geometry diagnostics, including codebook projection, usage-weighted projection,
  VIX-bucket code usage, token trajectories, and nearest-region or Voronoi views;
- token-level diagnostics, including active-code count, perplexity, entropy, and transition
  behaviour.

This profile keeps the method reportable: it separates path-level market quality from the
discrete representation quality, and it preserves a simple prior-facing interface.

## 2. Signature Branch Decision

The signature-conditioning branch is closed without changing the promoted default. The default
condition remains VIX-only.

The raw `logsig_l3_ctx20` condition remains research evidence because it explored whether richer
causal context could improve token modelling. It should not become the default condition without
generated-market evidence that improves the final diagnostics, not only token likelihood.

AdaLN-lite improved token likelihood in branch experiments, but it did not improve the generated
market diagnostics enough to justify replacing the additive VIX-only prior. It remains deferred.

The following alternatives are also deferred:

- cross-attention conditioning;
- Gumbel-Softmax training paths;
- signature-kernel training objectives.

Future signature work should be reopened only with a predeclared evaluation gate that requires
improvement in generated-market diagnostics, not only token negative log likelihood.

## 3. Tokenizer-Side Motivation

The standard VQ tokenizer currently has broad code usage and the simplest prior interface. The
S&P500/VIX standard VQ geometry uses 63 of 64 codes with global perplexity about 39, and its
VIX-bucket diagnostics show broader code usage in higher-volatility windows. This supports the
one-code-per-time-step representation as a stable baseline for the additive VIX-only causal AR
prior.

RVQ q2 is useful as an ablation because it gives an interpretable coarse/detail split. In the
regenerated S&P500/VIX diagnostics, q0 is low entropy and q1 is broad, but only 231 of 4096
same-time q0/q1 pairs are active. This sparse joint support explains why RVQ q2 can improve
reconstruction while making generation harder: a prior must model temporal dynamics,
VIX-dependent detail, and same-time code-pair compatibility.

The next phase should therefore tune standard VQ before adding grouped or multi-code tokenizers.
GroupedRVQ and MGVQ should be gated by evidence that standard VQ, after careful tuning and causal
decomposition tests, cannot address a measured failure mode.

## 4. Stage A: Standard VQ Hyperparameter Tuning

Stage A keeps the tokenizer family and prior interface fixed: one code per time step, a causal
encoder and decoder, and the additive VIX-only causal AR prior.

Candidate tokenizer sweeps should cover:

- codebook size, with emphasis on whether larger vocabularies improve reconstruction and
  generated-market diagnostics without creating fragile or underused codes;
- `codebook_dim`, to test whether the current low-dimensional codebook is too restrictive or
  already sufficient;
- commitment weight, to balance reconstruction fidelity against stable discrete assignment;
- decay and dead-code handling options, if supported by the wrapped quantizer backend;
- encoder and decoder capacity, including hidden width and depth;
- dilation and receptive field, including whether the causal stack captures enough local and
  medium-range temporal structure.

Selection should not promote a candidate solely from reconstruction loss. A tuned tokenizer must
remain compatible with the additive prior and must improve, or at least preserve, paper-style
generated-market diagnostics.

## 5. Stage B: Causal TimeVQVAE-Inspired Decomposition

Stage B may borrow the decomposition idea from TimeVQVAE, but not its bidirectional prior. The
goal is to test whether a causal split improves the tokenizer before introducing multi-code
generation.

The decomposition should remain strictly causal:

- a low-frequency or trend component represents slowly varying path structure;
- a high-frequency or residual component represents local movement not captured by the trend;
- each component is produced without future access;
- all priors remain causal and use only past tokens and admissible conditions.

This stage should be framed as tokenizer decomposition, not as a direct import of the TimeVQVAE
generation pipeline. Any smoothing, filtering, or component construction must pass no-leakage
checks before training or evaluation.

## 6. Stage C: GroupedResidualVQ

GroupedResidualVQ is deferred until the standard VQ and causal-decomposition results are
available. It should proceed only if those stages identify a specific representation failure that
grouped indices are expected to solve.

The gating requirements are:

- a stable multi-code prior interface;
- clear tensor conventions for groups, quantizer levels, and time;
- same-time compatibility diagnostics for grouped codes;
- causal no-leakage checks for tokenizer and prior paths;
- evidence that grouped structure improves generated-market diagnostics, not only
  reconstruction.

GroupedResidualVQ should be treated as a controlled ablation rather than as the next default.

## 7. Stage D: MGVQ

MGVQ is future work only. It should be implemented only after grouped evidence exists and after
the project has a stable multi-code prior interface.

There should be no direct image-tokenizer import without adaptation. A financial time-series MGVQ
variant would need causal encoders and decoders, causal conditioning, market-specific diagnostics,
and no bidirectional or spatial assumptions that conflict with the S&P500/VIX generation task.

## 8. Causality Checks

Every promoted tokenizer or prior candidate must pass explicit causality checks:

- tokenizer no-leakage: the encoded or reconstructed value at time `t` may not depend on future
  path values;
- prior no-leakage: token logits at time `t` may not depend on tokens after `t` or future
  conditions;
- condition provenance: each condition must have an auditable timestamp and must be known at the
  generation time where it is used;
- no bidirectional priors: bidirectional token models may be cited as references, but they cannot
  be used for the promoted causal generator.

These checks apply especially to trend/residual decomposition, signature-derived conditions, and
multi-code tokenizers because each can introduce hidden future access if implemented casually.

## 9. Evaluation

Model selection should combine path-level, token-level, and geometry-level evidence.

Path-level diagnostics:

- MMD and SWD;
- paper-style generated-market diagnostics for distributions, volatility, drawdown,
  autocorrelation, higher moments, tails, and VIX buckets;
- model-selection profiles that compare candidates across multiple criteria instead of promoting
  a single best scalar score.

Tokenizer and geometry diagnostics:

- reconstruction metrics on held-out data;
- active-code count, active-code ratio, token entropy, and perplexity;
- codebook projection and usage-weighted projection;
- condition-bucket code usage;
- token trajectory examples;
- latent geometry summaries for standard VQ and any ablations.

Transition diagnostics:

- one-step transition matrices or sparse summaries;
- transition entropy and rare-transition mass;
- VIX-bucket transition shifts;
- same-time code-pair support for RVQ, grouped, or other multi-code tokenizers;
- invalid or unsupported pair-rate estimates under sampled priors when applicable.

A candidate should advance only when its benefits survive this combined evaluation. In particular,
lower reconstruction error is insufficient if the prior becomes harder to sample from or if
generated paths lose market stylised facts.

## 10. W&B Execution

Every non-smoke run must use the following environment prefix:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread
```

For example, tokenizer and prior runs should be launched with that prefix before the relevant
`poetry run ...` command whenever W&B is enabled.

If W&B fails, rerun the same command with `--no-wandb` and document the failure and rerun path in
the relevant verification note. Local output files remain the reproducibility source of record.

## 11. Stage Gates

The execution order is:

1. Tune standard VQ while keeping the prior interface fixed.
2. Evaluate causal trend/residual decomposition if standard VQ tuning exposes a clear
   reconstruction or geometry limitation.
3. Revisit GroupedResidualVQ only after the prior interface and multi-code diagnostics are stable.
4. Treat MGVQ as future work until grouped evidence justifies a dedicated financial time-series
   adaptation.

The roadmap deliberately prioritises tokenizer evidence before architecture expansion. This keeps
the promoted method compact while preserving a clear path for more expressive tokenizers if the
standard VQ baseline reaches a measured limit.
