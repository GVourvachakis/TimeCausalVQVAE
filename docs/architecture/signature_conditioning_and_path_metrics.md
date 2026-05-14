# Signature Conditioning and Path Metrics

Status: extension roadmap only. This document does not change model code, add dependencies, train
models, modify notebooks, or implement deferred architectures.

## 1. Project state

The promoted public architecture remains:

- standard causal VQ tokenizer;
- additive scalar-conditioned causal AR prior;
- S&P500/VIX paper-style diagnostics;
- latent geometry diagnostics.

The current evidence supports this promoted path. The standard S&P500/VIX VQ tokenizer uses one
code per time step, uses 63 of 64 codes, and shows VIX-sensitive code usage. The RVQ q2 ablation
is informative, but its sparse same-time q0/q1 support makes generation harder than the standard
one-code interface.

The following ideas are deferred:

- diffusion;
- AdaLN;
- cross-attention;
- GroupedRVQ;
- MGVQ;
- transition-aware losses.

These items should remain future work unless a measured failure mode of the promoted standard VQ
path requires them.

## 2. Motivation

Market generators should be compared on path-level distributions rather than only pointwise errors.
For S&P500/VIX, the relevant stylised facts include return tails, volatility tails, volatility
clustering, terminal return, drawdown, and conditional behaviour across VIX regimes. These are
properties of full paths or path functionals, not isolated time steps.

This matters for downstream financial use. Dynamic hedging and portfolio problems are adapted
decision problems: a strategy observes information sequentially, updates its hedge or allocation,
and is judged by path-dependent payoffs and risks. A generator can match one-step marginals while
still failing to reproduce the sequential behaviour that such downstream tasks use.

Signature and log-signature methods are relevant because they summarise ordered path shape beyond
pointwise distances. Lead-lag signatures can expose area and quadratic-variation-like information,
while log-signatures give finite-dimensional coordinates for truncated path features. Signature
kernels extend this idea by comparing paths through the induced kernel without fixing a single
finite truncation level in the feature vector used by the model.

## 3. Signature feature conditioning

The first conditioning experiment should be deliberately small and should preserve the promoted
prior interface.

1. Compute historical or context features from each conditioning path. Candidate channels include
   recent returns, realised volatility, rolling drawdown, and a time channel. These features should
   be derived only from the observed context available at the conditioning time.
2. Apply a lead-lag transform to the context path to retain order and area information related to
   realised quadratic variation.
3. Compute truncated log-signatures from the lead-lag context path. The truncation level should
   be chosen by a smoke test for dimensionality, numerical stability, and conditioning usefulness.
4. Concatenate the resulting finite-dimensional log-signature vector to the existing scalar VIX
   condition.
5. Feed the expanded condition vector through the existing additive condition embedding before
   changing the prior architecture.

This keeps the first ablation close to the current scalar-conditioned setup. The only intended
model-facing change in a future implementation phase would be a wider condition vector, not a new
attention mechanism.

Cross-attention is deferred. It should only be reconsidered if the condition becomes a temporal
sequence, a set of market covariate tokens, or a multi-asset context whose order and alignment
cannot be represented responsibly by a single fixed-dimensional vector.

## 4. Signature-kernel metrics

The first signature-kernel path should be evaluation-only.

The intended metric is a signature-kernel MMD or distance between batches of real and generated
paths. For each checkpoint or sampling setting, the evaluator would:

- sample a generated path batch from the frozen tokenizer and additive token prior;
- select a matched real path batch from the same split or VIX bucket;
- apply consistent path preprocessing, such as return paths, cumulative log-return paths, a time
  channel, and optional lead-lag augmentation;
- compute a signature-kernel Gram matrix or batch MMD;
- report the signature-kernel distance beside the existing market diagnostics.

This should be used to choose checkpoints, temperatures, top-k values, or bucket-specific sampling
settings only after the implementation has been validated on small synthetic and S&P500/VIX
batches.

Kernel methods are useful here because they compare paths without explicitly committing the model
selection criterion to one manually chosen finite log-signature truncation. The numerical kernel
implementation still has solver, static-kernel, and batching choices, so those choices must be
recorded in the evaluation manifest.

## 5. Signature-kernel objective

Objective-level use is deferred.

The promoted generator samples hard VQ tokens from an autoregressive prior. Hard token sampling is
not differentiable, so a signature-kernel loss on final sampled paths cannot be naively inserted
into the existing training loop. Possible future approaches would require one of the following:

- soft token relaxation before decoding;
- Gumbel-softmax sampling;
- straight-through estimators;
- applying the objective to differentiable decoded paths before hard sampling;
- a separate fine-tuning stage whose estimator bias and variance are explicitly measured.

These alternatives add modelling and optimisation risk. The correct order is therefore evaluation
and model selection first, followed only later by objective-level signature losses if the
evaluation metric proves informative and stable.

## 6. Candidate packages

Compatibility is not claimed until tested in temporary environments. No package in this table
should be added to `pyproject.toml` until the compatibility checklist passes and a separate
implementation plan accepts the dependency.

The current package decision is:

- `signatory` is deferred. It remains a reference for differentiable signatures, but the
  compatibility probe found it does not fit the current Python 3.12 and PyTorch 2.x project path
  without a legacy or custom environment.
- `iisignature` is the leading candidate for offline CPU truncated signature and log-signature
  feature extraction. It is not a dependency yet, and the documented source-build workaround must
  be accepted before implementation.
- `sigkernel` is the leading candidate for evaluation-only signature-kernel metrics. It is not a
  dependency yet, and the documented Cython and `--no-build-isolation` workaround must be
  reviewed before implementation.
- `KSig` is deferred until a GPU-compatible environment and a compatible Python/NumPy setup are
  available.
- `pathsig` remains untested and deferred.

| Package | Purpose | CPU/GPU support expectation | PyTorch compatibility expectation | Candidate status | Temporary install command |
| --- | --- | --- | --- | --- | --- |
| `signatory` | Differentiable signature and log-signature features. | Documentation describes CPU and CUDA support. Current wheels appear tied to older Python and PyTorch combinations. | Compatibility probe failed for the current Python 3.12 and PyTorch 2.x path. | Deferred/reference-only. | `python -m pip install "signatory==1.2.7.1.11.0" --no-cache-dir --force-reinstall` |
| `iisignature` | Efficient CPU signature and log-signature computation through a C++ extension. | CPU-oriented expectation. GPU support should not be assumed. | Treat as NumPy/C++ feature extraction unless native tensor behaviour is separately verified. | Leading candidate for offline CPU feature extraction. | `python -m pip install numpy && python -m pip install iisignature --no-build-isolation` |
| `sigkernel` | Signature-PDE-kernel computation, MMD, and scoring-rule style path comparisons. | CPU smoke passed; GPU remains unverified because local CUDA discovery failed. | PyTorch-based evaluation package; isolated probe used PyTorch 2.12 and produced a finite CPU Gram matrix. | Leading candidate for evaluation-only signature-kernel metrics. | `python -m pip install Cython && python -m pip install "git+https://github.com/crispitagorico/sigkernel.git" --no-build-isolation` |
| `KSig` | Scikit-learn-compatible signature-kernel package, including GPU-accelerated algorithms. | Probe indicates practical GPU/CuPy dependence; CPU fallback was not demonstrated. | Normal install failed on Python 3.12 due to the pinned NumPy path. | Deferred until a GPU-compatible and Python/NumPy-compatible environment is available. | `python -m pip install "git+https://github.com/tgcsaba/ksig.git"` |
| `pathsig` | Optional experimental signature package to inspect. | Not tested. | Not tested. | Deferred. | `python -m pip install pathsig` |

## 7. Causal/path-space distances

Path-space evaluation should be staged from existing metrics to heavier causal metrics.

1. Existing MMD and SWD from the paper-style S&P500/VIX evaluator remain the baseline
   distributional metrics.
2. Wasserstein distances on financial path functionals should be added before heavier causal
   optimal transport. Candidate one-dimensional marginals are one-step returns, terminal returns,
   path volatility, rolling volatility, and maximum drawdown.
3. Autocorrelation and squared-autocorrelation diagnostics should remain explicit because they
   directly test sequential dependence and volatility clustering.
4. Signature-kernel distance should be added as a path-level metric once package compatibility and
   numerical stability are verified.
5. Adapted or causal Wasserstein should be treated as the final heavy metric. It requires a
   selected formulation, computational budget, and a clear convention for filtrations and
   conditioning. It should not be implemented in this phase.

This ordering keeps the metric stack interpretable and avoids introducing a computationally heavy
distance before the lighter path diagnostics have been stabilised.

## 8. Experiment roadmap

### A. Signature feature extraction smoke

Use `iisignature`, if installed, to run offline CPU feature extraction on a small S&P500/VIX batch.
Build return, cumulative-return, time-channel, and optional lead-lag paths. Compute truncated
log-signatures at levels 2 and 3, then inspect dimensionality, finite values, runtime, and
sensitivity across VIX buckets.

### B. Signature-kernel evaluation metric

Use `sigkernel`, if installed, to add an evaluation-only signature-kernel distance for real versus
generated path batches. Start with small CPU batches and record preprocessing choices, static
kernel settings, solver settings, device, dtype, and runtime.

### C. VIX-only versus VIX-plus-signature conditioning ablation

Once feature extraction and evaluation are stable, compare the existing VIX-only additive prior
against a VIX-plus-log-signature condition vector. Keep tokenizer, prior family, sampling settings,
and evaluation protocol otherwise fixed. The first conditioning ablation should use the existing
additive condition embedding only.

### D. Model selection by market score plus signature-kernel metric

Use the existing market diagnostics as the primary guardrail and signature-kernel distance as an
additional path-space score. A checkpoint or sampling setting should not be promoted by
signature-kernel distance alone if stylised facts, tails, or volatility clustering degrade.

### E. Future objective-level loss or architecture changes

Only after the metric proves useful should the project revisit differentiable signature-kernel
objectives, richer conditioning architectures, or new tokenizers. This is the point where
relaxations, straight-through estimators, cross-attention, or new architectures would need a
separate design document.

## 9. Non-goals

This phase does not implement:

- diffusion;
- MGVQ;
- GroupedRVQ;
- cross-attention;
- adapted Wasserstein;
- signature-kernel training objectives;
- transition-aware losses;
- source-code changes or new package dependencies.

The deliverable for this phase is a rigorous extension plan and a package-compatibility checklist
that can guide a later implementation thread.
