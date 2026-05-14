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

| Package | Purpose | CPU/GPU support expectation | PyTorch compatibility expectation | Candidate status | Temporary install command |
| --- | --- | --- | --- | --- | --- |
| `signatory` | Differentiable signature and log-signature features. | Documentation describes CPU and CUDA support. Current wheels appear tied to older Python and PyTorch combinations. | Inspect in a separate Python/PyTorch environment first; the current project uses Python 3.12 and PyTorch 2.x, which may be incompatible with the published wheel scheme. | Inspection-only unless a maintained compatible build is found. | `python -m pip install "signatory==1.2.7.1.11.0" --no-cache-dir --force-reinstall` |
| `iisignature` | Efficient CPU signature and log-signature computation through a C++ extension. | CPU-oriented expectation. GPU support should not be assumed. | May support machine-learning workflows but should be treated as NumPy/C++ feature extraction unless native tensor behaviour is verified. | Possible offline feature-extraction candidate, not an immediate model dependency. | `python -m pip install iisignature` |
| `sigkernel` | Signature-PDE-kernel computation, MMD, and scoring-rule style path comparisons. | Repository describes CPU and GPU support through PyTorch with automatic device selection. | Promising evaluation candidate, but current PyTorch 2.x compatibility must be tested. | Primary inspection candidate for signature-kernel metrics. | `python -m pip install "git+https://github.com/crispitagorico/sigkernel.git"` |
| `KSig` | Scikit-learn-compatible signature-kernel package, including GPU-accelerated algorithms. | Documentation emphasises GPU acceleration through CuPy and CUDA setup. CPU fallback must be tested rather than assumed. | Not a native PyTorch package; likely NumPy/CuPy/scikit-learn oriented. Use as a metric cross-check, not as a training dependency. | Inspection-only first, useful for numerical consistency checks against `sigkernel`. | `python -m pip install "git+https://github.com/tgcsaba/ksig.git"` |
| `pathsig` | Optional experimental signature package to inspect. | Current PyPI metadata should be inspected before assuming CPU or GPU behaviour. | PyTorch compatibility must be verified by import and tensor smoke tests. | Optional/current experimental package; inspection-only. | `python -m pip install pathsig` |

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

### A. Package compatibility checks

Create temporary environments and run import, CPU, GPU, and small numerical checks for each
candidate package. Record failures as compatibility results, not as project failures.

### B. Signature feature extraction smoke

On a small S&P500/VIX batch, build return, cumulative-return, time, and lead-lag paths. Compute
truncated log-signatures at low levels and inspect feature dimensionality, finite values, runtime,
and sensitivity across VIX buckets.

### C. VIX-only versus VIX-plus-signature conditioning ablation

Once feature extraction is stable, compare the existing VIX-only additive prior against a
VIX-plus-signature condition vector. Keep tokenizer, prior family, sampling settings, and
evaluation protocol otherwise fixed.

### D. Signature-kernel evaluation metric

Add an evaluation-only signature-kernel distance for real versus generated path batches. Start
with small batch sizes and record preprocessing choices, static kernel settings, solver settings,
device, dtype, and runtime.

### E. Model selection by market score plus signature-kernel metric

Use the existing market diagnostics as the primary guardrail and signature-kernel distance as an
additional path-space score. A checkpoint or sampling setting should not be promoted by
signature-kernel distance alone if stylised facts, tails, or volatility clustering degrade.

### F. Future objective-level loss or architecture changes

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
