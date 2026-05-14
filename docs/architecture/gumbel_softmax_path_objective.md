# Gumbel-Softmax and Path-Level Objectives

Status: architecture decision only. This document does not implement
Gumbel-Softmax, modify model code, train models, or add a new objective.

## 1. Current Training

The promoted discrete S&P500/VIX workflow trains a causal autoregressive token
prior over frozen VQ-token sequences. The tokenizer is trained separately and
then held fixed. Token-prior training uses teacher-forced cross-entropy against
the observed discrete code at each causal time step.

Under this objective, the training signal is already differentiable with
respect to the prior parameters. The model receives the previous ground-truth
tokens as context, predicts a categorical distribution over the next token, and
optimises the negative log-likelihood of the target token. No decoded path is
needed to compute the token cross-entropy loss.

## 2. Why ST-GS Is Not Needed For Current Training

Straight-through Gumbel-Softmax is not needed for the current token-prior
objective because there is no differentiability bottleneck in teacher-forced
cross-entropy. The loss is evaluated directly on logits before sampling.

Adding ST-GS to the current training loop would therefore solve the wrong
problem. It would introduce a sampled or relaxed token path even though the
existing objective does not require gradients through sampled tokens or through
the decoder. For the current baseline and signature-conditioning ablations, the
right comparison remains:

- frozen tokenizer;
- teacher-forced categorical cross-entropy;
- hard-token sampling only at evaluation time;
- decoded-path diagnostics reported after sampling.

## 3. When ST-GS Becomes Relevant

ST-GS becomes relevant only if a future training objective needs gradients
through sampled token choices and the frozen decoder into a path-level loss.
That would require a differentiable approximation to the discrete sampling
step:

```text
prior logits -> relaxed token/sample -> decoder -> decoded path -> path loss
```

Examples include objective-level losses on generated returns, volatility,
drawdown, or path-space distances. In that setting, hard categorical sampling
blocks gradients, so a relaxation such as Gumbel-Softmax or a
straight-through estimator could be considered.

This is a different training regime from the current token CE setup. It should
be designed as a separate fine-tuning or auxiliary-objective phase, not folded
into the baseline token-prior trainer by default.

## 4. Candidate Objective

The most plausible future objective is a signature-kernel score on decoded
paths, because the signature roadmap already treats signature-kernel MMD as an
evaluation-only path metric. However, objective-level use should be considered
only after the evaluation metric is stable.

The required order is:

1. install and validate `sigkernel` in an optional environment;
2. run synthetic and saved-batch signature-kernel MMD checks;
3. confirm that the metric gives useful model-selection signal beside MMD, SWD,
   Wasserstein path functionals, autocorrelation diagnostics, and tail
   diagnostics;
4. only then decide whether a differentiable signature-kernel training loss is
   worth the optimisation risk.

Until those steps are complete, ST-GS has no accepted path-level objective to
serve.

## 5. Risks

ST-GS and related relaxations carry several risks for this project:

- temperature sensitivity: the relaxation temperature can dominate training
  behaviour and requires its own schedule;
- train/inference mismatch: training on relaxed or straight-through codes may
  not match hard categorical sampling at evaluation time;
- biased gradients: straight-through estimators provide biased gradient
  estimates whose effect must be measured;
- soft-code decoder mismatch: the frozen decoder was trained on discrete code
  embeddings, so convex mixtures or relaxed code paths may produce
  out-of-distribution decoder inputs;
- extra hyperparameters: temperature, annealing, estimator choice, loss weight,
  path-batch size, and kernel settings add a large tuning surface;
- metric overfitting: optimising one path metric directly may improve that
  metric while degrading MMD, SWD, tail diagnostics, or sequential-dependence
  checks.

These risks are acceptable only if the chosen path-level objective is already
shown to be informative as an evaluation metric.

## 6. Decision

Defer Gumbel-Softmax and straight-through Gumbel-Softmax.

The current training objective remains teacher-forced cross-entropy over frozen
VQ tokens. ST-GS should be reconsidered only after a differentiable
path-level objective is selected, with signature-kernel scoring as the leading
candidate once the optional `sigkernel` metric is stable and useful for model
selection.
