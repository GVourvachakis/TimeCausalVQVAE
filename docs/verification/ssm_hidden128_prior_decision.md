# SSM Hidden128 Prior Decision

Status: decision note only. No source code was changed, no dependency was added, and no model was
trained for this decision.

## Current Best Discrete Research Model

The current best discrete research model remains the hidden128 standard-VQ tokenizer with the
causal conv-transformer k3 prior:

- tokenizer config: `configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml`;
- prior config:
  `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml`;
- prior type: `causal_conv_transformer`;
- local front-end: two strictly causal convolution layers, kernel size `3`, dilations `[1, 2]`;
- condition: scalar VIX with `condition_dim=1`;
- primary sampling: temperature `1.0`, unrestricted top-k.

The selected seed99 result has paper-style profile `0.186725`, with MMD `0.169510`, SWD
`0.010918`, terminal W1 `0.005086`, volatility W1 `0.001210`, transition L1 `0.193243`, and
run-length W1 `0.050189`. Seed robustness still shows sensitivity: seed99, seed1, and seed2 all
beat the hidden128 additive prior profile of `0.242020`, but the conv-transformer profile range is
`0.186725` to `0.237086`.

Therefore, hidden128 plus causal conv-transformer k3 remains the best current discrete research
model and the required comparison baseline for any SSM follow-up.

## Motivation For SSM

The SSM direction is motivated by prior calibration rather than tokenizer capacity. The hidden128
single-stream tokenizer remains the preferred discrete interface, uses one token per time step,
and avoids the sampled-token collapse observed in earlier multi-component prior directions.

An SSM-style prior is still worth a bounded implementation branch because it targets the residual
weaknesses of the conv-transformer:

- prior calibration: the best generated-market profile depends on sampling policy and still trails
  the continuous BetaCVAE reference;
- seed sensitivity: the conv-transformer k3 result holds directionally but has a non-trivial
  profile spread across seeds;
- persistent token dynamics: recurrent state may represent token persistence more directly than a
  short causal convolution front-end;
- local volatility dynamics: a prefix-only state update may improve volatility clustering and
  run-length behaviour without changing the hidden128 tokenizer.

## Package Results

The isolated compatibility probe created `/tmp/tcvae-ssm-probe` and did not modify the project
environment or `pyproject.toml`.

| Candidate | Result | Decision relevance |
| --- | --- | --- |
| `mamba-ssm` | Skipped locally. The active Poetry environment uses torch `2.11.0+cu130`, but `torch.cuda.is_available()` is `False` because the installed NVIDIA driver is too old for that torch CUDA build. | Do not implement now. The package remains high risk until tested in a CUDA-compatible environment with matching Python, torch, CUDA, and driver versions. |
| `causal-conv1d` | Skipped locally for the same CUDA-runtime reason. | Do not use as a prerequisite for the first SSM branch. It should be reprobed before any `mamba-ssm` implementation. |
| `mambapy` | Installed in the isolated CPU environment as `mambapy==1.2.0`; a small CPU forward pass produced shape `(2, 16, 32)`. | Useful as a reference probe, but not selected as a project dependency because its maintenance, inference-cache API, and strict stepwise causal behaviour still need audit. |
| Native PyTorch fallback | A GRU-style recurrent fallback smoke test produced logits with shape `(2, 16, 64)` for a 64-code vocabulary and `condition_dim=1`. | Selected for the first implementation branch because it requires no new dependency and can be tested directly against the existing causal token-prior contract. |

The package evidence does not justify adopting the reference Mamba CUDA stack yet. It does justify
a dependency-free native PyTorch recurrent or simple SSM fallback branch.

## Causality Requirements

Any selected SSM-style prior must satisfy the same autoregressive contract as the existing
hidden128 priors:

- logits at time `t` may depend only on tokens strictly before `t` and on the allowed scalar VIX
  condition;
- recurrent state must update left-to-right during sampling;
- no bidirectional scan, reverse scan, centred convolution, or full-sequence normalisation may
  enter the generation path;
- the training scan must match stepwise sampling within numerical tolerance;
- no-future-leakage checks must cover full-sequence evaluation and explicit stepwise generation.

The first implementation must treat state-cache behaviour as part of the public prior contract,
not as an optimisation to be added after training.

## Decision

Do not implement a `mamba-ssm` prior now. The CUDA package chain was not locally validated, and the
current machine cannot distinguish package failure from driver-runtime incompatibility.

Do not implement a `mambapy` prior now. The CPU forward pass succeeded, but adopting an additional
package for the project would require API, maintenance, and causal step-cache review. Passing a
shape smoke test is not enough to justify a new dependency.

Implement a native PyTorch recurrent or simple SSM fallback as the next prior-family branch. This
is the selected path because it preserves the hidden128 interface, uses the existing torch
dependency, avoids CUDA-extension packaging risk, and directly tests whether persistent
left-to-right state helps the seed-sensitivity and token-persistence bottleneck.

Do not defer SSM entirely. The compatibility results support a low-risk native fallback branch,
while leaving the reference Mamba stack for a later CUDA-compatible probe.

Until that branch produces trained and evaluated results, hidden128 plus causal conv-transformer k3
remains the best discrete research model.

## Proceeding Plan

Branch name:

```text
research/native-ssm-hidden128-prior
```

Dependency policy:

- do not add `mamba-ssm`, `causal-conv1d`, `mambapy`, or any other new package to
  `pyproject.toml`;
- implement the first candidate with existing PyTorch modules only;
- keep the public baseline and public defaults unchanged;
- keep the hidden128 tokenizer fixed.

Required smoke and no-leakage checks:

- source smoke for model construction, forward shape, loss compatibility, and sampling shape;
- no-future-leakage test where changing future tokens does not change logits at earlier times;
- equivalence test between batched full-sequence evaluation and stepwise left-to-right sampling;
- recurrent state-cache test that verifies exactly one update per generated time step;
- CPU execution smoke, because the selected fallback must not depend on CUDA;
- existing hidden128 sampling and evaluation pipeline compatibility checks.

First config:

```text
configs/experiments/sp500_vix_causal_token_prior_hidden128_native_ssm.yaml
```

The first config should keep:

- tokenizer path:
  `outputs/sp500_vix_discrete/vq_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_hidden128_seed99`;
- token-data path:
  `outputs/sp500_vix_discrete/token_prior/sp500_vix_causal_vq_tokenizer_hidden128_tokens`;
- `condition_dim: 1`;
- one hidden128 token per time step over the same 64-code vocabulary;
- the same sampling and paper-style evaluation pipeline as the conv-transformer k3 prior.

Initial reporting should compare the native SSM fallback against:

- hidden128 conv-transformer k3, temperature `1.0`, unrestricted top-k;
- hidden128 additive AR, temperature `0.8`, `top_k=20`;
- promoted standard-VQ additive AR public baseline.

Lower cross-entropy alone must not be sufficient for selection. The branch must improve or at
least preserve generated-market diagnostics, sampled token persistence, and code-usage stability
without introducing sampled-token collapse.
