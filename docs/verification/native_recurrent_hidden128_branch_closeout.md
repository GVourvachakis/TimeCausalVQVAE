# Native Recurrent Hidden128 Branch Closeout

Status: branch closeout completed on 2026-05-16. No code was implemented for this closeout, no
model was trained, no dependency was added, no tokenizer code was changed, and no public default
was changed.

## What Was Implemented

The branch implemented the first dependency-free native PyTorch recurrent prior for the hidden128
single-stream token interface:

- prior type: `native_recurrent`;
- recurrent core: one-layer `torch.nn.GRU` with hidden size `128`;
- autoregressive target convention: `input_t = BOS` for `t=0`, otherwise `token_{t-1}`, with
  `target_t = token_t`;
- token and learned position embeddings following the existing single-stream prior convention;
- additive scalar VIX condition projection with `condition_dim=1`;
- output projection to 64 codebook logits;
- stepwise sampling with recurrent state updated exactly once per generated token.

The implementation kept the hidden128 tokenizer fixed and did not introduce Mamba, `mambapy`,
`mamba-ssm`, `causal-conv1d`, GroupedRVQ, MGVQ, diffusion, signatures, cross-attention, or a new
objective.

## What Passed

The native recurrent prior passed the source and evaluation gates required for the branch:

| Check | Result |
| --- | --- |
| No-leakage | Passed. Perturbing tokens after cutoff `29` left prefix logits unchanged with `max_prefix_diff=0.00000000e+00`. |
| Full-sequence versus stepwise equivalence | Passed under deterministic evaluation with `max_stepwise_diff=0.00000000e+00`. |
| One-epoch smoke training | Passed with final eval CE `3.97471046`, eval accuracy `0.08564645`, and eval perplexity `53.28947747`. |
| Non-smoke train and evaluate | Passed. The seed99 checkpoint trained for 100 epochs with eval CE `1.351029`, eval accuracy `0.480260`, and eval perplexity `3.890610`. |
| Sampling ablation | Completed over temperatures `0.6`, `0.8`, and `1.0`, with `top_k` values `none`, `20`, and `40`, using `n_sample=1000` and seed `99`. |

The branch therefore produced valid negative evidence. The model obeyed the causal contract and
ran through the same train, decode, paper-style, and sampling-ablation workflow used for the other
hidden128 priors.

## Best Native Recurrent Result

The best native recurrent sampling setting by paper-style profile was:

```text
temperature=1.0
top_k=20
```

| Metric | Native recurrent, best setting |
| --- | ---: |
| Paper-style profile | 0.211580 |
| MMD | 0.193175 |
| SWD | 0.008756 |
| Terminal W1 | 0.007724 |
| Volatility W1 | 0.001926 |
| Returns W1 | 0.000899 |
| Drawdown W1 | 0.010284 |
| Squared-return AC L1 | 0.044374 |
| Transition L1 | 0.195417 |
| Run-length W1 | 0.244741 |
| Sampled active codes | 62/64 |
| Sampled token perplexity | 44.664780 |

This setting improved materially over unrestricted native recurrent sampling at `temperature=1.0`,
`top_k=none`, whose profile was `0.250494`. It was nevertheless not strong enough to displace the
current hidden128 conv-transformer k3 research prior.

## Rejection Rationale

Reject this exact native recurrent hidden128 prior as the next discrete research prior.

The best native recurrent profile, `0.211580`, is worse than the hidden128 causal conv-transformer
k3 profile of `0.186725`. The recurrent model also fails the persistence motivation for the
branch: its best transition L1, `0.195417`, is close to conv-transformer k3, but its run-length W1,
`0.244741`, is far worse than the conv-transformer k3 value of `0.050189`.

The unrestricted native recurrent evaluation also showed volatility and tail overproduction, with
volatility W1 `0.003969`, terminal W1 `0.016306`, and excessive sampled terminal-return outliers.
Top-k sampling improved the main profile, but did not repair run-length dynamics or produce a
better overall discrete prior.

There is no reason to run seed robustness for this exact GRU-128, one-layer configuration. The
branch should be closed as valid negative evidence rather than extended through seed1 and seed2
training.

## Dependency Decision

Do not adopt `mambapy` now. The isolated CPU probe showed that a small forward pass can run, but
the package still needs a maintenance, API, causal-cache, and stepwise-equivalence audit before it
can be considered as a project dependency.

Do not adopt `mamba-ssm` now. The local environment could not validate the compiled CUDA package
chain because `torch.cuda.is_available()` was `False` under the installed driver and torch build.
`causal-conv1d` was skipped for the same reason and should not be introduced as a hidden
dependency.

Keep the native recurrent implementation as negative research evidence. It is valuable because it
demonstrates that a dependency-free recurrent prior can satisfy the project causality contract,
but this particular architecture does not improve the hidden128 prior frontier.

## Current Best Discrete Research Model

The current best discrete research model remains:

```text
hidden128 tokenizer + causal conv-transformer k3 prior
temperature=1.0
top_k=none
```

Key seed99 metrics are profile `0.186725`, MMD `0.169510`, SWD `0.010918`, terminal W1
`0.005086`, volatility W1 `0.001210`, transition L1 `0.193243`, and run-length W1 `0.050189`.

The promoted public baseline remains unchanged. The conv-transformer k3 result should be treated
as the best discrete research prior, not as a public-default replacement.

## Future SSM Conditions

Revisit SSM-style priors only with a more principled state-space design than this simple GRU
baseline. A future branch should begin with a package, API, checkpoint-portability, and
stepwise-cache audit before implementation, especially if it considers `mambapy`, `mamba-ssm`, or
`causal-conv1d`.

Any future SSM candidate must compare directly against hidden128 causal conv-transformer k3 under
the same tokeniser, scalar VIX condition, sequence length, sampling policy grid, and paper-style
diagnostics. Lower token cross-entropy alone is not sufficient. The candidate must preserve or
improve generated-market profile, volatility and tail behaviour, transition dynamics, run-length
persistence, and sampled code usage.
