# Hawkes-Jump Log-Return Robustness Results

## Status

This note records the three-seed Hawkes/SVMHJD log-return robustness run on
branch `research/hawkes-jump-discrete-benchmark`. The run used the Ogata
backend, `n_sample=1024`, seeds `0/1/2`, 50 tokenizer epochs, 50 prior epochs,
50 requested continuous epochs, W&B disabled, `temperature=1.0`, and
`top_k=none`.

The discrete cb64 log-return path completed for all seeds. The matched
continuous log-return BetaCVAE baseline did not produce a usable model: the
runner's module invocation returned without artefacts, and explicit
`tcvae-train` runs for seeds `0/1/2` failed at the first training batch with
`ArithmeticError: NaN detected in train loss`. The likely configuration issue is
that the log-return continuous config still carries `transform: log` and
`inverse_transform: exp`, while log returns are signed. No config or model code
was changed in this prompt.

## Run Matrix

| Candidate | Seeds | Training result | Evaluation result |
|---|---:|---|---|
| Continuous log-return BetaCVAE | 0, 1, 2 | Failed at epoch 1, batch 1 with NaN loss | No metrics |
| hidden128 log-return cb64 + additive AR | 0, 1, 2 | Passed | Passed |
| hidden128 log-return cb64 + causal conv-transformer k3 | 0, 1, 2 | Passed | Passed |

The discrete stages wrote summaries under
`outputs/hawkes_jump_logreturn_robustness`. The generated outputs remain
uncommitted.

## Continuous Baseline

| Seed | Status | Failure mode |
|---:|---|---|
| 0 | Failed | `NaN detected in train loss` before completing epoch 1 |
| 1 | Failed | `NaN detected in train loss` before completing epoch 1 |
| 2 | Failed | `NaN detected in train loss` before completing epoch 1 |

Because the continuous log-return baseline failed for all seeds, this run cannot
decide whether the discrete advantage is robust against a matched continuous
log-return TC-VAE. It can only assess discrete robustness and record that the
continuous baseline setup needs repair before registry-level comparison.

## Per-Seed Smooth Metrics

| Candidate | Seed | MMD | SWD | Terminal W1 | Volatility W1 | Drawdown W1 |
|---|---:|---:|---:|---:|---:|---:|
| cb64 additive AR | 0 | 0.0877 | 0.0150 | 0.0189 | 0.0007 | 0.0114 |
| cb64 additive AR | 1 | 0.2153 | 0.0320 | 0.0487 | 0.0020 | 0.0160 |
| cb64 additive AR | 2 | 0.1671 | 0.0244 | 0.0285 | 0.0007 | 0.0044 |
| cb64 conv-transformer k3 | 0 | 0.0734 | 0.0127 | 0.0134 | 0.0005 | 0.0084 |
| cb64 conv-transformer k3 | 1 | 0.1389 | 0.0246 | 0.0354 | 0.0021 | 0.0171 |
| cb64 conv-transformer k3 | 2 | 0.1300 | 0.0185 | 0.0162 | 0.0003 | 0.0079 |

| Candidate | MMD mean/std | SWD mean/std | Terminal W1 mean/std | Volatility W1 mean/std | Drawdown W1 mean/std |
|---|---:|---:|---:|---:|---:|
| cb64 additive AR | 0.1567 / 0.0644 | 0.0238 / 0.0085 | 0.0320 / 0.0152 | 0.0011 / 0.0008 | 0.0106 / 0.0059 |
| cb64 conv-transformer k3 | 0.1141 / 0.0355 | 0.0186 / 0.0060 | 0.0217 / 0.0120 | 0.0010 / 0.0010 | 0.0111 / 0.0052 |

The conv-transformer has the stronger smooth profile across seeds: lower mean
MMD, SWD, and terminal W1, with comparable volatility and drawdown W1.

## Jump-Regime Metrics

The seed-0 real Ogata evaluation reference had mean detected jumps per path
`0.2422`, paths-with-jumps fraction `0.1943`, negative detected jump fraction
`0.9718`, VaR 1% `-0.0724`, and ES 1% `-0.1064`.

| Candidate | Seed | Jump-count W1 | Inter-arrival W1 | Jump-size W1 | Negative jump frac. | VaR 1% | ES 1% |
|---|---:|---:|---:|---:|---:|---:|---:|
| cb64 additive AR | 0 | 0.0352 | 3.7199 | 0.0128 | 1.0000 | -0.0717 | -0.0988 |
| cb64 additive AR | 1 | 0.0830 | 3.9512 | 0.0296 | 0.9864 | -0.0773 | -0.1147 |
| cb64 additive AR | 2 | 0.0225 | 11.3616 | 0.0116 | 1.0000 | -0.0744 | -0.1070 |
| cb64 conv-transformer k3 | 0 | 0.0430 | 7.3073 | 0.0121 | 1.0000 | -0.0746 | -0.1006 |
| cb64 conv-transformer k3 | 1 | 0.0947 | 1.8931 | 0.0293 | 0.9967 | -0.0775 | -0.1159 |
| cb64 conv-transformer k3 | 2 | 0.0352 | 15.1222 | 0.0117 | 1.0000 | -0.0724 | -0.1042 |

| Candidate | Jump-count W1 mean/std | Inter-arrival W1 mean/std | Jump-size W1 mean/std | Negative jump frac. mean/std | VaR 1% mean/std | ES 1% mean/std |
|---|---:|---:|---:|---:|---:|---:|
| cb64 additive AR | 0.0469 / 0.0319 | 6.3442 / 4.3467 | 0.0180 / 0.0101 | 0.9955 / 0.0078 | -0.0745 / 0.0028 | -0.1068 / 0.0080 |
| cb64 conv-transformer k3 | 0.0576 / 0.0324 | 8.1075 / 6.6507 | 0.0177 / 0.0101 | 0.9989 / 0.0019 | -0.0748 / 0.0026 | -0.1069 / 0.0080 |

Both discrete priors preserve the rare-event scale much better than the earlier
price-tokenizer comparison. Additive AR is slightly better on mean jump-count W1
and inter-arrival W1 in this run. The conv-transformer remains close on jump
metrics while improving the smooth profile.

## Token Diagnostics

The extracted tokenizer codebooks remained active across seeds:

| Seed | Extracted active codes | Extracted perplexity |
|---:|---:|---:|
| 0 | 63 / 64 | 44.80 |
| 1 | 64 / 64 | 44.00 |
| 2 | 64 / 64 | 43.16 |

| Candidate | Seed | Sampled active codes | Sampled perplexity | Transition L1 | Run-length W1 | Best CE | Best acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| cb64 additive AR | 0 | 64 | 44.93 | 0.4201 | 0.0022 | 3.6963 | 0.0640 |
| cb64 additive AR | 1 | 64 | 44.58 | 0.4504 | 0.0067 | 3.6658 | 0.0619 |
| cb64 additive AR | 2 | 64 | 43.58 | 0.4318 | 0.0004 | 3.6248 | 0.0646 |
| cb64 conv-transformer k3 | 0 | 64 | 45.18 | 0.4151 | 0.0015 | 3.6836 | 0.0626 |
| cb64 conv-transformer k3 | 1 | 64 | 44.50 | 0.4499 | 0.0059 | 3.6388 | 0.0623 |
| cb64 conv-transformer k3 | 2 | 64 | 43.58 | 0.4385 | 0.0037 | 3.6286 | 0.0625 |

| Candidate | Sampled perplexity mean/std | Transition L1 mean/std | Run-length W1 mean/std | Best CE mean/std |
|---|---:|---:|---:|---:|
| cb64 additive AR | 44.36 / 0.70 | 0.4341 / 0.0153 | 0.0031 / 0.0033 | 3.6623 / 0.0359 |
| cb64 conv-transformer k3 | 44.42 / 0.80 | 0.4345 / 0.0177 | 0.0037 / 0.0022 | 3.6503 / 0.0292 |

The tokenizer-utilisation fix is robust: both priors sampled all 64 codes for
all seeds, and sampled perplexity remained close to the extracted-token
perplexity. The conv-transformer has a small prior-fit edge on mean best CE, but
not a decisive transition or run-length advantage.

## Runtime

| Candidate | Train runtime mean/std, seconds | Evaluation runtime mean/std, seconds |
|---|---:|---:|
| cb64 additive AR | 96.49 / 0.26 | 36.13 / 0.12 |
| cb64 conv-transformer k3 | 112.99 / 0.18 | 42.65 / 0.39 |

Tokenizer training took approximately 38 seconds per seed including runner
overhead. Token extraction took approximately 5.5 seconds per seed. The
continuous baseline did not produce a meaningful runtime because it failed
before completing the first training batch.

## Failure Modes

- The continuous runner stage used a module invocation that returned without
  creating continuous artefacts. Explicit `tcvae-train` runs were therefore
  attempted for seeds `0/1/2`.
- All explicit continuous log-return BetaCVAE attempts failed with NaN loss at
  the first batch. The current log-return config applies a logarithmic transform
  to signed log returns, so the failure is consistent with invalid transformed
  inputs.
- The discrete priors still show noisy inter-arrival W1 across seeds because
  detected jumps are sparse; seed-level rankings should not be over-read.

## Decision

The discrete log-return result is internally robust across three seeds: code
usage does not collapse, sampled codebooks remain active, and both cb64 priors
recover the sparse downside-jump scale with VaR/ES close to the Ogata reference.
The cb64 conv-transformer is the best smooth-path candidate, while cb64 additive
AR is the slightly stronger jump-count/inter-arrival baseline.

The benchmark decision is nevertheless **inconclusive against a matched
continuous log-return baseline**, because the continuous BetaCVAE failed before
training. Do not update the registry yet. The next exact step is to repair the
continuous log-return baseline configuration without changing simulator
parameters, most likely by using an identity or otherwise signed-data-safe
transform for log returns, then rerun the same three-seed comparison.
