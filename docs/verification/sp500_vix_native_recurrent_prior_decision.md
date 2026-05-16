# S&P500/VIX Native Recurrent Prior Decision

Status: sampling ablation completed on 2026-05-16. No model was trained, no source code was
changed, no dependency was added, no tokenizer code was changed, and no public default was changed.

## Native Recurrent Result

The evaluated checkpoint is:

```text
outputs/sp500_vix_discrete/token_prior/hidden128_native_recurrent/sp500_vix_causal_token_prior_hidden128_native_recurrent_seed99/best_model
```

Training likelihood from the existing checkpoint:

| Metric | Value |
| --- | ---: |
| Best epoch | 100 |
| Eval CE | 1.351029 |
| Eval accuracy | 0.480260 |
| Eval perplexity | 3.890610 |

The initial unrestricted sampling result at `temperature=1.0`, `top_k=none` was weak:

| Metric | Value |
| --- | ---: |
| Paper-style profile | 0.250494 |
| MMD | 0.218727 |
| SWD | 0.011493 |
| Terminal W1 | 0.016306 |
| Volatility W1 | 0.003969 |
| Transition L1 | 0.246062 |
| Run-length W1 | 0.238527 |
| Sampled active codes | 64/64 |

The checkpoint did not collapse token usage, but it overproduced volatility and tails, and it did
not preserve token persistence.

## Sampling Grid

Paper-style diagnostics were run with `n_sample=1000` and `seed=99` for all settings.

| Setting | Profile | MMD | SWD | Terminal W1 | Vol W1 | Returns W1 | Drawdown W1 | Sq-ret AC L1 | Transition L1 | Run W1 | Active | Token PPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| temp10_topk20 | 0.211580 | 0.193175 | 0.008756 | 0.007724 | 0.001926 | 0.000899 | 0.010284 | 0.044374 | 0.195417 | 0.244741 | 62/64 | 44.664780 |
| temp10_topk40 | 0.233295 | 0.206751 | 0.012729 | 0.010016 | 0.003800 | 0.001499 | 0.020521 | 0.051359 | 0.257415 | 0.237101 | 64/64 | 46.201637 |
| temp08_topk20 | 0.241709 | 0.222396 | 0.007445 | 0.010636 | 0.001231 | 0.000944 | 0.009889 | 0.055843 | 0.204961 | 0.410562 | 61/64 | 42.186558 |
| temp10_topknone | 0.250494 | 0.218727 | 0.011493 | 0.016306 | 0.003969 | 0.001509 | 0.019013 | 0.049929 | 0.246062 | 0.238527 | 64/64 | 45.977097 |
| temp08_topknone | 0.257026 | 0.233841 | 0.007883 | 0.013953 | 0.001349 | 0.001060 | 0.011144 | 0.058922 | 0.206508 | 0.416440 | 62/64 | 41.505722 |
| temp08_topk40 | 0.258380 | 0.238302 | 0.007673 | 0.010967 | 0.001437 | 0.001025 | 0.011347 | 0.058069 | 0.223128 | 0.409369 | 63/64 | 41.763748 |
| temp06_topk20 | 0.427186 | 0.398182 | 0.011815 | 0.015133 | 0.002056 | 0.002025 | 0.021560 | 0.074716 | 0.288285 | 0.923761 | 59/64 | 36.519165 |
| temp06_topk40 | 0.439970 | 0.409023 | 0.012893 | 0.015944 | 0.002111 | 0.002071 | 0.022074 | 0.076017 | 0.326134 | 0.913079 | 61/64 | 36.240421 |
| temp06_topknone | 0.442725 | 0.412365 | 0.012560 | 0.015702 | 0.002097 | 0.002086 | 0.022276 | 0.077456 | 0.296378 | 0.940340 | 61/64 | 36.112846 |

The low-temperature settings are clearly poor. They reduce token entropy but worsen the main
profile and run-length diagnostics. The `temperature=0.8` settings improve SWD and volatility W1
in some cases, but run-length W1 becomes very large. The only competitive setting is
`temperature=1.0`, `top_k=20`.

## Best Setting

The best native recurrent setting by paper-style profile is:

```text
temperature=1.0
top_k=20
```

| Metric | Best native recurrent |
| --- | ---: |
| Profile | 0.211580 |
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

This setting substantially improves over unrestricted sampling by reducing MMD, terminal W1,
volatility W1, transition L1, and tail exceedance. However, it still does not solve the persistence
problem: run-length W1 remains `0.244741`, much worse than the hidden128 conv-transformer k3 value
of `0.050189`.

## Comparison

| Model | Profile | MMD | SWD | Terminal W1 | Vol W1 | Drawdown W1 | Sq-ret AC L1 | Transition L1 | Run W1 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| continuous BetaCVAE | 0.172891 | 0.154421 | 0.008785 | 0.009051 | 0.000634 | 0.007667 | 0.029462 | n/a | n/a | strongest non-discrete reference |
| hidden128 conv-transformer k3 | 0.186725 | 0.169510 | 0.010918 | 0.005086 | 0.001210 | 0.007687 | 0.050871 | 0.193243 | 0.050189 | best discrete research model |
| native recurrent, temp 1.0, top-k 20 | 0.211580 | 0.193175 | 0.008756 | 0.007724 | 0.001926 | 0.010284 | 0.044374 | 0.195417 | 0.244741 | best native recurrent setting |
| hidden128 additive | 0.242020 | 0.229078 | 0.007175 | 0.004509 | 0.001258 | 0.008910 | 0.060885 | n/a | n/a | former hidden128 prior |
| promoted standard VQ + additive AR | 0.298020 | 0.279341 | 0.007674 | 0.009817 | 0.001188 | 0.010502 | 0.041300 | n/a | n/a | public baseline |

The sampling ablation improves the native recurrent prior enough to beat hidden128 additive and the
promoted public baseline on the main profile. It still does not beat the hidden128 conv-transformer
k3 primary setting. The decisive failure is not code usage or MMD alone, but persistence and
stylised dynamics: run-length W1 remains much worse than conv-transformer k3, and volatility W1,
terminal W1, and drawdown W1 are all weaker than the conv-transformer result.

## Decision

Reject the current native recurrent hidden128 prior as the next discrete research prior.

Return to the hidden128 causal conv-transformer k3 as the best discrete research prior. The
conv-transformer k3 remains the best discrete research model after the native recurrent sampling
ablation.

Do not continue with seed robustness for this exact native recurrent checkpoint. The best sampling
setting is not strong enough to justify seed1 and seed2 training runs. Do not change public
defaults.

The recurrent direction is not closed permanently, but it should not continue as a sampling-only
exercise. If revisited, tune architecture before more sampling: use a larger recurrent hidden
state, more recurrent layers with dropout, or a more controlled native SSM-style transition, then
repeat the no-leakage, stepwise-equivalence, and full paper-style evaluation checks. No exact next
config or seed robustness step is specified because this checkpoint is rejected.
