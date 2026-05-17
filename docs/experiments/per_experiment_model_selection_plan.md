# Per-Experiment Model-Selection Plan

## Scope

This plan defines how to select continuous and discrete TimeCausalVQVAE variants for
Black-Scholes, Heston, PDV4, and S&P500/VIX. It is a documentation plan only. It does not train
models, add generated outputs, modify `main`, change objectives, or introduce MGVQ, GroupedRVQ,
diffusion, signatures, or other new model families.

The current branch contains selected continuous configs for all four experiments, public or smoke
discrete configs for Black-Scholes, PDV4, and S&P500/VIX, and no committed Heston discrete
experiment config. The S&P500/VIX public discrete baseline remains the promoted public baseline in
`README.md` and `trained_models/model_registry.yaml`.

## Experiments

| Experiment | Dataset name in configs | Conditioning policy | Selection emphasis |
| --- | --- | --- | --- |
| Black-Scholes | `black_scholes` | Use the committed continuous condition; keep the public discrete smoke path unconditioned unless a later config explicitly adds an observable scalar condition. | Terminal distribution, path distribution, volatility fit, and no-leakage reconstruction. |
| Heston | `heston` | Use the committed continuous condition; there is no committed public discrete Heston config on this branch. | Volatility dynamics, terminal distribution, path distribution, and autocorrelation diagnostics. |
| PDV4 | `path_dependent_volatility` | Use `r2_volatility_feature` for conditional continuous and discrete variants. | Volatility W1, drawdown W1, squared-return AC L1, condition-bucket stability, and distributional guardrails. |
| S&P500/VIX | `sp500_vix` | Use observed VIX context through scalar additive conditioning. | Market-style profile, VIX-bucket diagnostics, token geometry, tail risk, and sequential dependence. |

## Baselines

### Continuous TC-VAE Selected Configs

| Experiment | Continuous baseline | Objective | Notes |
| --- | --- | --- | --- |
| Black-Scholes | `configs/experiments/black_scholes_beta_cvae.yaml` | `beta_cvae` | Selected reproduction wrapper: `scripts/reproduce_black_scholes.py`. |
| Heston | `configs/experiments/heston_info_cvae.yaml` | `info_cvae` | Selected reproduction wrapper: `scripts/reproduce_heston.py`. |
| PDV4 | `configs/experiments/pdv_info_cvae.yaml` | `info_cvae` | Conditional baseline with `r2_volatility_feature`. |
| S&P500/VIX | `configs/experiments/sp500_vix_beta_cvae.yaml` | `beta_cvae` | Continuous BetaCVAE remains the strongest overall reference in current S&P500/VIX evidence. |

### Public Discrete Baseline

| Experiment | Public discrete baseline status | Configs |
| --- | --- | --- |
| Black-Scholes | Available as a small discrete smoke workflow. Treat it as a local baseline, not as a promoted public result. | `configs/experiments/black_scholes_causal_vq_tokenizer.yaml`; `configs/experiments/black_scholes_causal_token_prior.yaml`. |
| Heston | Not present on this branch. A future Heston discrete baseline must first add a standard VQ plus causal AR config in a separate experiment branch. | None committed. |
| PDV4 | Available as a conditional standard-VQ plus additive causal AR workflow. Treat it as the PDV4 public discrete baseline for selection once trained locally. | `configs/experiments/pdv_causal_vq_tokenizer_codebook64_codebookdim16.yaml`; `configs/experiments/pdv_causal_token_prior_additive_seed1.yaml`. |
| S&P500/VIX | Promoted public discrete baseline: standard causal VQ tokenizer plus scalar VIX additive causal AR prior. | `configs/experiments/sp500_vix_causal_vq_tokenizer.yaml`; `configs/experiments/sp500_vix_causal_token_prior_additive.yaml`. |

## Candidate Discrete Variants

All candidates keep the one-code-per-time-step standard VQ interface unless explicitly stated. For
unconditioned experiments, "additive AR" reduces to the same causal AR prior without a condition
embedding. For PDV4 and S&P500/VIX, additive conditioning uses only the causal scalar context that
is available at the relevant time step.

| Candidate | Applies to | Purpose | Required comparison |
| --- | --- | --- | --- |
| Standard VQ plus additive AR | Black-Scholes, PDV4, S&P500/VIX; Heston after a future config exists. | Public or smoke discrete baseline with the simplest causal discrete interface. | Compare against the selected continuous TC-VAE config and all stronger discrete candidates for the same experiment. |
| Hidden128 plus additive AR | All experiments after local configs exist; S&P500/VIX is documented on `research/vq-tokenizer-tuning`. | Wider standard-VQ tokenizer capacity while preserving the additive AR prior. | Compare against standard VQ plus additive AR, keeping token likelihood and path metrics visible. |
| Hidden128 plus conv-transformer k3 | All experiments after hidden128 additive is stable; S&P500/VIX is documented on `research/stronger-hidden128-prior`. | Test whether a causal convolutional front end improves token-prior calibration and path dynamics without changing objectives. | Compare against hidden128 additive and the public discrete baseline. |
| Conditional standard VQ plus additive AR | PDV4 and S&P500/VIX. | Use the committed causal scalar condition: `r2_volatility_feature` for PDV4 and VIX for S&P500/VIX. | Compare condition-bucket metrics against unconditional or weaker conditional variants where available. |
| Conditional hidden128 plus additive AR | PDV4 and S&P500/VIX. | Combine wider tokenizer capacity with scalar additive conditioning. | Require no-leakage checks for tokenizer and prior conditioning. |
| Conditional hidden128 plus conv-transformer k3 | PDV4 and S&P500/VIX. | Research candidate for conditional sequential dynamics after hidden128 additive is credible. | Report seed spread and sampling policy; do not promote from one MMD improvement alone. |

For S&P500/VIX, the research summaries give the current ordering:

- the promoted public baseline remains standard VQ plus additive AR with `temperature=0.8` and
  `top_k=40`;
- hidden128 plus additive AR is the leading standard-VQ ablation, selected at `temperature=0.8`
  and `top_k=20` in the tuning note;
- hidden128 plus conv-transformer k3 is the best current discrete research model at
  `temperature=1.0` and unrestricted top-k, but it is not the public default and remains
  seed-sensitive.

## Excluded Variants

| Excluded variant | Reason |
| --- | --- |
| RVQ q2 | It changes the token interface to multiple codes per time step, increases prior and geometry complexity, and was not promoted by the research-branch evidence. It is outside this per-experiment standard-VQ selection plan. |
| VQ-Diffusion | It would add a diffusion sampling and training path, which is deferred in the public README and outside the requested objective and architecture scope. |
| Frequency tokenizers | They add a decomposition-specific tokenizer family and are excluded from this documentation branch. The S&P500/VIX separate-frequency hierarchical prior evidence also showed sampled-token and path collapse. |
| Native recurrent GRU | It introduces another prior family without current branch evidence or public config support. The current comparison is restricted to additive causal AR and causal conv-transformer k3 priors. |
| Signatures | Signature conditioning is not part of the default plan. It may be reconsidered only if PDV4-specific conditioning is explicitly selected in a later branch; it must not replace the VIX-only S&P500/VIX default here. |

## Selection Metrics

Keep all metrics visible for every candidate. Lower values are better for the listed distance and
error metrics.

Core path metrics:

- MMD;
- SWD;
- terminal W1 or `terminal_return_wasserstein`;
- volatility W1 or `volatility_wasserstein`;
- drawdown W1 or `maximum_drawdown_wasserstein`;
- return AC L1 or `return_autocorrelation_within_path_l1`;
- squared-return AC L1 or `squared_return_autocorrelation_within_path_l1`.

Discrete-token metrics:

- tokenizer reconstruction L1 and L2 when available;
- tokenizer terminal and volatility reconstruction errors when available;
- active token count and active token ratio;
- token usage entropy and codebook perplexity;
- prior eval cross-entropy, accuracy, and perplexity;
- sampled-token active codes and sampled-token perplexity for generated samples;
- condition-bucket token usage and VIX or PDV4 condition-bucket diagnostics where applicable.

Experiment-specific metrics from TC-VAE notebooks and reproduction workflows:

- Black-Scholes: terminal distribution, path-level MMD/SWD, volatility fit, and smoke discrete
  reconstruction diagnostics;
- Heston: stochastic-volatility fit, terminal distribution, path-level MMD/SWD, and return or
  squared-return autocorrelation where available;
- PDV4: condition-feature calibration, volatility W1, drawdown W1, squared-return AC L1, and
  conditional bucket stability;
- S&P500/VIX: paper-style path metrics, VIX-bucket diagnostics, latent-geometry diagnostics,
  maximum drawdown, return and squared-return autocorrelation, and report-facing comparison tables.

## Selection Rule

Do not select a model by MMD alone. Use MMD as one visible component of a profile, not as a gate.
For each experiment:

1. Confirm the candidate preserves no anticipation with the relevant tokenizer, prior, and
   conditioning checks.
2. Confirm the notebook or reproduction workflow can be re-run from stripped notebooks and local
   output paths.
3. Compare the selected continuous TC-VAE baseline, the public discrete baseline, and all available
   candidate discrete variants in one table.
4. Keep all metrics visible, including those that regress.
5. Select by the profile relevant to the experiment:
   - distributional profile: MMD, SWD, and terminal W1;
   - volatility and tail-risk profile: volatility W1, drawdown W1, and terminal W1;
   - sequential-dependence profile: return AC L1 and squared-return AC L1;
   - discrete-geometry profile: token usage, codebook perplexity, and condition-bucket usage;
   - balanced-market profile: the visible aggregate of distributional, tail-risk,
     sequential-dependence, and token diagnostics.
6. Require seed and sampling-policy disclosure for any promoted discrete research variant.
7. Prefer the simpler public baseline when a stronger candidate improves one profile but regresses
   token likelihood, reproducibility, no-leakage status, or the experiment's primary diagnostics.

## Output Layout

Future model-selection outputs should use the following ignored local layout:

```text
outputs/per_experiment_selection/<experiment>/<candidate>/
```

Suggested candidate directory names:

```text
standard_vq_additive_ar/
hidden128_additive_ar/
hidden128_conv_transformer_k3/
conditional_standard_vq_additive_ar/
conditional_hidden128_additive_ar/
conditional_hidden128_conv_transformer_k3/
```

Each candidate directory should contain local summaries, paper-style or notebook metrics, selection
tables, no-leakage check logs, and links to local tokenizer or prior output directories. These
files must remain ignored generated outputs.

After a selection is made outside this documentation-only pass, update
`trained_models/model_registry.yaml` with selected metadata only:

- experiment name;
- selected continuous and discrete config paths;
- output directory conventions;
- sampling policy;
- condition policy;
- selection profile and selected metric values;
- local-only weight status.

Do not commit weights, checkpoints, arrays, notebooks with outputs, W&B run exports, or other
generated artefacts.

## W&B Profile

Use the following W&B profiles when training is intentionally run outside this documentation task:

| Stage | W&B project |
| --- | --- |
| Tokenizers | `time-causal-vq-tokenizer` |
| Token priors | `time-causal-token-prior` |

Use entity `tc_vae`. If W&B initialisation times out or blocks a reproducibility run, retry the
same command with `--no-wandb` and record that fallback in the local candidate summary. The
S&P500/VIX hidden128 conv-transformer seed-robustness note already used this fallback for timed-out
W&B initialisation, so the fallback is acceptable when it is documented.

## Documentation Stop Point

This branch stops after this documentation plan. It should not add configs, run training, update
the registry, commit weights, or generate new outputs.
