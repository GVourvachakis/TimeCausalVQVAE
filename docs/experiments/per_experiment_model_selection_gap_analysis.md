# Per-Experiment Model-Selection Gap Analysis

## Scope

This audit reviews the current per-experiment model-selection outputs after the non-smoke
tokenizer and token-prior run documented in
`docs/experiments/per_experiment_model_selection_results.md`. It does not train models, change
source code, update `trained_models/model_registry.yaml`, or modify `main`.

The current trained-model registry is present and records S&P500/VIX continuous, public discrete,
and optional hidden128 research metadata. It has not been updated with the per-experiment
token-run selections from this branch.

## Provisional Token-Run Selections

The current selections are provisional because they are based on tokenizer and token-prior
summaries only. They are useful for prioritising final evaluation, but they are not final
trained-model registry selections.

| Experiment | Provisional token-run selection | Basis available from runner |
| --- | --- | --- |
| Black-Scholes | `hidden128_conv_transformer_k3` | Lowest observed prior validation CE among the three Black-Scholes candidates, with full code usage. |
| Heston | `standard_vq_additive_ar` | Lowest observed prior validation CE among the three Heston candidates, with full code usage. |
| PDV4 | `conditional_standard_vq_additive_ar` | Lowest observed prior validation CE among the three PDV4 candidates. |
| S&P500/VIX | `conditional_hidden128_conv_transformer_k3` | Lowest observed prior validation CE among the two S&P500/VIX candidates, with stronger token usage than the standard public baseline. |

These selections should be treated as a candidate shortlist for path-level evaluation. They should
not replace the public S&P500/VIX registry metadata, nor should they add Black-Scholes, Heston, or
PDV4 final discrete selections to the registry yet.

## What The Runner Measured

`scripts/run_per_experiment_model_selection.py` validated the candidate config paths, trained or
reused tokenizer artefacts, extracted frozen token datasets, trained token priors, and wrote
aggregate JSON and CSV summaries under `outputs/per_experiment_selection/`.

The runner measured:

- tokenizer code usage through `codebook_summary.json`, including active-code counts, active-code
  ratios, codebook perplexity, and tokenizer training losses where available;
- token dataset summaries through `token_dataset_summary.json`, including train/eval token shapes,
  token counts, active-code counts, entropy, and codebook perplexity;
- prior validation cross-entropy, accuracy, and perplexity through
  `best_checkpoint_summary.json`;
- runtime and checkpoint summaries through tokenizer and prior `runtime_summary.json` files;
- stage status for tokenizer training, token extraction, prior training, skipped existing
  artefacts, and W&B fallback status in the aggregate results.

The output is therefore a discrete-token and prior-likelihood audit. It is not a generated-path
model-selection report.

## What The Runner Did Not Measure

The runner did not execute generated-path evaluation or notebook reproduction. In particular, it
did not measure:

- MMD;
- SWD;
- terminal W1;
- volatility W1;
- drawdown W1;
- return AC L1;
- squared-return AC L1;
- experiment-level condition-bucket metrics, such as PDV4 volatility-feature bucket diagnostics
  or S&P500/VIX VIX-bucket path diagnostics;
- no-leakage checks for tokenizer, prior, causal convolution, or conditional paths;
- notebook execution, report notebook rendering, or reproduction-wrapper execution.

Some token dataset summaries may include token-usage breakdowns by condition bucket, but those are
not substitutes for path-level condition-bucket diagnostics. They do not assess generated-path
calibration, volatility behaviour, drawdown behaviour, or autocorrelation profiles.

## Decision

The current selections are provisional token-run selections only. They are suitable for deciding
which discrete candidates should enter the final evaluation pass. They are not sufficient for
final registry promotion.

Do not update `trained_models/model_registry.yaml` yet. The registry should only be updated after
the final evaluation shows, per experiment, that the selected candidate has an acceptable
path-level profile, passes no-leakage checks, and can be reproduced from the documented notebook
or wrapper workflow.

Path-level evaluation is required because the selection plan explicitly rejects decisions based on
a single metric or token-prior likelihood alone. Prior validation CE can improve while generated
paths still regress on volatility, drawdown, terminal distribution, or autocorrelation.

## Required Final Evaluation

Each experiment needs one final comparison table that keeps the continuous baseline, public
discrete baseline where available, and provisional best discrete candidate visible together.

| Experiment | Continuous selected baseline | Public discrete baseline if available | Provisional best discrete candidate | Required path metrics | Required checks |
| --- | --- | --- | --- | --- | --- |
| Black-Scholes | `configs/experiments/black_scholes_beta_cvae.yaml` | Local/smoke standard VQ baseline: `configs/experiments/black_scholes_causal_vq_tokenizer.yaml` plus `configs/experiments/black_scholes_causal_token_prior.yaml`, if regenerated for comparison. | `hidden128_conv_transformer_k3` | MMD, SWD, terminal W1, volatility W1, drawdown W1 where available, return AC L1, squared-return AC L1. | Tokenizer and prior no-leakage check; continuous and discrete notebook or reproduction check. |
| Heston | `configs/experiments/heston_info_cvae.yaml` | No registry public discrete baseline is currently promoted; use `standard_vq_additive_ar` as the standard discrete comparison candidate. | `standard_vq_additive_ar` | MMD, SWD, terminal W1, volatility W1, drawdown W1 where available, return AC L1, squared-return AC L1. | Tokenizer and prior no-leakage check; continuous and discrete notebook or reproduction check. |
| PDV4 | `configs/experiments/pdv_info_cvae.yaml` | Conditional standard VQ plus additive prior: `configs/experiments/pdv_causal_vq_tokenizer_codebook64_codebookdim16.yaml` plus `configs/experiments/pdv_causal_token_prior_additive_seed1.yaml`. | `conditional_standard_vq_additive_ar` | MMD, SWD, terminal W1, volatility W1, drawdown W1, return AC L1, squared-return AC L1, PDV4 condition-feature bucket diagnostics. | Conditional tokenizer no-leakage check; conditional prior no-leakage check; notebook or reproduction check. |
| S&P500/VIX | `configs/experiments/sp500_vix_beta_cvae.yaml` | Public discrete baseline: `configs/experiments/sp500_vix_causal_vq_tokenizer.yaml` plus `configs/experiments/sp500_vix_causal_token_prior_additive.yaml`. | `conditional_hidden128_conv_transformer_k3` | MMD, SWD, terminal W1, volatility W1, drawdown W1, return AC L1, squared-return AC L1, VIX-bucket path diagnostics. | Conditional tokenizer and prior no-leakage checks; `scripts/evaluate_sp500_vix_paper_style.py`; discrete/report notebook reproduction check. |

The final evaluation should keep regressions visible. A candidate that wins prior CE but regresses
on the experiment's main generated-path profile should not be promoted without an explicit
justification.

## Registry Promotion Gate

Only after the final evaluation passes should `trained_models/model_registry.yaml` be updated.
The update should contain selected metadata only, not weights or generated artefacts.

The registry promotion gate is:

- selected continuous and discrete config paths;
- selection profile and visible component metrics, including token metrics and path metrics;
- sampling policy, including temperature, top-k, seed policy, and condition policy;
- local checkpoint and output-directory convention, marked as local outputs only;
- no-leakage check status and notebook or reproduction status;
- W&B status or documented `--no-wandb` fallback;
- explicit note that weights, checkpoints, token tensors, arrays, executed notebooks, W&B exports,
  and generated figures are not committed;
- optional future candidate scan support, limited to metadata fields that can point to additional
  ignored local candidate summaries without changing the promoted model identity.

Until those gates are satisfied, the branch should keep the registry unchanged and treat
`docs/experiments/per_experiment_model_selection_results.md` as a provisional token-run report.

