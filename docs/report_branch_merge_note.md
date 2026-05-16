# Best Discrete Report Branch Merge Note

Status: validation note for merging `docs/best-discrete-research-report` into
`feat/causal-vq-vae`. This note documents the branch boundary only. It does not merge branches,
modify `main`, add model code, add configs, train models, or commit generated outputs.

## Branch Purpose

This branch integrates the current S&P500/VIX best discrete research model into public-facing
documentation and report notebooks while preserving the public default workflow. It adds
report-ready language for the hidden128 VQ tokenizer with the causal conv-transformer k3 prior,
but keeps that model separate from the public baseline.

The intended reporting distinction is:

- public baseline: standard VQ tokenizer plus additive VIX-only causal AR prior;
- best discrete research model: hidden128 VQ tokenizer plus causal conv-transformer k3 prior;
- strongest overall reference: continuous BetaCVAE.

## Safe To Merge Into `feat/causal-vq-vae`

The following changes are documentation and notebook-facing only and are safe to merge into
`feat/causal-vq-vae`:

- `docs/report_ready_best_discrete_research_model.md`, which records the public baseline,
  best discrete research model, continuous reference, seed robustness, sampling policy, and
  caveats;
- `README.md`, which keeps the public quickstart on standard VQ plus additive AR and mentions the
  hidden128 conv-transformer only as a local-checkout research variant;
- `notebooks/README.md`, which clarifies the continuous, discrete, and report-notebook roles;
- `notebooks/discrete/sp500_vix.ipynb`, which remains a public discrete-baseline workflow and
  includes only a Markdown note about the optional research variant;
- `notebooks/report/sp500_vix_report_figures.ipynb`, which reads optional local paper-style
  output directories for the public baseline, best discrete research model, and continuous
  reference without embedding outputs or running training by default.

The default commands remain on:

```text
configs/experiments/sp500_vix_causal_vq_tokenizer.yaml
configs/experiments/sp500_vix_causal_token_prior_additive.yaml
```

No default quickstart command uses the hidden128 conv-transformer variant.

## Remains On `research/stronger-hidden128-prior`

The following items remain research-branch or local-checkout artefacts and are not added on this
report branch:

- `configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml`;
- `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml`;
- hidden128 tokenizer checkpoints;
- hidden128 conv-transformer prior checkpoints;
- token artefacts, paper-style outputs, W&B runs, tensors, NumPy arrays, pickles, and generated
  figures under ignored local output paths.

References to the hidden128 conv-transformer config are documentation-only and should be read as
research-branch or local-checkout pointers, not as public-baseline requirements.

## Validation Summary

The branch was validated for the intended merge boundary:

- the public default remains standard VQ plus additive AR;
- the hidden128 conv-transformer is described only as the best discrete research model;
- the continuous BetaCVAE remains the strongest overall reference;
- generated artefacts are not committed;
- the S&P500/VIX notebooks are output-stripped;
- the report notebook continues when optional local output directories are missing.

`main` is not modified by this branch or by this validation note.
