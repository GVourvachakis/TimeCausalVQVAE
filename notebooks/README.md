# Notebook Guide

The notebooks are organised by demonstration role.

Notebooks should prefer the metadata registry over hard-coded optimal model identifiers. Continuous
and discrete notebooks now call the same registry selector before constructing commands, so each
notebook can pick the registered best candidate for its experiment and family. Use
`trained_models/model_registry.yaml`, or the helper CLI below, to discover selected config paths,
local checkpoint conventions, sampling policy, visible metrics, and missing metrics:

```bash
poetry run python scripts/select_registered_model.py --experiment sp500_vix --family discrete
```

## `continuous/`

Refactored continuous TC-VAE baseline demos. These notebooks select the registered continuous
candidate for their own experiment, show the selected continuous config, print dry-run training
and evaluation commands, and avoid released-checkpoint requirements by default. For S&P500/VIX,
the continuous notebook is the TC-VAE baseline and the continuous BetaCVAE remains the strongest
overall reference in the current report evidence. The Hawkes/SVMHJD continuous notebook is a
guarded log-return comparator demo for the optional research-candidate benchmark.

- `black_scholes.ipynb`
- `hawkes_jump.ipynb`
- `heston.ipynb`
- `pdv.ipynb`
- `sp500_vix.ipynb`

## `discrete/`

TC-VQVAE discrete-latent demos. The S&P500/VIX notebook is a public discrete-baseline demo:
standard causal VQ tokenizer, additive scalar-conditioned causal AR prior, paper-style diagnostics,
and latent-geometry diagnostics. The Hawkes/SVMHJD discrete notebook is a guarded log-return
tokenizer and token-prior demo for the optional research-candidate benchmark. For Hawkes/SVMHJD,
the hidden128 log-return cb64 tokenizer + causal conv-transformer k3 prior is the selected
research candidate under the balanced/smooth profile, while the additive AR prior remains the
required ablation and is slightly stronger on jump-count and inter-arrival diagnostics. RVQ q2 was
evaluated on research branches and is not part of the public baseline.

- `black_scholes.ipynb`
- `hawkes_jump.ipynb`
- `heston.ipynb`
- `pdv.ipynb`
- `sp500_vix.ipynb`
- `discrete_latent_geometry.ipynb`

## `benchmarks/`

Public dataset-analysis notebooks. The Hawkes/SVMHJD benchmark notebook generates synthetic data in
memory, compares simulator diagnostics, and does not train models or commit generated artefacts.

- `benchmarks/hawkes_jump_dataset.ipynb`

## `report/`

Report-specific figure notebooks. These read figures from local `outputs/` paths and do not
train models by default. The S&P500/VIX report notebook can compare the public discrete baseline,
the best discrete research model, and the continuous BetaCVAE reference when the corresponding
local paper-style output directories are available. The Hawkes/SVMHJD report notebook is a
side-by-side model-comparison notebook for the continuous comparator, additive AR ablation, and
conv-transformer k3 research candidate. Both report notebooks also profile the selected YAML
architectures by parameter count and CPU generation time using randomly initialised weights, so
the timing tables measure architecture cost rather than checkpoint quality.

The final sample-geometry report notebook is registry-aware and output-stripped. It can compare
Black-Scholes, Heston, PDV4, S&P500/VIX, and Hawkes/SVMHJD continuous/discrete candidates when
local output batches exist. It creates t-SNE and KDE/ECDF diagnostics under local `outputs/`
paths, and it does not train or evaluate models by default.

- `report/sp500_vix_report_figures.ipynb`
- `report/hawkes_jump_model_comparison.ipynb`
- `report/final_sample_geometry_report.ipynb`

## Output Policy

Executed notebook previews are available on the `docs/executed-notebook-previews` branch. The
`main` branch keeps notebooks output-stripped for reproducibility and package size. Preview outputs
depend on local artefacts and checkpoints, are not the package source of truth, and the preview
branch is not intended to merge into `main`.

Committed notebooks should remain output-stripped. Generated figures, executed notebooks,
checkpoints, tensors, JSON summaries, logs, and local data belong under local `outputs/` or
`data/processed/` paths and should not be committed.

The S&P500/VIX data file is local and is not committed:

```text
data/processed/sp500vix/sp500vix_normalized.npy
```
