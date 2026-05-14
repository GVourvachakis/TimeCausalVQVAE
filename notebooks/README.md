# Notebook Guide

The notebooks are organised by demonstration role.

## `continuous/`

Refactored continuous TC-VAE baseline demos. These notebooks show the selected continuous config,
print dry-run training and evaluation commands, and avoid released-checkpoint requirements by
default.

- `black_scholes.ipynb`
- `heston.ipynb`
- `pdv.ipynb`
- `sp500_vix.ipynb`

## `discrete/`

TC-VQVAE discrete-latent demos. The S&P500/VIX notebook is the promoted public method demo:
standard causal VQ tokenizer, additive scalar-conditioned causal AR prior, paper-style diagnostics,
and latent-geometry diagnostics. RVQ q2 remains ablation/future evidence only.

- `black_scholes.ipynb`
- `heston.ipynb`
- `pdv.ipynb`
- `sp500_vix.ipynb`
- `discrete_latent_geometry_demo.ipynb`

## `report/`

Report-specific figure notebooks. These read figures from ignored `outputs/` paths and do not
train models by default.

- `sp500_vix_report_figures.ipynb`

## Output Policy

Committed notebooks should remain output-stripped. Generated figures, executed notebooks,
checkpoints, tensors, JSON summaries, logs, and local data belong under ignored `outputs/` or
`data/processed/` paths.

The S&P500/VIX data file is local and is not committed:

```text
data/processed/sp500vix/sp500vix_normalized.npy
```
