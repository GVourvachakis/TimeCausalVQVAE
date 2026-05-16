# Notebook Guide

The notebooks are organised by demonstration role.

## `continuous/`

Refactored continuous TC-VAE baseline demos. These notebooks show the selected continuous config,
print dry-run training and evaluation commands, and avoid released-checkpoint requirements by
default. For S&P500/VIX, the continuous notebook is the TC-VAE baseline and the continuous
BetaCVAE remains the strongest overall reference in the current report evidence.

- `black_scholes.ipynb`
- `heston.ipynb`
- `pdv.ipynb`
- `sp500_vix.ipynb`

## `discrete/`

TC-VQVAE discrete-latent demos. The S&P500/VIX notebook is a public discrete-baseline demo:
standard causal VQ tokenizer, additive scalar-conditioned causal AR prior, paper-style diagnostics,
and latent-geometry diagnostics. The hidden128 causal conv-transformer k3 prior is an optional
research variant for report comparison, not the default notebook workflow. RVQ q2 was evaluated
on research branches and is not part of the public baseline.

- `black_scholes.ipynb`
- `heston.ipynb`
- `pdv.ipynb`
- `sp500_vix.ipynb`
- `discrete_latent_geometry.ipynb`

## `report/`

Report-specific figure notebooks. These read figures from ignored `outputs/` paths and do not
train models by default. The S&P500/VIX report notebook can compare the public discrete baseline,
the best discrete research model, and the continuous BetaCVAE reference when the corresponding
local paper-style output directories are available.

- `sp500_vix_report_figures.ipynb`

## Output Policy

Committed notebooks should remain output-stripped. Generated figures, executed notebooks,
checkpoints, tensors, JSON summaries, logs, and local data belong under ignored `outputs/` or
`data/processed/` paths.

The S&P500/VIX data file is local and is not committed:

```text
data/processed/sp500vix/sp500vix_normalized.npy
```
