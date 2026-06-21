# TimeCausalVAE

[![PyPI](https://img.shields.io/pypi/v/time-causal-vae.svg)](https://pypi.org/project/time-causal-vae/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Ruff](https://github.com/GVourvachakis/TimeCausalVQVAE/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/GVourvachakis/TimeCausalVQVAE/actions/workflows/lint.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)

Time-causal financial generative models: refactored TC-VAE baselines with
causal VQ/RVQ tokenizers, token priors, S&P500/VIX, Hawkes/SVMHJD,
multi-dimensional benchmarks, and path-risk diagnostics.

`time-causal-vae` is a research package for time-causal financial generative
modelling across synthetic and empirical market time series.

The Python distribution is `time-causal-vae`; the import package is
`time_causal_vae`. The GitHub repository remains `TimeCausalVQVAE` because it
also hosts the discrete VQ extension work.

Release notes: [`0.1.1`](https://github.com/GVourvachakis/TimeCausalVQVAE/blob/main/docs/release/0.1.1.md).

![Discrete time-causal VQ-VAE architecture](assets/figures/time_causal_vqvae_pipeline.svg)

*Discrete time-causal VQ-VAE architecture. The diagram shows the S&P 500/VIX
input window, causal convolutional encoder and decoder stacks, vector
quantization, the VIX conditioning branch, and the receptive-field structure
used to preserve no-anticipation behaviour.*

## Installation

Install the package from PyPI:

```bash
pip install time-causal-vae
```

Wheel installs include the runtime package only. From a source checkout, use
Poetry groups for development tools, local empirical data access, notebooks,
and optional tracking:

```bash
poetry install --only main
poetry install --with dev
poetry install --with notebooks
poetry install --with data
poetry install --with tracking
```

The `docs` URL currently points to the repository documentation directory. No
hosted Sphinx documentation is published yet.

## Quickstart

Check the installed package:

```bash
python - <<'PY'
import time_causal_vae

print(time_causal_vae.__version__)
PY
```

Inspect installed command-line entry points:

```bash
tcvae-train --help
tcvae-train-tokenizer --help
tcvae-train-token-prior --help
tcvae-evaluate --help
tcvae-select-model --help
```

Repository examples use configs, scripts, and registry files from the source
tree. Clone the repository when running the public workflows:

```bash
git clone https://github.com/GVourvachakis/TimeCausalVQVAE.git
cd TimeCausalVQVAE
poetry install --with dev,data
```

Inspect the public S&P500/VIX registry entry:

```bash
poetry run python scripts/select_registered_model.py \
  --experiment sp500_vix \
  --family discrete
```

Run a dry-run continuous S&P500/VIX smoke command:

```bash
poetry run tcvae-train \
  --config configs/experiments/sp500_vix_beta_cvae.yaml \
  --output-dir outputs/sp500_vix_continuous \
  --epochs 1 \
  --no-wandb \
  --dry-run
```

Remove `--dry-run` only when you intentionally want to train locally.

## Public Status

S&P500/VIX is the stable public default one-dimensional workflow.
Hawkes/SVMHJD is an optional research benchmark with research-candidate
metadata. Multidimensional benchmarks are experimental infrastructure, and no
multidimensional model is selected in
[`trained_models/model_registry.yaml`](https://github.com/GVourvachakis/TimeCausalVQVAE/blob/main/trained_models/model_registry.yaml).
Experimental multidimensional profile metadata is kept in
[`trained_models/multidim_profiles.yaml`](trained_models/multidim_profiles.yaml).

No downloaded data, trained weights, checkpoints, token tensors, generated
paths, W&B runs, notebooks with outputs, or local result summaries are shipped
with the package.

## Stable Benchmarks

| Benchmark | Role | Public status |
| --- | --- | --- |
| [S&P500/VIX](docs/benchmarks/sp500_vix.md) | Empirical one-dimensional market workflow with VIX conditioning and a local processed data convention. | Public default. Uses local-only processed data and selected continuous/discrete registry metadata. |
| [Black-Scholes](docs/benchmarks/black_scholes.md) | Synthetic geometric Brownian motion baseline for smoke tests and one-dimensional generation checks. | Stable baseline config and registry metadata. |
| [Heston](docs/benchmarks/heston.md) | Synthetic stochastic-volatility baseline with a latent variance channel. | Stable baseline config and registry metadata. |
| [Path-dependent volatility](docs/benchmarks/path_dependent_volatility.md) | Conditional synthetic volatility baseline with a prefix volatility feature. | Stable baseline config and registry metadata. |

The selected public S&P500/VIX discrete baseline is a standard causal VQ
tokenizer plus an additive scalar-conditioned causal autoregressive token
prior:

```text
configs/experiments/sp500_vix_causal_vq_tokenizer.yaml
configs/experiments/sp500_vix_causal_token_prior_additive.yaml
```

## Optional Research Benchmark

| Benchmark | Description | Public status |
| --- | --- | --- |
| [Hawkes/SVMHJD](docs/benchmarks/hawkes_jump.md) | Marked Hawkes jump-diffusion benchmark with Ogata event simulation and fixed-grid observation. | Optional rare-event research benchmark with `public_default: false`. No weights or generated outputs are committed. |

## Experimental Benchmarks

| Benchmark | Description | Public status |
| --- | --- | --- |
| [Multifactor market](docs/benchmarks/multifactor_market.md) | 50-dimensional low-rank factor market with sector structure and optional common/sector jumps. | Experimental infrastructure for shape, covariance, and no-leakage checks. |
| [S&P500 50-stock panel](docs/benchmarks/sp500_50_panel.md) | Local-only `yfinance`/Yahoo-backed daily 50-stock equity panel. | Experimental infrastructure. Downloaded Yahoo-backed data must remain local and is not redistributed. |

The benchmark notes live under
[`docs/benchmarks`](https://github.com/GVourvachakis/TimeCausalVQVAE/tree/main/docs/benchmarks).

## Benchmark Data Conventions

| Benchmark | Data convention |
| --- | --- |
| [S&P500/VIX](docs/benchmarks/sp500_vix.md) | Local processed benchmark data is expected at `data/processed/sp500vix/sp500vix_normalized.npy`. |
| [Hawkes/SVMHJD](docs/benchmarks/hawkes_jump.md) | Synthetic paths are generated locally from the marked Hawkes jump-diffusion simulator. |
| [Multifactor market](docs/benchmarks/multifactor_market.md) | Synthetic 50D panels are generated locally from the low-rank sector-factor simulator. |
| [S&P500 50-stock panel](docs/benchmarks/sp500_50_panel.md) | Daily panels are downloaded locally through optional `yfinance` access and must not be redistributed. |

## Models And Features

| Area | Included | Release status |
| --- | --- | --- |
| Continuous TC-VAE | No-anticipation continuous VAE baseline, RealNVP-compatible prior paths, and financial dataset conventions. | Stable baseline surface. |
| Causal VQ tokenizers | Causal convolutional tokenizers with vector-quantized latent codes. | Public S&P500/VIX discrete baseline. |
| RVQ and multi-code tokenizers | Residual and multi-code tokenizer infrastructure. | Experimental. No multidimensional model is registry-selected. |
| Token priors | Additive autoregressive priors and causal conv-transformer research variants. | Additive prior is the public S&P500/VIX default; conv-transformer variants are research candidates. |
| Registry metadata | Selected configs, local checkpoint conventions, metrics, caveats, and no-leakage status. | Metadata only. It does not contain weights. |
| Notebook demos | Output-stripped notebooks that print guarded commands and read local outputs when available. | Demonstration only. They should not train or evaluate by default. |

Executed notebook previews are available on the `docs/executed-notebook-previews` branch. The
`main` branch keeps notebooks output-stripped for reproducibility and package size. Preview outputs
depend on local artefacts and checkpoints and are not the package source of truth.

## Diagnostics

| Diagnostic family | Examples | Notes |
| --- | --- | --- |
| Distributional distances | MMD, sliced Wasserstein, terminal and volatility Wasserstein distances. | Used for registry summaries and model comparison. |
| Path-risk summaries | Drawdown, return autocorrelation, squared-return autocorrelation, VaR, and ES. | Intended for generated-vs-real path checks, not investment advice. |
| Conditional checks | VIX-bucket summaries and prefix-safe condition handling. | Used by the public S&P500/VIX workflow. |
| Token diagnostics | Codebook usage, active codes, token perplexity, transition summaries, and latent geometry. | Used to inspect discrete-token behaviour. |
| Jump diagnostics | Jump count, inter-arrival, jump-size, and lower-tail summaries. | Used by the optional Hawkes/SVMHJD benchmark. |
| Cross-sectional checks | Covariance, correlation, eigenspectrum, sector-block, and portfolio-risk summaries. | Experimental multidimensional infrastructure. |

## Local Data Policy

The package does not redistribute empirical market data. The S&P500/VIX data
file is expected locally at:

```text
data/processed/sp500vix/sp500vix_normalized.npy
```

The S&P500 50-stock panel downloader uses optional `yfinance` access and writes
local raw and processed files under `data/raw/` and `data/processed/`.
Yahoo-backed data is subject to Yahoo's terms and must not be redistributed or
committed.

Generated artefacts belong under local paths such as `outputs/`, `wandb/`, or
`data/processed/`. They are intentionally excluded from the public repository
and package.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/time_causal_vae` | Importable package source. |
| `configs/experiments` | Repository workflow configs used by scripts and notebooks. |
| `scripts` | Inspection, extraction, evaluation, no-leakage, and smoke helpers. |
| `trained_models` | Lightweight registry metadata and model cards only. |
| `docs/benchmarks` | Public benchmark notes. |
| `assets/figures` | Small curated README figures generated from local runs. |
| `notebooks` | Output-stripped demos and report-facing notebooks. |

## Background

TimeCausalVAE keeps the no-anticipation contract from upstream TC-VAE: at time
`t`, encoders, tokenizers, priors, and diagnostics should only use observations
and conditions available up to that point. The public branch preserves the
continuous TC-VAE baseline and adds a discrete two-stage path: causal tokenizer
first, causal token prior second.

The package is research software for generative modelling diagnostics. It is
not a calibrated pricing library, a trading system, or a source of financial
advice.

## Citation And Acknowledgement

This repository refactors selected parts of the original Time-Causal VAE code
and extends the public workflow with causal VQ-style discrete latent models.
Please cite or acknowledge the relevant upstream work when using the package:

- **Time-Causal VAE: Robust Financial Time Series Generator** - Beatrice
  Acciaio, Stephan Eckstein, and Songyan Hou. DOI:
  [10.48550/arXiv.2411.02947](https://doi.org/10.48550/arXiv.2411.02947);
  code: [justinhou95/TimeCausalVAE](https://github.com/justinhou95/TimeCausalVAE).
- **Neural Discrete Representation Learning** - Aaron van den Oord, Oriol
  Vinyals, and Koray Kavukcuoglu. DOI:
  [10.48550/arXiv.1711.00937](https://doi.org/10.48550/arXiv.1711.00937).
- **Vector Quantized Time Series Generation with a Bidirectional Prior Model** -
  Daesoo Lee, Sara Malacarne, and Erlend Aune. DOI:
  [10.48550/arXiv.2303.04743](https://doi.org/10.48550/arXiv.2303.04743);
  code: [ML4ITS/TimeVQVAE](https://github.com/ML4ITS/TimeVQVAE).
- **vector-quantize-pytorch** - lucidrains. Repository:
  [lucidrains/vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch).

## License

This project is released under the GNU General Public License v3. See
[`LICENSE`](https://github.com/GVourvachakis/TimeCausalVQVAE/blob/main/LICENSE).
