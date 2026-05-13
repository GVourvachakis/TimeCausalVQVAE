# TimeCausalVQVAE

Public-minimal research code for financial time-series generation with a refactored
continuous TC-VAE baseline and a promoted discrete architecture:

```text
causal VQ-family tokenizer + additive conditional causal AR prior
```

The current empirical focus is S&P500/VIX. Black-Scholes, Heston, and
path-dependent-volatility configs remain as compact continuous baselines and tokenizer/prior
sanity checks. Diffusion and transition-constrained sampling were evaluated during development,
but they are not part of the promoted public method.

## Installation

```bash
poetry install
```

Optional W&B tracking is available through the tracking dependency group used by Poetry. Generated
outputs, local data, W&B runs, checkpoints, NumPy arrays, pickles, and logs are ignored by Git.

## Minimal Configs

Continuous TC-VAE baselines:

- `configs/experiments/black_scholes_beta_cvae.yaml`
- `configs/experiments/heston_info_cvae.yaml`
- `configs/experiments/pdv_info_cvae.yaml`
- `configs/experiments/sp500_vix_beta_cvae.yaml`

Discrete tokenizer and token-prior configs:

- `configs/experiments/black_scholes_causal_vq_tokenizer.yaml`
- `configs/experiments/black_scholes_causal_token_prior.yaml`
- `configs/experiments/pdv_causal_vq_tokenizer_codebook64_codebookdim16.yaml`
- `configs/experiments/pdv_causal_token_prior_additive_seed1.yaml`
- `configs/experiments/sp500_vix_causal_vq_tokenizer.yaml`
- `configs/experiments/sp500_vix_causal_token_prior_additive.yaml`
- `configs/experiments/sp500_vix_causal_rvq_tokenizer_q2.yaml`
- `configs/experiments/sp500_vix_causal_rvq_token_prior_q2.yaml`

Inspect the selected YAML files with:

```bash
poetry run python scripts/inspect_selected_configs.py
```

## Data

The S&P500/VIX experiment expects the local normalised array at:

```text
data/processed/sp500vix/sp500vix_normalized.npy
```

This file is not committed. Use `--base-data-dir data/processed` for tokenizer training,
token extraction, token-prior evaluation, and S&P500/VIX diagnostics.

## Training And Evaluation

Continuous baseline smoke run:

```bash
poetry run tcvae-train \
  --config configs/experiments/sp500_vix_beta_cvae.yaml \
  --output-dir outputs/sp500_vix_continuous \
  --epochs 1 \
  --no-wandb \
  --dry-run
```

Remove `--dry-run` and use the full epoch count in the selected config to run real training.

Promoted S&P500/VIX discrete path:

```bash
poetry run tcvae-train-tokenizer \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer.yaml \
  --base-data-dir data/processed \
  --output-dir outputs/sp500_vix_discrete/tokenizer \
  --no-wandb

poetry run python scripts/extract_token_indices.py \
  --config configs/experiments/sp500_vix_causal_vq_tokenizer.yaml \
  --tokenizer-dir <tokenizer-dir> \
  --output-dir outputs/sp500_vix_discrete/token_prior/tokens_codebook64_codebookdim16 \
  --base-data-dir data/processed

poetry run tcvae-train-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_additive.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/additive \
  --no-wandb

poetry run python scripts/evaluate_sp500_vix_paper_style.py \
  --discrete-config configs/experiments/sp500_vix_causal_token_prior_additive.yaml \
  --discrete-prior-dir <prior-dir> \
  --discrete-tokenizer-dir <tokenizer-dir> \
  --continuous-config configs/experiments/sp500_vix_beta_cvae.yaml \
  --continuous-model-dir <continuous-final-model-dir> \
  --output-dir outputs/sp500_vix_discrete/paper_style \
  --base-data-dir data/processed \
  --n-sample 1000 \
  --temperature 1.0 \
  --top-k 40
```

Optional RVQ q2 ablation uses the matching S&P500/VIX RVQ q2 tokenizer and token-prior configs.

W&B can be enabled with:

```bash
--wandb --wandb-project ... --wandb-entity tc_vae
```

## Verification Scripts

Core public checks:

```bash
poetry run python scripts/check_causal_conv_no_leakage.py
poetry run python scripts/check_conditional_vq_tokenizer_no_leakage.py \
  --config configs/experiments/pdv_causal_vq_tokenizer_codebook64_codebookdim16.yaml
poetry run python scripts/check_conditional_token_prior_no_leakage.py \
  --config configs/experiments/sp500_vix_causal_token_prior_additive.yaml
poetry run python scripts/check_multicode_token_prior_no_leakage.py \
  --config configs/experiments/sp500_vix_causal_rvq_token_prior_q2.yaml
poetry run python scripts/check_vq_family_tokenizer_shapes.py
poetry run python scripts/check_vq_tokenizer_shapes.py
```

The reproduction wrappers under `scripts/reproduce_*.py` print or run the selected continuous
baseline train/evaluate commands. The S&P500/VIX tokenizer and token-prior ablation helpers are
kept for compact public ablation runs below ignored `outputs/` paths.
