# TimeCausalVAE Refactored Baseline

This repository contains a cleaned, modular refactor of the original Time-Causal VAE
implementation for continuous financial time-series generation. The public baseline keeps
the continuous BetaCVAE and InfoCVAE training and evaluation path, with the code organised
under `time_causal_vae.data`, `time_causal_vae.models`, `time_causal_vae.training`, and
`time_causal_vae.evaluation`.

The selected continuous experiment configs are:

- `configs/experiments/black_scholes_beta_cvae.yaml`
- `configs/experiments/heston_info_cvae.yaml`
- `configs/experiments/pdv_info_cvae.yaml`
- `configs/experiments/sp500_vix_beta_cvae.yaml`

The training and evaluation CLIs preserve the selected checkpoint layout used by the
refactored baseline. Existing compatible `final_model/` directories containing
`model.pt`, `encoder_model.pt`, and `decoder_model.pt` can be inspected with
`tcvae-inspect-checkpoint` and evaluated with `tcvae-evaluate`.

## Installation

```bash
poetry install
```

## One-Epoch Dry Run

```bash
poetry run tcvae-train \
  --config configs/experiments/black_scholes_beta_cvae.yaml \
  --output-dir outputs/baseline_cleanup_smoke/bs_continuous \
  --epochs 1 \
  --no-wandb \
  --dry-run
```

The command builds the Black-Scholes dataset, model, and trainer configuration, prints the
resulting tensor shapes and parameter count, and exits without writing checkpoints.

## S&P500/VIX Data

The S&P500/VIX config expects a local NumPy array at:

```text
data/processed/sp500vix/sp500vix_normalized.npy
```

Raw and processed market data are intentionally ignored by Git. Place the file locally
before running `configs/experiments/sp500_vix_beta_cvae.yaml`.

## Command-Line Tools

```bash
poetry run tcvae-train --config configs/experiments/black_scholes_beta_cvae.yaml --output-dir outputs/black_scholes_beta_cvae --epochs 1 --no-wandb
poetry run tcvae-evaluate --config configs/experiments/black_scholes_beta_cvae.yaml --model-dir outputs/<run>/final_model --output-dir outputs/<run>/evaluation --dry-run
poetry run tcvae-inspect-checkpoint outputs/<run>/final_model
```

Generated checkpoints, metrics, logs, and local datasets belong under ignored paths such
as `outputs/`, `wandb/`, `data/raw/`, and `data/processed/`.

## References

This project is based on and extends the following papers and codebases.

### Papers

| | Reference |
| --- | --- |
| **TC-VAE** | Acciaio, Eckstein & Hou. *Time-Causal VAE: Robust Financial Time Series Generator.* [arXiv:2411.02947](https://doi.org/10.48550/arXiv.2411.02947) |
| **VQ-VAE** | van den Oord et al. *Neural Discrete Representation Learning.* NeurIPS 2017. [arXiv:1711.00937](https://doi.org/10.48550/arXiv.1711.00937) |
| **TimeVQVAE** | Lee & Kim. *Vector Quantized Time Series Generation with a Bidirectional Prior Model.* [arXiv:2303.04743](https://doi.org/10.48550/arXiv.2303.04743) |
| **VQ-Diffusion** | Gu et al. *Vector Quantized Diffusion Model for Text-to-Image Synthesis.* CVPR 2022. [arXiv:2111.14822](https://doi.org/10.48550/arXiv.2111.14822) |
| **CausalFusion** | Deng et al. *Causal Diffusion Transformers for Generative Modeling.* [arXiv:2412.12095](https://doi.org/10.48550/arXiv.2412.12095) |
| **DeepVol** | Graziani et al. *DeepVol: Volatility Forecasting from High-Frequency Data with Dilated Causal Convolutions.* [arXiv:2210.04797](https://doi.org/10.48550/arXiv.2210.04797) |
| **QINCo / RVQ** | Huijben et al. *RVQ-VAE: Residual Vector Quantization for Controllable Speech Synthesis.* [arXiv:2401.14732](https://doi.org/10.48550/arXiv.2401.14732) |
| **MGVQ** | Jia et al. *MGVQ: Multi-Group Vector Quantization.* [arXiv:2507.07997](https://doi.org/10.48550/arXiv.2507.07997) |
| **Chronos** | Ansari et al. *Chronos: Learning the Language of Time Series.* [arXiv:2403.07815](https://doi.org/10.48550/arXiv.2403.07815) |

### Code

| | Repository |
| --- | --- |
| **TimeCausalVAE** | [github.com/justinhou95/TimeCausalVAE](https://github.com/justinhou95/TimeCausalVAE) |
| **vector-quantize-pytorch** | [github.com/lucidrains/vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) |
| **TimeVQVAE** | [github.com/ML4ITS/TimeVQVAE](https://github.com/ML4ITS/TimeVQVAE) |
| **QINCo** | [github.com/facebookresearch/Qinco](https://github.com/facebookresearch/Qinco) |
| **MGVQ** | [github.com/MKJia/MGVQ](https://github.com/MKJia/MGVQ) |
| **aotnumerics** | [github.com/stephaneckstein/aotnumerics](https://github.com/stephaneckstein/aotnumerics) |
