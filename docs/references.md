# References

This file records canonical references used by the implementation. Source files should cite these
keys in compact docstrings rather than duplicating full bibliographic text.

## Generative market modelling

### [tcvae_2024]

**Title:** Time-Causal VAE: Robust Financial Time Series Generator  
**Link:** [arXiv:2411.02947](https://arxiv.org/abs/2411.02947)  
**Code:** [github.com/justinhou95/TimeCausalVAE](https://github.com/justinhou95/TimeCausalVAE)  
**Used for:** no-anticipation architecture, continuous TC-VAE baseline,
conditional PDV and S&P500/VIX benchmarks, financial evaluation protocol.
**Not used for:** RealNVP prior in the promoted public discrete-token baseline.

## Vector-quantized representations

### [vqvae_2017]

**Title:** Neural Discrete Representation Learning  
**Link:** [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)  
**Used for:** discrete latent codes, commitment loss, tokenizer-prior separation.

### [timevqvae_2023]

**Title:** Vector Quantized Time Series Generation with a Bidirectional Prior Model  
**Link:** [arXiv:2303.04743](https://arxiv.org/abs/2303.04743)  
**Code:** [github.com/ML4ITS/TimeVQVAE](https://github.com/ML4ITS/TimeVQVAE)  
**Used for:** two-stage VQ time-series generation reference.  
**Not used for:** bidirectional prior; our prior must remain causal.

### [vector_quantize_pytorch]

**Repository:** [github.com/lucidrains/vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)  
**Used for:** VectorQuantize, ResidualVQ, and already wrapped GroupedResidualVQ backends.

### [qinco_2024]

**Title:** Residual Quantization with Implicit Neural Codebooks  
**Link:** [arXiv:2401.14732](https://arxiv.org/abs/2401.14732)  
**Code:** [github.com/facebookresearch/Qinco](https://github.com/facebookresearch/Qinco)  
**Used for:** residual-quantization background only.
**Not used for:** implicit neural codebook implementation.

### [mgvq_2025]

**Title:** MGVQ: Could VQ-VAE Beat VAE? A Generalizable Tokenizer with Multi-group Quantization  
**Link:** [arXiv:2507.07997](https://arxiv.org/abs/2507.07997)  
**Code:** [github.com/MKJia/MGVQ](https://github.com/MKJia/MGVQ)  
**Used for:** future grouped-tokenizer motivation.
**Not used for:** current implementation; MGVQ is deferred.

## Causal / diffusion references

### [vqdiffusion_2021]

**Title:** Vector Quantized Diffusion Model for Text-to-Image Synthesis  
**Link:** [arXiv:2111.14822](https://arxiv.org/abs/2111.14822)  
**Used for:** mask-and-replace discrete diffusion background.  
**Not used for:** promoted public method; diffusion is deferred.

### [causal_diffusion_transformers_2024]

**Title:** Causal Diffusion Transformers for Generative Modeling  
**Link:** [arXiv:2412.12095](https://arxiv.org/abs/2412.12095)  
**Used for:** future causal diffusion motivation only.
**Not used for:** current token prior; diffusion and CausalFusion-style generation are deferred.

## Financial time-series and downstream diagnostics

### [deepvol_2022]

**Title:** DeepVol: Volatility Forecasting from High-Frequency Data with Dilated Causal Convolutions  
**Link:** [arXiv:2210.04797](https://arxiv.org/abs/2210.04797)  
**Used for:** dilated causal convolution motivation in financial time-series encoders.

### [aotnumerics]

**Repository:** [github.com/stephaneckstein/aotnumerics](https://github.com/stephaneckstein/aotnumerics)  
**Used for:** adapted/causal optimal transport background.
**Not used for:** vendored implementation code.

### [chronos_2024]

**Title:** Chronos: Learning the Language of Time Series  
**Link:** [arXiv:2403.07815](https://arxiv.org/abs/2403.07815)  
**Code:** [github.com/amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)  
**Used for:** contrast with forecasting foundation models and scalar-value tokenisation.
**Not used for:** forecasting-only pretrained foundation-model workflow.
