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

### [market_generators_2026]

**Title:** Market Generators: A Paradigm Shift in Financial Modeling  
**Link:** [Springer DOI 10.1007/978-3-031-97239-3\_4](https://doi.org/10.1007/978-3-031-97239-3_4)  
**Used for:** market-generator motivation, path-wise evaluation, conditioning discussion, and
signature-kernel-backed performance-evaluation context.  
**Not used for:** implementation code or dependency choices.

## Path signatures and signature kernels

### [signatory_2021]

**Title:** Signatory: differentiable computations of the signature and logsignature transforms, on
both CPU and GPU  
**Link:** [arXiv:2001.00706](https://arxiv.org/abs/2001.00706)  
**Code:** [github.com/patrick-kidger/signatory](https://github.com/patrick-kidger/signatory)  
**Used for:** future signature and log-signature feature-extraction background.  
**Not used for:** current dependencies; compatibility with the current Python/PyTorch environment
must be tested before use.

### [iisignature_2020]

**Title:** Algorithm 1004: The iisignature Library: Efficient Calculation of Iterated-Integral
Signatures and Log Signatures  
**Link:** [ACM DOI 10.1145/3371237](https://doi.org/10.1145/3371237)  
**Code:** [github.com/bottler/iisignature](https://github.com/bottler/iisignature)  
**Used for:** CPU-oriented signature and log-signature package inspection.  
**Not used for:** current dependencies or GPU assumptions.

### [signature_kernel_goursat_pde_2021]

**Title:** The Signature Kernel is the solution of a Goursat PDE  
**Link:** [arXiv:2006.14794](https://arxiv.org/abs/2006.14794)  
**Code:** [github.com/crispitagorico/sigkernel](https://github.com/crispitagorico/sigkernel)  
**Used for:** future evaluation-only signature-kernel distances, MMD, and path-space scoring-rule
background.  
**Not used for:** current training objectives or dependencies.

### [ksig_2025]

**Title:** A User's Guide to KSig: GPU-Accelerated Computation of the Signature Kernel  
**Link:** [arXiv:2501.07145](https://arxiv.org/abs/2501.07145)  
**Code:** [github.com/tgcsaba/KSig](https://github.com/tgcsaba/KSig)  
**Used for:** candidate signature-kernel package inspection and possible numerical cross-checks.  
**Not used for:** current dependencies.

### [signature_kernel_scores_neuralsde_2023]

**Title:** Non-adversarial training of Neural SDEs with signature kernel scores  
**Link:** [NeurIPS 2023 proceedings](https://papers.nips.cc/paper_files/paper/2023/hash/2460396f2d0d421885997dd1612ac56b-Abstract-Conference.html)  
**Used for:** future background on objective-level signature-kernel scoring rules for path-space
generators.  
**Not used for:** the current hard-token VQ prior objective.

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
