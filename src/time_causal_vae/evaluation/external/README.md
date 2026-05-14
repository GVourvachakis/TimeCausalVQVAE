# External Evaluation Algorithms

This directory isolates optional borrowed or adapted evaluation algorithms from the
upstream Time-Causal VAE research release. The standard selected reproduction path does
not require these modules: released-checkpoint reproduction uses Gaussian MMD and SWD.
The notebooks under `notebooks/` expose these routines behind
`RUN_UPSTREAM_DIAGNOSTICS` and `RUN_HEAVY_OPTIONAL_DIAGNOSTICS` flags so the upstream
paper-analysis workflow remains available without making optional diagnostics part of
ordinary notebook execution.

## Expected-Signature Metrics

`signatures.py` contains expected-signature helpers adapted from the upstream evaluation
code. The upstream audit records references to Sig-Wasserstein-GANs by Ni et al. (2021)
and Randomised-Signature-TimeSeries-Generation by Biagini, Gonon, and Walter.

This path is an upstream optional expected-signature diagnostic. It may require
`signatory`, and it is not part of the promoted public S&P500/VIX workflow. Missing
optional dependencies should surface only when a caller explicitly invokes the
signature metric. New signature-conditioning and signature-kernel work should be
implemented outside `evaluation.external` so this upstream compatibility area remains
isolated from project-native metrics.

`signatory` is not part of the standard environment because it is not compatible with
the current Python/PyTorch constraints used by this project. `log-signatures-pytorch` is
also not added by default because current releases require Python >=3.13 and PyTorch
>=2.9. Until a compatible backend is selected, notebooks should skip expected-signature
diagnostics with a clear message.

## Adapted Wasserstein / SAWD

`awd/` contains the adapted-Wasserstein implementation copied from the upstream borrowed
algorithm code. It is optional, may require POT (`ot`) and solver-specific packages, and
is not part of selected reproduction metrics.

## Optimal Stopping

`optimal_stopping/` contains optional downstream optimal-stopping algorithms copied from
the upstream evaluation tree. These routines are not required for selected checkpoint
reproduction and should be treated as optional financial-analysis code.

## NDB

`ndb.py` contains the optional NDB diagnostic copied from the upstream implementation,
which references `gans-n-gmms`. It is not required for selected reproduction.

## Maintenance

These modules are intentionally isolated from the core target evaluator. Changes here
should include attribution notes and dedicated parity checks before being used for paper
claims.
