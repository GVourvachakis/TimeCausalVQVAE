# Trained Model Metadata

This public branch does not store trained checkpoints. It keeps only metadata for
the S&P500/VIX continuous reference, public discrete baseline, and optional best
discrete research variant.

Expected checkpoints and generated samples live under local `outputs/` paths after
a user runs the training or evaluation commands. Do not commit model
weights, token arrays, processed data, W&B artefacts, or generated output files.

See `model_registry.yaml` for the public config paths, research-branch config
paths, expected local output locations, and sampling policies.
