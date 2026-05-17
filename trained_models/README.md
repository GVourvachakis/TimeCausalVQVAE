# Trained Model Metadata

This directory stores lightweight metadata for selected continuous and discrete latent-variable
models. It does not store trained checkpoints.

`model_registry.yaml` is the notebook-facing registry. It records, per experiment:

- selected continuous and discrete candidate ids;
- config paths under `configs/experiments/`;
- local checkpoint conventions such as `outputs/.../<training-run>/final_model`;
- selection profiles, visible metrics, missing metrics, no-leakage status, and caveats;
- optional comparison candidates needed for dynamic selection.

The per-experiment model cards provide a compact human-readable summary:

- `black_scholes/model_card.md`;
- `heston/model_card.md`;
- `pdv/model_card.md`;
- `sp500_vix/model_card.md`.

Expected checkpoints, token arrays, generated samples, CSV/JSON summaries, W&B artefacts, and
processed data live under local `outputs/` or `data/processed/` paths after a user runs training
or evaluation commands. Do not commit those artefacts.

Use the selector to inspect registered metadata:

```bash
poetry run python scripts/select_registered_model.py --experiment sp500_vix --family discrete
poetry run python scripts/select_registered_model.py --experiment sp500_vix --family discrete --metric mmd
poetry run python scripts/select_registered_model.py --experiment pdv --family continuous
```
