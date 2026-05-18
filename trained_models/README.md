# Trained Model Metadata

This directory stores lightweight metadata for selected continuous and discrete latent-variable models. It does not store trained checkpoints or weights.

`model_registry.yaml` is the notebook-facing registry. It records, per experiment:

- selected continuous and discrete candidate ids for the current registry state;
- config paths under `configs/experiments/`;
- local checkpoint conventions such as `outputs/.../<training-run>/final_model`;
- selection profiles, visible metrics, missing metrics, no-leakage status, and caveats;
- optional comparison candidates needed for dynamic selection.

Selected entries are current registry selections for public workflows, not universal mathematical optima. They reflect the metrics and caveats recorded in the registry and model cards.

Hawkes/SVMHJD is an optional synthetic benchmark. No Hawkes/SVMHJD trained model is registered yet,
and registry updates require seed-robust model-selection evidence.

The per-experiment model cards provide a compact human-readable summary:

- `black_scholes/model_card.md`;
- `heston/model_card.md`;
- `pdv/model_card.md`;
- `sp500_vix/model_card.md`.

Expected checkpoints, token arrays, generated samples, CSV/JSON summaries, W&B artefacts, and processed data live under local `outputs/` or `data/processed/` paths after a user runs training or evaluation commands. Do not commit those artefacts.

Use the selector to inspect registered metadata by experiment family, profile, or metric:

```bash
poetry run python scripts/select_registered_model.py --experiment sp500_vix --family discrete
poetry run python scripts/select_registered_model.py --experiment sp500_vix --family discrete --profile balanced_market
poetry run python scripts/select_registered_model.py --experiment sp500_vix --family discrete --metric mmd
poetry run python scripts/select_registered_model.py --experiment pdv --family continuous
```
