# Multidimensional Public Smoke Validation

Current branch: `feature/multidim-experimental-benchmarks`.

This validation checks the public-safe multidimensional benchmark
infrastructure. It does not run non-smoke training, does not merge the branch,
and does not commit generated data or output artefacts.

## Synthetic Multifactor Smoke

Command:

```bash
poetry run python scripts/smoke_multifactor_market_dataset.py \
  --n-samples 128 \
  --n-assets 50 \
  --n-factors 5 \
  --n-timesteps 60 \
  --seed 99 \
  --output-dir outputs/multidim_public_smoke/multifactor
```

Status: passed.

Observed tensors:

- data: `(128, 60, 50)`;
- raw log returns: `(128, 60, 50)`;
- labels: `(128, 1)`;
- loadings: `(50, 5)`;
- common and sector jump counts: `0`.

Generated local files:

- `outputs/multidim_public_smoke/multifactor/summary.json`;
- `outputs/multidim_public_smoke/multifactor/summary.md`.

These files are local smoke artefacts and are not tracked.

## Empirical Download Smoke

Dependency check:

```bash
poetry run python -c "import yfinance as yf; print(yf.__version__)"
```

Status: passed with `yfinance` version `1.4.1`.

Initial sandboxed command:

```bash
poetry run python scripts/download_sp500_50_panel.py \
  --start 2020-01-01 \
  --end 2020-06-30 \
  --output-root data \
  --include-sector-etfs
```

Status: failed in the sandbox because Yahoo host resolution was unavailable
(`Could not resolve host: guce.yahoo.com`).

Approved network-enabled retry of the same short-range command: passed.

Observed processed tensors:

- data: `(64, 60, 50)`;
- labels: `(64, 2)`;
- aligned price dates: `124`;
- window start: `2020-01-03`;
- window end: `2020-06-29`;
- conditions: `spy_log_return_start`, `log_vix_level_start`.

Generated local files under `data/raw/sp500_50_panel/` and
`data/processed/sp500_50_panel/` are downloaded or processed empirical data and
must not be tracked.

## No-Leakage And Shape Smokes

Multifactor tokenizer no-leakage command:

```bash
poetry run python scripts/check_conditional_vq_tokenizer_no_leakage.py \
  --config configs/experiments/multifactor_market_causal_vq_tokenizer.yaml
```

Status: passed.

Observed tensors:

- input: `(8, 60, 50)`;
- conditions: `(8, 1)`;
- indices: `(8, 60)`;
- reconstruction: `(8, 60, 50)`;
- maximum prefix difference after future perturbation: `0.00000000e+00`.

Generic tokenizer shape command:

```bash
poetry run python scripts/check_vq_tokenizer_shapes.py
```

Status: passed.

Empirical dataset smoke command, adjusted to the short-range panel's available
split sizes:

```bash
poetry run python scripts/smoke_sp500_50_panel_dataset.py \
  --n-samples 13 \
  --train-n-samples 32 \
  --eval-n-samples 13 \
  --n-timesteps 60 \
  --base-data-dir data/processed \
  --output-dir outputs/multidim_public_smoke/sp500_50_panel \
  --standardize-returns
```

Status: passed.

Observed tensors:

- train data: `(32, 60, 50)`;
- train labels: `(32, 2)`;
- eval data: `(13, 60, 50)`;
- eval labels: `(13, 2)`.

The default empirical tokenizer config requests more windows than the
short-range smoke panel contains. For this validation only, a temporary
`/tmp` config was made from
`configs/experiments/sp500_50_panel_causal_vq_tokenizer.yaml` with
`n_samples: 13`.

Empirical tokenizer no-leakage command:

```bash
poetry run python scripts/check_conditional_vq_tokenizer_no_leakage.py \
  --config /tmp/sp500_50_panel_public_smoke_causal_vq_tokenizer.yaml \
  --batch-size 8
```

Status: passed.

Observed tensors:

- input: `(8, 60, 50)`;
- conditions: `(8, 2)`;
- indices: `(8, 60)`;
- reconstruction: `(8, 60, 50)`;
- maximum prefix difference after future perturbation: `0.00000000e+00`.

## Readiness Decision

The public multidimensional benchmark infrastructure is smoke-ready for branch
review. Synthetic dataset generation, empirical short-range local processing,
cross-sectional dataset diagnostics, and tokenizer no-leakage checks all pass.

The branch should not be merged automatically. It remains an experimental
benchmark integration with no multidimensional registry-selected model.
Generated outputs and downloaded or processed empirical data remain local only.

## Final Merge Readiness

Final validation was run on branch
`feature/multidim-experimental-benchmarks` before manual merge review.

Commands run:

```bash
poetry run python scripts/smoke_multifactor_market_dataset.py \
  --n-samples 128 \
  --n-assets 50 \
  --n-factors 5 \
  --n-timesteps 60 \
  --seed 99 \
  --output-dir outputs/multidim_public_smoke_final/multifactor
```

Status: passed. The synthetic smoke produced data and raw-return tensors with
shape `(128, 60, 50)`, labels with shape `(128, 1)`, and loadings with shape
`(50, 5)`.

```bash
poetry run python scripts/smoke_sp500_50_panel_dataset.py \
  --n-samples 13 \
  --train-n-samples 32 \
  --eval-n-samples 13 \
  --n-timesteps 60 \
  --base-data-dir data/processed \
  --output-dir outputs/multidim_public_smoke_final/sp500_50_panel \
  --standardize-returns
```

Status: passed because a local short-range processed panel was available. The
empirical smoke produced train tensors with shape `(32, 60, 50)` and eval
tensors with shape `(13, 60, 50)`, each with two prefix-safe condition labels.

```bash
poetry run python scripts/check_conditional_vq_tokenizer_no_leakage.py \
  --config configs/experiments/multifactor_market_causal_vq_tokenizer.yaml \
  --batch-size 8
```

Status: passed with maximum prefix reconstruction difference
`0.00000000e+00`.

For the empirical tokenizer no-leakage check, a temporary smoke-sized config was
created under `/tmp` from
`configs/experiments/sp500_50_panel_causal_vq_tokenizer.yaml` with
`n_samples: 13`.

```bash
poetry run python scripts/check_conditional_vq_tokenizer_no_leakage.py \
  --config /tmp/sp500_50_panel_final_smoke_causal_vq_tokenizer.yaml \
  --batch-size 8
```

Status: passed with maximum prefix reconstruction difference
`0.00000000e+00`.

Additional public-branch checks:

```bash
poetry run python - <<'PY'
from time_causal_vae.data.multifactor_market import MultifactorMarketDataset
from time_causal_vae.data.sp500_panel import SP50050PanelDataset
from time_causal_vae.evaluation.cross_sectional_diagnostics import compare_cross_sectional_diagnostics
print("multidim imports ok")
PY
poetry run python scripts/select_registered_model.py --experiment sp500_vix --family discrete
poetry check
poetry run ruff check src scripts configs docs --fix
poetry run mypy src/time_causal_vae
git ls-files | grep -E '(^outputs/|^wandb/|^data/raw/|^data/processed/|\.npy$|\.npz$|\.pt$|\.pkl$|\.pyc$|__pycache__)' || true
```

Status: passed. The registry selector still resolves the existing
one-dimensional `sp500_vix` discrete public baseline, `poetry check` passed,
`ruff` passed, and `mypy` reported no issues in `116` source files.

The forbidden artefact scan reported no tracked outputs, W&B artefacts,
downloaded raw or processed financial data, tensor dumps, pickle files, bytecode
files, or `__pycache__` entries. No generated outputs or downloaded data are
tracked by git. No multidimensional model entry has been added to
`trained_models/model_registry.yaml`.

Readiness decision: the branch is ready for manual review and manual merge as
an experimental benchmark infrastructure branch. It should still not be merged
automatically, and it does not claim a selected multidimensional generator.
