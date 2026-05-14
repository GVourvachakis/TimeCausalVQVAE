# Signature Feature Extraction Smoke

Status: synthetic smoke run completed with the optional `iisignature` dependency missing from the
Poetry environment.

## Command run

```bash
poetry run python scripts/extract_signature_features.py \
  --synthetic \
  --output-dir outputs/signature_features_smoke \
  --depth 2 \
  --context-length 8 \
  --use-lead-lag \
  --include-time
```

## Dependency status

`iisignature` was not installed in the Poetry environment. The script printed the expected optional
dependency message:

```text
Optional dependency missing: iisignature
Optional signature feature extraction requires 'iisignature'. It is not installed by default. In a temporary or opt-in environment, install NumPy first and then try: pip install iisignature --no-build-isolation
Wrote missing-dependency summary to outputs/signature_features_smoke
```

## Outputs

Generated files under ignored `outputs/signature_features_smoke/`:

- `signature_feature_summary.json`
- `signature_feature_summary.md`

No `train_signature_features.npz` or `eval_signature_features.npz` files were produced because the
optional backend was missing. This is the intended behaviour: the smoke run records the missing
optional dependency rather than pretending feature extraction passed.

## Requested settings

- Dataset: `sp500_vix`
- Synthetic: `True`
- Depth: `2`
- Context length: `8`
- Lead-lag: `True`
- Include time: `True`
- Include VIX: `False`
- Seed: `99`

## Feature status

- Feature shape: unavailable because `iisignature` was missing.
- Finite-value check: not run because no feature matrix was produced.
- Summary status: `missing_optional_dependency`.

The optional install hint remains:

```bash
pip install iisignature --no-build-isolation
```

after installing NumPy in the target opt-in environment.
