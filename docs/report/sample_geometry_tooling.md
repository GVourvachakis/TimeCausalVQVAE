# Sample Geometry Tooling

This note describes the sample-geometry utilities used to prepare final report
t-SNE and KDE/ECDF figures. The tooling is diagnostic only: it does not train
models, select models, or write committed artefacts.

## Utilities

- `src/time_causal_vae/evaluation/sample_geometry.py` builds one feature vector
  per path, fits a qualitative two-dimensional projection, and summarises
  feature distributions with ECDF points plus optional KDE values.
- `scripts/plot_sample_geometry.py` loads real and generated tensor payloads,
  accepts repeated `--generated-batch NAME:PATH` arguments, and writes local
  feature summaries and figures below `outputs/`.

The CLI supports `--dataset sp500_vix`, `--dataset hawkes_jump`, and
`--dataset generic`. Output directories outside `outputs/` are rejected.

## Feature Set

Common path features are:

- terminal return;
- realised volatility;
- maximum drawdown;
- return skewness;
- return excess kurtosis;
- mean one-step return;
- maximum absolute one-step return;
- return autocorrelation at configured lags;
- squared-return autocorrelation at configured lags.

For Hawkes/SVMHJD batches, the feature matrix also includes:

- detected jump count;
- detected jump fraction;
- one-path 1% return VaR;
- one-path 1% return expected shortfall;
- raw path increment.

When Hawkes paths contain non-positive values, the utilities treat the input as
a signed log-return series and derive drawdown features from the corresponding
normalised price path. Positive Hawkes paths are treated as price paths.

## Interpretation

t-SNE is a qualitative visual diagnostic only. It can reveal broad feature-space
separation between real and generated samples, but distances, clusters, and
neighbourhoods should not be used as model-selection evidence by themselves.
If scikit-learn is unavailable or too few samples are supplied, the projection
falls back to a deterministic PCA-style projection and records the fallback
reason in JSON metadata.

KDE/ECDF panels are the primary distributional view for report figures. ECDF
overlays are always written when `--kde` is requested. KDE overlays are written
only when `scipy.stats.gaussian_kde` is importable and the feature distribution
is non-degenerate. If KDE cannot be computed, the summary records an ECDF-only
fallback instead of labelling the result as KDE.

## Dependency Status

The current Poetry environment reports:

```text
sklearn True
scipy True
```

No dependency changes were made. If a fresh notebook environment lacks
scikit-learn, add `scikit-learn = ">=1.4,<2.0"` only to the notebook dependency
group. If SciPy is unavailable, use the ECDF fallback rather than adding SciPy
for this report tooling.

## Smoke Result

A local smoke run used the registered S&P500/VIX continuous evaluation batch as
both the real and generated payload source. The loader selected `real_data` for
the real sample and `fake_data` for the generated sample:

```bash
poetry run python scripts/plot_sample_geometry.py \
  --real-batch outputs/per_experiment_final_evaluation/sp500_vix/continuous/evaluation_batch.pt \
  --generated-batch beta_cvae:outputs/per_experiment_final_evaluation/sp500_vix/continuous/evaluation_batch.pt \
  --dataset sp500_vix \
  --output-dir outputs/sample_geometry_smoke \
  --tsne \
  --kde
```

The run completed successfully with feature shapes `real_shape: [1000, 17]` and
`generated_shapes: {'beta_cvae': [1000, 17]}`. It produced local, uncommitted
diagnostic files:

- `sample_geometry_projection.png`;
- `sample_geometry_ecdf.png`;
- `sample_geometry_kde.png`;
- JSON feature and summary payloads.
