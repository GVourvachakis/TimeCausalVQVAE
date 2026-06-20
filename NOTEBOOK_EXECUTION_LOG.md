# Notebook Execution Log

- Manifest: `notebooks/full_preview_notebook_manifest.yaml`
- Parameter mode: `full-preview`
- Started: 2026-06-20T02:37:43+00:00
- Last update: 2026-06-20T02:40:47+00:00
- Status: complete
- Initial estimated total: 1h 6m 0s
- Remaining ETA: 0s
- Runtime guard: 6.0 hours

## Message

Initial full-preview pass failed for notebooks/continuous/hawkes_jump.ipynb because an existing local evaluation batch exposed a tensor truth-value check. The notebook was patched to select tensors without boolean coercion and rerun only; the rerun passed.

| Priority | Notebook | Bucket | Status | Runtime | Started | Ended | ETA after | Reason | Expensive metrics | Traceback |
| ---: | --- | --- | --- | ---: | --- | --- | ---: | --- | --- | --- |
| 10 | `notebooks/benchmarks/hawkes_jump_dataset.ipynb` | short | passed | 8s | 2026-06-20T02:37:43+00:00 | 2026-06-20T02:37:51+00:00 | 31m 28s |  | skipped by manifest |  |
| 20 | `notebooks/continuous/black_scholes.ipynb` | short | passed | 5s | 2026-06-20T02:37:51+00:00 | 2026-06-20T02:37:56+00:00 | 31m 4s |  | skipped by manifest |  |
| 30 | `notebooks/continuous/heston.ipynb` | short | passed | 5s | 2026-06-20T02:37:56+00:00 | 2026-06-20T02:38:01+00:00 | 30m 53s |  | skipped by manifest |  |
| 40 | `notebooks/continuous/pdv.ipynb` | short | passed | 5s | 2026-06-20T02:38:01+00:00 | 2026-06-20T02:38:05+00:00 | 30m 45s |  | skipped by manifest |  |
| 50 | `notebooks/continuous/sp500_vix.ipynb` | short | passed | 5s | 2026-06-20T02:38:05+00:00 | 2026-06-20T02:38:10+00:00 | 30m 38s |  | skipped by manifest |  |
| 60 | `notebooks/continuous/hawkes_jump.ipynb` | short | passed | 5s | 2026-06-20T02:40:41+00:00 | 2026-06-20T02:40:47+00:00 | 30m 33s | passed after tensor-safe missing-output fix and rerun | skipped by manifest |  |
| 70 | `notebooks/discrete/black_scholes.ipynb` | short | passed | 5s | 2026-06-20T02:38:15+00:00 | 2026-06-20T02:38:21+00:00 | 30m 27s |  | skipped by manifest |  |
| 80 | `notebooks/discrete/heston.ipynb` | short | passed | 5s | 2026-06-20T02:38:21+00:00 | 2026-06-20T02:38:25+00:00 | 30m 21s |  | skipped by manifest |  |
| 90 | `notebooks/discrete/pdv.ipynb` | short | passed | 5s | 2026-06-20T02:38:25+00:00 | 2026-06-20T02:38:30+00:00 | 30m 16s |  | skipped by manifest |  |
| 100 | `notebooks/discrete/sp500_vix.ipynb` | short | passed | 5s | 2026-06-20T02:38:30+00:00 | 2026-06-20T02:38:35+00:00 | 30m 10s |  | skipped by manifest |  |
| 110 | `notebooks/discrete/hawkes_jump.ipynb` | short | passed | 6s | 2026-06-20T02:38:35+00:00 | 2026-06-20T02:38:41+00:00 | 30m 5s |  | skipped by manifest |  |
| 120 | `notebooks/discrete/discrete_latent_geometry.ipynb` | short | passed | 3s | 2026-06-20T02:38:41+00:00 | 2026-06-20T02:38:45+00:00 | 30m 0s |  | skipped by manifest |  |
| 130 | `notebooks/report/hawkes_jump_model_comparison.ipynb` | medium | passed | 18s | 2026-06-20T02:38:45+00:00 | 2026-06-20T02:39:03+00:00 | 36s |  | skipped by manifest |  |
| 140 | `notebooks/report/sp500_vix_report_figures.ipynb` | medium | passed | 23s | 2026-06-20T02:39:03+00:00 | 2026-06-20T02:39:25+00:00 | 20s |  | skipped by manifest |  |
| 150 | `notebooks/report/sample_geometry_diagnostics.ipynb` | medium | passed | 21s | 2026-06-20T02:39:25+00:00 | 2026-06-20T02:39:47+00:00 | 0s |  | skipped by manifest |  |
