# Signature-Kernel Metric Smoke

## Purpose

This note records the first evaluation-only signature-kernel metric smoke. The
implementation is optional and uses `sigkernel` only when it is installed. No
dependency was added to `pyproject.toml`, no model was trained, no
signature-kernel loss was implemented, and no Gumbel-Softmax relaxation was
introduced.

## Command

```bash
poetry run python scripts/evaluate_signature_kernel_metrics.py \
  --synthetic \
  --output-dir outputs/signature_kernel_smoke \
  --max-samples 16 \
  --dyadic-order 1 \
  --include-time
```

## Result

`sigkernel` is not installed in the current Poetry environment.

The script exited cleanly and wrote:

```text
outputs/signature_kernel_smoke/signature_kernel_summary.json
outputs/signature_kernel_smoke/signature_kernel_summary.md
```

Summary status:

| Field | Value |
| --- | --- |
| Status | `missing_dependency` |
| `sigkernel` installed | `false` |
| Synthetic mode | `true` |
| Max samples | `16` |
| Dyadic order | `1` |
| RBF sigma | `1.0` |
| Include time | `true` |
| Lead-lag | `false` |

Missing dependency message:

```text
Optional signature-kernel metrics require 'sigkernel'. It is not installed by default. In a temporary or opt-in environment, try: pip install Cython && pip install 'git+https://github.com/crispitagorico/sigkernel.git' --no-build-isolation
```

## Metric Availability

No signature-kernel MMD value was produced in this Poetry smoke because the
optional package is absent. Therefore the finite, symmetry, and positive
diagonal Gram checks were not executed in this environment.

The implementation path is ready to exercise those checks once `sigkernel` is
installed. The module computes:

- finite checks for `Kxx`, `Kyy`, and `Kxy`;
- symmetry checks for `Kxx` and `Kyy`;
- positive-diagonal checks for `Kxx` and `Kyy`;
- a biased signature-kernel MMD from the three Gram matrices.

## Decision

Use this metric in the next branch only after explicitly opting into
`sigkernel` in a temporary or dependency-review environment. The next branch
should run the same synthetic command again and then compare saved
paper-style batches, starting with small `--max-samples` values before using it
for model selection.
