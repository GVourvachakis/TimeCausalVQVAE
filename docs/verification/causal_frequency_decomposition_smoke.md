# Causal Frequency Decomposition Smoke Check

Status: completed for deterministic causal EMA decomposition utilities.

This smoke check verifies the first stage recommended in
`docs/architecture/causal_frequency_tokenizer.md`: a deterministic causal EMA split into low- and
high-frequency components. It does not train tokenizers or priors and does not modify the
promoted S&P500/VIX baseline.

## Recurrence

For each path `x`, the low-frequency component is computed causally:

```text
low_0 = x_0
low_t = alpha * x_t + (1 - alpha) * low_{t-1}
high_t = x_t - low_t
```

Composition is the additive inverse of the split:

```text
compose_low_high(low, high) = low + high
```

## Command

```bash
poetry run python scripts/check_causal_frequency_decomposition_no_leakage.py \
  --batch-size 8 \
  --length 60 \
  --alpha 0.2 \
  --cutoff 29 \
  --seed 99
```

## Parameters and Shapes

- Alpha tested: `0.2`.
- Batch size: `8`.
- Path length: `60`.
- Inclusive cutoff: `29`.
- Random seed: `99`.
- Tensor layouts tested:
  - `[batch, time]`: `(8, 60)`.
  - `[batch, time, channels]`: `(8, 60, 1)`.

## No-Leakage Result

The script created synthetic positive paths, decomposed them, perturbed only values after the
inclusive cutoff, and recomputed the low/high components. Prefix values through the cutoff were
unchanged:

| Shape | Max low prefix difference | Max high prefix difference |
| --- | ---: | ---: |
| `(8, 60)` | `0.00000000e+00` | `0.00000000e+00` |
| `(8, 60, 1)` | `0.00000000e+00` | `0.00000000e+00` |

This confirms prefix invariance for the deterministic EMA decomposition under the tested
synthetic perturbation.

## Reconstruction Result

The optional reconstruction smoke also passed. For both tested tensor layouts,
`compose_low_high(low, high)` exactly reconstructed the original path up to the script tolerance:

| Shape | Max reconstruction difference |
| --- | ---: |
| `(8, 60)` | `0.00000000e+00` |
| `(8, 60, 1)` | `0.00000000e+00` |

## Caveats

- This verifies only the deterministic decomposition utilities, not any tokenizer, decoder, or
  token prior.
- The check uses synthetic positive paths rather than the private local S&P500/VIX array.
- The check covers one EMA smoothing value, `alpha = 0.2`; future tuning should use a
  predeclared validation grid.
- The decomposition is causal by construction, but downstream users still need separate
  no-leakage checks after connecting it to tokenizers or priors.
- The command emitted a local Matplotlib cache warning because the default user config directory
  was not writable in this environment; the warning did not affect the decomposition check.
