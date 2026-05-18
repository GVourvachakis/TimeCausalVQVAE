# Hawkes-Jump Continuous Log-Return Evaluator

## Status

The generic `tcvae-evaluate` path is not sufficient for the Hawkes log-return
continuous baseline because it evaluates tensors in the model output convention.
For `data_output: log_return`, market and jump diagnostics must first convert
generated and real log-return paths to normalised prices.

`scripts/evaluate_hawkes_jump_continuous.py` provides that Hawkes-specific
evaluation path. No models were trained, no continuous architecture was changed,
simulator parameters were unchanged, and the registry was not updated.

## Evaluator

The script accepts:

```bash
poetry run python scripts/evaluate_hawkes_jump_continuous.py \
  --config configs/experiments/hawkes_jump_beta_cvae_logreturn.yaml \
  --model-dir <final_model-dir> \
  --output-dir <outputs-dir> \
  --n-sample 1024 \
  --seed 99 \
  --base-data-dir data/processed
```

For real evaluations it:

- loads the selected continuous config and `final_model` checkpoint;
- generates paths through the existing continuous model evaluator;
- reads `data_output` from `data.params.data_output`;
- converts real, generated, and reconstructed log returns to normalised prices
  when `data_output: log_return`;
- fits robust detected-jump thresholds on the real Ogata evaluation price paths;
- computes smooth path metrics, market diagnostics, jump diagnostics, VaR/ES,
  detected jump-count W1, and detected jump-size W1;
- writes `evaluation_summary.json`, `evaluation_summary.md`, and
  `evaluation_batch.pt`.

For `data_output: price`, the script preserves price-space behaviour and checks
that decoded paths are finite and positive.

## Dry-Run

No log-return continuous smoke checkpoint currently exists. The only detected
Hawkes continuous checkpoint under `outputs/` is from the earlier price-level
first-comparison run. Therefore the evaluator setup was verified with dry-run:

```bash
poetry run python scripts/evaluate_hawkes_jump_continuous.py \
  --config configs/experiments/hawkes_jump_beta_cvae_logreturn.yaml \
  --output-dir outputs/hawkes_jump_continuous_logreturn_evaluator_dry \
  --n-sample 16 \
  --seed 99 \
  --dry-run
```

The dry-run completed and wrote:

- `outputs/hawkes_jump_continuous_logreturn_evaluator_dry/evaluation_summary.json`;
- `outputs/hawkes_jump_continuous_logreturn_evaluator_dry/evaluation_summary.md`.

The summary confirms `data_output: log_return` and
`log_return_to_price_conversion: true`.

## Robustness Use

After continuous log-return checkpoints are trained, use this evaluator for the
continuous baseline in the Hawkes log-return robustness study. Report the same
path and jump metrics as the discrete token-prior evaluator:

- MMD;
- SWD;
- terminal W1;
- volatility W1;
- drawdown W1;
- jump-count W1;
- jump-size W1;
- paths-with-jumps fraction;
- negative jump fraction;
- VaR/ES;
- return and squared-return autocorrelation diagnostics.

The evaluator keeps the no-arbitrage caveat unchanged: generated prices are
scenario paths for diagnostics, not an arbitrage-free pricing model.
