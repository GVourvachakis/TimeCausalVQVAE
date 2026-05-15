# S&P500/VIX Hidden128 Prior Robustness Setup

Status: setup completed. No non-smoke training was run. No tokenizer architecture, prior
architecture, signature conditioning, GroupedRVQ, MGVQ, diffusion, or objective change was made.

## Motivation

The `hidden128` standard-VQ tokenizer is the leading tokenizer candidate after:

- candidate-prior paper-style profile `0.266589` versus promoted baseline `0.298020`;
- broad latent geometry support with 64/64 active codes and global perplexity `50.3587`;
- VIX-bucket support of at least 61 active codes in every bucket.

The remaining gate is repeatability. This setup prepares seed-level prior robustness checks and
a sampling-temperature/top-k ablation for the existing seed-99 `hidden128` prior.

## Configs Created

Two hidden128 prior configs were added:

- `configs/experiments/sp500_vix_causal_token_prior_hidden128_candidate_seed1.yaml`
- `configs/experiments/sp500_vix_causal_token_prior_hidden128_candidate_seed2.yaml`

Relative to `configs/experiments/sp500_vix_causal_token_prior_hidden128_candidate.yaml`, each
config changes only:

- `experiment.name`;
- `experiment.seed`.

Both keep:

- tokenizer directory
  `outputs/sp500_vix_discrete/vq_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_hidden128_seed99`;
- token-data directory
  `outputs/sp500_vix_discrete/token_prior/sp500_vix_causal_vq_tokenizer_hidden128_tokens`;
- `condition_dim=1`;
- additive VIX-only conditioning;
- the same causal AR prior hyperparameters.

## Runner

The setup adds:

```text
scripts/run_sp500_vix_token_prior_candidate_ablation.py
```

The runner:

- validates S&P500/VIX additive scalar-conditioned token-prior configs;
- trains each config through `time_causal_vae.cli.train_token_prior`;
- evaluates the best checkpoint through `time_causal_vae.cli.evaluate_token_prior`;
- optionally runs `scripts/evaluate_sp500_vix_paper_style.py` with `--paper-style`;
- writes aggregate JSON and CSV summaries;
- records the model-selection profile score
  `MMD + SWD + volatility_wasserstein + terminal_return_wasserstein`;
- unsets `WANDB_MODE` and applies `MPLBACKEND=Agg`,
  `WANDB_DISABLE_SERVICE=true`, and `WANDB_START_METHOD=thread` to subprocesses;
- supports `--wandb`, `--no-wandb`, W&B project/entity options, and `--wandb-mode`.

Aggregate files are:

- `token_prior_candidate_ablation_summary.json`
- `token_prior_candidate_ablation_summary.csv`

## Dry-Run Status

The requested dry run was executed:

```bash
MPLBACKEND=Agg poetry run python scripts/run_sp500_vix_token_prior_candidate_ablation.py \
  --configs \
    configs/experiments/sp500_vix_causal_token_prior_hidden128_candidate_seed1.yaml \
    configs/experiments/sp500_vix_causal_token_prior_hidden128_candidate_seed2.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/hidden128_seed_ablation_dry \
  --base-data-dir data/processed \
  --epochs 1 \
  --n-sample 128 \
  --seed 99 \
  --temperature 0.8 \
  --top-k 40 \
  --dry-run \
  --no-wandb
```

Result:

| Config | Training seed | Dry-run status | Train tokens | Eval tokens | Conditions | Parameters |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `hidden128_candidate_seed1` | 1 | passed | `[2457, 60]` | `[2457, 60]` | `[2457, 1]` train/eval | 554432 |
| `hidden128_candidate_seed2` | 2 | passed | `[2457, 60]` | `[2457, 60]` | `[2457, 1]` train/eval | 554432 |

The dry run printed `Dry run complete. No training was started and no artifacts were written.`
for both configs. The runner summary was written under:

```text
outputs/sp500_vix_discrete/token_prior/hidden128_seed_ablation_dry/
```

Because the experiment names include `seed1` and `seed2`, the package training CLI's standard
run-directory convention yields dry-run target directories ending in `_seed1_seed1` and
`_seed2_seed2`. This is only a naming detail; the config seeds are correctly `1` and `2`.

## Non-Smoke Runner Command

The planned non-smoke robustness command is:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread \
poetry run python scripts/run_sp500_vix_token_prior_candidate_ablation.py \
  --configs \
    configs/experiments/sp500_vix_causal_token_prior_hidden128_candidate_seed1.yaml \
    configs/experiments/sp500_vix_causal_token_prior_hidden128_candidate_seed2.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/hidden128_seed_ablation \
  --base-data-dir data/processed \
  --epochs 100 \
  --n-sample 1000 \
  --seed 99 \
  --temperature 0.8 \
  --top-k 40 \
  --paper-style \
  --wandb \
  --wandb-project time-causal-token-prior \
  --wandb-entity tc_vae
```

If live W&B fails with connection, socket, or `CommError` failures, rerun the same command with
`--no-wandb` and document the failure path in the results note.

## Sampling Ablation Plan

The existing `scripts/run_token_prior_sampling_ablation.py` supports the requested seed0
hidden128 sampling grid via:

- `--temperatures`;
- `--top-k-values`;
- `none` as the unrestricted top-k value.

The full sampling ablation was not run in this setup prompt. The planned command is:

```bash
MPLBACKEND=Agg poetry run python scripts/run_token_prior_sampling_ablation.py \
  --config configs/experiments/sp500_vix_causal_token_prior_hidden128_candidate.yaml \
  --prior-dir outputs/sp500_vix_discrete/token_prior/candidate_priors/sp500_vix_causal_token_prior_hidden128_candidate_seed99/best_model \
  --tokenizer-dir outputs/sp500_vix_discrete/vq_tokenizer_ablation/sp500_vix_causal_vq_tokenizer_hidden128_seed99 \
  --output-dir outputs/sp500_vix_discrete/token_prior/hidden128_sampling_ablation \
  --base-data-dir data/processed \
  --n-sample 1000 \
  --seed 99 \
  --temperatures 0.6 0.8 1.0 \
  --top-k-values none 20 40
```

The expected aggregate files are:

- `sampling_ablation_summary.json`
- `sampling_ablation_summary.csv`

## W&B Profile

Every non-smoke robustness run should use:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread
```

The runner forwards W&B settings only to training runs. Evaluation and paper-style diagnostics
remain local artifact generation steps. If W&B remains unavailable, use `--no-wandb`; do not set
`WANDB_MODE=offline` for the promoted non-smoke attempt unless the run is explicitly being marked
as offline-only.

## Next Step

Run the non-smoke seed1/seed2 robustness command, then compare:

- token likelihood stability;
- decoded model-selection profile score;
- paper-style profile score;
- VIX-bucket sampled code usage;
- whether `hidden128` remains ahead of the promoted baseline across seeds.
