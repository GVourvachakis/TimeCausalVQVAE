# S&P500/VIX VQ Tokenizer Ablation Setup

Status: setup complete for controlled standard-VQ tokenizer hyperparameter ablations. No
non-smoke training was run.

## Scope

This setup follows `docs/architecture/vq_tokenizer_tuning_roadmap.md` Stage A. It keeps the
promoted baseline config unchanged and adds only standard VQ tokenizer variants for the
S&P500/VIX benchmark.

The runner rejects configs outside the controlled surface:

- dataset: `sp500_vix`;
- condition feature: VIX with `condition_dim: 1`;
- model family: `causal_vq_tokenizer`;
- quantizer type: standard one-code vector VQ;
- no RVQ, GroupedRVQ, MGVQ, diffusion, or signature conditioning.

## Supported Config Fields

The inspected tokenizer code supports the planned Stage A fields.

| Field | Support path |
| --- | --- |
| `codebook_size` | `VQTokenizerConfig`, `VectorQuantizerAdapter`, training config loader |
| `embedding_dim` | `VQTokenizerConfig`, encoder/decoder stack, quantizer adapter |
| `codebook_dim` | `VQTokenizerConfig`, `VectorQuantize` backend |
| `commitment_weight` | `VQTokenizerConfig`, `VectorQuantize` backend |
| `decay` | `VQTokenizerConfig`, `VectorQuantize` backend |
| `threshold_ema_dead_code` | `VQTokenizerConfig`, `VectorQuantize` backend |
| `kmeans_init` | `VQTokenizerConfig`, `VectorQuantize` backend |
| `kmeans_iters` | `VQTokenizerConfig`, `VectorQuantize` backend |
| `encoder_hidden_dim` | `VQTokenizerConfig`, causal encoder stack |
| `decoder_hidden_dim` | `VQTokenizerConfig`, causal decoder stack |
| `num_layers` | `VQTokenizerConfig.layer_dilations`, encoder/decoder stack |
| `dilations` | `VQTokenizerConfig.layer_dilations`, encoder/decoder stack |

No new tokenizer config fields were added.

## Config List

All variants preserve S&P500/VIX data settings, `condition_dim: 1`, causal encoder/decoder
operation, `kmeans_init: true`, `kmeans_iters: 10`, `embedding_dim: 64`, and the baseline
training settings unless noted.

| Config | Intended change |
| --- | --- |
| `configs/experiments/sp500_vix_causal_vq_tokenizer_cb32_dim16.yaml` | `codebook_size: 32` |
| `configs/experiments/sp500_vix_causal_vq_tokenizer_cb64_dim8.yaml` | `codebook_dim: 8` |
| `configs/experiments/sp500_vix_causal_vq_tokenizer_cb64_dim32.yaml` | `codebook_dim: 32` |
| `configs/experiments/sp500_vix_causal_vq_tokenizer_cb128_dim16.yaml` | `codebook_size: 128` |
| `configs/experiments/sp500_vix_causal_vq_tokenizer_commitment005.yaml` | `commitment_weight: 0.05` |
| `configs/experiments/sp500_vix_causal_vq_tokenizer_commitment025.yaml` | `commitment_weight: 0.25` |
| `configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml` | `encoder_hidden_dim: 128`, `decoder_hidden_dim: 128` |
| `configs/experiments/sp500_vix_causal_vq_tokenizer_dilations124816.yaml` | `num_layers: 5`, `dilations: [1, 2, 4, 8, 16]` |

The dilation variant changes both `num_layers` and `dilations` because the tokenizer uses
`VQTokenizerConfig.layer_dilations`, which truncates the dilation schedule to `num_layers`.

## Runner

The runner is `scripts/run_sp500_vix_tokenizer_ablation.py`. It accepts multiple configs and
writes aggregate summaries under the requested `outputs/` directory:

- `tokenizer_ablation_summary.json`;
- `tokenizer_ablation_summary.csv`.

The aggregate schema includes:

- reconstruction L1 and L2;
- terminal-return error;
- volatility reconstruction error;
- active code count and active code ratio;
- codebook perplexity and entropy;
- VIX-bucket code usage fields when evaluation has been run.

During `--dry-run`, evaluation is intentionally skipped, so these metric fields are present with
null values.

## Dry-Run Results

Command run:

```bash
MPLBACKEND=Agg poetry run python scripts/run_sp500_vix_tokenizer_ablation.py \
  --configs \
    configs/experiments/sp500_vix_causal_vq_tokenizer_cb32_dim16.yaml \
    configs/experiments/sp500_vix_causal_vq_tokenizer_cb64_dim8.yaml \
    configs/experiments/sp500_vix_causal_vq_tokenizer_cb64_dim32.yaml \
    configs/experiments/sp500_vix_causal_vq_tokenizer_cb128_dim16.yaml \
    configs/experiments/sp500_vix_causal_vq_tokenizer_commitment005.yaml \
    configs/experiments/sp500_vix_causal_vq_tokenizer_commitment025.yaml \
    configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml \
    configs/experiments/sp500_vix_causal_vq_tokenizer_dilations124816.yaml \
  --output-dir outputs/sp500_vix_discrete/vq_tokenizer_ablation_dry \
  --base-data-dir data/processed \
  --epochs 1 \
  --n-sample-test 128 \
  --seed 99 \
  --dry-run \
  --no-wandb
```

Result:

- all eight configs loaded successfully;
- each dry-run built the S&P500/VIX train and eval datasets with 2457 samples and shape
  `[2457, 60, 1]`;
- each dry-run built a CPU tokenizer and printed its config summary;
- no training was started;
- evaluation was skipped as intended;
- aggregate JSON and CSV were written under
  `outputs/sp500_vix_discrete/vq_tokenizer_ablation_dry/`.

Observed environment notes:

- Matplotlib used temporary cache directories under `/tmp` because
  `/home/georgios-vourvachakis/.config/matplotlib` was not writable in the sandbox;
- PyTorch selected CPU because local CUDA initialisation warned that the NVIDIA driver was too
  old for the installed PyTorch build.

Neither warning blocked the setup dry-run.

## W&B Execution Profile

Every non-smoke run must use:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread
```

The runner also applies the same environment to its tokenizer train/evaluation subprocesses. Its
default W&B destination is:

- project: `time-causal-vq-tokenizer`;
- entity: `tc_vae`.

If W&B fails, rerun the same command with `--no-wandb` and document the rerun in the relevant
verification note.

## Next Non-Smoke Command

The next controlled non-smoke run should remove `--dry-run`, keep the same config list, and enable
W&B with the required environment profile:

```bash
env -u WANDB_MODE MPLBACKEND=Agg WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread \
poetry run python scripts/run_sp500_vix_tokenizer_ablation.py \
  --configs \
    configs/experiments/sp500_vix_causal_vq_tokenizer_cb32_dim16.yaml \
    configs/experiments/sp500_vix_causal_vq_tokenizer_cb64_dim8.yaml \
    configs/experiments/sp500_vix_causal_vq_tokenizer_cb64_dim32.yaml \
    configs/experiments/sp500_vix_causal_vq_tokenizer_cb128_dim16.yaml \
    configs/experiments/sp500_vix_causal_vq_tokenizer_commitment005.yaml \
    configs/experiments/sp500_vix_causal_vq_tokenizer_commitment025.yaml \
    configs/experiments/sp500_vix_causal_vq_tokenizer_hidden128.yaml \
    configs/experiments/sp500_vix_causal_vq_tokenizer_dilations124816.yaml \
  --output-dir outputs/sp500_vix_discrete/vq_tokenizer_ablation \
  --base-data-dir data/processed \
  --epochs 100 \
  --n-sample-test 512 \
  --seed 99 \
  --wandb \
  --wandb-project time-causal-vq-tokenizer \
  --wandb-entity tc_vae
```

If the W&B service fails in the local environment, rerun with `--no-wandb` and preserve the
aggregate JSON/CSV as the reproducibility record.
