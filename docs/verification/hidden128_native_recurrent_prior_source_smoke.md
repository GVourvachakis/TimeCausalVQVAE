# Hidden128 Native Recurrent Prior Source Smoke

Status: source smoke completed. No dependency was added, no tokenizer code was changed, no public
default was changed, and no non-smoke model was trained.

## Source Files Changed

- `src/time_causal_vae/token_prior/config.py`
- `src/time_causal_vae/token_prior/causal_transformer.py`
- `src/time_causal_vae/token_prior/__init__.py`
- `src/time_causal_vae/cli/train_token_prior.py`
- `src/time_causal_vae/evaluation/token_prior.py`
- `scripts/check_native_recurrent_prior_no_leakage.py`
- `configs/experiments/sp500_vix_causal_token_prior_hidden128_native_recurrent.yaml`
- `docs/verification/hidden128_native_recurrent_prior_source_smoke.md`

## Prior Type

The new prior type is:

```yaml
model:
  prior_type: native_recurrent
```

The default `single_code` behaviour remains unchanged.

## Architecture

The implementation is a dependency-free native PyTorch recurrent prior for the hidden128
single-stream token interface. It uses:

- a tokenizer-code embedding with BOS support;
- the same learned position embedding convention used by the existing single-stream priors;
- additive scalar VIX condition projection;
- `torch.nn.GRU` with `recurrent_hidden_dim=128`, `recurrent_num_layers=1`, and
  `recurrent_dropout=0.0`;
- a linear projection from recurrent hidden states to 64 codebook logits.

The model follows the existing BOS-shifted target contract:

```text
input_t = BOS for t=0, else token_{t-1}
target_t = token_t
```

Stepwise sampling updates the GRU state once for each generated token.

## No-Leakage And Stepwise Check

Command:

```bash
poetry run python scripts/check_native_recurrent_prior_no_leakage.py \
  --config configs/experiments/sp500_vix_causal_token_prior_hidden128_native_recurrent.yaml \
  --batch-size 8 \
  --cutoff 29 \
  --seed 99
```

Result:

```text
PASS native recurrent prior no-leakage and stepwise-equivalence check
tokens=(8, 60)
conditions=(8, 1)
logits=(8, 60, 64)
samples=(8, 60)
cutoff=29
cross_entropy=4.17962313
accuracy=0.02708333
perplexity=65.34122467
max_prefix_diff=0.00000000e+00
max_stepwise_diff=0.00000000e+00
```

The no-future-leakage check perturbed tokens after the inclusive cutoff and verified unchanged
prefix logits through the cutoff. The teacher-forced stepwise logits matched the full-sequence GRU
logits exactly under deterministic evaluation mode on this run.

## Smoke Training

Command:

```bash
poetry run tcvae-train-token-prior \
  --config configs/experiments/sp500_vix_causal_token_prior_hidden128_native_recurrent.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/hidden128_native_recurrent_smoke \
  --epochs 1 \
  --no-wandb
```

Result:

```text
epoch=1 train_ce=4.09711490 train_acc=0.03827160 eval_ce=3.97471046 eval_acc=0.08564645
runtime_seconds: 2.241
final_eval_cross_entropy: 3.97471046
final_eval_accuracy: 0.08564645
final_eval_perplexity: 53.28947747
best_epoch: 1
```

The run wrote the smoke artefacts under:

```text
outputs/sp500_vix_discrete/token_prior/hidden128_native_recurrent_smoke/sp500_vix_causal_token_prior_hidden128_native_recurrent_seed99
```

## Backward Compatibility

Existing `single_code`, `causal_conv_transformer`, `factorised_multi_code`, and
`hierarchical_rvq_q2` build paths remain available. The new recurrent fields have defaults and are
only used by `native_recurrent`. Existing hidden128 additive and conv-transformer configs are not
modified. Tokenizer code and tokenizer configs are untouched.

No new dependency was added to `pyproject.toml`; the implementation uses the repository's existing
PyTorch dependency only.
