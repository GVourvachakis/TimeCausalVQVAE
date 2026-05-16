# Separate Frequency Hierarchical Prior Source Smoke

Status: source-level implementation smoke for the separate low/high frequency-token prior. This
stage does not train a non-smoke model and does not add GroupedRVQ, MGVQ, signatures, diffusion,
cross-attention, or new objectives.

## Source Changes

The implementation added or updated:

```text
src/time_causal_vae/token_prior/config.py
src/time_causal_vae/token_prior/causal_transformer.py
src/time_causal_vae/token_prior/__init__.py
src/time_causal_vae/cli/train_token_prior.py
src/time_causal_vae/evaluation/token_prior.py
scripts/check_separate_frequency_prior_no_leakage.py
configs/experiments/sp500_vix_separate_frequency_hierarchical_prior_alpha02.yaml
```

The existing `tcvae-train-token-prior` entry point supports the new paired-token dataset cleanly,
so no dedicated CLI was added.

## Factorisation

The new prior type is:

```yaml
model:
  prior_type: separate_frequency_hierarchical
```

It implements:

```text
p(low_t | low_<t, high_<t, condition)
p(high_t | low_<=t, high_<t, condition)
```

The shared causal transformer trunk receives the shifted previous-time block
`[low_{t-1}, high_{t-1}]`. The low head reads the trunk hidden state `h_t`. The high head reads
`h_t + embedding(low_t)`, where `low_t` is teacher-forced during training and sampled during
generation.

The loss is the specified objective:

```text
loss = CE_low + CE_high
```

The model reports stream-specific cross-entropy, accuracy, and perplexity, plus aggregate
cross-entropy, accuracy, and perplexity.

## Token Shapes

The paired S&P500/VIX alpha 0.2 dataset path is:

```text
outputs/sp500_vix_discrete/token_prior/separate_freq_alpha02_tokens
```

The training CLI loads:

```text
train_low_tokens.pt   -> [2457, 60]
train_high_tokens.pt  -> [2457, 60]
eval_low_tokens.pt    -> [2457, 60]
eval_high_tokens.pt   -> [2457, 60]
train_labels.pt       -> [2457, 1]
eval_labels.pt        -> [2457, 1]
```

It packs the model input as:

```text
tokens:     [batch, 60, 2]
conditions: [batch, 1]
```

Sampling exposes the stream tensors:

```text
sampled_low_tokens:  [batch, 60]
sampled_high_tokens: [batch, 60]
```

and also returns a packed tensor for compatibility with existing token-prior utilities.

## No-Leakage Check

Command:

```text
poetry run python scripts/check_separate_frequency_prior_no_leakage.py \
  --batch-size 8 \
  --length 60 \
  --cutoff 29 \
  --seed 99
```

Result:

```text
PASS separate frequency prior no-leakage check
tokens=(8, 60, 2)
conditions=(8, 1)
low_logits=(8, 60, 64)
high_logits=(8, 60, 64)
sampled_low_tokens=(8, 60)
sampled_high_tokens=(8, 60)
cutoff=29
cross_entropy=9.77163124
low_ce=4.83861876
high_ce=4.93301201
low_accuracy=0.02500000
high_accuracy=0.01041667
low_perplexity=126.29478455
high_perplexity=138.79693604
same_time_pair_perplexity=439.68270874
max_low_prefix_diff_after_future_perturb=0.00000000e+00
max_high_prefix_diff_after_future_perturb=0.00000000e+00
current_low_edge: low_logit_diff_at_t=0.00000000e+00 high_logit_diff_at_t=2.83020282e+00 allowed=True
```

The source check perturbs only future low/high tokens after the inclusive cutoff and verifies that
low and high logits through the cutoff are unchanged. It also changes the current low token at
time `t`; the high logits at `t` are allowed to change because `low_t -> high_t` is the intended
same-time hierarchical edge. The condition tensor is held fixed throughout the token-perturbation
check.

## Smoke Training

Command:

```text
poetry run tcvae-train-token-prior \
  --config configs/experiments/sp500_vix_separate_frequency_hierarchical_prior_alpha02.yaml \
  --output-dir outputs/sp500_vix_discrete/token_prior/separate_freq_alpha02_smoke \
  --epochs 1 \
  --no-wandb
```

Dry-run loader check before training:

```text
train_tokens_shape: (2457, 60, 2)
train_conditions_shape: (2457, 1)
eval_tokens_shape: (2457, 60, 2)
eval_conditions_shape: (2457, 1)
parameters: 579200
device: cpu
```

One-epoch smoke output:

```text
epoch=1 train_ce=9.16369724 train_acc=0.04089337 eval_ce=8.30710157 eval_acc=0.07827296
training_complete: outputs/sp500_vix_discrete/token_prior/separate_freq_alpha02_smoke/sp500_vix_separate_frequency_hierarchical_prior_alpha02_seed0
runtime_seconds: 13.312
final_eval_cross_entropy: 8.30710157
final_eval_accuracy: 0.07827296
final_eval_perplexity: 4172.18701102
best_epoch: 1
best_eval_cross_entropy: 8.30710157
```

Detailed final metrics:

| Split | CE | Accuracy | Perplexity | Low CE | Low acc. | Low perplexity | High CE | High acc. | High perplexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 9.16369724 | 0.04089337 | 10820.92429673 | 4.32321638 | 0.06150455 | 79.33853311 | 4.84048092 | 0.02028219 | 128.66568540 |
| Eval | 8.30710157 | 0.07827296 | 4172.18701102 | 3.74645961 | 0.12739113 | 42.87065852 | 4.56064207 | 0.02915480 | 96.13520037 |

These metrics are smoke-only diagnostics from one epoch. They establish that the source path,
paired-token loader, loss decomposition, checkpoint writing, and best-model selection run end to
end. They are not a prior-quality result.

## Backward Compatibility

Default behaviour remains the original single-token prior:

```text
prior_type: single_code
```

The existing single-token loader still reads `train_tokens.pt` and `eval_tokens.pt` payloads with
`indices` and optional `labels`. The new paired-token loader is activated only when
`prior_type: separate_frequency_hierarchical`. Existing `factorised_multi_code` and
`hierarchical_rvq_q2` configuration paths are left intact.

## Caveats

The smoke ran on CPU because the local CUDA driver was not usable by PyTorch. The commands also
emitted a local Matplotlib cache warning. Neither warning blocked the no-leakage check or
one-epoch smoke.
