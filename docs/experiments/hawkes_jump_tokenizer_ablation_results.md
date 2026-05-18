# Hawkes-Jump Tokenizer Ablation Results

## Status

The non-smoke Hawkes-jump tokenizer utilisation ablations were run with the Ogata
simulator, 1024 train paths, 1024 eval paths, 60 timesteps, seed `0`, W&B disabled,
and `50` tokenizer epochs. No token priors were trained, no model code was changed,
and the registry was not updated.

An initial run exposed a post-hoc oracle metadata lookup issue in the ablation
runner after the first tokenizer had trained and evaluated. The runner was corrected
to rebuild `HawkesJumpDataset` directly for oracle-only diagnostics, matching the
pipeline train/eval seed convention. The table below reports the completed rerun.

The successful run wrote aggregate outputs under
`outputs/hawkes_jump_tokenizer_ablation`. The summed subprocess runtime recorded in
`aggregate_summary.json` was `223.7` seconds, about `3.7` minutes on CPU.

## Utilisation And Reconstruction

| Config | Data | Active codes | Perplexity | Recon L1 | Recon L2 | Vol recon |
|---|---:|---:|---:|---:|---:|---:|
| `logreturn_cb32` | log return | `32/32` | `25.11` | `0.001589` | `0.002563` | `0.001194` |
| `logreturn_cb64` | log return | `64/64` | `47.81` | `0.001897` | `0.002661` | `0.000891` |
| `hidden128_logreturn_cb32` | log return | `31/32` | `25.78` | `0.002033` | `0.002937` | `0.000950` |
| `hidden128_logreturn_cb64` | log return | `63/64` | `44.82` | `0.001375` | `0.002437` | `0.000928` |
| `price_cb32` | price | `32/32` | `27.27` | `0.010798` | `0.014010` | `0.001921` |
| `hidden128_price_cb32` | price | `32/32` | `28.40` | `0.007747` | `0.010654` | `0.001558` |

The active-code collapse from the first comparison is resolved for these tokenizer
settings. The previous diagnostic comparison had only `6/64` active codes for the
standard tokenizer and `4/64` active codes for the hidden128 tokenizer. In this
ablation, all variants exceed the minimum success threshold of 10 active codes, and
the log-return variants use either all codes or all but one code.

The log-return configs are the main improvement. They make the jump-relevant return
increments directly visible to the tokenizer and produce high code usage without
requiring a prior-family change. The price configs also avoid collapse, but their
reconstruction errors are naturally on the price scale and their jump alignment is
less compelling.

## Jump-Code Alignment

Oracle jump labels were used only after token extraction. They were not model-visible
during tokenizer training.

| Config | Jump/non-jump code L1 | Rare-code lift near jumps | Token-change jump | Token-change non-jump | Change lift |
|---|---:|---:|---:|---:|---:|
| `logreturn_cb32` | `0.321` | `4.30` | `0.956` | `0.950` | `1.006` |
| `logreturn_cb64` | `0.335` | `1.37` | `0.969` | `0.968` | `1.001` |
| `hidden128_logreturn_cb32` | `0.324` | `8.32` | `0.957` | `0.953` | `1.004` |
| `hidden128_logreturn_cb64` | `0.327` | `1.14` | `0.968` | `0.967` | `1.001` |
| `price_cb32` | `0.258` | `1.20` | `0.657` | `0.600` | `1.094` |
| `hidden128_price_cb32` | `0.303` | `2.22` | `0.718` | `0.689` | `1.042` |

The strongest code-distribution separation is `logreturn_cb64` with L1 distance
`0.335`, followed closely by `hidden128_logreturn_cb64` and the cb32 log-return
variants. The strongest rare-code jump enrichment is `hidden128_logreturn_cb32`
with an `8.32x` rare-code activation lift. The log-return tokenizers change token
almost every timestep in both jump and non-jump windows, so token-change lift is not
the main discriminant for those configs. The price tokenizers show larger relative
token-change lift near jumps, but weaker reconstruction and lower code-distribution
separation.

## Candidate

The primary tokenizer candidate for prior training is
`hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb64`.

It gives the best reconstruction L1 among the log-return runs, uses `63/64` codes,
has high perplexity, and retains meaningful jump/non-jump code-distribution
separation. It is also the closest match to the intended hidden128 discrete path for
the Hawkes benchmark.

The secondary diagnostic candidate is
`hawkes_jump_causal_vq_tokenizer_hidden128_logreturn_cb32`. It has slightly weaker
reconstruction, but the strongest rare-code enrichment near jumps. It is useful if
the 64-code prior remains too diffuse or if rare-code dynamics matter more than raw
reconstruction.

## Decision

Proceed to token-prior training, using the hidden128 log-return tokenizer as the
default representation. The first prior pass should train no more than the already
planned additive AR and causal conv-transformer k3 priors; this ablation does not
justify changing prior families.

Keep the Hawkes branch active. There is no evidence here to stop the branch, and no
need to return to price-level tokenisation as the default. Additional tokenizer
tuning can wait until the prior-training comparison shows whether high utilisation
translates into better jump-regime generation.

Recommended next prior candidates:

- primary: `hidden128_logreturn_cb64`;
- secondary: `hidden128_logreturn_cb32`;
- optional baseline: `logreturn_cb64`.

The next evaluation should check whether the improved tokenizer usage produces
better detected jump counts, inter-arrival distributions, tail exceedance, VaR/ES,
run-length diagnostics, and transition diagnostics, not only lower reconstruction
error.
