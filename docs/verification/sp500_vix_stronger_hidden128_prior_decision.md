# Stronger Hidden128 Prior Decision

Status: decision note only. No code was implemented and no models were trained.

## Inputs

This decision uses the hidden128 prior roadmap, the conv-transformer source-quality run, the
conv-transformer sampling ablation, the hidden128 additive prior robustness report, and the
standard-VQ tuning final decision. The separate-frequency prior-quality note was not present in
this checkout; the failed separate-frequency metrics below are taken from the conv-transformer
quality note, which recorded the available separate-frequency evaluation artefacts.

Profile is:

```text
MMD + SWD + terminal_return_wasserstein + volatility_wasserstein
```

Lower is better.

## Baseline Hidden128 Additive Prior

The baseline hidden128 research prior is the unchanged additive VIX-only causal AR transformer:

- tokenizer: hidden128 single-stream standard VQ;
- prior config: `configs/experiments/sp500_vix_causal_token_prior_hidden128_candidate.yaml`;
- condition: scalar VIX, additive injection, `condition_dim=1`;
- selected sampling: temperature `0.8`, `top_k=20`;
- paper-style profile: `0.242020`;
- MMD/SWD: `0.229078` / `0.007175`;
- terminal W1 / volatility W1: `0.004509` / `0.001258`;
- maximum drawdown W1: `0.008910`;
- return AC L1 / squared-return AC L1: `0.038262` / `0.060885`;
- transition L1 / run-length distance: `0.223900` / `0.390197`;
- sampled active codes / token perplexity: `63/64` / `42.898106`.

The hidden128 additive prior is a strong research baseline because it improves the promoted public
standard-VQ baseline on the overall paper-style profile, MMD, SWD, terminal W1, maximum-drawdown
W1, and return autocorrelation. Its residual weaknesses are token likelihood, volatility W1, and
squared-return autocorrelation.

## Conv-Transformer Prior Result

The causal conv-transformer prior keeps the hidden128 tokenizer fixed and inserts a strictly
causal local-convolutional front-end before the transformer trunk:

- prior config:
  `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer.yaml`;
- checkpoint:
  `outputs/sp500_vix_discrete/token_prior/hidden128_conv_transformer/sp500_vix_causal_token_prior_hidden128_conv_transformer_seed99/best_model`;
- convolution: two causal layers, kernel size `3`, dilations `[1, 2]`;
- best checkpoint epoch: `100`;
- eval CE / accuracy / perplexity: `1.129030` / `0.557089` / `3.107617`.

At the inherited additive-prior sampling setting, temperature `0.8` and `top_k=20`, the
conv-transformer improved MMD but did not dominate the additive hidden128 prior:

| Prior | Profile | MMD | SWD | Terminal W1 | Vol W1 | Drawdown W1 | Sq-return AC L1 | Transition L1 | Run-length W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hidden128 additive, temp 0.8, top-k 20 | 0.242020 | 0.229078 | 0.007175 | 0.004509 | 0.001258 | 0.008910 | 0.060885 | 0.223900 | 0.390197 |
| conv-transformer, temp 0.8, top-k 20 | 0.217425 | 0.200300 | 0.007683 | 0.008033 | 0.001409 | 0.011835 | 0.063155 | 0.211512 | 0.468005 |

This first result was not promotion-complete: it improved profile and transition L1, but regressed
terminal W1, volatility W1, drawdown W1, squared-return autocorrelation, and run-length matching.

## Sampling And Capacity Ablation

The sampling ablation changed the conclusion. Raising temperature to `1.0` removed the
over-persistent sampled-token runs seen at lower temperatures.

| Setting | Profile | MMD | SWD | Terminal W1 | Vol W1 | Drawdown W1 | Sq-return AC L1 | Transition L1 | Run-length W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| conv-transformer, temp 1.0, top-k none | 0.186725 | 0.169510 | 0.010918 | 0.005086 | 0.001210 | 0.007687 | 0.050871 | 0.193243 | 0.050189 |
| conv-transformer, temp 1.0, top-k 20 | 0.205630 | 0.184628 | 0.011476 | 0.008584 | 0.000942 | 0.007858 | 0.050510 | 0.183255 | 0.097464 |
| hidden128 additive, temp 0.8, top-k 20 | 0.242020 | 0.229078 | 0.007175 | 0.004509 | 0.001258 | 0.008910 | 0.060885 | 0.223900 | 0.390197 |

The best profile setting is `temperature=1.0, top_k=none`. It improves MMD, volatility W1,
drawdown W1, squared-return autocorrelation, transition L1, and run-length W1 relative to the
hidden128 additive prior. It does not improve SWD or terminal W1. The secondary setting
`temperature=1.0, top_k=20` is useful when volatility W1, transition L1, or squared-return
autocorrelation is the main target, but its terminal W1 is weaker.

Two follow-up capacity configs exist for later controlled runs:

- `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer_k5.yaml`;
- `configs/experiments/sp500_vix_causal_token_prior_hidden128_conv_transformer_dilations124.yaml`.

No non-smoke capacity model has been trained for either config. They remain follow-up options, not
evidence in this decision.

## Cross-Model Comparison

| Model or prior | Profile | MMD | SWD | Terminal W1 | Vol W1 | Drawdown W1 | Sq-return AC L1 | Main interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| continuous BetaCVAE | 0.172891 | 0.154421 | 0.008785 | 0.009051 | 0.000634 | 0.007667 | 0.029462 | strongest continuous reference |
| conv-transformer, temp 1.0, top-k none | 0.186725 | 0.169510 | 0.010918 | 0.005086 | 0.001210 | 0.007687 | 0.050871 | best hidden128 prior profile |
| conv-transformer, temp 1.0, top-k 20 | 0.205630 | 0.184628 | 0.011476 | 0.008584 | 0.000942 | 0.007858 | 0.050510 | best conv calibration for volatility and transitions |
| hidden128 additive, temp 0.8, top-k 20 | 0.242020 | 0.229078 | 0.007175 | 0.004509 | 0.001258 | 0.008910 | 0.060885 | former best hidden128 prior |
| promoted standard-VQ baseline | 0.298020 | 0.279341 | 0.007674 | 0.009817 | 0.001188 | 0.010502 | 0.041300 | current public discrete baseline |
| joint EMA alpha02, temp 0.8, top-k 20 | 0.320110 | 0.301908 | 0.009780 | 0.007312 | 0.001110 | 0.005988 | 0.032284 | good drawdown and autocorrelation, weak profile |
| separate-frequency hierarchical prior | 3.242975 | 2.333169 | 0.145358 | 0.717744 | 0.046704 | 0.073700 | 0.049859 | failed sampled-token/path composition |

The conv-transformer is now the best hidden128 prior by the paper-style profile and by local token
persistence diagnostics. It does not replace the continuous BetaCVAE, which remains stronger on
MMD, volatility W1, drawdown W1, and autocorrelation. It also does not cleanly dominate every
discrete metric: the promoted baseline and joint EMA alpha02 remain stronger on squared-return
autocorrelation, and the additive hidden128 prior remains stronger on SWD and terminal W1.

## Decision

Promote the causal conv-transformer as the best hidden128 prior for research reporting, using
`temperature=1.0, top_k=none` as the primary paper-style setting and `temperature=1.0, top_k=20` as
a secondary calibration setting for volatility and transition-sensitive comparisons.

Do not replace the promoted public standard-VQ baseline. The public baseline remains the simpler
standard-VQ tokenizer with additive VIX-only AR prior because it is more established, has stronger
token likelihood, and still wins some stylised-fact diagnostics.

Keep the hidden128 additive prior as the ablation reference, not as the preferred hidden128 prior.
The additive prior remains important because it has better SWD and terminal W1, but the
conv-transformer materially improves the main profile, MMD, volatility W1, drawdown W1,
squared-return autocorrelation, transition L1, and run-length matching after sampling calibration.

Do not run a wider/deeper transformer next unless the conv-transformer k5 follow-up fails to
improve SWD and terminal W1. The next non-smoke prior capacity run, if one is run, should start
with the existing `k5` conv-transformer config because it is the smallest targeted change to local
receptive field.

Do not test Mamba or another selective-SSM package yet. Package compatibility, CUDA constraints,
licensing, checkpoint portability, and causal generation semantics should be inspected only after
the native PyTorch conv-transformer follow-ups are exhausted.

Do not return to tokenizer variants now. The hidden128 tokenizer uses all 64 codes, has strong
latent geometry, and does not show tokenizer collapse. The separate-frequency branch failed through
sampled-token and composed-path collapse, not because hidden128 lacked token capacity.

## Bottleneck Conclusion

The dominant bottleneck is token-prior calibration, not tokenizer limitation.

The evidence is that changing the prior architecture and sampling policy, while keeping the
hidden128 tokenizer fixed, moved the paper-style profile from `0.242020` for the additive prior to
`0.186725` for the conv-transformer. It also reduced run-length W1 from `0.390197` to `0.050189`
and transition L1 from `0.223900` to `0.193243`. Those gains are prior-side calibration gains.

The remaining weaknesses, especially SWD and terminal W1, are not enough to reopen tokenizer
variants before completing the local-convolution prior follow-ups. They should be treated as
residual prior calibration targets rather than evidence that the hidden128 tokenizer has collapsed
or reached a hard representation limit.
