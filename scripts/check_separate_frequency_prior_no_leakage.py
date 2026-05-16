"""Check separate low/high hierarchical token-prior prefix causality."""

from __future__ import annotations

import argparse
from typing import cast

import torch
from torch import Tensor

from time_causal_vae.token_prior import (
    CausalTokenPriorConfig,
    SeparateFrequencyHierarchicalPrior,
    assert_separate_frequency_prior_no_future_leakage,
)
from time_causal_vae.utils.random import set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the no-leakage check parser."""
    parser = argparse.ArgumentParser(
        description="Check separate frequency hierarchical prior no-future-leakage.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for the check.")
    parser.add_argument("--length", type=int, default=60, help="Token sequence length.")
    parser.add_argument("--cutoff", type=int, default=29, help="Inclusive output-logit cutoff.")
    parser.add_argument("--seed", type=int, default=99, help="Random seed.")
    parser.add_argument("--device", default="cpu", help="Device for the check.")
    return parser


def main() -> int:
    """Run a synthetic separate-frequency prior no-leakage check."""
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.length <= 0:
        raise SystemExit("--length must be positive.")
    if not 0 <= args.cutoff < args.length:
        raise SystemExit("--cutoff must satisfy 0 <= cutoff < length.")

    set_seed(args.seed)
    device = torch.device(args.device)
    config = CausalTokenPriorConfig(
        codebook_size=64,
        low_codebook_size=64,
        high_codebook_size=64,
        sequence_length=args.length,
        token_embedding_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_hidden_dim=128,
        dropout=0.0,
        condition_dim=1,
        condition_injection="additive",
        prior_type="separate_frequency_hierarchical",
        index_shape=[args.length, 2],
    )
    model = SeparateFrequencyHierarchicalPrior(config).to(device)
    tokens = torch.randint(
        low=0,
        high=config.codebook_size,
        size=(args.batch_size, args.length, 2),
        dtype=torch.long,
        device=device,
    )
    conditions = torch.randn(args.batch_size, config.condition_dim, device=device)

    try:
        output = model(tokens, conditions=conditions)
        reference, changed = assert_separate_frequency_prior_no_future_leakage(
            model,
            tokens,
            args.cutoff,
            conditions=conditions,
        )
        edge = same_time_low_to_high_edge(
            model=model,
            tokens=tokens,
            conditions=conditions,
            time_index=args.cutoff,
        )
        samples = model.sample_streams(
            batch_size=args.batch_size,
            device=device,
            temperature=1.0,
            top_k=8,
            conditions=conditions,
        )
    except Exception as exc:
        print(f"FAIL separate frequency prior no-leakage check: {exc}")
        return 1

    low_prefix_diff = prefix_max_abs_diff(
        cast(Tensor, reference["low_logits"]),
        cast(Tensor, changed["low_logits"]),
        cutoff=args.cutoff,
    )
    high_prefix_diff = prefix_max_abs_diff(
        cast(Tensor, reference["high_logits"]),
        cast(Tensor, changed["high_logits"]),
        cutoff=args.cutoff,
    )
    print("PASS separate frequency prior no-leakage check")
    print(f"tokens={tuple(tokens.shape)}")
    print(f"conditions={tuple(conditions.shape)}")
    print(f"low_logits={tuple(cast(Tensor, output.low_logits).shape)}")
    print(f"high_logits={tuple(cast(Tensor, output.high_logits).shape)}")
    print(f"sampled_low_tokens={tuple(cast(Tensor, samples.sampled_low_tokens).shape)}")
    print(f"sampled_high_tokens={tuple(cast(Tensor, samples.sampled_high_tokens).shape)}")
    print(f"cutoff={args.cutoff}")
    print(f"cross_entropy={output.cross_entropy.item():.8f}")
    print(f"low_ce={output.component_cross_entropy_low.item():.8f}")
    print(f"high_ce={output.component_cross_entropy_high.item():.8f}")
    print(f"low_accuracy={output.component_accuracy_low.item():.8f}")
    print(f"high_accuracy={output.component_accuracy_high.item():.8f}")
    print(f"low_perplexity={output.component_perplexity_low.item():.8f}")
    print(f"high_perplexity={output.component_perplexity_high.item():.8f}")
    print(f"same_time_pair_perplexity={output.same_time_pair_perplexity.item():.8f}")
    print(f"max_low_prefix_diff_after_future_perturb={low_prefix_diff:.8e}")
    print(f"max_high_prefix_diff_after_future_perturb={high_prefix_diff:.8e}")
    print(
        "current_low_edge: "
        f"low_logit_diff_at_t={edge['low_logit_diff_at_t']:.8e} "
        f"high_logit_diff_at_t={edge['high_logit_diff_at_t']:.8e} "
        "allowed=True"
    )
    return 0


def prefix_max_abs_diff(reference: Tensor, changed: Tensor, *, cutoff: int) -> float:
    """Return max absolute prefix-logit difference through an inclusive cutoff."""
    return float((reference[:, : cutoff + 1] - changed[:, : cutoff + 1]).abs().max().item())


def same_time_low_to_high_edge(
    *,
    model: SeparateFrequencyHierarchicalPrior,
    tokens: Tensor,
    conditions: Tensor,
    time_index: int,
) -> dict[str, float]:
    """Measure the allowed same-time edge from current low token to high logits."""
    changed = tokens.clone()
    changed[:, time_index, 0] = (changed[:, time_index, 0] + 1) % model.low_codebook_size
    model.eval()
    with torch.no_grad():
        reference = model(tokens, conditions=conditions)
        changed_output = model(changed, conditions=conditions)
    low_reference = cast(Tensor, reference["low_logits"])[:, time_index]
    low_changed = cast(Tensor, changed_output["low_logits"])[:, time_index]
    high_reference = cast(Tensor, reference["high_logits"])[:, time_index]
    high_changed = cast(Tensor, changed_output["high_logits"])[:, time_index]
    high_diff = float((high_reference - high_changed).abs().max().item())
    if high_diff <= 0.0:
        raise AssertionError("Expected high logits at time t to change when current low_t changes.")
    return {
        "low_logit_diff_at_t": float((low_reference - low_changed).abs().max().item()),
        "high_logit_diff_at_t": high_diff,
    }


if __name__ == "__main__":
    raise SystemExit(main())
