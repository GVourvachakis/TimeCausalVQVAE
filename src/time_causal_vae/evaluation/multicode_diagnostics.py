"""Diagnostics for multi-code discrete token tensors."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from time_causal_vae.evaluation.tokenizer import condition_bucket_label, summarise_code_usage


def compare_rvq_q2_pairs(
    *,
    real_tokens: Tensor,
    sampled_tokens: Tensor,
    codebook_size: int,
    real_conditions: Tensor | None = None,
    sampled_conditions: Tensor | None = None,
    n_buckets: int = 5,
    top_k: int = 20,
) -> dict[str, Any]:
    """Compare same-time ``(q0, q1)`` pair usage for RVQ q2 token tensors."""
    validate_rvq_q2_tokens(real_tokens, name="real_tokens")
    validate_rvq_q2_tokens(sampled_tokens, name="sampled_tokens")
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    real_cpu = real_tokens.detach().cpu().long()
    sampled_cpu = sampled_tokens.detach().cpu().long()
    real_q0_counts = component_counts(real_cpu, codebook_size, component=0)
    real_q1_counts = component_counts(real_cpu, codebook_size, component=1)
    sampled_q0_counts = component_counts(sampled_cpu, codebook_size, component=0)
    sampled_q1_counts = component_counts(sampled_cpu, codebook_size, component=1)
    real_pair_counts = pair_count_matrix(real_cpu, codebook_size)
    sampled_pair_counts = pair_count_matrix(sampled_cpu, codebook_size)

    result: dict[str, Any] = {
        "shapes": {
            "real_tokens": list(real_cpu.shape),
            "sampled_tokens": list(sampled_cpu.shape),
        },
        "q0": component_comparison(real_q0_counts, sampled_q0_counts),
        "q1": component_comparison(real_q1_counts, sampled_q1_counts),
        "pairs": pair_comparison(
            real_pair_counts,
            sampled_pair_counts,
            top_k=top_k,
        ),
    }
    if real_conditions is not None and sampled_conditions is not None:
        result["condition_buckets"] = condition_bucket_pair_comparisons(
            real_tokens=real_cpu,
            sampled_tokens=sampled_cpu,
            real_conditions=real_conditions,
            sampled_conditions=sampled_conditions,
            codebook_size=codebook_size,
            n_buckets=n_buckets,
            top_k=top_k,
        )
    return result


def validate_rvq_q2_tokens(tokens: Tensor, *, name: str) -> None:
    """Validate RVQ q2 token tensors."""
    if tokens.ndim != 3 or tokens.shape[2] != 2:
        raise ValueError(f"{name} must have shape [batch, length, 2]; got {tuple(tokens.shape)}.")
    if tokens.dtype != torch.long:
        raise ValueError(f"{name} must have dtype torch.long; got {tokens.dtype}.")


def component_counts(tokens: Tensor, codebook_size: int, *, component: int) -> Tensor:
    """Return code counts for one RVQ component."""
    return torch.bincount(
        tokens[:, :, component].reshape(-1),
        minlength=codebook_size,
    )[:codebook_size].float()


def pair_count_matrix(tokens: Tensor, codebook_size: int) -> Tensor:
    """Return same-time q0/q1 pair counts with shape ``[codebook_size, codebook_size]``."""
    q0 = tokens[:, :, 0].reshape(-1)
    q1 = tokens[:, :, 1].reshape(-1)
    flat_pairs = q0 * codebook_size + q1
    counts = torch.bincount(flat_pairs, minlength=codebook_size * codebook_size).float()
    return counts.reshape(codebook_size, codebook_size)


def probability(counts: Tensor) -> Tensor:
    """Normalise counts into a probability tensor."""
    total = counts.sum()
    if float(total.item()) <= 0.0:
        return torch.zeros_like(counts, dtype=torch.float32)
    return counts.float() / total


def component_comparison(real_counts: Tensor, sampled_counts: Tensor) -> dict[str, Any]:
    """Return marginal component usage diagnostics."""
    real_prob = probability(real_counts)
    sampled_prob = probability(sampled_counts)
    return {
        "real": summarise_code_usage(real_counts),
        "sampled": summarise_code_usage(sampled_counts),
        "distribution_l1": float((real_prob - sampled_prob).abs().sum().item()),
        "top_overrepresented_codes": top_component_differences(
            real_prob,
            sampled_prob,
            direction="sampled_minus_real",
        ),
        "top_underrepresented_codes": top_component_differences(
            real_prob,
            sampled_prob,
            direction="real_minus_sampled",
        ),
    }


def pair_comparison(
    real_counts: Tensor,
    sampled_counts: Tensor,
    *,
    top_k: int,
) -> dict[str, Any]:
    """Return same-time pair diagnostics."""
    real_prob = probability(real_counts)
    sampled_prob = probability(sampled_counts)
    absent_mask = real_counts == 0
    sampled_absent_counts = torch.where(
        absent_mask, sampled_counts, torch.zeros_like(sampled_counts)
    )
    sampled_absent_mass = torch.where(absent_mask, sampled_prob, torch.zeros_like(sampled_prob))
    return {
        "real_active_pair_count": int((real_counts > 0).sum().item()),
        "sampled_active_pair_count": int((sampled_counts > 0).sum().item()),
        "sampled_absent_pair_count": int((sampled_absent_counts > 0).sum().item()),
        "sampled_absent_pair_mass": float(sampled_absent_mass.sum().item()),
        "real_pair_entropy": entropy(real_prob),
        "sampled_pair_entropy": entropy(sampled_prob),
        "real_pair_perplexity": perplexity(real_prob),
        "sampled_pair_perplexity": perplexity(sampled_prob),
        "pair_distribution_l1": float((real_prob - sampled_prob).abs().sum().item()),
        "top_sampled_absent_pairs": top_pairs(
            sampled_absent_mass,
            top_k=top_k,
            include_zero=False,
        ),
        "top_overrepresented_pairs": top_pairs(
            sampled_prob - real_prob,
            top_k=top_k,
            include_zero=False,
        ),
        "top_underrepresented_pairs": top_pairs(
            real_prob - sampled_prob,
            top_k=top_k,
            include_zero=False,
        ),
        "top_real_pairs": top_pairs(real_prob, top_k=top_k, include_zero=False),
        "top_sampled_pairs": top_pairs(sampled_prob, top_k=top_k, include_zero=False),
    }


def entropy(probabilities: Tensor) -> float:
    """Return entropy of a probability tensor."""
    positive = probabilities[probabilities > 0.0]
    if positive.numel() == 0:
        return 0.0
    return float((-(positive * positive.log())).sum().item())


def perplexity(probabilities: Tensor) -> float:
    """Return perplexity implied by a probability tensor."""
    return float(torch.exp(torch.tensor(entropy(probabilities))).item())


def top_component_differences(
    real_prob: Tensor,
    sampled_prob: Tensor,
    *,
    direction: str,
    top_k: int = 20,
) -> list[dict[str, float | int]]:
    """Return top component-level probability differences."""
    if direction == "sampled_minus_real":
        diff = sampled_prob - real_prob
    elif direction == "real_minus_sampled":
        diff = real_prob - sampled_prob
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    values, indices = torch.topk(diff, k=min(top_k, diff.numel()))
    rows: list[dict[str, float | int]] = []
    for value, index in zip(values.tolist(), indices.tolist(), strict=True):
        if value <= 0.0:
            continue
        rows.append({
            "code": int(index),
            "difference": float(value),
            "real_probability": float(real_prob[index].item()),
            "sampled_probability": float(sampled_prob[index].item()),
        })
    return rows


def top_pairs(
    scores: Tensor,
    *,
    top_k: int,
    include_zero: bool,
) -> list[dict[str, float | int]]:
    """Return top q0/q1 pairs from a score matrix."""
    flat_scores = scores.reshape(-1)
    values, indices = torch.topk(flat_scores, k=min(top_k, flat_scores.numel()))
    codebook_size = scores.shape[0]
    rows: list[dict[str, float | int]] = []
    for value, flat_index in zip(values.tolist(), indices.tolist(), strict=True):
        if not include_zero and value <= 0.0:
            continue
        q0 = int(flat_index // codebook_size)
        q1 = int(flat_index % codebook_size)
        rows.append({"q0": q0, "q1": q1, "score": float(value)})
    return rows


def condition_bucket_pair_comparisons(
    *,
    real_tokens: Tensor,
    sampled_tokens: Tensor,
    real_conditions: Tensor,
    sampled_conditions: Tensor,
    codebook_size: int,
    n_buckets: int,
    top_k: int,
) -> list[dict[str, Any]]:
    """Return pair diagnostics by condition quantile."""
    real_values = condition_values(real_conditions)
    sampled_values = condition_values(sampled_conditions)
    if real_values.shape[0] != real_tokens.shape[0]:
        raise ValueError("real_conditions must share batch size with real_tokens.")
    if sampled_values.shape[0] != sampled_tokens.shape[0]:
        raise ValueError("sampled_conditions must share batch size with sampled_tokens.")
    bucket_count = min(n_buckets, real_values.shape[0], sampled_values.shape[0])
    real_sorted = torch.argsort(real_values)
    sampled_sorted = torch.argsort(sampled_values)
    buckets: list[dict[str, Any]] = []
    for bucket_index, (real_positions, sampled_positions) in enumerate(
        zip(
            torch.tensor_split(real_sorted, bucket_count),
            torch.tensor_split(sampled_sorted, bucket_count),
            strict=True,
        )
    ):
        if real_positions.numel() == 0 or sampled_positions.numel() == 0:
            continue
        real_bucket_tokens = real_tokens.index_select(0, real_positions)
        sampled_bucket_tokens = sampled_tokens.index_select(0, sampled_positions)
        real_pair_counts = pair_count_matrix(real_bucket_tokens, codebook_size)
        sampled_pair_counts = pair_count_matrix(sampled_bucket_tokens, codebook_size)
        comparison = pair_comparison(real_pair_counts, sampled_pair_counts, top_k=top_k)
        bucket_real_values = real_values.index_select(0, real_positions)
        bucket_sampled_values = sampled_values.index_select(0, sampled_positions)
        buckets.append({
            "bucket_index": bucket_index,
            "bucket_label": condition_bucket_label(bucket_index, bucket_count),
            "real_n": int(real_positions.numel()),
            "sampled_n": int(sampled_positions.numel()),
            "real_condition_min": float(bucket_real_values.min().item()),
            "real_condition_max": float(bucket_real_values.max().item()),
            "sampled_condition_min": float(bucket_sampled_values.min().item()),
            "sampled_condition_max": float(bucket_sampled_values.max().item()),
            **comparison,
        })
    return buckets


def condition_values(conditions: Tensor) -> Tensor:
    """Collapse scalar or temporal conditions to one value per path."""
    cpu_conditions = conditions.detach().cpu().float()
    if cpu_conditions.ndim == 1:
        return cpu_conditions
    if cpu_conditions.ndim == 2:
        return cpu_conditions.reshape(cpu_conditions.shape[0], -1).mean(dim=1)
    if cpu_conditions.ndim == 3:
        return cpu_conditions.mean(dim=(1, 2))
    raise ValueError(f"Unsupported condition shape: {tuple(cpu_conditions.shape)}.")
