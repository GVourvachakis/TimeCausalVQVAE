"""Code-usage attribution diagnostics for discrete token priors."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from time_causal_vae.evaluation.token_diagnostics import (
    marginal_code_histogram,
    transition_matrix,
    validate_tokens,
)


def compare_code_usage(
    *,
    real_tokens: Tensor,
    sampled_tokens: Tensor,
    codebook_size: int,
    top_k: int = 20,
) -> dict[str, Any]:
    """Return code-set, marginal-rank, and transition-pair diagnostics.

    ``real_tokens`` and ``sampled_tokens`` use shape ``[batch, length]``.
    Extra codes are sampled active codes that are absent from the real active
    code set. Missing codes are real active codes that are absent from sampled
    tokens.
    """
    validate_tokens(real_tokens)
    validate_tokens(sampled_tokens)
    real_counts = marginal_code_histogram(real_tokens, codebook_size)
    sampled_counts = marginal_code_histogram(sampled_tokens, codebook_size)
    real_active = active_code_set(real_counts)
    sampled_active = active_code_set(sampled_counts)
    extra_codes = sorted(sampled_active - real_active)
    missing_codes = sorted(real_active - sampled_active)
    sampled_total = float(sampled_counts.sum().item())
    extra_mass = (
        float(sampled_counts[extra_codes].sum().item()) / sampled_total
        if extra_codes and sampled_total > 0.0
        else 0.0
    )
    return {
        "real_active_codes": sorted(real_active),
        "sampled_active_codes": sorted(sampled_active),
        "extra_sampled_codes": extra_codes,
        "missing_real_codes": missing_codes,
        "real_active_code_count": len(real_active),
        "sampled_active_code_count": len(sampled_active),
        "extra_sampled_code_count": len(extra_codes),
        "missing_real_code_count": len(missing_codes),
        "extra_sampled_code_mass": extra_mass,
        "top_real_codes": top_code_frequencies(real_counts, top_k=top_k),
        "top_sampled_codes": top_code_frequencies(sampled_counts, top_k=top_k),
        "frequency_rank_correlation": frequency_rank_correlation(real_counts, sampled_counts),
        "transition_top_overlap": transition_top_overlap(
            real_tokens=real_tokens,
            sampled_tokens=sampled_tokens,
            codebook_size=codebook_size,
            top_k=top_k,
        ),
        "rare_code_transition_contribution": rare_code_transition_contribution(
            sampled_tokens=sampled_tokens,
            codebook_size=codebook_size,
            rare_codes=extra_codes,
        ),
    }


def active_code_set(counts: Tensor) -> set[int]:
    """Return active code ids from a count vector."""
    return {int(index) for index in torch.nonzero(counts > 0, as_tuple=False).flatten().tolist()}


def top_code_frequencies(counts: Tensor, *, top_k: int) -> list[dict[str, float | int]]:
    """Return the most frequent code ids and probabilities."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    total = float(counts.sum().item())
    if total == 0.0:
        return []
    values, indices = torch.topk(counts.float(), k=min(top_k, counts.numel()))
    rows: list[dict[str, float | int]] = []
    for code, count in zip(indices.tolist(), values.tolist(), strict=True):
        if count <= 0.0:
            continue
        rows.append(
            {
                "code": int(code),
                "count": int(count),
                "probability": float(count / total),
            }
        )
    return rows


def frequency_rank_correlation(real_counts: Tensor, sampled_counts: Tensor) -> float:
    """Return Pearson correlation between descending frequency ranks.

    Codes with high frequency receive low numeric ranks. The correlation is
    computed across the full codebook, so absent codes also affect the score.
    """
    if real_counts.shape != sampled_counts.shape:
        raise ValueError("real_counts and sampled_counts must have the same shape.")
    real_ranks = descending_ranks(real_counts.float())
    sampled_ranks = descending_ranks(sampled_counts.float())
    real_centered = real_ranks - real_ranks.mean()
    sampled_centered = sampled_ranks - sampled_ranks.mean()
    denominator = torch.linalg.vector_norm(real_centered) * torch.linalg.vector_norm(
        sampled_centered
    )
    if float(denominator.item()) == 0.0:
        return 0.0
    return float((real_centered * sampled_centered).sum().div(denominator).item())


def descending_ranks(values: Tensor) -> Tensor:
    """Return stable descending ranks for a one-dimensional tensor."""
    if values.ndim != 1:
        raise ValueError(f"values must be one-dimensional; got {tuple(values.shape)}.")
    order = sorted(range(values.numel()), key=lambda index: (-float(values[index].item()), index))
    ranks = torch.empty(values.numel(), dtype=torch.float32)
    for rank, index in enumerate(order, start=1):
        ranks[index] = float(rank)
    return ranks


def transition_top_overlap(
    *,
    real_tokens: Tensor,
    sampled_tokens: Tensor,
    codebook_size: int,
    top_k: int,
) -> dict[str, Any]:
    """Return top transition-pair overlap diagnostics."""
    real_pairs = top_transition_pairs(real_tokens, codebook_size, top_k=top_k)
    sampled_pairs = top_transition_pairs(sampled_tokens, codebook_size, top_k=top_k)
    real_pair_set = {(int(row["from"]), int(row["to"])) for row in real_pairs}
    sampled_pair_set = {(int(row["from"]), int(row["to"])) for row in sampled_pairs}
    overlap = sorted(real_pair_set & sampled_pair_set)
    return {
        "top_k": top_k,
        "overlap_count": len(overlap),
        "overlap_ratio": len(overlap) / float(top_k),
        "overlap_pairs": [{"from": pair[0], "to": pair[1]} for pair in overlap],
        "top_real_transition_pairs": real_pairs,
        "top_sampled_transition_pairs": sampled_pairs,
    }


def top_transition_pairs(
    tokens: Tensor,
    codebook_size: int,
    *,
    top_k: int,
) -> list[dict[str, float | int]]:
    """Return the most common adjacent transition pairs."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    matrix = transition_matrix(tokens, codebook_size)
    total = float(matrix.sum().item())
    if total == 0.0:
        return []
    flattened = matrix.reshape(-1)
    values, flat_indices = torch.topk(flattened, k=min(top_k, flattened.numel()))
    pairs: list[dict[str, float | int]] = []
    for flat_index, count in zip(flat_indices.tolist(), values.tolist(), strict=True):
        if count <= 0.0:
            continue
        pairs.append(
            {
                "from": int(flat_index // codebook_size),
                "to": int(flat_index % codebook_size),
                "count": int(count),
                "probability": float(count / total),
            }
        )
    return pairs


def rare_code_transition_contribution(
    *,
    sampled_tokens: Tensor,
    codebook_size: int,
    rare_codes: list[int],
) -> dict[str, float | int]:
    """Return sampled transition mass involving sampled-only extra codes."""
    matrix = transition_matrix(sampled_tokens, codebook_size)
    total = float(matrix.sum().item())
    if not rare_codes or total == 0.0:
        return {
            "rare_code_count": len(rare_codes),
            "transition_count": 0,
            "transition_mass": 0.0,
            "outgoing_mass": 0.0,
            "incoming_mass": 0.0,
        }
    rare_index = torch.tensor(rare_codes, dtype=torch.long)
    outgoing = float(matrix[rare_index, :].sum().item())
    incoming = float(matrix[:, rare_index].sum().item())
    internal = float(matrix[rare_index][:, rare_index].sum().item())
    involving = outgoing + incoming - internal
    return {
        "rare_code_count": len(rare_codes),
        "transition_count": int(involving),
        "transition_mass": float(involving / total),
        "outgoing_mass": float(outgoing / total),
        "incoming_mass": float(incoming / total),
    }
