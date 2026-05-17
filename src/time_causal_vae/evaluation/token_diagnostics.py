"""Token-level diagnostics for discrete causal priors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from time_causal_vae.evaluation.tokenizer import summarise_code_usage


def marginal_code_histogram(tokens: Tensor, codebook_size: int) -> Tensor:
    """Return code counts over token tensors with shape ``[batch, length]``."""
    validate_tokens(tokens)
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive.")
    return torch.bincount(tokens.detach().cpu().reshape(-1), minlength=codebook_size)[
        :codebook_size
    ]


def marginal_code_distribution(tokens: Tensor, codebook_size: int) -> Tensor:
    """Return a probability vector over code indices."""
    counts = marginal_code_histogram(tokens, codebook_size).float()
    total = counts.sum()
    if float(total.item()) == 0.0:
        return torch.zeros_like(counts)
    return counts / total


def transition_matrix(tokens: Tensor, codebook_size: int) -> Tensor:
    """Return a full codebook transition-count matrix.

    Rows are previous tokens and columns are next tokens. The returned tensor
    has shape ``[codebook_size, codebook_size]``.
    """
    validate_tokens(tokens)
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive.")
    matrix = torch.zeros((codebook_size, codebook_size), dtype=torch.float32)
    if tokens.shape[1] <= 1:
        return matrix
    previous_tokens = tokens.detach().cpu()[:, :-1].reshape(-1).long()
    next_tokens = tokens.detach().cpu()[:, 1:].reshape(-1).long()
    flat_indices = previous_tokens * codebook_size + next_tokens
    counts = torch.bincount(flat_indices, minlength=codebook_size * codebook_size)
    return counts.reshape(codebook_size, codebook_size).float()


def row_normalize_transition_matrix(matrix: Tensor) -> Tensor:
    """Return a row-normalised transition matrix with zero rows left at zero."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square; got {tuple(matrix.shape)}.")
    row_sums = matrix.sum(dim=1, keepdim=True)
    return torch.where(row_sums > 0.0, matrix / row_sums.clamp_min(1.0), torch.zeros_like(matrix))


def transition_l1_distance(
    real_tokens: Tensor, sampled_tokens: Tensor, codebook_size: int
) -> float:
    """Return mean row L1 distance between row-normalised transition matrices."""
    real_matrix = row_normalize_transition_matrix(transition_matrix(real_tokens, codebook_size))
    sampled_matrix = row_normalize_transition_matrix(
        transition_matrix(sampled_tokens, codebook_size)
    )
    return float((real_matrix - sampled_matrix).abs().sum(dim=1).mean().item())


def run_lengths(tokens: Tensor) -> Tensor:
    """Return aggregated consecutive run lengths across all token paths."""
    validate_tokens(tokens)
    cpu_tokens = tokens.detach().cpu().long()
    lengths: list[int] = []
    for row in cpu_tokens:
        current_length = 1
        for position in range(1, row.numel()):
            if int(row[position].item()) == int(row[position - 1].item()):
                current_length += 1
            else:
                lengths.append(current_length)
                current_length = 1
        lengths.append(current_length)
    if not lengths:
        return torch.empty(0, dtype=torch.float32)
    return torch.tensor(lengths, dtype=torch.float32)


def run_length_histogram(tokens: Tensor, max_length: int | None = None) -> Tensor:
    """Return an aggregated run-length probability histogram."""
    lengths = run_lengths(tokens)
    if lengths.numel() == 0:
        width = 1 if max_length is None else max_length
        return torch.zeros(width, dtype=torch.float32)
    observed_max = int(lengths.max().item())
    histogram_width = observed_max if max_length is None else max(max_length, observed_max)
    counts = torch.bincount(lengths.long(), minlength=histogram_width + 1)[1:].float()
    total = counts.sum()
    if float(total.item()) == 0.0:
        return counts
    return counts / total


def run_length_wasserstein(real_tokens: Tensor, sampled_tokens: Tensor) -> float:
    """Return equal-weight one-dimensional Wasserstein distance between run lengths."""
    real_lengths = run_lengths(real_tokens)
    sampled_lengths = run_lengths(sampled_tokens)
    if real_lengths.numel() == 0 or sampled_lengths.numel() == 0:
        return 0.0
    n_values = min(real_lengths.numel(), sampled_lengths.numel())
    distance = (
        real_lengths.sort().values[:n_values] - sampled_lengths.sort().values[:n_values]
    ).abs()
    return float(distance.mean().item())


def run_length_histogram_l1(real_tokens: Tensor, sampled_tokens: Tensor) -> float:
    """Return L1 distance between aggregated run-length histograms."""
    max_length = max(real_tokens.shape[1], sampled_tokens.shape[1])
    real_histogram = run_length_histogram(real_tokens, max_length=max_length)
    sampled_histogram = run_length_histogram(sampled_tokens, max_length=max_length)
    return float((real_histogram - sampled_histogram).abs().sum().item())


def compare_token_sequences(
    *,
    real_tokens: Tensor,
    sampled_tokens: Tensor,
    codebook_size: int,
) -> dict[str, Any]:
    """Return marginal, transition, and run-length diagnostics."""
    validate_tokens(real_tokens)
    validate_tokens(sampled_tokens)
    real_distribution = marginal_code_distribution(real_tokens, codebook_size)
    sampled_distribution = marginal_code_distribution(sampled_tokens, codebook_size)
    real_counts = marginal_code_histogram(real_tokens, codebook_size)
    sampled_counts = marginal_code_histogram(sampled_tokens, codebook_size)
    real_transition = row_normalize_transition_matrix(transition_matrix(real_tokens, codebook_size))
    sampled_transition = row_normalize_transition_matrix(
        transition_matrix(sampled_tokens, codebook_size)
    )
    run_wasserstein = run_length_wasserstein(real_tokens, sampled_tokens)
    run_histogram_l1 = run_length_histogram_l1(real_tokens, sampled_tokens)
    return {
        "marginal_code_l1": float((real_distribution - sampled_distribution).abs().sum().item()),
        "transition_matrix_l1": float(
            (real_transition - sampled_transition).abs().sum(dim=1).mean().item()
        ),
        "run_length_distance": run_wasserstein,
        "run_length_wasserstein": run_wasserstein,
        "run_length_histogram_l1": run_histogram_l1,
        "real_token": summarise_code_usage(real_counts),
        "sampled_token": summarise_code_usage(sampled_counts),
        "real_run_length_mean": mean_run_length(real_tokens),
        "sampled_run_length_mean": mean_run_length(sampled_tokens),
        "real_run_length_max": max_run_length(real_tokens),
        "sampled_run_length_max": max_run_length(sampled_tokens),
    }


def flatten_token_comparison_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten selected token-comparison diagnostics for summary JSON."""
    real_token = metrics["real_token"]
    sampled_token = metrics["sampled_token"]
    return {
        "marginal_code_l1": metrics["marginal_code_l1"],
        "transition_matrix_l1": metrics["transition_matrix_l1"],
        "run_length_distance": metrics["run_length_distance"],
        "run_length_wasserstein": metrics["run_length_wasserstein"],
        "run_length_histogram_l1": metrics["run_length_histogram_l1"],
        "real_active_code_count": real_token["active_code_count"],
        "sampled_active_code_count": sampled_token["active_code_count"],
        "real_token_perplexity": real_token["codebook_perplexity"],
        "sampled_token_perplexity": sampled_token["codebook_perplexity"],
        "real_token_index_entropy": real_token["index_entropy"],
        "sampled_token_index_entropy": sampled_token["index_entropy"],
        "real_run_length_mean": metrics["real_run_length_mean"],
        "sampled_run_length_mean": metrics["sampled_run_length_mean"],
        "real_run_length_max": metrics["real_run_length_max"],
        "sampled_run_length_max": metrics["sampled_run_length_max"],
    }


def mean_run_length(tokens: Tensor) -> float:
    """Return mean aggregated run length."""
    lengths = run_lengths(tokens)
    if lengths.numel() == 0:
        return 0.0
    return float(lengths.mean().item())


def max_run_length(tokens: Tensor) -> int:
    """Return maximum aggregated run length."""
    lengths = run_lengths(tokens)
    if lengths.numel() == 0:
        return 0
    return int(lengths.max().item())


def validate_tokens(tokens: Tensor) -> None:
    """Validate token tensors in ``[batch, length]`` format."""
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be [batch, length]; got {tuple(tokens.shape)}.")
    if tokens.dtype != torch.long:
        raise ValueError(f"tokens must have dtype torch.long; got {tokens.dtype}.")
