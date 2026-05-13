"""Attention masks for causal token-prior models."""

from __future__ import annotations

import torch
from torch import Tensor


def causal_attention_mask(
    sequence_length: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return an additive causal attention mask for PyTorch transformers.

    The returned tensor has shape ``[sequence_length, sequence_length]`` and
    uses additive float values: ``0`` permits attention and ``-inf`` blocks
    attention. It is suitable for ``torch.nn.TransformerEncoder`` and
    ``torch.nn.MultiheadAttention`` attention-mask arguments.

    Row ``t`` may attend only to columns ``<= t``. In the default shifted
    convention, input position ``t`` contains ``[BOS, k_0, ..., k_{t-1}]`` and
    logits at position ``t`` predict ``k_t`` from past tokens only.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    mask = torch.zeros((sequence_length, sequence_length), device=device, dtype=dtype)
    blocked_positions = torch.ones(
        (sequence_length, sequence_length),
        device=device,
        dtype=torch.bool,
    ).triu(diagonal=1)
    return mask.masked_fill(blocked_positions, float("-inf"))
