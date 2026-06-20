"""Condition handling shared by discrete encoder and decoder modules."""

from __future__ import annotations

import torch
from torch import Tensor


def prepare_conditioned_sequence(
    inputs: Tensor,
    conditions: Tensor | None,
    *,
    data_dim: int,
    condition_dim: int,
    module_name: str,
) -> Tensor:
    """Concatenate optional scalar or temporal conditions to a sequence."""
    if inputs.ndim != 3:
        raise ValueError(
            f"{module_name} expects [batch, length, channels] inputs; got shape "
            f"{tuple(inputs.shape)}."
        )
    if inputs.shape[-1] != data_dim:
        raise ValueError(f"{module_name} expected {data_dim} channels; got {inputs.shape[-1]}.")
    if condition_dim == 0:
        return inputs
    if conditions is None:
        raise ValueError(f"{module_name} requires conditions with condition_dim={condition_dim}.")

    batch_size, length, _ = inputs.shape
    if conditions.ndim == 2:
        if conditions.shape != (batch_size, condition_dim):
            raise ValueError(
                f"{module_name} expected scalar conditions of shape "
                f"{(batch_size, condition_dim)}; got {tuple(conditions.shape)}."
            )
        prepared_conditions = conditions[:, None, :].expand(batch_size, length, condition_dim)
    elif conditions.ndim == 3:
        if conditions.shape != (batch_size, length, condition_dim):
            raise ValueError(
                f"{module_name} expected temporal conditions of shape "
                f"{(batch_size, length, condition_dim)}; got {tuple(conditions.shape)}."
            )
        prepared_conditions = conditions
    else:
        raise ValueError(
            f"{module_name} conditions must be [batch, condition_dim] or "
            f"[batch, length, condition_dim]; got {tuple(conditions.shape)}."
        )
    return torch.cat(
        [inputs, prepared_conditions.to(device=inputs.device, dtype=inputs.dtype)], dim=-1
    )
