"""Dilated causal convolution layers for time-series tokenizers.

The public convention in this module is ``[batch, length, channels]``. The
wrappers transpose internally to PyTorch's ``[batch, channels, length]`` layout
only around ``torch.nn.Conv1d`` calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

ActivationName = Literal["relu", "gelu", "silu", "tanh"]


def _make_activation(name: ActivationName | None) -> nn.Module:
    if name is None:
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _validate_sequence_input(inputs: Tensor, expected_channels: int, module_name: str) -> None:
    if inputs.ndim != 3:
        raise ValueError(
            f"{module_name} expects a [batch, length, channels] tensor; got shape "
            f"{tuple(inputs.shape)}."
        )
    if inputs.shape[-1] != expected_channels:
        raise ValueError(
            f"{module_name} expects {expected_channels} input channels; got {inputs.shape[-1]}."
        )


class CausalConv1d(nn.Module):
    """One-dimensional convolution with left padding only.

    The layer accepts inputs in ``[batch, length, channels]`` format and returns
    outputs in the same format. It pads only on the left by
    ``dilation * (kernel_size - 1)`` positions before applying ``Conv1d``. For
    stride-one convolutions this preserves the sequence length and prevents
    output time ``t`` from depending on input times greater than ``t``.
    """

    in_channels: int
    out_channels: int
    kernel_size: int
    dilation: int
    left_padding: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialise the causal convolution."""
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if dilation <= 0:
            raise ValueError("dilation must be positive.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the causal convolution."""
        _validate_sequence_input(inputs, self.in_channels, self.__class__.__name__)
        channels_first = inputs.transpose(1, 2)
        if self.left_padding > 0:
            channels_first = functional.pad(channels_first, (self.left_padding, 0))
        outputs = cast(Tensor, self.conv(channels_first))
        return outputs.transpose(1, 2)


class DilatedCausalConvBlock(nn.Module):
    """Dilated causal convolution block without future-leaking normalisation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        activation: ActivationName | None = "gelu",
        dropout: float = 0.0,
        use_residual: bool = True,
    ) -> None:
        """Initialise the block."""
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0.0 <= dropout < 1.0.")

        self.conv = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.activation = _make_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.use_residual = use_residual and in_channels == out_channels

    def forward(self, inputs: Tensor) -> Tensor:
        """Return block outputs in ``[batch, length, channels]`` format."""
        outputs = self.conv(inputs)
        outputs = cast(Tensor, self.activation(outputs))
        outputs = cast(Tensor, self.dropout(outputs))
        if self.use_residual:
            outputs = outputs + inputs
        return outputs


class CausalConvStack(nn.Module):
    """Stack of dilated causal convolution blocks."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        *,
        out_channels: int | None = None,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
        activation: ActivationName | None = "gelu",
        dropout: float = 0.0,
        use_residual: bool = True,
    ) -> None:
        """Initialise the stack with a dilation schedule."""
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")
        if not dilations:
            raise ValueError("dilations must contain at least one value.")
        if any(dilation <= 0 for dilation in dilations):
            raise ValueError("all dilations must be positive.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels if out_channels is None else out_channels
        self.dilations = tuple(dilations)

        blocks: list[DilatedCausalConvBlock] = []
        block_in_channels = in_channels
        for dilation in self.dilations:
            blocks.append(
                DilatedCausalConvBlock(
                    in_channels=block_in_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation=activation,
                    dropout=dropout,
                    use_residual=use_residual,
                )
            )
            block_in_channels = hidden_channels
        self.blocks = nn.ModuleList(blocks)
        self.output_projection: nn.Module
        if self.out_channels == hidden_channels:
            self.output_projection = nn.Identity()
        else:
            self.output_projection = CausalConv1d(hidden_channels, self.out_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the stack and return ``[batch, length, channels_out]`` outputs."""
        _validate_sequence_input(inputs, self.in_channels, self.__class__.__name__)
        outputs = inputs
        for block in self.blocks:
            outputs = cast(Tensor, block(outputs))
        return cast(Tensor, self.output_projection(outputs))


def assert_no_future_leakage(
    module: nn.Module,
    reference_inputs: Tensor,
    changed_future_inputs: Tensor,
    cutoff: int,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Assert that output prefixes are unchanged when only future inputs change.

    The cutoff is zero-indexed and inclusive. Both input tensors must be in
    ``[batch, length, channels]`` format and must match through
    ``reference_inputs[:, : cutoff + 1]``. The helper temporarily evaluates the
    module in deterministic evaluation mode and restores its original training
    state afterwards.
    """
    if reference_inputs.shape != changed_future_inputs.shape:
        raise ValueError(
            "reference_inputs and changed_future_inputs must have the same shape; "
            f"got {tuple(reference_inputs.shape)} and {tuple(changed_future_inputs.shape)}."
        )
    if reference_inputs.ndim != 3:
        raise ValueError(
            "assert_no_future_leakage expects [batch, length, channels] inputs; got "
            f"{tuple(reference_inputs.shape)}."
        )
    length = reference_inputs.shape[1]
    if not 0 <= cutoff < length:
        raise ValueError(f"cutoff must satisfy 0 <= cutoff < {length}; got {cutoff}.")

    prefix = slice(None, cutoff + 1)
    if not torch.allclose(reference_inputs[:, prefix], changed_future_inputs[:, prefix]):
        raise AssertionError("Inputs differ at or before the cutoff.")

    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            reference_outputs = module(reference_inputs)
            changed_future_outputs = module(changed_future_inputs)
    finally:
        module.train(was_training)

    if reference_outputs.ndim < 2 or changed_future_outputs.ndim < 2:
        raise AssertionError("Module outputs must include a sequence-length dimension.")
    if reference_outputs.shape != changed_future_outputs.shape:
        raise AssertionError(
            "Module output shapes differ: "
            f"{tuple(reference_outputs.shape)} vs {tuple(changed_future_outputs.shape)}."
        )
    if reference_outputs.shape[1] <= cutoff:
        raise AssertionError(
            f"Output length {reference_outputs.shape[1]} is too short for cutoff {cutoff}."
        )

    reference_prefix = reference_outputs[:, prefix]
    changed_future_prefix = changed_future_outputs[:, prefix]
    if not torch.allclose(reference_prefix, changed_future_prefix, atol=atol, rtol=rtol):
        max_diff = (reference_prefix - changed_future_prefix).abs().max().item()
        raise AssertionError(
            "Future leakage detected before or at the cutoff: "
            f"max absolute difference is {max_diff:.6g}."
        )
    return reference_outputs, changed_future_outputs
