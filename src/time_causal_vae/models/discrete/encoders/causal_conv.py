"""Causal-convolution encoder for discrete VQ tokenizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from time_causal_vae.models.discrete.conditioning import prepare_conditioned_sequence
from time_causal_vae.models.layers import CausalConvStack

if TYPE_CHECKING:
    from time_causal_vae.models.discrete.tokenizers.config import VQTokenizerConfig


class CausalVQEncoder(nn.Module):
    """Causal convolutional encoder for tokenizer latents."""

    def __init__(self, config: VQTokenizerConfig) -> None:
        """Initialise the encoder from tokenizer config."""
        super().__init__()
        self.config = config
        self.input_channels = config.data_dim + config.condition_dim
        self.stack = CausalConvStack(
            in_channels=self.input_channels,
            hidden_channels=config.encoder_hidden_dim,
            out_channels=config.embedding_dim,
            kernel_size=3,
            dilations=config.layer_dilations,
            dropout=config.dropout,
        )

    def forward(self, inputs: Tensor, conditions: Tensor | None = None) -> Tensor:
        """Encode ``[batch, length, data_dim]`` inputs into ``z_e``."""
        encoder_inputs = prepare_conditioned_sequence(
            inputs,
            conditions,
            data_dim=self.config.data_dim,
            condition_dim=self.config.condition_dim,
            module_name=self.__class__.__name__,
        )
        return cast(Tensor, self.stack(encoder_inputs))
