"""Causal-convolution decoder for discrete VQ tokenizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from time_causal_vae.models.discrete.conditioning import prepare_conditioned_sequence
from time_causal_vae.models.layers import CausalConvStack

if TYPE_CHECKING:
    from time_causal_vae.models.discrete.tokenizers.config import VQTokenizerConfig


class CausalVQDecoder(nn.Module):
    """Causal convolutional decoder for quantized latent paths."""

    def __init__(self, config: VQTokenizerConfig) -> None:
        """Initialise the decoder from tokenizer config."""
        super().__init__()
        self.config = config
        self.input_channels = config.embedding_dim + config.condition_dim
        self.stack = CausalConvStack(
            in_channels=self.input_channels,
            hidden_channels=config.decoder_hidden_dim,
            out_channels=config.data_dim,
            kernel_size=3,
            dilations=config.layer_dilations,
            dropout=config.dropout,
        )

    def forward(self, quantized: Tensor, conditions: Tensor | None = None) -> Tensor:
        """Decode ``[batch, length, embedding_dim]`` quantized latents."""
        decoder_inputs = prepare_conditioned_sequence(
            quantized,
            conditions,
            data_dim=self.config.embedding_dim,
            condition_dim=self.config.condition_dim,
            module_name=self.__class__.__name__,
        )
        return cast(Tensor, self.stack(decoder_inputs))
