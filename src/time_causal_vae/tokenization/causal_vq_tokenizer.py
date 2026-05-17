"""Causal convolutional VQ tokenizer for financial time-series windows.

References
----------
- Time-Causal VAE: Robust Financial Time Series Generator, Acciaio, Eckstein, and Hou
(DOI: 10.1137/24M1711650; arXiv DOI: 10.48550/arXiv.2411.02947) - adapted
no-anticipation financial time-series generation and diagnostics.

- Neural Discrete Representation Learning, van den Oord, Vinyals, and Kavukcuoglu
(DOI: 10.5555/3295222.3295378) - adapted vector-quantized latent tokenisation.

- Vector Quantized Time Series Generation with a Bidirectional Prior Model, Lee, Malacarne,
and Aune (PMLR 206; arXiv DOI: 10.48550/arXiv.2303.04743) - used as a
two-stage VQ time-series contrast; bidirectional priors are not used.

- DeepVol: Volatility Forecasting from High-Frequency Data with Dilated Causal Convolutions
(arXiv DOI: 10.48550/arXiv.2210.04797) - adapted dilated causal-convolution motivation.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from time_causal_vae.models.layers import CausalConvStack, assert_no_future_leakage
from time_causal_vae.tokenization.config import VQTokenizerConfig
from time_causal_vae.tokenization.quantizers import QuantizerAdapter, build_quantizer_adapter
from time_causal_vae.utils.output import ModelOutput


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


class CausalVQTokenizer(nn.Module):
    """Standalone causal vector-quantized autoencoder.

    Shape convention:

    ```text
    x: [batch, length, data_dim]
    z_e: [batch, length, embedding_dim]
    z_q: [batch, length, embedding_dim]
    indices:
      vector: [batch, length]
      residual_vq: [batch, length, num_quantizers]
      grouped_residual_vq: [batch, length, groups, num_quantizers]
    recon_x: [batch, length, data_dim]
    ```
    """

    def __init__(self, config: VQTokenizerConfig) -> None:
        """Initialise encoder, quantizer, and decoder."""
        super().__init__()
        self.config = config
        self.encoder = CausalVQEncoder(config)
        self.quantizer = cast(
            QuantizerAdapter,
            build_quantizer_adapter(
                quantizer_type=config.quantizer_type,
                embedding_dim=config.embedding_dim,
                codebook_size=config.codebook_size,
                commitment_weight=config.commitment_weight,
                num_quantizers=config.num_quantizers,
                groups=config.groups,
                kmeans_init=config.kmeans_init,
                kmeans_iters=config.kmeans_iters,
                use_cosine_sim=config.use_cosine_sim,
                codebook_dim=config.codebook_dim,
                threshold_ema_dead_code=config.threshold_ema_dead_code,
                decay=config.decay,
                shared_codebook=config.shared_codebook,
                stochastic_sample_codes=config.stochastic_sample_codes,
                sample_codebook_temp=config.sample_codebook_temp,
            ),
        )
        self.decoder = CausalVQDecoder(config)

    def forward(self, inputs: Tensor, conditions: Tensor | None = None) -> ModelOutput:
        """Return reconstruction, token, and loss outputs for ``inputs``."""
        validate_observation_sequence(inputs, self.config)
        z_e = self.encoder(inputs, conditions)
        quantizer_output = self.quantizer(z_e)
        recon_x = self.decoder(quantizer_output.quantized, conditions)
        recon_loss = functional.l1_loss(recon_x, inputs)
        usage_loss = code_usage_entropy_loss(
            quantizer_output.indices,
            codebook_size=self.config.codebook_size,
            reference=recon_loss,
        )
        loss = recon_loss + quantizer_output.commitment_loss + quantizer_output.codebook_loss
        return ModelOutput(
            recon_x=recon_x,
            z_e=z_e,
            z_q=quantizer_output.quantized,
            indices=quantizer_output.indices,
            recon_loss=recon_loss,
            commitment_loss=quantizer_output.commitment_loss,
            codebook_loss=quantizer_output.codebook_loss,
            usage_loss=usage_loss,
            usage_regularization_applied=False,
            quantizer_type=quantizer_output.quantizer_type,
            index_shape=list(quantizer_output.index_shape),
            loss=loss,
        )

    def decode_indices(self, indices: Tensor, conditions: Tensor | None = None) -> Tensor:
        """Decode tokenizer indices through the quantizer helper and causal decoder."""
        quantized = self.quantizer.decode_indices(indices)
        return cast(Tensor, self.decoder(quantized, conditions))


def validate_observation_sequence(inputs: Tensor, config: VQTokenizerConfig) -> None:
    """Validate tokenizer input observations."""
    if inputs.ndim != 3:
        raise ValueError(
            "CausalVQTokenizer expects [batch, length, data_dim] inputs; "
            f"got shape {tuple(inputs.shape)}."
        )
    if inputs.shape[1] != config.data_length:
        raise ValueError(f"Expected sequence length {config.data_length}; got {inputs.shape[1]}.")
    if inputs.shape[-1] != config.data_dim:
        raise ValueError(f"Expected data_dim={config.data_dim}; got {inputs.shape[-1]}.")


def code_usage_entropy_loss(
    indices: Tensor,
    *,
    codebook_size: int,
    reference: Tensor,
) -> Tensor:
    """Return diagnostic ``-entropy`` from hard code indices.

    The installed quantizer exposes hard indices, not differentiable assignment
    probabilities. This value is therefore detached diagnostic telemetry and is
    intentionally not added to the tokenizer training loss.
    """
    with torch.no_grad():
        code_counts = torch.bincount(
            indices.detach().reshape(-1).to(device=torch.device("cpu")),
            minlength=codebook_size,
        )[:codebook_size]
        total = code_counts.sum()
        if int(total.item()) == 0:
            value = 0.0
        else:
            probabilities = code_counts.float() / total.float()
            active_probabilities = probabilities[probabilities > 0.0]
            entropy = -(active_probabilities * active_probabilities.log()).sum()
            value = -float(entropy.item())
    return reference.detach().new_tensor(value)


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


def assert_tokenizer_no_future_leakage(
    tokenizer: CausalVQTokenizer,
    reference_inputs: Tensor,
    changed_future_inputs: Tensor,
    cutoff: int,
    *,
    conditions: Tensor | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Assert no future leakage in the tokenizer reconstruction prefix."""

    class ReconstructionOnly(nn.Module):
        def __init__(self, wrapped_tokenizer: CausalVQTokenizer) -> None:
            super().__init__()
            self.wrapped_tokenizer = wrapped_tokenizer

        def forward(self, inputs: Tensor) -> Tensor:
            output = self.wrapped_tokenizer(inputs, conditions)
            return cast(Tensor, output.recon_x)

    return assert_no_future_leakage(
        ReconstructionOnly(tokenizer),
        reference_inputs,
        changed_future_inputs,
        cutoff,
        atol=atol,
        rtol=rtol,
    )
