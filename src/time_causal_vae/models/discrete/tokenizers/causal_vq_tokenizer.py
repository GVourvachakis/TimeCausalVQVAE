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

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from time_causal_vae.models.discrete.decoders import CausalVQDecoder
from time_causal_vae.models.discrete.encoders import CausalVQEncoder
from time_causal_vae.models.discrete.tokenizers.config import VQTokenizerConfig
from time_causal_vae.models.discrete.tokenizers.quantizers import (
    QuantizerAdapter,
    build_quantizer_adapter,
)
from time_causal_vae.models.layers import assert_no_future_leakage
from time_causal_vae.utils.output import ModelOutput


@dataclass(frozen=True)
class TokenizerAuxiliaryLossContext:
    """Optional dataset-level metadata for factor-tokenizer auxiliary losses."""

    projection_basis: Tensor | None = None
    projection_mean: Tensor | None = None
    standardization_mean: Tensor | None = None
    standardization_std: Tensor | None = None
    sector_labels: Tensor | None = None
    inverse_project_to_raw: bool = True


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

    def forward(
        self,
        inputs: Tensor,
        conditions: Tensor | None = None,
        auxiliary_loss_context: TokenizerAuxiliaryLossContext | None = None,
    ) -> ModelOutput:
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
        auxiliary_losses = tokenizer_auxiliary_losses(
            inputs=inputs,
            reconstructions=recon_x,
            config=self.config,
            context=auxiliary_loss_context,
            reference=recon_loss,
        )
        auxiliary_loss = auxiliary_losses["auxiliary_loss"]
        loss = (
            recon_loss
            + quantizer_output.commitment_loss
            + quantizer_output.codebook_loss
            + auxiliary_loss
        )
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
            **auxiliary_losses,
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


def tokenizer_auxiliary_losses(
    *,
    inputs: Tensor,
    reconstructions: Tensor,
    config: VQTokenizerConfig,
    context: TokenizerAuxiliaryLossContext | None,
    reference: Tensor,
) -> dict[str, Tensor | bool]:
    """Compute optional factor-aware and cross-sectional tokenizer losses."""
    zero = reference.new_zeros(())
    losses = {
        "factor_reconstruction_aux_loss": zero,
        "factor_covariance_loss": zero,
        "factor_correlation_loss": zero,
        "inverse_projected_covariance_loss": zero,
        "inverse_projected_correlation_loss": zero,
        "sector_block_loss": zero,
        "equal_weight_portfolio_vol_loss": zero,
    }

    if config.factor_reconstruction_loss_weight > 0.0:
        losses["factor_reconstruction_aux_loss"] = functional.l1_loss(reconstructions, inputs)
    if config.factor_covariance_loss_weight > 0.0:
        losses["factor_covariance_loss"] = covariance_matrix_loss(inputs, reconstructions)
    if config.factor_correlation_loss_weight > 0.0:
        losses["factor_correlation_loss"] = correlation_matrix_loss(inputs, reconstructions)

    projected_context_available = (
        context is not None
        and context.projection_basis is not None
        and context.projection_mean is not None
    )
    needs_projection_losses = any(
        weight > 0.0
        for weight in (
            config.inverse_projected_covariance_loss_weight,
            config.inverse_projected_correlation_loss_weight,
            config.sector_block_loss_weight,
            config.equal_weight_portfolio_vol_loss_weight,
        )
    )
    if projected_context_available and needs_projection_losses:
        reference_returns = inverse_project_factor_coordinates(
            inputs, cast(TokenizerAuxiliaryLossContext, context)
        )
        candidate_returns = inverse_project_factor_coordinates(
            reconstructions,
            cast(TokenizerAuxiliaryLossContext, context),
        )
        if config.inverse_projected_covariance_loss_weight > 0.0:
            losses["inverse_projected_covariance_loss"] = covariance_matrix_loss(
                reference_returns,
                candidate_returns,
            )
        if config.inverse_projected_correlation_loss_weight > 0.0:
            losses["inverse_projected_correlation_loss"] = correlation_matrix_loss(
                reference_returns,
                candidate_returns,
            )
        if config.sector_block_loss_weight > 0.0 and context is not None:
            losses["sector_block_loss"] = sector_block_correlation_loss(
                reference_returns,
                candidate_returns,
                context.sector_labels,
            )
        if config.equal_weight_portfolio_vol_loss_weight > 0.0:
            losses["equal_weight_portfolio_vol_loss"] = equal_weight_volatility_loss(
                reference_returns,
                candidate_returns,
            )

    auxiliary_loss = (
        losses["factor_reconstruction_aux_loss"] * config.factor_reconstruction_loss_weight
        + losses["factor_covariance_loss"] * config.factor_covariance_loss_weight
        + losses["factor_correlation_loss"] * config.factor_correlation_loss_weight
        + losses["inverse_projected_covariance_loss"]
        * config.inverse_projected_covariance_loss_weight
        + losses["inverse_projected_correlation_loss"]
        * config.inverse_projected_correlation_loss_weight
        + losses["sector_block_loss"] * config.sector_block_loss_weight
        + losses["equal_weight_portfolio_vol_loss"] * config.equal_weight_portfolio_vol_loss_weight
    )
    return {
        **losses,
        "auxiliary_loss": auxiliary_loss,
        "auxiliary_loss_applied": bool(float(auxiliary_loss.detach().cpu()) != 0.0),
        "auxiliary_context_available": bool(projected_context_available),
    }


def inverse_project_factor_coordinates(
    factor_coordinates: Tensor,
    context: TokenizerAuxiliaryLossContext,
) -> Tensor:
    """Map factor coordinates to inverse-projected 50D returns for auxiliary losses."""
    if context.projection_basis is None or context.projection_mean is None:
        raise ValueError("projection_basis and projection_mean are required.")
    basis = context.projection_basis.to(
        device=factor_coordinates.device, dtype=factor_coordinates.dtype
    )
    mean = context.projection_mean.to(
        device=factor_coordinates.device, dtype=factor_coordinates.dtype
    )
    if factor_coordinates.shape[-1] != basis.shape[-1]:
        raise ValueError(
            f"Expected {basis.shape[-1]} factor coordinates; got {factor_coordinates.shape[-1]}."
        )
    projected = torch.matmul(factor_coordinates, basis.T) + mean.view(1, 1, -1)
    if (
        context.inverse_project_to_raw
        and context.standardization_mean is not None
        and context.standardization_std is not None
    ):
        standardization_mean = context.standardization_mean.to(
            device=factor_coordinates.device,
            dtype=factor_coordinates.dtype,
        )
        standardization_std = context.standardization_std.to(
            device=factor_coordinates.device,
            dtype=factor_coordinates.dtype,
        )
        projected = projected * standardization_std.view(1, 1, -1)
        projected = projected + standardization_mean.view(1, 1, -1)
    return projected


def covariance_matrix_loss(reference: Tensor, candidate: Tensor) -> Tensor:
    """Return normalized MSE between pooled covariance matrices."""
    reference_covariance = pooled_covariance(reference).detach()
    candidate_covariance = pooled_covariance(candidate)
    return normalized_matrix_mse(reference_covariance, candidate_covariance)


def correlation_matrix_loss(reference: Tensor, candidate: Tensor) -> Tensor:
    """Return normalized MSE between pooled correlation matrices."""
    reference_correlation = covariance_to_correlation(pooled_covariance(reference)).detach()
    candidate_correlation = covariance_to_correlation(pooled_covariance(candidate))
    return normalized_matrix_mse(reference_correlation, candidate_correlation)


def sector_block_correlation_loss(
    reference: Tensor,
    candidate: Tensor,
    sector_labels: Tensor | None,
) -> Tensor:
    """Return MSE between sector-block mean correlations."""
    if sector_labels is None:
        return reference.new_zeros(())
    reference_correlation = covariance_to_correlation(pooled_covariance(reference)).detach()
    candidate_correlation = covariance_to_correlation(pooled_covariance(candidate))
    labels = sector_labels.detach().to(device=reference.device).long().reshape(-1)
    if labels.numel() != reference.shape[-1]:
        raise ValueError("sector_labels length must match the projected asset dimension.")
    errors: list[Tensor] = []
    for row_sector in torch.unique(labels, sorted=True).tolist():
        row_mask = labels == int(row_sector)
        for col_sector in torch.unique(labels, sorted=True).tolist():
            col_mask = labels == int(col_sector)
            reference_block = reference_correlation[row_mask][:, col_mask]
            candidate_block = candidate_correlation[row_mask][:, col_mask]
            if int(row_sector) == int(col_sector):
                keep_mask = ~torch.eye(
                    reference_block.shape[0],
                    dtype=torch.bool,
                    device=reference.device,
                )
                reference_values = reference_block[keep_mask]
                candidate_values = candidate_block[keep_mask]
            else:
                reference_values = reference_block.reshape(-1)
                candidate_values = candidate_block.reshape(-1)
            if reference_values.numel() > 0:
                errors.append((candidate_values.mean() - reference_values.mean()).square())
    if not errors:
        return reference.new_zeros(())
    return torch.stack(errors).mean()


def equal_weight_volatility_loss(reference: Tensor, candidate: Tensor) -> Tensor:
    """Return normalized squared error between equal-weight portfolio volatilities."""
    reference_portfolio = reference.mean(dim=-1).reshape(-1)
    candidate_portfolio = candidate.mean(dim=-1).reshape(-1)
    reference_vol = reference_portfolio.std(unbiased=False).detach().clamp_min(1e-12)
    candidate_vol = candidate_portfolio.std(unbiased=False)
    return (candidate_vol - reference_vol).square() / reference_vol.square()


def pooled_covariance(values: Tensor) -> Tensor:
    """Return pooled feature covariance for ``[batch, time, dim]`` values."""
    flat_values = values.reshape(-1, values.shape[-1])
    centred = flat_values - flat_values.mean(dim=0, keepdim=True)
    denominator = max(flat_values.shape[0] - 1, 1)
    return centred.T.matmul(centred) / float(denominator)


def covariance_to_correlation(covariance: Tensor) -> Tensor:
    """Convert a covariance matrix to a correlation matrix."""
    diagonal = covariance.diag().clamp_min(1e-12).sqrt()
    raw_correlation = covariance / (diagonal.view(-1, 1) * diagonal.view(1, -1))
    identity = torch.eye(
        covariance.shape[0],
        device=covariance.device,
        dtype=covariance.dtype,
    )
    return raw_correlation * (1.0 - identity) + identity


def normalized_matrix_mse(reference: Tensor, candidate: Tensor) -> Tensor:
    """Return matrix MSE normalized by reference scale."""
    denominator = reference.square().mean().detach().clamp_min(1e-12)
    return functional.mse_loss(candidate, reference) / denominator


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


def assert_tokenizer_no_future_leakage(
    tokenizer: CausalVQTokenizer,
    reference_inputs: Tensor,
    changed_future_inputs: Tensor,
    cutoff: int,
    *,
    conditions: Tensor | None = None,
    auxiliary_loss_context: TokenizerAuxiliaryLossContext | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Assert no future leakage in the tokenizer reconstruction prefix."""

    class ReconstructionOnly(nn.Module):
        def __init__(self, wrapped_tokenizer: CausalVQTokenizer) -> None:
            super().__init__()
            self.wrapped_tokenizer = wrapped_tokenizer

        def forward(self, inputs: Tensor) -> Tensor:
            output = self.wrapped_tokenizer(inputs, conditions, auxiliary_loss_context)
            return cast(Tensor, output.recon_x)

    return assert_no_future_leakage(
        ReconstructionOnly(tokenizer),
        reference_inputs,
        changed_future_inputs,
        cutoff,
        atol=atol,
        rtol=rtol,
    )
