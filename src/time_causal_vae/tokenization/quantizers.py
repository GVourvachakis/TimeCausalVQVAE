"""Stable adapters around VQ-family quantizer backends.

References
----------
    [vqvae_2017], [vector_quantize_pytorch], [qinco_2024], [mgvq_2025] in docs/references.md.
Borrowed idea:
    Expose standard, residual, and grouped residual codebooks behind one tokenizer-facing API.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
from torch import Tensor, nn
from vector_quantize_pytorch import GroupedResidualVQ, ResidualVQ, VectorQuantize


@dataclass(frozen=True)
class VectorQuantizerOutput:
    """Stable output structure returned by quantizer adapters."""

    quantized: Tensor
    indices: Tensor
    commitment_loss: Tensor
    codebook_loss: Tensor
    quantizer_type: str
    index_shape: tuple[int, ...]


class QuantizerAdapter(Protocol):
    """Protocol shared by tokenizer quantizer adapters."""

    backend: nn.Module
    embedding_dim: int
    codebook_size: int
    quantizer_type: str

    def __call__(self, inputs: Tensor) -> VectorQuantizerOutput:
        """Quantize input embeddings."""
        ...

    def decode_indices(self, indices: Tensor) -> Tensor:
        """Decode code indices to quantized embeddings."""
        ...


class VectorQuantizerAdapter(nn.Module):
    """Wrap ``vector_quantize_pytorch.VectorQuantize`` behind a stable API."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        codebook_size: int,
        commitment_weight: float,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        use_cosine_sim: bool = False,
        codebook_dim: int | None = None,
        threshold_ema_dead_code: float = 0.0,
        decay: float = 0.8,
        stochastic_sample_codes: bool = False,
        sample_codebook_temp: float = 0.0,
    ) -> None:
        """Initialise the backend vector quantizer."""
        super().__init__()
        _validate_common_quantizer_args(
            embedding_dim=embedding_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            kmeans_iters=kmeans_iters,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=threshold_ema_dead_code,
            decay=decay,
            sample_codebook_temp=sample_codebook_temp,
        )

        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.quantizer_type = "vector"
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.use_cosine_sim = use_cosine_sim
        self.codebook_dim = codebook_dim
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.decay = decay
        self.stochastic_sample_codes = stochastic_sample_codes
        self.sample_codebook_temp = sample_codebook_temp
        self.backend = VectorQuantize(
            dim=embedding_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            use_cosine_sim=use_cosine_sim,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=threshold_ema_dead_code,
            decay=decay,
            channel_last=True,
            stochastic_sample_codes=stochastic_sample_codes,
            sample_codebook_temp=sample_codebook_temp,
        )

    def forward(self, inputs: Tensor) -> VectorQuantizerOutput:
        """Quantize ``[batch, length, embedding_dim]`` inputs."""
        if inputs.ndim != 3:
            raise ValueError(
                "VectorQuantizerAdapter expects [batch, length, embedding_dim] inputs; "
                f"got shape {tuple(inputs.shape)}."
            )
        if inputs.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}; got {inputs.shape[-1]}."
            )

        backend_output = cast(
            tuple[Tensor, Tensor, Tensor, Any],
            self.backend(inputs, return_loss_breakdown=True),
        )
        quantized, indices, commitment_loss, loss_breakdown = backend_output
        codebook_loss = _codebook_loss_from_breakdown(loss_breakdown, inputs)
        return VectorQuantizerOutput(
            quantized=quantized,
            indices=indices,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            quantizer_type=self.quantizer_type,
            index_shape=tuple(indices.shape),
        )

    def decode_indices(self, indices: Tensor) -> Tensor:
        """Decode ``[batch, length]`` indices to quantized embeddings."""
        if indices.ndim != 2:
            raise ValueError(f"Vector indices must be [batch, length]; got {tuple(indices.shape)}.")
        return _decode_with_backend(self.backend, indices)


class ResidualVQAdapter(nn.Module):
    """Wrap ``vector_quantize_pytorch.ResidualVQ`` behind a stable API."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        codebook_size: int,
        commitment_weight: float,
        num_quantizers: int,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        use_cosine_sim: bool = False,
        codebook_dim: int | None = None,
        threshold_ema_dead_code: float = 0.0,
        decay: float = 0.8,
        shared_codebook: bool = False,
        stochastic_sample_codes: bool = False,
        sample_codebook_temp: float = 0.0,
    ) -> None:
        """Initialise the backend residual quantizer."""
        super().__init__()
        _validate_common_quantizer_args(
            embedding_dim=embedding_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            kmeans_iters=kmeans_iters,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=threshold_ema_dead_code,
            decay=decay,
            sample_codebook_temp=sample_codebook_temp,
        )
        if num_quantizers <= 0:
            raise ValueError("num_quantizers must be positive.")
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.num_quantizers = num_quantizers
        self.quantizer_type = "residual_vq"
        self.backend = ResidualVQ(
            dim=embedding_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            codebook_dim=codebook_dim,
            shared_codebook=shared_codebook,
            commitment_weight=commitment_weight,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            use_cosine_sim=use_cosine_sim,
            threshold_ema_dead_code=threshold_ema_dead_code,
            decay=decay,
            stochastic_sample_codes=stochastic_sample_codes,
            sample_codebook_temp=sample_codebook_temp,
        )

    def forward(self, inputs: Tensor) -> VectorQuantizerOutput:
        """Quantize ``[batch, length, embedding_dim]`` inputs."""
        _validate_quantizer_inputs(
            inputs, embedding_dim=self.embedding_dim, module_name=self.__class__.__name__
        )
        quantized, indices, commitment_loss = cast(
            tuple[Tensor, Tensor, Tensor],
            self.backend(inputs),
        )
        if indices.ndim != 3 or indices.shape[-1] != self.num_quantizers:
            raise ValueError(
                "ResidualVQ indices must be [batch, length, num_quantizers]; "
                f"got {tuple(indices.shape)}."
            )
        return VectorQuantizerOutput(
            quantized=quantized,
            indices=indices,
            commitment_loss=_reduce_loss_tensor(commitment_loss, reference=inputs),
            codebook_loss=inputs.new_zeros(()),
            quantizer_type=self.quantizer_type,
            index_shape=tuple(indices.shape),
        )

    def decode_indices(self, indices: Tensor) -> Tensor:
        """Decode ``[batch, length, num_quantizers]`` indices to embeddings."""
        if indices.ndim != 3 or indices.shape[-1] != self.num_quantizers:
            raise ValueError(
                "ResidualVQ indices must be [batch, length, num_quantizers]; "
                f"got {tuple(indices.shape)}."
            )
        return _decode_with_backend(self.backend, indices)


class GroupedResidualVQAdapter(nn.Module):
    """Wrap ``GroupedResidualVQ`` with project-normalised index layout."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        codebook_size: int,
        commitment_weight: float,
        num_quantizers: int,
        groups: int,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        use_cosine_sim: bool = False,
        codebook_dim: int | None = None,
        threshold_ema_dead_code: float = 0.0,
        decay: float = 0.8,
        shared_codebook: bool = False,
        stochastic_sample_codes: bool = False,
        sample_codebook_temp: float = 0.0,
    ) -> None:
        """Initialise the backend grouped residual quantizer."""
        super().__init__()
        _validate_common_quantizer_args(
            embedding_dim=embedding_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            kmeans_iters=kmeans_iters,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=threshold_ema_dead_code,
            decay=decay,
            sample_codebook_temp=sample_codebook_temp,
        )
        if num_quantizers <= 0:
            raise ValueError("num_quantizers must be positive.")
        if groups <= 0:
            raise ValueError("groups must be positive.")
        if embedding_dim % groups != 0:
            raise ValueError("embedding_dim must be divisible by groups.")
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.num_quantizers = num_quantizers
        self.groups = groups
        self.quantizer_type = "grouped_residual_vq"
        self.backend = GroupedResidualVQ(
            dim=embedding_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            groups=groups,
            codebook_dim=codebook_dim,
            shared_codebook=shared_codebook,
            commitment_weight=commitment_weight,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            use_cosine_sim=use_cosine_sim,
            threshold_ema_dead_code=threshold_ema_dead_code,
            decay=decay,
            stochastic_sample_codes=stochastic_sample_codes,
            sample_codebook_temp=sample_codebook_temp,
        )

    def forward(self, inputs: Tensor) -> VectorQuantizerOutput:
        """Quantize inputs and normalise indices to ``[batch, length, group, level]``."""
        _validate_quantizer_inputs(
            inputs, embedding_dim=self.embedding_dim, module_name=self.__class__.__name__
        )
        quantized, backend_indices, commitment_loss = cast(
            tuple[Tensor, Tensor, Tensor],
            self.backend(inputs),
        )
        expected_backend_shape = (
            self.groups,
            inputs.shape[0],
            inputs.shape[1],
            self.num_quantizers,
        )
        if tuple(backend_indices.shape) != expected_backend_shape:
            raise ValueError(
                "GroupedResidualVQ backend indices must be "
                "[group, batch, length, num_quantizers]; got "
                f"{tuple(backend_indices.shape)}."
            )
        indices = backend_indices.permute(1, 2, 0, 3).contiguous()
        return VectorQuantizerOutput(
            quantized=quantized,
            indices=indices,
            commitment_loss=_reduce_loss_tensor(commitment_loss, reference=inputs),
            codebook_loss=inputs.new_zeros(()),
            quantizer_type=self.quantizer_type,
            index_shape=tuple(indices.shape),
        )

    def decode_indices(self, indices: Tensor) -> Tensor:
        """Decode project-normalised ``[batch, length, group, level]`` indices."""
        if indices.ndim != 4 or indices.shape[2:] != (self.groups, self.num_quantizers):
            raise ValueError(
                "GroupedResidualVQ indices must be [batch, length, groups, num_quantizers]; "
                f"got {tuple(indices.shape)}."
            )
        backend_indices = indices.permute(2, 0, 1, 3).contiguous()
        return _decode_with_backend(self.backend, backend_indices)


def build_quantizer_adapter(
    *,
    quantizer_type: str,
    embedding_dim: int,
    codebook_size: int,
    commitment_weight: float,
    num_quantizers: int = 1,
    groups: int = 1,
    kmeans_init: bool = False,
    kmeans_iters: int = 10,
    use_cosine_sim: bool = False,
    codebook_dim: int | None = None,
    threshold_ema_dead_code: float = 0.0,
    decay: float = 0.8,
    shared_codebook: bool = False,
    stochastic_sample_codes: bool = False,
    sample_codebook_temp: float = 0.0,
) -> nn.Module:
    """Build a configured tokenizer quantizer adapter."""
    if quantizer_type == "vector":
        return VectorQuantizerAdapter(
            embedding_dim=embedding_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            use_cosine_sim=use_cosine_sim,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=threshold_ema_dead_code,
            decay=decay,
            stochastic_sample_codes=stochastic_sample_codes,
            sample_codebook_temp=sample_codebook_temp,
        )
    if quantizer_type == "residual_vq":
        return ResidualVQAdapter(
            embedding_dim=embedding_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            num_quantizers=num_quantizers,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            use_cosine_sim=use_cosine_sim,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=threshold_ema_dead_code,
            decay=decay,
            shared_codebook=shared_codebook,
            stochastic_sample_codes=stochastic_sample_codes,
            sample_codebook_temp=sample_codebook_temp,
        )
    if quantizer_type == "grouped_residual_vq":
        return GroupedResidualVQAdapter(
            embedding_dim=embedding_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            num_quantizers=num_quantizers,
            groups=groups,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            use_cosine_sim=use_cosine_sim,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=threshold_ema_dead_code,
            decay=decay,
            shared_codebook=shared_codebook,
            stochastic_sample_codes=stochastic_sample_codes,
            sample_codebook_temp=sample_codebook_temp,
        )
    raise ValueError(f"Unsupported quantizer_type: {quantizer_type}")


def _codebook_loss_from_breakdown(loss_breakdown: Any, reference: Tensor) -> Tensor:
    """Return backend codebook losses when exposed, otherwise a zero scalar."""
    codebook_loss = reference.new_zeros(())
    for attribute_name in ("codebook_loss", "codebook_diversity", "orthogonal_reg"):
        value = getattr(loss_breakdown, attribute_name, None)
        if isinstance(value, Tensor):
            codebook_loss = codebook_loss + value.to(device=reference.device, dtype=reference.dtype)
    if not torch.isfinite(codebook_loss):
        raise ValueError("Backend returned a non-finite codebook loss.")
    return codebook_loss


def _validate_common_quantizer_args(
    *,
    embedding_dim: int,
    codebook_size: int,
    commitment_weight: float,
    kmeans_iters: int,
    codebook_dim: int | None,
    threshold_ema_dead_code: float,
    decay: float,
    sample_codebook_temp: float,
) -> None:
    """Validate scalar quantizer constructor arguments."""
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive.")
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive.")
    if commitment_weight < 0.0:
        raise ValueError("commitment_weight must be non-negative.")
    if kmeans_iters <= 0:
        raise ValueError("kmeans_iters must be positive.")
    if codebook_dim is not None and codebook_dim <= 0:
        raise ValueError("codebook_dim must be positive when provided.")
    if threshold_ema_dead_code < 0.0:
        raise ValueError("threshold_ema_dead_code must be non-negative.")
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must satisfy 0.0 < decay <= 1.0.")
    if sample_codebook_temp < 0.0:
        raise ValueError("sample_codebook_temp must be non-negative.")


def _validate_quantizer_inputs(
    inputs: Tensor,
    *,
    embedding_dim: int,
    module_name: str,
) -> None:
    """Validate common quantizer input shape."""
    if inputs.ndim != 3:
        raise ValueError(
            f"{module_name} expects [batch, length, embedding_dim] inputs; "
            f"got shape {tuple(inputs.shape)}."
        )
    if inputs.shape[-1] != embedding_dim:
        raise ValueError(f"Expected embedding_dim={embedding_dim}; got {inputs.shape[-1]}.")


def _reduce_loss_tensor(value: Tensor, *, reference: Tensor) -> Tensor:
    """Reduce backend per-level loss tensors to a finite scalar."""
    reduced = value.to(device=reference.device, dtype=reference.dtype).mean()
    if not torch.isfinite(reduced):
        raise ValueError("Backend returned a non-finite commitment loss.")
    return reduced


def _decode_with_backend(backend: nn.Module, indices: Tensor) -> Tensor:
    """Decode indices with the backend helper and validate output rank."""
    if not hasattr(backend, "get_output_from_indices"):
        raise RuntimeError(f"{backend.__class__.__name__} does not expose get_output_from_indices.")
    decode = cast(Callable[[Tensor], Tensor], backend.get_output_from_indices)
    decoded = decode(indices)
    if decoded.ndim != 3:
        raise ValueError(
            "Decoded embeddings must be [batch, length, embedding_dim]; "
            f"got {tuple(decoded.shape)}."
        )
    return decoded
