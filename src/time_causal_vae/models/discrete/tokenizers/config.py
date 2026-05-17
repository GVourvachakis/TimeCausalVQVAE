"""Configuration objects for causal VQ tokenizers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

QuantizerType = Literal["vector", "residual_vq", "grouped_residual_vq"]


@dataclass(frozen=True)
class VQTokenizerConfig:
    """Configuration for the standalone causal VQ tokenizer.

    The public tensor convention is:

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

    data_dim: int
    data_length: int
    embedding_dim: int
    codebook_size: int
    commitment_weight: float
    encoder_hidden_dim: int
    decoder_hidden_dim: int
    num_layers: int
    dilations: Sequence[int] = field(default_factory=lambda: (1, 2, 4, 8))
    dropout: float = 0.0
    condition_dim: int = 0
    kmeans_init: bool = False
    kmeans_iters: int = 10
    use_cosine_sim: bool = False
    codebook_dim: int | None = None
    threshold_ema_dead_code: float = 0.0
    decay: float = 0.8
    usage_regularization_weight: float = 0.0
    usage_regularization_type: Literal["none", "entropy"] = "none"
    quantizer_type: QuantizerType = "vector"
    num_quantizers: int = 1
    groups: int = 1
    shared_codebook: bool = False
    stochastic_sample_codes: bool = False
    sample_codebook_temp: float = 0.0

    def __post_init__(self) -> None:
        """Validate scalar dimensions and dilation schedule."""
        positive_fields = {
            "data_dim": self.data_dim,
            "data_length": self.data_length,
            "embedding_dim": self.embedding_dim,
            "codebook_size": self.codebook_size,
            "encoder_hidden_dim": self.encoder_hidden_dim,
            "decoder_hidden_dim": self.decoder_hidden_dim,
            "num_layers": self.num_layers,
            "kmeans_iters": self.kmeans_iters,
            "num_quantizers": self.num_quantizers,
            "groups": self.groups,
        }
        for field_name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if self.commitment_weight < 0.0:
            raise ValueError("commitment_weight must be non-negative.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0.0 <= dropout < 1.0.")
        if self.condition_dim < 0:
            raise ValueError("condition_dim must be non-negative.")
        if self.codebook_dim is not None and self.codebook_dim <= 0:
            raise ValueError("codebook_dim must be positive when provided.")
        if self.threshold_ema_dead_code < 0.0:
            raise ValueError("threshold_ema_dead_code must be non-negative.")
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("decay must satisfy 0.0 < decay <= 1.0.")
        if self.usage_regularization_weight < 0.0:
            raise ValueError("usage_regularization_weight must be non-negative.")
        if self.usage_regularization_type not in {"none", "entropy"}:
            raise ValueError("usage_regularization_type must be 'none' or 'entropy'.")
        if self.usage_regularization_type == "none" and self.usage_regularization_weight != 0.0:
            raise ValueError(
                "usage_regularization_weight must be 0.0 when usage_regularization_type='none'."
            )
        if self.quantizer_type not in {"vector", "residual_vq", "grouped_residual_vq"}:
            raise ValueError(
                "quantizer_type must be 'vector', 'residual_vq', or 'grouped_residual_vq'."
            )
        if self.quantizer_type == "vector" and self.num_quantizers != 1:
            raise ValueError("num_quantizers must be 1 when quantizer_type='vector'.")
        if self.quantizer_type != "grouped_residual_vq" and self.groups != 1:
            raise ValueError("groups must be 1 unless quantizer_type='grouped_residual_vq'.")
        if self.quantizer_type == "grouped_residual_vq" and self.embedding_dim % self.groups != 0:
            raise ValueError("embedding_dim must be divisible by groups for GroupedResidualVQ.")
        if self.sample_codebook_temp < 0.0:
            raise ValueError("sample_codebook_temp must be non-negative.")
        if not self.dilations:
            raise ValueError("dilations must contain at least one value.")
        if any(dilation <= 0 for dilation in self.dilations):
            raise ValueError("all dilations must be positive.")

    @property
    def layer_dilations(self) -> tuple[int, ...]:
        """Return a dilation schedule with exactly ``num_layers`` entries."""
        dilation_values = tuple(self.dilations)
        repeats = (self.num_layers + len(dilation_values) - 1) // len(dilation_values)
        return (dilation_values * repeats)[: self.num_layers]
