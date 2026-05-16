"""Configuration objects for causal token priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CausalTokenPriorConfig:
    """Configuration for a causal autoregressive token prior.

    The prior receives tokenizer indices with shape ``[batch, sequence_length]``.
    With the default ``bos_shifted_next_token`` convention, the model builds the
    shifted input internally:

    ```text
    input tokens:  [BOS, k_0, k_1, ..., k_{T-2}]
    target tokens: [k_0, k_1, k_2, ..., k_{T-1}]
    ```
    """

    codebook_size: int
    sequence_length: int
    token_embedding_dim: int
    num_layers: int
    num_heads: int
    mlp_hidden_dim: int
    dropout: float
    bos_token_id: int | None = None
    pad_token_id: int | None = None
    prediction_convention: str = "bos_shifted_next_token"
    condition_dim: int = 0
    condition_injection: Literal["none", "additive", "adaln_lite"] = "none"
    condition_hidden_dim: int | None = None
    adaln_hidden_dim: int | None = None
    prior_type: Literal[
        "single_code",
        "factorised_multi_code",
        "hierarchical_rvq_q2",
        "separate_frequency_hierarchical",
    ] = "single_code"
    index_shape: list[int] | None = None
    num_quantizers: int = 1
    groups: int = 1
    component_loss_weights: list[float] | None = None
    low_codebook_size: int | None = None
    high_codebook_size: int | None = None

    def __post_init__(self) -> None:
        """Validate dimensions and token identifiers."""
        positive_fields = {
            "codebook_size": self.codebook_size,
            "sequence_length": self.sequence_length,
            "token_embedding_dim": self.token_embedding_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_hidden_dim": self.mlp_hidden_dim,
        }
        for field_name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if self.token_embedding_dim % self.num_heads != 0:
            raise ValueError("token_embedding_dim must be divisible by num_heads.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0.0 <= dropout < 1.0.")
        if self.condition_dim < 0:
            raise ValueError("condition_dim must be non-negative.")
        if self.condition_injection not in {"none", "additive", "adaln_lite"}:
            raise ValueError("condition_injection must be 'none', 'additive', or 'adaln_lite'.")
        if self.condition_injection == "none" and self.condition_dim != 0:
            raise ValueError("condition_dim must be 0 when condition_injection='none'.")
        if self.condition_injection == "additive" and self.condition_dim <= 0:
            raise ValueError("condition_dim must be positive for additive conditioning.")
        if self.condition_injection == "adaln_lite" and self.condition_dim <= 0:
            raise ValueError("condition_dim must be positive for AdaLN-lite conditioning.")
        if self.condition_hidden_dim is not None and self.condition_hidden_dim <= 0:
            raise ValueError("condition_hidden_dim must be positive when provided.")
        if self.adaln_hidden_dim is not None and self.adaln_hidden_dim <= 0:
            raise ValueError("adaln_hidden_dim must be positive when provided.")
        if self.prior_type not in {
            "single_code",
            "factorised_multi_code",
            "hierarchical_rvq_q2",
            "separate_frequency_hierarchical",
        }:
            raise ValueError(
                "prior_type must be 'single_code', 'factorised_multi_code', "
                "'hierarchical_rvq_q2', or 'separate_frequency_hierarchical'."
            )
        if self.num_quantizers <= 0:
            raise ValueError("num_quantizers must be positive.")
        if self.groups <= 0:
            raise ValueError("groups must be positive.")
        if self.index_shape is not None:
            if not self.index_shape:
                raise ValueError("index_shape must not be empty when provided.")
            if any(value <= 0 for value in self.index_shape):
                raise ValueError("index_shape values must be positive.")
            if self.index_shape[0] != self.sequence_length:
                raise ValueError("index_shape must start with sequence_length.")
        if self.prior_type == "single_code":
            if self.index_shape is not None and self.index_shape != [self.sequence_length]:
                raise ValueError("single_code index_shape must be [sequence_length] when provided.")
            if self.num_quantizers != 1:
                raise ValueError("num_quantizers must be 1 for single_code priors.")
            if self.groups != 1:
                raise ValueError("groups must be 1 for single_code priors.")
            if self.component_loss_weights is not None:
                raise ValueError("component_loss_weights is only valid for factorised priors.")
        elif self.prior_type == "factorised_multi_code":
            expected_shape = [self.sequence_length, *self.component_shape]
            if self.index_shape is not None and self.index_shape != expected_shape:
                raise ValueError(
                    "factorised_multi_code index_shape must match "
                    f"{expected_shape}; got {self.index_shape}."
                )
            if self.condition_injection == "adaln_lite":
                raise ValueError("AdaLN-lite is not supported for factorised_multi_code yet.")
            if self.component_loss_weights is not None:
                if len(self.component_loss_weights) != self.component_count:
                    raise ValueError(
                        "component_loss_weights length must equal the number of components."
                    )
                if any(weight < 0.0 for weight in self.component_loss_weights):
                    raise ValueError("component_loss_weights must be non-negative.")
                if sum(self.component_loss_weights) <= 0.0:
                    raise ValueError("component_loss_weights must contain positive mass.")
        elif self.prior_type == "hierarchical_rvq_q2":
            expected_shape = [self.sequence_length, 2]
            if self.groups != 1:
                raise ValueError("groups must be 1 for hierarchical_rvq_q2 priors.")
            if self.num_quantizers != 2:
                raise ValueError("num_quantizers must be 2 for hierarchical_rvq_q2 priors.")
            if self.index_shape is not None and self.index_shape != expected_shape:
                raise ValueError(
                    "hierarchical_rvq_q2 index_shape must match "
                    f"{expected_shape}; got {self.index_shape}."
                )
            if self.condition_injection == "adaln_lite":
                raise ValueError("AdaLN-lite is not supported for hierarchical_rvq_q2 yet.")
            if self.component_loss_weights is not None:
                if len(self.component_loss_weights) != 2:
                    raise ValueError(
                        "component_loss_weights length must be 2 for hierarchical_rvq_q2."
                    )
                if any(weight < 0.0 for weight in self.component_loss_weights):
                    raise ValueError("component_loss_weights must be non-negative.")
                if sum(self.component_loss_weights) <= 0.0:
                    raise ValueError("component_loss_weights must contain positive mass.")
        else:
            expected_shape = [self.sequence_length, 2]
            if self.groups != 1:
                raise ValueError("groups must be 1 for separate_frequency_hierarchical priors.")
            if self.num_quantizers != 1:
                raise ValueError(
                    "num_quantizers must remain 1 for separate_frequency_hierarchical priors."
                )
            if self.index_shape is not None and self.index_shape != expected_shape:
                raise ValueError(
                    "separate_frequency_hierarchical index_shape must match "
                    f"{expected_shape}; got {self.index_shape}."
                )
            if self.condition_injection == "adaln_lite":
                raise ValueError(
                    "AdaLN-lite is not supported for separate_frequency_hierarchical yet."
                )
            low_codebook_size = (
                self.codebook_size if self.low_codebook_size is None else self.low_codebook_size
            )
            high_codebook_size = (
                self.codebook_size if self.high_codebook_size is None else self.high_codebook_size
            )
            if low_codebook_size <= 0:
                raise ValueError("low_codebook_size must be positive.")
            if high_codebook_size <= 0:
                raise ValueError("high_codebook_size must be positive.")
            object.__setattr__(self, "low_codebook_size", low_codebook_size)
            object.__setattr__(self, "high_codebook_size", high_codebook_size)
            if self.component_loss_weights is not None:
                raise ValueError(
                    "component_loss_weights is not supported for "
                    "separate_frequency_hierarchical; use equal CE_low + CE_high."
                )
        if self.prediction_convention != "bos_shifted_next_token":
            raise ValueError("Only prediction_convention='bos_shifted_next_token' is supported.")
        if self.bos_token_id is None:
            object.__setattr__(self, "bos_token_id", self.codebook_size)
        bos_token_id = self.codebook_size if self.bos_token_id is None else self.bos_token_id
        if not 0 <= bos_token_id <= self.codebook_size:
            raise ValueError("bos_token_id must be in [0, codebook_size].")
        if self.pad_token_id is not None and not 0 <= self.pad_token_id <= self.codebook_size:
            raise ValueError("pad_token_id must be in [0, codebook_size] when provided.")

    @property
    def component_shape(self) -> tuple[int, ...]:
        """Return the non-time component shape for this prior."""
        if self.prior_type == "single_code":
            return ()
        if self.prior_type == "separate_frequency_hierarchical":
            return (2,)
        if self.groups == 1:
            return (self.num_quantizers,)
        return (self.groups, self.num_quantizers)

    @property
    def component_count(self) -> int:
        """Return the number of factorised categorical heads."""
        count = 1
        for value in self.component_shape:
            count *= value
        return count
