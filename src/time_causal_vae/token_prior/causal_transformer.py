"""Causal autoregressive transformer priors over tokenizer indices.

References
----------
    [vqvae_2017], [timevqvae_2023], [chronos_2024] in docs/references.md.
Borrowed idea:
    Model discrete time-series tokens with a causal next-token objective and scalar conditioning.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from time_causal_vae.token_prior.config import CausalTokenPriorConfig
from time_causal_vae.token_prior.masks import causal_attention_mask
from time_causal_vae.utils.output import ModelOutput


def build_token_prior_model(
    config: CausalTokenPriorConfig,
) -> (
    CausalTokenTransformerPrior
    | FactorisedMultiCodeTokenPrior
    | HierarchicalRVQ2TokenPrior
    | SeparateFrequencyHierarchicalPrior
):
    """Build the configured token-prior module."""
    if config.prior_type == "single_code":
        return CausalTokenTransformerPrior(config)
    if config.prior_type == "factorised_multi_code":
        return FactorisedMultiCodeTokenPrior(config)
    if config.prior_type == "hierarchical_rvq_q2":
        return HierarchicalRVQ2TokenPrior(config)
    if config.prior_type == "separate_frequency_hierarchical":
        return SeparateFrequencyHierarchicalPrior(config)
    raise ValueError(f"Unsupported prior_type={config.prior_type!r}.")


class CausalTokenTransformerPrior(nn.Module):
    """Decoder-only causal transformer prior for discrete tokenizer indices.

    Inputs and samples use tokenizer-code indices with shape
    ``[batch, sequence_length]``. The module builds the BOS-shifted input
    internally and returns logits with shape
    ``[batch, sequence_length, codebook_size]``.
    """

    def __init__(self, config: CausalTokenPriorConfig) -> None:
        """Initialise token embeddings, causal transformer, and projection."""
        super().__init__()
        if config.prior_type != "single_code":
            raise ValueError("CausalTokenTransformerPrior requires prior_type='single_code'.")
        self.config = config
        self.input_vocab_size = config.codebook_size + 1
        self.token_embedding = nn.Embedding(self.input_vocab_size, config.token_embedding_dim)
        self.position_embedding = nn.Embedding(config.sequence_length, config.token_embedding_dim)
        self.condition_projection = build_condition_projection(config)
        self.transformer: nn.TransformerEncoder | None = None
        self.adaln_blocks = nn.ModuleList()
        if config.condition_injection == "adaln_lite":
            self.adaln_blocks = nn.ModuleList(
                [AdaLNCausalTransformerBlock(config) for _ in range(config.num_layers)]
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.token_embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.mlp_hidden_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=config.num_layers,
            )
        self.output_projection = nn.Linear(config.token_embedding_dim, config.codebook_size)

    def forward(
        self,
        tokens: Tensor,
        targets: Tensor | None = None,
        conditions: Tensor | None = None,
    ) -> ModelOutput:
        """Return logits and next-token training metrics.

        ``tokens`` must have shape ``[batch, sequence_length]`` and contain
        tokenizer code indices. If ``targets`` is omitted, the original
        ``tokens`` tensor is used as the target sequence under the
        BOS-shifted teacher-forcing convention.

        Optional ``conditions`` may be provided when conditioning is enabled.
        Scalar conditions have shape ``[batch, condition_dim]`` and are
        repeated over token positions. Temporal conditions have shape
        ``[batch, sequence_length, condition_dim]`` and are consumed per
        position. Temporal condition causality is enforced by the same causal
        transformer mask as token causality.
        """
        validate_token_sequence(tokens, self.config, tensor_name="tokens")
        target_tokens = tokens if targets is None else targets
        validate_token_sequence(target_tokens, self.config, tensor_name="targets")

        shifted_inputs = self.build_shifted_inputs(tokens)
        logits = self.logits_from_shifted_inputs(shifted_inputs, conditions=conditions)
        cross_entropy = token_cross_entropy(
            logits,
            target_tokens,
            pad_token_id=self.config.pad_token_id,
        )
        accuracy = token_accuracy(
            logits,
            target_tokens,
            pad_token_id=self.config.pad_token_id,
        )
        perplexity = torch.exp(cross_entropy.detach())
        return ModelOutput(
            logits=logits,
            loss=cross_entropy,
            cross_entropy=cross_entropy,
            accuracy=accuracy,
            perplexity=perplexity,
        )

    def build_shifted_inputs(self, tokens: Tensor) -> Tensor:
        """Build ``[BOS, k_0, ..., k_{T-2}]`` inputs from target tokens."""
        validate_token_sequence(tokens, self.config, tensor_name="tokens")
        batch_size = tokens.shape[0]
        bos_token_id = self.config.codebook_size
        if self.config.bos_token_id is not None:
            bos_token_id = self.config.bos_token_id
        bos_tokens = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        return torch.cat([bos_tokens, tokens[:, :-1]], dim=1)

    def logits_from_shifted_inputs(
        self,
        shifted_inputs: Tensor,
        *,
        conditions: Tensor | None = None,
    ) -> Tensor:
        """Return logits from already shifted token inputs."""
        validate_shifted_sequence(shifted_inputs, self.config)
        batch_size, sequence_length = shifted_inputs.shape
        positions = torch.arange(sequence_length, device=shifted_inputs.device)
        positions = positions.unsqueeze(0).expand(batch_size, sequence_length)
        hidden = self.token_embedding(shifted_inputs) + self.position_embedding(positions)
        condition_sequence = self.prepare_conditions(
            conditions,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=shifted_inputs.device,
            dtype=hidden.dtype,
        )
        if self.config.condition_injection == "additive":
            hidden = hidden + self.condition_embedding(condition_sequence)
        attention_mask = causal_attention_mask(
            sequence_length,
            device=shifted_inputs.device,
            dtype=hidden.dtype,
        )
        if self.config.condition_injection == "adaln_lite":
            encoded = self.encode_with_adaln(hidden, condition_sequence, attention_mask)
        else:
            if self.transformer is None:
                raise RuntimeError("transformer encoder is required for this condition mode.")
            encoded = self.transformer(hidden, mask=attention_mask)
        return cast(Tensor, self.output_projection(encoded))

    def prepare_conditions(
        self,
        conditions: Tensor | None,
        *,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        """Return prepared condition sequences or ``None`` for unconditional mode."""
        if self.config.condition_injection == "none":
            if conditions is not None:
                raise ValueError("conditions were provided but condition_injection='none'.")
            return None
        return prepare_condition_sequence(
            conditions,
            config=self.config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=device,
            dtype=dtype,
        )

    def condition_embedding(self, condition_sequence: Tensor | None) -> Tensor:
        """Return additive condition embeddings."""
        if condition_sequence is None:
            raise RuntimeError("condition_sequence is required for additive conditioning.")
        if self.condition_projection is None:
            raise RuntimeError("condition_projection is required for additive conditioning.")
        return cast(Tensor, self.condition_projection(condition_sequence))

    def encode_with_adaln(
        self,
        hidden: Tensor,
        condition_sequence: Tensor | None,
        attention_mask: Tensor,
    ) -> Tensor:
        """Encode with AdaLN-lite causal blocks."""
        if condition_sequence is None:
            raise RuntimeError("condition_sequence is required for AdaLN-lite conditioning.")
        encoded = hidden
        for block in self.adaln_blocks:
            encoded = block(encoded, condition_sequence, attention_mask)
        return encoded

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        temperature: float = 1.0,
        top_k: int | None = None,
        conditions: Tensor | None = None,
    ) -> Tensor:
        """Generate token sequences from left to right.

        Sampling uses temperature and optional top-k filtering only. It does
        not implement diffusion, masked iterative refinement, or transition
        constraints.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if top_k is not None and not 0 < top_k <= self.config.codebook_size:
            raise ValueError("top_k must satisfy 0 < top_k <= codebook_size.")

        sample_device = torch.device(device)
        prepared_conditions = None
        if self.config.condition_injection != "none":
            prepared_conditions = prepare_condition_sequence(
                conditions,
                config=self.config,
                batch_size=batch_size,
                sequence_length=self.config.sequence_length,
                device=sample_device,
                dtype=torch.float32,
            )
        elif conditions is not None:
            raise ValueError("conditions were provided but condition_injection='none'.")
        was_training = self.training
        self.eval()
        try:
            generated = torch.empty(
                (batch_size, 0),
                dtype=torch.long,
                device=sample_device,
            )
            for position in range(self.config.sequence_length):
                prefix_inputs = torch.full(
                    (batch_size, self.config.sequence_length),
                    fill_value=0,
                    dtype=torch.long,
                    device=sample_device,
                )
                if position > 0:
                    prefix_inputs[:, :position] = generated
                logits = cast(Tensor, self(prefix_inputs, conditions=prepared_conditions).logits)
                next_logits = logits[:, position, :] / temperature
                next_token = sample_from_logits(next_logits, top_k=top_k)
                generated = torch.cat([generated, next_token[:, None]], dim=1)
        finally:
            self.train(was_training)
        return generated


class FactorisedMultiCodeTokenPrior(nn.Module):
    """Causal AR prior with factorised output heads for multi-code tokens.

    Inputs use native multi-index tokenizer layout, for example ResidualVQ q2
    tokens with shape ``[batch, sequence_length, num_quantizers]``. The model
    applies the same calendar-time shifted convention as the single-code prior:
    at time ``t`` it receives the full code block from ``t - 1`` and predicts
    all code components for time ``t`` from a shared causal transformer trunk.
    """

    def __init__(self, config: CausalTokenPriorConfig) -> None:
        """Initialise per-component embeddings, shared trunk, and output heads."""
        super().__init__()
        if config.prior_type != "factorised_multi_code":
            raise ValueError(
                "FactorisedMultiCodeTokenPrior requires prior_type='factorised_multi_code'."
            )
        if config.component_count <= 1:
            raise ValueError("factorised_multi_code requires at least two components.")
        self.config = config
        self.input_vocab_size = config.codebook_size + 1
        self.component_embeddings = nn.ModuleList(
            [
                nn.Embedding(self.input_vocab_size, config.token_embedding_dim)
                for _ in range(config.component_count)
            ]
        )
        self.position_embedding = nn.Embedding(config.sequence_length, config.token_embedding_dim)
        self.condition_projection = build_condition_projection(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.token_embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
        )
        self.component_names = component_names(config)
        self.output_heads = nn.ModuleList(
            [
                nn.Linear(config.token_embedding_dim, config.codebook_size)
                for _component_name in self.component_names
            ]
        )
        weights = component_loss_weights(config)
        self.register_buffer("component_loss_weights", weights, persistent=False)

    def forward(
        self,
        tokens: Tensor,
        targets: Tensor | None = None,
        conditions: Tensor | None = None,
    ) -> ModelOutput:
        """Return factorised component logits and aggregate training metrics."""
        validate_multicode_sequence(tokens, self.config, tensor_name="tokens")
        target_tokens = tokens if targets is None else targets
        validate_multicode_sequence(target_tokens, self.config, tensor_name="targets")

        shifted_inputs = self.build_shifted_inputs(tokens)
        logits = self.logits_from_shifted_inputs(shifted_inputs, conditions=conditions)
        target_components = flatten_components(target_tokens, self.config)
        component_losses: list[Tensor] = []
        component_accuracies: list[Tensor] = []
        logits_by_component: dict[str, Tensor] = {}
        output_payload: dict[str, Tensor] = {}
        for component_index, component_name in enumerate(self.component_names):
            component_logits = logits[:, :, component_index, :]
            component_targets = target_components[:, :, component_index]
            ce = token_cross_entropy(
                component_logits,
                component_targets,
                pad_token_id=self.config.pad_token_id,
            )
            accuracy = token_accuracy(
                component_logits,
                component_targets,
                pad_token_id=self.config.pad_token_id,
            )
            component_losses.append(ce)
            component_accuracies.append(accuracy)
            logits_by_component[component_name] = component_logits
            output_payload[f"component_cross_entropy_{component_name}"] = ce
            output_payload[f"component_accuracy_{component_name}"] = accuracy
            output_payload[f"component_perplexity_{component_name}"] = torch.exp(ce.detach())

        weights = cast(Tensor, self.component_loss_weights).to(
            device=tokens.device,
            dtype=logits.dtype,
        )
        loss = torch.stack(component_losses).mul(weights).sum()
        accuracy = torch.stack(component_accuracies).mul(weights).sum()
        perplexity = torch.exp(loss.detach())
        return ModelOutput(
            logits=logits,
            logits_by_component=logits_by_component,
            loss=loss,
            cross_entropy=loss,
            accuracy=accuracy,
            perplexity=perplexity,
            **output_payload,
        )

    def build_shifted_inputs(self, tokens: Tensor) -> Tensor:
        """Build BOS-block shifted inputs from target multi-code tokens."""
        validate_multicode_sequence(tokens, self.config, tensor_name="tokens")
        batch_size = tokens.shape[0]
        bos_token_id = self.config.codebook_size
        if self.config.bos_token_id is not None:
            bos_token_id = self.config.bos_token_id
        bos_shape = (batch_size, 1, *self.config.component_shape)
        bos_tokens = torch.full(
            bos_shape,
            bos_token_id,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        return torch.cat([bos_tokens, tokens[:, :-1]], dim=1)

    def logits_from_shifted_inputs(
        self,
        shifted_inputs: Tensor,
        *,
        conditions: Tensor | None = None,
    ) -> Tensor:
        """Return ``[batch, time, component, codebook]`` logits."""
        validate_multicode_shifted_sequence(shifted_inputs, self.config)
        batch_size, sequence_length = shifted_inputs.shape[:2]
        component_inputs = flatten_components(shifted_inputs, self.config)
        hidden = self.component_embeddings[0](component_inputs[:, :, 0])
        for component_index in range(1, self.config.component_count):
            hidden = hidden + self.component_embeddings[component_index](
                component_inputs[:, :, component_index]
            )
        positions = torch.arange(sequence_length, device=shifted_inputs.device)
        positions = positions.unsqueeze(0).expand(batch_size, sequence_length)
        hidden = hidden + self.position_embedding(positions)
        condition_sequence = prepare_condition_sequence(
            conditions,
            config=self.config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=shifted_inputs.device,
            dtype=hidden.dtype,
        )
        if self.config.condition_injection == "additive":
            if self.condition_projection is None:
                raise RuntimeError("condition_projection is required for additive conditioning.")
            hidden = hidden + cast(Tensor, self.condition_projection(condition_sequence))
        attention_mask = causal_attention_mask(
            sequence_length,
            device=shifted_inputs.device,
            dtype=hidden.dtype,
        )
        encoded = self.transformer(hidden, mask=attention_mask)
        component_logits = [head(encoded) for head in self.output_heads]
        return torch.stack(component_logits, dim=2)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        temperature: float = 1.0,
        top_k: int | None = None,
        conditions: Tensor | None = None,
    ) -> Tensor:
        """Sample native multi-code tensors from left to right."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if top_k is not None and not 0 < top_k <= self.config.codebook_size:
            raise ValueError("top_k must satisfy 0 < top_k <= codebook_size.")

        sample_device = torch.device(device)
        prepared_conditions = None
        if self.config.condition_injection != "none":
            prepared_conditions = prepare_condition_sequence(
                conditions,
                config=self.config,
                batch_size=batch_size,
                sequence_length=self.config.sequence_length,
                device=sample_device,
                dtype=torch.float32,
            )
        elif conditions is not None:
            raise ValueError("conditions were provided but condition_injection='none'.")

        was_training = self.training
        self.eval()
        try:
            generated = torch.empty(
                (batch_size, 0, *self.config.component_shape),
                dtype=torch.long,
                device=sample_device,
            )
            for position in range(self.config.sequence_length):
                prefix_inputs = torch.zeros(
                    (batch_size, self.config.sequence_length, *self.config.component_shape),
                    dtype=torch.long,
                    device=sample_device,
                )
                if position > 0:
                    prefix_inputs[:, :position] = generated
                logits = cast(
                    Tensor,
                    self(prefix_inputs, conditions=prepared_conditions).logits,
                )
                next_component_tokens = []
                for component_index in range(self.config.component_count):
                    component_logits = logits[:, position, component_index, :] / temperature
                    next_component_tokens.append(sample_from_logits(component_logits, top_k=top_k))
                next_block = torch.stack(next_component_tokens, dim=1).reshape(
                    batch_size,
                    1,
                    *self.config.component_shape,
                )
                generated = torch.cat([generated, next_block], dim=1)
        finally:
            self.train(was_training)
        return generated


class HierarchicalRVQ2TokenPrior(nn.Module):
    """Causal RVQ q2 prior with within-time hierarchical q0 then q1 heads.

    The calendar-time trunk receives only the previous time block at each
    position. The q0 head predicts the first RVQ component from the trunk
    hidden state. The q1 head predicts the second component from the same
    hidden state plus a teacher-forced or sampled same-time q0 embedding.
    """

    def __init__(self, config: CausalTokenPriorConfig) -> None:
        """Initialise previous-block embeddings, shared trunk, and hierarchical heads."""
        super().__init__()
        if config.prior_type != "hierarchical_rvq_q2":
            raise ValueError(
                "HierarchicalRVQ2TokenPrior requires prior_type='hierarchical_rvq_q2'."
            )
        if config.groups != 1 or config.num_quantizers != 2:
            raise ValueError("hierarchical_rvq_q2 requires groups=1 and num_quantizers=2.")
        self.config = config
        self.input_vocab_size = config.codebook_size + 1
        self.component_embeddings = nn.ModuleList(
            [
                nn.Embedding(self.input_vocab_size, config.token_embedding_dim)
                for _component_index in range(2)
            ]
        )
        self.position_embedding = nn.Embedding(config.sequence_length, config.token_embedding_dim)
        self.condition_projection = build_condition_projection(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.token_embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
        )
        self.q0_head = nn.Linear(config.token_embedding_dim, config.codebook_size)
        self.q0_condition_embedding = nn.Embedding(
            config.codebook_size,
            config.token_embedding_dim,
        )
        self.q1_head = nn.Linear(config.token_embedding_dim, config.codebook_size)
        self.component_names = ["q0", "q1"]
        weights = component_loss_weights(config)
        self.register_buffer("component_loss_weights", weights, persistent=False)

    def forward(
        self,
        tokens: Tensor,
        targets: Tensor | None = None,
        conditions: Tensor | None = None,
    ) -> ModelOutput:
        """Return hierarchical q0/q1 logits and training metrics."""
        validate_multicode_sequence(tokens, self.config, tensor_name="tokens")
        target_tokens = tokens if targets is None else targets
        validate_multicode_sequence(target_tokens, self.config, tensor_name="targets")

        shifted_inputs = self.build_shifted_inputs(tokens)
        target_components = flatten_components(target_tokens, self.config)
        q0_targets = target_components[:, :, 0]
        q1_targets = target_components[:, :, 1]
        logits = self.logits_from_shifted_inputs(
            shifted_inputs,
            q0_tokens=q0_targets,
            conditions=conditions,
        )
        q0_logits = logits[:, :, 0, :]
        q1_logits = logits[:, :, 1, :]

        q0_ce = token_cross_entropy(
            q0_logits,
            q0_targets,
            pad_token_id=self.config.pad_token_id,
        )
        q1_ce = token_cross_entropy(
            q1_logits,
            q1_targets,
            pad_token_id=self.config.pad_token_id,
        )
        q0_accuracy = token_accuracy(
            q0_logits,
            q0_targets,
            pad_token_id=self.config.pad_token_id,
        )
        q1_accuracy = token_accuracy(
            q1_logits,
            q1_targets,
            pad_token_id=self.config.pad_token_id,
        )
        weights = cast(Tensor, self.component_loss_weights).to(
            device=tokens.device,
            dtype=logits.dtype,
        )
        loss = torch.stack([q0_ce, q1_ce]).mul(weights).sum()
        accuracy = torch.stack([q0_accuracy, q1_accuracy]).mul(weights).sum()
        perplexity = torch.exp(loss.detach())
        pair_perplexity = same_time_pair_perplexity(target_tokens, self.config)
        return ModelOutput(
            logits=logits,
            logits_by_component={"q0": q0_logits, "q1": q1_logits},
            loss=loss,
            cross_entropy=loss,
            accuracy=accuracy,
            perplexity=perplexity,
            component_cross_entropy_q0=q0_ce,
            component_accuracy_q0=q0_accuracy,
            component_perplexity_q0=torch.exp(q0_ce.detach()),
            component_cross_entropy_q1=q1_ce,
            component_accuracy_q1=q1_accuracy,
            component_perplexity_q1=torch.exp(q1_ce.detach()),
            same_time_pair_perplexity=pair_perplexity,
        )

    def build_shifted_inputs(self, tokens: Tensor) -> Tensor:
        """Build BOS-block shifted inputs from target RVQ q2 tokens."""
        validate_multicode_sequence(tokens, self.config, tensor_name="tokens")
        batch_size = tokens.shape[0]
        bos_token_id = self.config.codebook_size
        if self.config.bos_token_id is not None:
            bos_token_id = self.config.bos_token_id
        bos_tokens = torch.full(
            (batch_size, 1, 2),
            bos_token_id,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        return torch.cat([bos_tokens, tokens[:, :-1]], dim=1)

    def encode_shifted_inputs(
        self,
        shifted_inputs: Tensor,
        *,
        conditions: Tensor | None = None,
    ) -> Tensor:
        """Encode BOS-shifted previous-time RVQ blocks with a causal trunk."""
        validate_multicode_shifted_sequence(shifted_inputs, self.config)
        batch_size, sequence_length = shifted_inputs.shape[:2]
        component_inputs = flatten_components(shifted_inputs, self.config)
        hidden = self.component_embeddings[0](component_inputs[:, :, 0])
        hidden = hidden + self.component_embeddings[1](component_inputs[:, :, 1])
        positions = torch.arange(sequence_length, device=shifted_inputs.device)
        positions = positions.unsqueeze(0).expand(batch_size, sequence_length)
        hidden = hidden + self.position_embedding(positions)
        condition_sequence = prepare_condition_sequence(
            conditions,
            config=self.config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=shifted_inputs.device,
            dtype=hidden.dtype,
        )
        if self.config.condition_injection == "additive":
            if self.condition_projection is None:
                raise RuntimeError("condition_projection is required for additive conditioning.")
            hidden = hidden + cast(Tensor, self.condition_projection(condition_sequence))
        attention_mask = causal_attention_mask(
            sequence_length,
            device=shifted_inputs.device,
            dtype=hidden.dtype,
        )
        return cast(Tensor, self.transformer(hidden, mask=attention_mask))

    def logits_from_shifted_inputs(
        self,
        shifted_inputs: Tensor,
        *,
        q0_tokens: Tensor,
        conditions: Tensor | None = None,
    ) -> Tensor:
        """Return ``[batch, time, 2, codebook]`` logits conditioned on q0 tokens."""
        validate_token_sequence(q0_tokens, self.config, tensor_name="q0_tokens")
        encoded = self.encode_shifted_inputs(shifted_inputs, conditions=conditions)
        q0_logits = self.q0_head(encoded)
        q1_hidden = encoded + self.q0_condition_embedding(q0_tokens)
        q1_logits = self.q1_head(q1_hidden)
        return torch.stack([q0_logits, q1_logits], dim=2)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        temperature: float = 1.0,
        top_k: int | None = None,
        conditions: Tensor | None = None,
    ) -> Tensor:
        """Sample RVQ q2 blocks by drawing q0 first, then q1 conditioned on q0."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if top_k is not None and not 0 < top_k <= self.config.codebook_size:
            raise ValueError("top_k must satisfy 0 < top_k <= codebook_size.")

        sample_device = torch.device(device)
        prepared_conditions = None
        if self.config.condition_injection != "none":
            prepared_conditions = prepare_condition_sequence(
                conditions,
                config=self.config,
                batch_size=batch_size,
                sequence_length=self.config.sequence_length,
                device=sample_device,
                dtype=torch.float32,
            )
        elif conditions is not None:
            raise ValueError("conditions were provided but condition_injection='none'.")

        was_training = self.training
        self.eval()
        try:
            generated = torch.empty(
                (batch_size, 0, 2),
                dtype=torch.long,
                device=sample_device,
            )
            for position in range(self.config.sequence_length):
                prefix_inputs = torch.zeros(
                    (batch_size, self.config.sequence_length, 2),
                    dtype=torch.long,
                    device=sample_device,
                )
                if position > 0:
                    prefix_inputs[:, :position] = generated
                encoded = self.encode_shifted_inputs(
                    prefix_inputs,
                    conditions=prepared_conditions,
                )
                current_hidden = encoded[:, position, :]
                q0_logits = self.q0_head(current_hidden) / temperature
                q0_tokens = sample_from_logits(q0_logits, top_k=top_k)
                q1_hidden = current_hidden + self.q0_condition_embedding(q0_tokens)
                q1_logits = self.q1_head(q1_hidden) / temperature
                q1_tokens = sample_from_logits(q1_logits, top_k=top_k)
                next_block = torch.stack([q0_tokens, q1_tokens], dim=1)[:, None, :]
                generated = torch.cat([generated, next_block], dim=1)
        finally:
            self.train(was_training)
        return generated


class SeparateFrequencyHierarchicalPrior(nn.Module):
    """Hierarchical causal prior for separate low/high frequency token streams.

    The shared calendar-time trunk receives only the previous low/high token
    block at each position. The low head predicts ``low_t`` from the trunk
    hidden state. The high head predicts ``high_t`` from the same hidden state
    plus a same-time low-token embedding. Training uses teacher-forced true
    low tokens; sampling uses sampled low tokens.
    """

    def __init__(self, config: CausalTokenPriorConfig) -> None:
        """Initialise previous-block embeddings, shared trunk, and stream heads."""
        super().__init__()
        if config.prior_type != "separate_frequency_hierarchical":
            raise ValueError(
                "SeparateFrequencyHierarchicalPrior requires "
                "prior_type='separate_frequency_hierarchical'."
            )
        if config.low_codebook_size is None or config.high_codebook_size is None:
            raise ValueError("Separate frequency priors require low/high codebook sizes.")
        self.config = config
        self.low_codebook_size = config.low_codebook_size
        self.high_codebook_size = config.high_codebook_size
        self.low_input_vocab_size = self.low_codebook_size + 1
        self.high_input_vocab_size = self.high_codebook_size + 1
        self.low_previous_embedding = nn.Embedding(
            self.low_input_vocab_size,
            config.token_embedding_dim,
        )
        self.high_previous_embedding = nn.Embedding(
            self.high_input_vocab_size,
            config.token_embedding_dim,
        )
        self.position_embedding = nn.Embedding(config.sequence_length, config.token_embedding_dim)
        self.condition_projection = build_condition_projection(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.token_embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
        )
        self.low_head = nn.Linear(config.token_embedding_dim, self.low_codebook_size)
        self.current_low_embedding = nn.Embedding(
            self.low_codebook_size,
            config.token_embedding_dim,
        )
        self.high_head = nn.Linear(config.token_embedding_dim, self.high_codebook_size)

    def forward(
        self,
        tokens: Tensor,
        targets: Tensor | None = None,
        conditions: Tensor | None = None,
    ) -> ModelOutput:
        """Return low/high logits and hierarchical training metrics."""
        validate_separate_frequency_sequence(tokens, self.config, tensor_name="tokens")
        target_tokens = tokens if targets is None else targets
        validate_separate_frequency_sequence(target_tokens, self.config, tensor_name="targets")

        shifted_inputs = self.build_shifted_inputs(tokens)
        low_targets = target_tokens[:, :, 0]
        high_targets = target_tokens[:, :, 1]
        low_logits, high_logits = self.logits_from_shifted_inputs(
            shifted_inputs,
            current_low_tokens=low_targets,
            conditions=conditions,
        )
        low_ce = token_cross_entropy(
            low_logits,
            low_targets,
            pad_token_id=self.config.pad_token_id,
        )
        high_ce = token_cross_entropy(
            high_logits,
            high_targets,
            pad_token_id=self.config.pad_token_id,
        )
        low_accuracy = token_accuracy(
            low_logits,
            low_targets,
            pad_token_id=self.config.pad_token_id,
        )
        high_accuracy = token_accuracy(
            high_logits,
            high_targets,
            pad_token_id=self.config.pad_token_id,
        )
        loss = low_ce + high_ce
        accuracy = (low_accuracy + high_accuracy) * 0.5
        pair_perplexity = separate_frequency_pair_perplexity(target_tokens, self.config)
        return ModelOutput(
            logits_by_component={"low": low_logits, "high": high_logits},
            low_logits=low_logits,
            high_logits=high_logits,
            loss=loss,
            cross_entropy=loss,
            accuracy=accuracy,
            perplexity=torch.exp(loss.detach()),
            component_cross_entropy_low=low_ce,
            component_accuracy_low=low_accuracy,
            component_perplexity_low=torch.exp(low_ce.detach()),
            component_cross_entropy_high=high_ce,
            component_accuracy_high=high_accuracy,
            component_perplexity_high=torch.exp(high_ce.detach()),
            same_time_pair_perplexity=pair_perplexity,
        )

    def build_shifted_inputs(self, tokens: Tensor) -> Tensor:
        """Build BOS-block shifted inputs from target low/high tokens."""
        validate_separate_frequency_sequence(tokens, self.config, tensor_name="tokens")
        batch_size = tokens.shape[0]
        low_bos = self.low_codebook_size
        high_bos = self.high_codebook_size
        bos_tokens = torch.empty(
            (batch_size, 1, 2),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        bos_tokens[:, :, 0] = low_bos
        bos_tokens[:, :, 1] = high_bos
        return torch.cat([bos_tokens, tokens[:, :-1]], dim=1)

    def encode_shifted_inputs(
        self,
        shifted_inputs: Tensor,
        *,
        conditions: Tensor | None = None,
    ) -> Tensor:
        """Encode BOS-shifted previous-time low/high blocks with a causal trunk."""
        validate_separate_frequency_shifted_sequence(shifted_inputs, self.config)
        batch_size, sequence_length = shifted_inputs.shape[:2]
        low_inputs = shifted_inputs[:, :, 0]
        high_inputs = shifted_inputs[:, :, 1]
        hidden = self.low_previous_embedding(low_inputs) + self.high_previous_embedding(high_inputs)
        positions = torch.arange(sequence_length, device=shifted_inputs.device)
        positions = positions.unsqueeze(0).expand(batch_size, sequence_length)
        hidden = hidden + self.position_embedding(positions)
        condition_sequence = prepare_condition_sequence(
            conditions,
            config=self.config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=shifted_inputs.device,
            dtype=hidden.dtype,
        )
        if self.config.condition_injection == "additive":
            if self.condition_projection is None:
                raise RuntimeError("condition_projection is required for additive conditioning.")
            hidden = hidden + cast(Tensor, self.condition_projection(condition_sequence))
        attention_mask = causal_attention_mask(
            sequence_length,
            device=shifted_inputs.device,
            dtype=hidden.dtype,
        )
        return cast(Tensor, self.transformer(hidden, mask=attention_mask))

    def logits_from_shifted_inputs(
        self,
        shifted_inputs: Tensor,
        *,
        current_low_tokens: Tensor,
        conditions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return low and high logits using teacher-forced current low tokens."""
        validate_token_sequence(current_low_tokens, self.config, tensor_name="current_low_tokens")
        if int(current_low_tokens.max().item()) >= self.low_codebook_size:
            raise ValueError(
                f"current_low_tokens values must be in [0, {self.low_codebook_size - 1}]."
            )
        encoded = self.encode_shifted_inputs(shifted_inputs, conditions=conditions)
        low_logits = self.low_head(encoded)
        high_hidden = encoded + self.current_low_embedding(current_low_tokens)
        high_logits = self.high_head(high_hidden)
        return low_logits, high_logits

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        temperature: float = 1.0,
        top_k: int | None = None,
        conditions: Tensor | None = None,
    ) -> Tensor:
        """Sample packed ``[batch, time, 2]`` low/high token blocks."""
        streams = self.sample_streams(
            batch_size=batch_size,
            device=device,
            temperature=temperature,
            top_k=top_k,
            conditions=conditions,
        )
        return cast(Tensor, streams["sampled_tokens"])

    @torch.no_grad()
    def sample_streams(
        self,
        batch_size: int,
        device: torch.device | str,
        temperature: float = 1.0,
        top_k: int | None = None,
        conditions: Tensor | None = None,
    ) -> ModelOutput:
        """Sample low tokens first, then high tokens conditioned on sampled low."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if top_k is not None:
            max_top_k = min(self.low_codebook_size, self.high_codebook_size)
            if not 0 < top_k <= max_top_k:
                raise ValueError("top_k must satisfy both stream codebook sizes.")

        sample_device = torch.device(device)
        prepared_conditions = None
        if self.config.condition_injection != "none":
            prepared_conditions = prepare_condition_sequence(
                conditions,
                config=self.config,
                batch_size=batch_size,
                sequence_length=self.config.sequence_length,
                device=sample_device,
                dtype=torch.float32,
            )
        elif conditions is not None:
            raise ValueError("conditions were provided but condition_injection='none'.")

        was_training = self.training
        self.eval()
        try:
            generated = torch.empty(
                (batch_size, 0, 2),
                dtype=torch.long,
                device=sample_device,
            )
            for position in range(self.config.sequence_length):
                prefix_inputs = torch.zeros(
                    (batch_size, self.config.sequence_length, 2),
                    dtype=torch.long,
                    device=sample_device,
                )
                if position > 0:
                    prefix_inputs[:, :position] = generated
                encoded = self.encode_shifted_inputs(
                    prefix_inputs,
                    conditions=prepared_conditions,
                )
                current_hidden = encoded[:, position, :]
                low_logits = self.low_head(current_hidden) / temperature
                low_tokens = sample_from_logits(low_logits, top_k=top_k)
                high_hidden = current_hidden + self.current_low_embedding(low_tokens)
                high_logits = self.high_head(high_hidden) / temperature
                high_tokens = sample_from_logits(high_logits, top_k=top_k)
                next_block = torch.stack([low_tokens, high_tokens], dim=1)[:, None, :]
                generated = torch.cat([generated, next_block], dim=1)
        finally:
            self.train(was_training)
        return ModelOutput(
            sampled_tokens=generated,
            sampled_low_tokens=generated[:, :, 0],
            sampled_high_tokens=generated[:, :, 1],
        )


class AdaLNCausalTransformerBlock(nn.Module):
    """Causal transformer block with scalar-condition AdaLN-lite modulation.

    The block keeps self-attention causal via the supplied additive attention
    mask. Normalisation is per token over channels, so it does not aggregate
    statistics across future time steps. The condition network emits per-token
    channel scale and shift for the attention and feed-forward sublayers.
    """

    def __init__(self, config: CausalTokenPriorConfig) -> None:
        """Initialise attention, feed-forward, and condition modulation layers."""
        super().__init__()
        hidden_dim = config.token_embedding_dim
        condition_hidden_dim = (
            config.adaln_hidden_dim
            if config.adaln_hidden_dim is not None
            else config.condition_hidden_dim
            if config.condition_hidden_dim is not None
            else hidden_dim
        )
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.norm_mlp = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, config.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, hidden_dim),
        )
        self.dropout_attention = nn.Dropout(config.dropout)
        self.dropout_mlp = nn.Dropout(config.dropout)
        self.condition_modulation = nn.Sequential(
            nn.Linear(config.condition_dim, condition_hidden_dim),
            nn.GELU(),
            nn.Linear(condition_hidden_dim, hidden_dim * 4),
        )
        final_linear = self.condition_modulation[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def forward(self, hidden: Tensor, condition_sequence: Tensor, attention_mask: Tensor) -> Tensor:
        """Return causally encoded hidden states."""
        attn_scale, attn_shift, mlp_scale, mlp_shift = self.modulation_parameters(
            condition_sequence
        )
        attention_input = modulate_adaln(
            self.norm_attention(hidden),
            scale=attn_scale,
            shift=attn_shift,
        )
        attention_output, _weights = self.self_attention(
            attention_input,
            attention_input,
            attention_input,
            attn_mask=attention_mask,
            need_weights=False,
        )
        hidden = hidden + self.dropout_attention(attention_output)
        mlp_input = modulate_adaln(
            self.norm_mlp(hidden),
            scale=mlp_scale,
            shift=mlp_shift,
        )
        mlp_output = cast(Tensor, self.mlp(mlp_input))
        return cast(Tensor, hidden + self.dropout_mlp(mlp_output))

    def modulation_parameters(
        self, condition_sequence: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return per-token scale and shift tensors for both sublayers."""
        raw = cast(Tensor, self.condition_modulation(condition_sequence))
        attn_scale, attn_shift, mlp_scale, mlp_shift = raw.chunk(4, dim=-1)
        return attn_scale, attn_shift, mlp_scale, mlp_shift


def modulate_adaln(hidden: Tensor, *, scale: Tensor, shift: Tensor) -> Tensor:
    """Apply AdaLN-lite channel modulation without sequence-level statistics."""
    return hidden * (1.0 + scale) + shift


def build_condition_projection(config: CausalTokenPriorConfig) -> nn.Module | None:
    """Build an additive condition projection when conditioning is enabled."""
    if config.condition_injection != "additive":
        return None
    if config.condition_hidden_dim is None:
        return nn.Linear(config.condition_dim, config.token_embedding_dim)
    return nn.Sequential(
        nn.Linear(config.condition_dim, config.condition_hidden_dim),
        nn.GELU(),
        nn.Linear(config.condition_hidden_dim, config.token_embedding_dim),
    )


def validate_token_sequence(
    tokens: Tensor,
    config: CausalTokenPriorConfig,
    *,
    tensor_name: str,
) -> None:
    """Validate tokenizer-code tensors in ``[batch, sequence_length]`` format."""
    if tokens.ndim != 2:
        raise ValueError(f"{tensor_name} must be [batch, sequence_length]; got {tokens.shape}.")
    if tokens.shape[1] != config.sequence_length:
        raise ValueError(
            f"{tensor_name} length must be {config.sequence_length}; got {tokens.shape[1]}."
        )
    if not torch.is_floating_point(tokens) and tokens.numel() > 0:
        min_value = int(tokens.min().item())
        max_value = int(tokens.max().item())
        if min_value < 0 or max_value >= config.codebook_size:
            raise ValueError(
                f"{tensor_name} values must be in [0, {config.codebook_size - 1}]; "
                f"observed [{min_value}, {max_value}]."
            )
    if tokens.dtype != torch.long:
        raise ValueError(f"{tensor_name} must have dtype torch.long; got {tokens.dtype}.")


def validate_multicode_sequence(
    tokens: Tensor,
    config: CausalTokenPriorConfig,
    *,
    tensor_name: str,
) -> None:
    """Validate multi-code tensors in native ``[batch, time, ...]`` layout."""
    expected_shape = (config.sequence_length, *config.component_shape)
    if tokens.ndim != 2 + len(config.component_shape):
        raise ValueError(
            f"{tensor_name} must be [batch, {', '.join(str(v) for v in expected_shape)}]; "
            f"got {tuple(tokens.shape)}."
        )
    if tuple(tokens.shape[1:]) != expected_shape:
        raise ValueError(
            f"{tensor_name} shape after batch must be {expected_shape}; "
            f"got {tuple(tokens.shape[1:])}."
        )
    if tokens.dtype != torch.long:
        raise ValueError(f"{tensor_name} must have dtype torch.long; got {tokens.dtype}.")
    if tokens.numel() == 0:
        return
    min_value = int(tokens.min().item())
    max_value = int(tokens.max().item())
    if min_value < 0 or max_value >= config.codebook_size:
        raise ValueError(
            f"{tensor_name} values must be in [0, {config.codebook_size - 1}]; "
            f"observed [{min_value}, {max_value}]."
        )


def validate_multicode_shifted_sequence(
    shifted_inputs: Tensor,
    config: CausalTokenPriorConfig,
) -> None:
    """Validate BOS-shifted multi-code inputs."""
    expected_shape = (config.sequence_length, *config.component_shape)
    if shifted_inputs.ndim != 2 + len(config.component_shape):
        raise ValueError(
            "shifted_inputs must be [batch, time, ...] with shape after batch "
            f"{expected_shape}; got {tuple(shifted_inputs.shape)}."
        )
    if tuple(shifted_inputs.shape[1:]) != expected_shape:
        raise ValueError(
            f"shifted_inputs shape after batch must be {expected_shape}; "
            f"got {tuple(shifted_inputs.shape[1:])}."
        )
    if shifted_inputs.dtype != torch.long:
        raise ValueError(f"shifted_inputs must have dtype torch.long; got {shifted_inputs.dtype}.")
    if shifted_inputs.numel() == 0:
        return
    min_value = int(shifted_inputs.min().item())
    max_value = int(shifted_inputs.max().item())
    if min_value < 0 or max_value > config.codebook_size:
        raise ValueError(
            "shifted_inputs values must be tokenizer codes or BOS in "
            f"[0, {config.codebook_size}]; observed [{min_value}, {max_value}]."
        )


def validate_separate_frequency_sequence(
    tokens: Tensor,
    config: CausalTokenPriorConfig,
    *,
    tensor_name: str,
) -> None:
    """Validate packed low/high tokens in ``[batch, time, 2]`` layout."""
    if config.low_codebook_size is None or config.high_codebook_size is None:
        raise ValueError("separate frequency validation requires low/high codebook sizes.")
    expected_shape = (config.sequence_length, 2)
    if tokens.ndim != 3:
        raise ValueError(
            f"{tensor_name} must be [batch, {config.sequence_length}, 2]; "
            f"got {tuple(tokens.shape)}."
        )
    if tuple(tokens.shape[1:]) != expected_shape:
        raise ValueError(
            f"{tensor_name} shape after batch must be {expected_shape}; "
            f"got {tuple(tokens.shape[1:])}."
        )
    if tokens.dtype != torch.long:
        raise ValueError(f"{tensor_name} must have dtype torch.long; got {tokens.dtype}.")
    if tokens.numel() == 0:
        return
    validate_stream_token_range(
        tokens[:, :, 0],
        codebook_size=config.low_codebook_size,
        tensor_name=f"{tensor_name} low stream",
    )
    validate_stream_token_range(
        tokens[:, :, 1],
        codebook_size=config.high_codebook_size,
        tensor_name=f"{tensor_name} high stream",
    )


def validate_separate_frequency_shifted_sequence(
    shifted_inputs: Tensor,
    config: CausalTokenPriorConfig,
) -> None:
    """Validate BOS-shifted low/high token blocks."""
    if config.low_codebook_size is None or config.high_codebook_size is None:
        raise ValueError("separate frequency validation requires low/high codebook sizes.")
    expected_shape = (config.sequence_length, 2)
    if shifted_inputs.ndim != 3:
        raise ValueError(
            f"shifted_inputs must be [batch, time, 2]; got {tuple(shifted_inputs.shape)}."
        )
    if tuple(shifted_inputs.shape[1:]) != expected_shape:
        raise ValueError(
            f"shifted_inputs shape after batch must be {expected_shape}; "
            f"got {tuple(shifted_inputs.shape[1:])}."
        )
    if shifted_inputs.dtype != torch.long:
        raise ValueError(f"shifted_inputs must have dtype torch.long; got {shifted_inputs.dtype}.")
    if shifted_inputs.numel() == 0:
        return
    validate_stream_token_range(
        shifted_inputs[:, :, 0],
        codebook_size=config.low_codebook_size,
        tensor_name="shifted low stream",
        allow_bos=True,
    )
    validate_stream_token_range(
        shifted_inputs[:, :, 1],
        codebook_size=config.high_codebook_size,
        tensor_name="shifted high stream",
        allow_bos=True,
    )


def validate_stream_token_range(
    tokens: Tensor,
    *,
    codebook_size: int,
    tensor_name: str,
    allow_bos: bool = False,
) -> None:
    """Validate token range for one categorical stream."""
    min_value = int(tokens.min().item())
    max_value = int(tokens.max().item())
    upper_bound = codebook_size if allow_bos else codebook_size - 1
    if min_value < 0 or max_value > upper_bound:
        suffix = " including BOS" if allow_bos else ""
        raise ValueError(
            f"{tensor_name} values must be in [0, {upper_bound}]{suffix}; "
            f"observed [{min_value}, {max_value}]."
        )


def flatten_components(tokens: Tensor, config: CausalTokenPriorConfig) -> Tensor:
    """Return multi-code tokens as ``[batch, time, component]``."""
    batch_size = tokens.shape[0]
    return tokens.reshape(batch_size, config.sequence_length, config.component_count)


def component_names(config: CausalTokenPriorConfig) -> list[str]:
    """Return stable component names for factorised output heads."""
    if config.groups == 1:
        return [f"q{index}" for index in range(config.num_quantizers)]
    return [
        f"g{group_index}_q{quantizer_index}"
        for group_index in range(config.groups)
        for quantizer_index in range(config.num_quantizers)
    ]


def component_loss_weights(config: CausalTokenPriorConfig) -> Tensor:
    """Return normalised component loss weights."""
    if config.component_loss_weights is None:
        weights = torch.ones(config.component_count, dtype=torch.float32)
    else:
        weights = torch.tensor(config.component_loss_weights, dtype=torch.float32)
    return weights / weights.sum().clamp_min(1e-12)


def same_time_pair_perplexity(tokens: Tensor, config: CausalTokenPriorConfig) -> Tensor:
    """Return empirical same-time q0/q1 pair perplexity for RVQ q2 tensors."""
    validate_multicode_sequence(tokens, config, tensor_name="tokens")
    if config.component_count != 2:
        raise ValueError("same_time_pair_perplexity requires exactly two components.")
    components = flatten_components(tokens, config)
    pair_ids = components[:, :, 0] * config.codebook_size + components[:, :, 1]
    counts = torch.bincount(
        pair_ids.detach().cpu().reshape(-1),
        minlength=config.codebook_size * config.codebook_size,
    ).float()
    total = counts.sum()
    if float(total.item()) <= 0.0:
        return tokens.new_zeros((), dtype=torch.float32)
    probabilities = counts / total
    positive = probabilities > 0.0
    entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
    return torch.exp(entropy).to(device=tokens.device)


def separate_frequency_pair_perplexity(tokens: Tensor, config: CausalTokenPriorConfig) -> Tensor:
    """Return empirical same-time low/high pair perplexity for packed stream tensors."""
    validate_separate_frequency_sequence(tokens, config, tensor_name="tokens")
    if config.low_codebook_size is None or config.high_codebook_size is None:
        raise ValueError("separate frequency pair perplexity requires low/high codebook sizes.")
    pair_ids = tokens[:, :, 0] * config.high_codebook_size + tokens[:, :, 1]
    counts = torch.bincount(
        pair_ids.detach().cpu().reshape(-1),
        minlength=config.low_codebook_size * config.high_codebook_size,
    ).float()
    total = counts.sum()
    if float(total.item()) <= 0.0:
        return tokens.new_zeros((), dtype=torch.float32)
    probabilities = counts / total
    positive = probabilities > 0.0
    entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
    return torch.exp(entropy).to(device=tokens.device)


def validate_shifted_sequence(shifted_inputs: Tensor, config: CausalTokenPriorConfig) -> None:
    """Validate BOS-shifted input tensors."""
    if shifted_inputs.ndim != 2:
        raise ValueError(
            f"shifted_inputs must be [batch, sequence_length]; got {shifted_inputs.shape}."
        )
    if shifted_inputs.shape[1] != config.sequence_length:
        raise ValueError(
            f"shifted_inputs length must be {config.sequence_length}; "
            f"got {shifted_inputs.shape[1]}."
        )
    if shifted_inputs.dtype != torch.long:
        raise ValueError(f"shifted_inputs must have dtype torch.long; got {shifted_inputs.dtype}.")
    if shifted_inputs.numel() == 0:
        return
    min_value = int(shifted_inputs.min().item())
    max_value = int(shifted_inputs.max().item())
    if min_value < 0 or max_value > config.codebook_size:
        raise ValueError(
            "shifted_inputs values must be tokenizer codes or BOS in "
            f"[0, {config.codebook_size}]; observed [{min_value}, {max_value}]."
        )


def prepare_condition_sequence(
    conditions: Tensor | None,
    *,
    config: CausalTokenPriorConfig,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Validate and broadcast scalar or temporal conditions for conditional modes."""
    if config.condition_injection == "none":
        if conditions is not None:
            raise ValueError("conditions were provided but condition_injection='none'.")
        return torch.zeros(
            (batch_size, sequence_length, config.condition_dim),
            device=device,
            dtype=dtype,
        )
    if conditions is None:
        raise ValueError("conditions are required when condition_injection is enabled.")
    if conditions.ndim == 2:
        if conditions.shape != (batch_size, config.condition_dim):
            raise ValueError(
                "scalar conditions must have shape "
                f"{(batch_size, config.condition_dim)}; got {tuple(conditions.shape)}."
            )
        prepared = conditions[:, None, :].expand(batch_size, sequence_length, config.condition_dim)
    elif conditions.ndim == 3:
        if conditions.shape != (batch_size, sequence_length, config.condition_dim):
            raise ValueError(
                "temporal conditions must have shape "
                f"{(batch_size, sequence_length, config.condition_dim)}; "
                f"got {tuple(conditions.shape)}."
            )
        prepared = conditions
    else:
        raise ValueError(
            "conditions must be [batch, condition_dim] or "
            f"[batch, sequence_length, condition_dim]; got {tuple(conditions.shape)}."
        )
    return prepared.to(device=device, dtype=dtype)


def token_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    *,
    pad_token_id: int | None,
) -> Tensor:
    """Return cross-entropy over token logits."""
    ignore_index = -100 if pad_token_id is None else pad_token_id
    return functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def token_accuracy(
    logits: Tensor,
    targets: Tensor,
    *,
    pad_token_id: int | None,
) -> Tensor:
    """Return mean token accuracy, excluding optional padding tokens."""
    predictions = logits.argmax(dim=-1)
    if pad_token_id is None:
        mask = torch.ones_like(targets, dtype=torch.bool)
    else:
        mask = targets != pad_token_id
    total = mask.sum()
    if int(total.item()) == 0:
        return logits.new_zeros(())
    correct = (predictions == targets) & mask
    return correct.float().sum() / total.float()


def sample_from_logits(logits: Tensor, *, top_k: int | None) -> Tensor:
    """Sample one token per batch row from logits."""
    if top_k is not None:
        top_values, top_indices = torch.topk(logits, k=top_k, dim=-1)
        probabilities = functional.softmax(top_values, dim=-1)
        sampled_positions = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
        return top_indices.gather(dim=-1, index=sampled_positions[:, None]).squeeze(-1)
    probabilities = functional.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)


def assert_token_prior_no_future_leakage(
    model: CausalTokenTransformerPrior,
    tokens: Tensor,
    cutoff: int,
    *,
    conditions: Tensor | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Assert prefix logits are invariant to token changes after ``cutoff``.

    The cutoff is zero-indexed and inclusive in output-logit space. Tokens
    through ``cutoff`` are kept identical, and tokens after ``cutoff`` are
    changed. Because logits at position ``t`` predict ``k_t`` from shifted
    inputs ``[BOS, k_0, ..., k_{t-1}]``, this verifies logits through the
    cutoff cannot depend on future target tokens after that cutoff.
    """
    validate_token_sequence(tokens, model.config, tensor_name="tokens")
    length = tokens.shape[1]
    if not 0 <= cutoff < length:
        raise ValueError(f"cutoff must satisfy 0 <= cutoff < {length}; got {cutoff}.")

    changed_tokens = tokens.clone()
    if cutoff + 1 < length:
        future = changed_tokens[:, cutoff + 1 :]
        changed_tokens[:, cutoff + 1 :] = (future + 1) % model.config.codebook_size

    changed_conditions = None
    if conditions is not None and conditions.ndim == 3:
        changed_conditions = conditions.clone()
        if cutoff + 1 < length:
            changed_conditions[:, cutoff + 1 :] = changed_conditions[:, cutoff + 1 :] + 1.0

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            reference_logits = cast(Tensor, model(tokens, conditions=conditions).logits)
            changed_logits = cast(Tensor, model(changed_tokens, conditions=conditions).logits)
            if changed_conditions is not None:
                changed_condition_logits = cast(
                    Tensor,
                    model(tokens, conditions=changed_conditions).logits,
                )
    finally:
        model.train(was_training)

    prefix = slice(None, cutoff + 1)
    reference_prefix = reference_logits[:, prefix, :]
    changed_prefix = changed_logits[:, prefix, :]
    if not torch.allclose(reference_prefix, changed_prefix, atol=atol, rtol=rtol):
        max_difference = (reference_prefix - changed_prefix).abs().max().item()
        raise AssertionError(
            "Token prior prefix logits changed after future-token perturbation; "
            f"max_difference={max_difference}."
        )
    if changed_conditions is not None:
        changed_condition_prefix = changed_condition_logits[:, prefix, :]
        if not torch.allclose(
            reference_prefix,
            changed_condition_prefix,
            atol=atol,
            rtol=rtol,
        ):
            max_difference = (reference_prefix - changed_condition_prefix).abs().max().item()
            raise AssertionError(
                "Token prior prefix logits changed after future-condition perturbation; "
                f"max_difference={max_difference}."
            )
    return reference_logits, changed_logits


def assert_multicode_token_prior_no_future_leakage(
    model: FactorisedMultiCodeTokenPrior,
    tokens: Tensor,
    cutoff: int,
    *,
    conditions: Tensor | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Assert multi-code prefix logits are invariant to future block changes."""
    validate_multicode_sequence(tokens, model.config, tensor_name="tokens")
    length = tokens.shape[1]
    if not 0 <= cutoff < length:
        raise ValueError(f"cutoff must satisfy 0 <= cutoff < {length}; got {cutoff}.")

    changed_tokens = tokens.clone()
    if cutoff + 1 < length:
        future = changed_tokens[:, cutoff + 1 :]
        changed_tokens[:, cutoff + 1 :] = (future + 1) % model.config.codebook_size

    changed_conditions = None
    if conditions is not None and conditions.ndim == 3:
        changed_conditions = conditions.clone()
        if cutoff + 1 < length:
            changed_conditions[:, cutoff + 1 :] = changed_conditions[:, cutoff + 1 :] + 1.0

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            reference_logits = cast(Tensor, model(tokens, conditions=conditions).logits)
            changed_logits = cast(Tensor, model(changed_tokens, conditions=conditions).logits)
            if changed_conditions is not None:
                changed_condition_logits = cast(
                    Tensor,
                    model(tokens, conditions=changed_conditions).logits,
                )
    finally:
        model.train(was_training)

    prefix = slice(None, cutoff + 1)
    reference_prefix = reference_logits[:, prefix, :, :]
    changed_prefix = changed_logits[:, prefix, :, :]
    if not torch.allclose(reference_prefix, changed_prefix, atol=atol, rtol=rtol):
        max_difference = (reference_prefix - changed_prefix).abs().max().item()
        raise AssertionError(
            "Multi-code token-prior prefix logits changed after future-token perturbation; "
            f"max_difference={max_difference}."
        )
    if changed_conditions is not None:
        changed_condition_prefix = changed_condition_logits[:, prefix, :, :]
        if not torch.allclose(
            reference_prefix,
            changed_condition_prefix,
            atol=atol,
            rtol=rtol,
        ):
            max_difference = (reference_prefix - changed_condition_prefix).abs().max().item()
            raise AssertionError(
                "Multi-code token-prior prefix logits changed after future-condition "
                f"perturbation; max_difference={max_difference}."
            )
    return reference_logits, changed_logits


def assert_hierarchical_rvq_prior_no_future_leakage(
    model: HierarchicalRVQ2TokenPrior,
    tokens: Tensor,
    cutoff: int,
    *,
    conditions: Tensor | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Assert hierarchical RVQ q2 prefix logits are invariant to future blocks."""
    validate_multicode_sequence(tokens, model.config, tensor_name="tokens")
    length = tokens.shape[1]
    if not 0 <= cutoff < length:
        raise ValueError(f"cutoff must satisfy 0 <= cutoff < {length}; got {cutoff}.")

    changed_tokens = tokens.clone()
    if cutoff + 1 < length:
        future = changed_tokens[:, cutoff + 1 :]
        changed_tokens[:, cutoff + 1 :] = (future + 1) % model.config.codebook_size

    changed_conditions = None
    if conditions is not None and conditions.ndim == 3:
        changed_conditions = conditions.clone()
        if cutoff + 1 < length:
            changed_conditions[:, cutoff + 1 :] = changed_conditions[:, cutoff + 1 :] + 1.0

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            reference_logits = cast(Tensor, model(tokens, conditions=conditions).logits)
            changed_logits = cast(Tensor, model(changed_tokens, conditions=conditions).logits)
            if changed_conditions is not None:
                changed_condition_logits = cast(
                    Tensor,
                    model(tokens, conditions=changed_conditions).logits,
                )
    finally:
        model.train(was_training)

    prefix = slice(None, cutoff + 1)
    reference_prefix = reference_logits[:, prefix, :, :]
    changed_prefix = changed_logits[:, prefix, :, :]
    if not torch.allclose(reference_prefix, changed_prefix, atol=atol, rtol=rtol):
        max_difference = (reference_prefix - changed_prefix).abs().max().item()
        raise AssertionError(
            "Hierarchical RVQ q2 prefix logits changed after future-token perturbation; "
            f"max_difference={max_difference}."
        )
    if changed_conditions is not None:
        changed_condition_prefix = changed_condition_logits[:, prefix, :, :]
        if not torch.allclose(
            reference_prefix,
            changed_condition_prefix,
            atol=atol,
            rtol=rtol,
        ):
            max_difference = (reference_prefix - changed_condition_prefix).abs().max().item()
            raise AssertionError(
                "Hierarchical RVQ q2 prefix logits changed after future-condition "
                f"perturbation; max_difference={max_difference}."
            )
    return reference_logits, changed_logits


def assert_separate_frequency_prior_no_future_leakage(
    model: SeparateFrequencyHierarchicalPrior,
    tokens: Tensor,
    cutoff: int,
    *,
    conditions: Tensor | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> tuple[ModelOutput, ModelOutput]:
    """Assert low/high prefix logits are invariant to future stream perturbations."""
    validate_separate_frequency_sequence(tokens, model.config, tensor_name="tokens")
    length = tokens.shape[1]
    if not 0 <= cutoff < length:
        raise ValueError(f"cutoff must satisfy 0 <= cutoff < {length}; got {cutoff}.")

    changed_tokens = tokens.clone()
    if cutoff + 1 < length:
        future_low = changed_tokens[:, cutoff + 1 :, 0]
        future_high = changed_tokens[:, cutoff + 1 :, 1]
        changed_tokens[:, cutoff + 1 :, 0] = (future_low + 1) % model.low_codebook_size
        changed_tokens[:, cutoff + 1 :, 1] = (future_high + 1) % model.high_codebook_size

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            reference = model(tokens, conditions=conditions)
            changed = model(changed_tokens, conditions=conditions)
    finally:
        model.train(was_training)

    prefix = slice(None, cutoff + 1)
    for stream_name in ("low", "high"):
        key = f"{stream_name}_logits"
        reference_prefix = cast(Tensor, reference[key])[:, prefix, :]
        changed_prefix = cast(Tensor, changed[key])[:, prefix, :]
        if not torch.allclose(reference_prefix, changed_prefix, atol=atol, rtol=rtol):
            max_difference = (reference_prefix - changed_prefix).abs().max().item()
            raise AssertionError(
                f"Separate frequency {stream_name} prefix logits changed after "
                f"future-token perturbation; max_difference={max_difference}."
            )
    return reference, changed
