"""Causal autoregressive token-prior modules."""

from time_causal_vae.models.discrete.priors.causal_transformer import (
    CausalConvTransformerPrior,
    CausalTokenTransformerPrior,
    FactorisedMultiCodeTokenPrior,
    HierarchicalRVQ2TokenPrior,
    assert_hierarchical_rvq_prior_no_future_leakage,
    assert_multicode_token_prior_no_future_leakage,
    assert_token_prior_no_future_leakage,
    build_token_prior_model,
)
from time_causal_vae.models.discrete.priors.config import CausalTokenPriorConfig
from time_causal_vae.models.discrete.priors.masks import causal_attention_mask

__all__ = [
    "CausalConvTransformerPrior",
    "CausalTokenPriorConfig",
    "CausalTokenTransformerPrior",
    "FactorisedMultiCodeTokenPrior",
    "HierarchicalRVQ2TokenPrior",
    "assert_hierarchical_rvq_prior_no_future_leakage",
    "assert_multicode_token_prior_no_future_leakage",
    "assert_token_prior_no_future_leakage",
    "build_token_prior_model",
    "causal_attention_mask",
]
