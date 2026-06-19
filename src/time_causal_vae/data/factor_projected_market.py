"""Factor-projected views of the synthetic multifactor market benchmark.

The view exposes low-dimensional factor coordinates as model-visible data while
retaining raw 50D returns and projection metadata for diagnostics. Train-fitted
PCA projection states can be reused on eval data to avoid fitting a basis on
held-out paths.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from time_causal_vae.data.multifactor_market import (
    ConditionMode,
    MultifactorMarketDataset,
    coerce_standardization_stats,
    inverse_standardize_multifactor_returns,
)

__all__ = [
    "FactorProjectedMultifactorMarketDataset",
    "FactorProjectionMode",
    "FactorProjectionState",
    "coerce_factor_projection_state",
    "fit_train_pca_projection",
    "project_returns_to_factors",
    "reconstruct_model_visible_returns_from_factors",
    "reconstruct_raw_returns_from_factors",
]

FactorProjectionMode = Literal["train_pca", "oracle_loadings"]


@dataclass(frozen=True)
class FactorProjectionState:
    """Frozen cross-sectional projection used by factor-tokenizer datasets."""

    mode: FactorProjectionMode
    basis: Tensor
    mean: Tensor
    n_factors: int
    input_dim: int
    input_scale: str
    is_oracle: bool
    metadata: dict[str, Any]


class FactorProjectedMultifactorMarketDataset(TensorDataset):
    """Expose factor coordinates for the synthetic multifactor market dataset."""

    def __init__(
        self,
        n_sample: int,
        n_timestep: int = 60,
        *,
        n_assets: int = 50,
        n_factors: int = 5,
        n_sectors: int = 5,
        seed: int | None = None,
        structure_seed: int | None = None,
        path_seed: int | None = None,
        condition_mode: ConditionMode = "constant",
        with_jumps: bool = False,
        common_jumps: bool = True,
        sector_jumps: bool = True,
        idiosyncratic_jumps: bool = False,
        standardize_returns: bool = True,
        standardization_stats: dict[str, Tensor] | None = None,
        standardization_epsilon: float = 1e-6,
        projection_mode: FactorProjectionMode = "train_pca",
        projection_n_factors: int | None = None,
        projection_state: FactorProjectionState | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Build a factor-projected synthetic dataset.

        ``projection_mode="train_pca"`` fits a PCA basis on the current sample
        only when no ``projection_state`` is supplied. The data pipeline supplies
        the train-fitted state to eval datasets, so eval paths do not fit their
        own basis. ``projection_mode="oracle_loadings"`` uses the simulator
        loading span and is marked as oracle diagnostic metadata.
        """
        self.base_dataset = MultifactorMarketDataset(
            n_sample,
            n_timestep,
            n_assets=n_assets,
            n_factors=n_factors,
            n_sectors=n_sectors,
            seed=seed,
            structure_seed=structure_seed,
            path_seed=path_seed,
            condition_mode=condition_mode,
            with_jumps=with_jumps,
            common_jumps=common_jumps,
            sector_jumps=sector_jumps,
            idiosyncratic_jumps=idiosyncratic_jumps,
            standardize_returns=standardize_returns,
            standardization_stats=standardization_stats,
            standardization_epsilon=standardization_epsilon,
            **kwargs,
        )
        factor_count = int(projection_n_factors if projection_n_factors is not None else n_factors)
        _validate_projection_count(factor_count, n_assets=n_assets)

        if projection_state is None:
            state = build_factor_projection_state(
                self.base_dataset,
                projection_mode=projection_mode,
                n_factors=factor_count,
            )
            state_source = "fit_on_current_sample"
        else:
            state = coerce_factor_projection_state(projection_state)
            if state.mode != projection_mode:
                raise ValueError(
                    f"projection_state mode {state.mode!r} does not match {projection_mode!r}."
                )
            if state.n_factors != factor_count:
                raise ValueError(
                    f"projection_state has {state.n_factors} factors; expected {factor_count}."
                )
            state_source = "provided"

        self.projection_mode = projection_mode
        self.projection_state = state
        self.factor_returns = project_returns_to_factors(self.base_dataset.data, state)
        self.data = self.factor_returns
        self.labels = self.base_dataset.labels
        self.raw_log_returns = self.base_dataset.raw_log_returns
        self.model_visible_returns = self.base_dataset.data
        self.standardization_stats = self.base_dataset.standardization_stats
        self.loadings = self.base_dataset.loadings
        self.sector_labels = self.base_dataset.sector_labels
        self.factor_vol_paths = self.base_dataset.factor_vol_paths
        self.true_covariance = self.base_dataset.true_covariance
        self.true_correlation = self.base_dataset.true_correlation
        self.metadata = dict(self.base_dataset.metadata)
        self.metadata["factor_projection"] = factor_projection_metadata(
            state,
            state_source=state_source,
            factor_returns=self.factor_returns,
        )
        super().__init__(self.data)

    def reconstruct_model_visible_returns(self, factor_coordinates: Tensor | None = None) -> Tensor:
        """Map factor coordinates back to the model-visible 50D return scale."""
        coordinates = self.factor_returns if factor_coordinates is None else factor_coordinates
        return reconstruct_model_visible_returns_from_factors(coordinates, self.projection_state)

    def reconstruct_raw_returns(self, factor_coordinates: Tensor | None = None) -> Tensor:
        """Map factor coordinates back to raw 50D log-return scale."""
        coordinates = self.factor_returns if factor_coordinates is None else factor_coordinates
        return reconstruct_raw_returns_from_factors(
            coordinates,
            self.projection_state,
            standardization_stats=self.standardization_stats,
        )


def build_factor_projection_state(
    dataset: MultifactorMarketDataset,
    *,
    projection_mode: FactorProjectionMode,
    n_factors: int,
) -> FactorProjectionState:
    """Build a projection state from a synthetic multifactor dataset."""
    if projection_mode == "train_pca":
        return fit_train_pca_projection(
            dataset.data,
            n_factors=n_factors,
            input_scale=_dataset_input_scale(dataset),
        )
    if projection_mode == "oracle_loadings":
        return oracle_loading_projection(
            loadings=dataset.loadings,
            n_factors=n_factors,
            standardization_stats=dataset.standardization_stats,
            input_dim=dataset.data.shape[-1],
            input_scale=_dataset_input_scale(dataset),
        )
    raise ValueError(f"Unsupported projection_mode: {projection_mode}")


def fit_train_pca_projection(
    returns: Tensor,
    *,
    n_factors: int,
    input_scale: str = "standardized_returns",
) -> FactorProjectionState:
    """Fit a PCA basis on pooled train returns with shape ``[batch, time, assets]``."""
    validated = _validate_return_tensor(returns)
    _validate_projection_count(n_factors, n_assets=validated.shape[-1])
    flat_returns = validated.reshape(-1, validated.shape[-1])
    mean = flat_returns.mean(dim=0)
    centred = flat_returns - mean.view(1, -1)
    _, singular_values, vh = torch.linalg.svd(centred, full_matrices=False)
    basis = _canonicalise_basis_signs(vh[:n_factors].T.contiguous())
    explained_variance = singular_values[:n_factors].square()
    total_variance = singular_values.square().sum().clamp_min(1e-12)
    return FactorProjectionState(
        mode="train_pca",
        basis=basis.detach().cpu(),
        mean=mean.detach().cpu(),
        n_factors=int(n_factors),
        input_dim=int(validated.shape[-1]),
        input_scale=input_scale,
        is_oracle=False,
        metadata={
            "basis_source": "train_pca",
            "fit_scope": "train_split_only_when_supplied_by_pipeline",
            "explained_variance_ratio": (explained_variance / total_variance).detach().cpu(),
        },
    )


def oracle_loading_projection(
    *,
    loadings: Tensor,
    n_factors: int,
    standardization_stats: dict[str, Tensor] | None,
    input_dim: int,
    input_scale: str,
) -> FactorProjectionState:
    """Build an oracle loading-basis projection for synthetic controlled ablations."""
    loadings_tensor = _validate_loading_matrix(loadings, input_dim=input_dim)
    _validate_projection_count(n_factors, n_assets=loadings_tensor.shape[0])
    loading_basis_source = loadings_tensor[:, :n_factors].detach().float()
    if standardization_stats is not None:
        stats = coerce_standardization_stats(standardization_stats)
        std = stats["std"].view(-1, 1).clamp_min(1e-12)
        loading_basis_source = loading_basis_source / std
    basis = torch.linalg.qr(loading_basis_source, mode="reduced").Q[:, :n_factors].contiguous()
    basis = _canonicalise_basis_signs(basis)
    return FactorProjectionState(
        mode="oracle_loadings",
        basis=basis.detach().cpu(),
        mean=torch.zeros(input_dim, dtype=torch.float32),
        n_factors=int(n_factors),
        input_dim=int(input_dim),
        input_scale=input_scale,
        is_oracle=True,
        metadata={
            "basis_source": "simulator_loadings",
            "fit_scope": "oracle_synthetic_controlled_ablation",
            "oracle_warning": (
                "Uses simulator loading metadata and is not a valid empirical-data convention."
            ),
        },
    )


def project_returns_to_factors(returns: Tensor, state: FactorProjectionState) -> Tensor:
    """Project ``[batch, time, assets]`` returns to factor coordinates."""
    validated = _validate_return_tensor(returns)
    if validated.shape[-1] != state.input_dim:
        raise ValueError(
            f"Projection state expects input_dim={state.input_dim}; got {validated.shape[-1]}."
        )
    basis = state.basis.to(device=validated.device, dtype=validated.dtype)
    mean = state.mean.to(device=validated.device, dtype=validated.dtype)
    return torch.matmul(validated - mean.view(1, 1, -1), basis)


def reconstruct_model_visible_returns_from_factors(
    factor_coordinates: Tensor,
    state: FactorProjectionState,
) -> Tensor:
    """Reconstruct 50D returns in the projection input scale from factor coordinates."""
    coordinates = _validate_factor_tensor(factor_coordinates, n_factors=state.n_factors)
    basis = state.basis.to(device=coordinates.device, dtype=coordinates.dtype)
    mean = state.mean.to(device=coordinates.device, dtype=coordinates.dtype)
    return torch.matmul(coordinates, basis.T) + mean.view(1, 1, -1)


def reconstruct_raw_returns_from_factors(
    factor_coordinates: Tensor,
    state: FactorProjectionState,
    *,
    standardization_stats: dict[str, Tensor] | None = None,
) -> Tensor:
    """Reconstruct raw 50D log returns from factor coordinates."""
    model_visible_returns = reconstruct_model_visible_returns_from_factors(
        factor_coordinates, state
    )
    if standardization_stats is None:
        return model_visible_returns
    return inverse_standardize_multifactor_returns(model_visible_returns, standardization_stats)


def coerce_factor_projection_state(
    state: FactorProjectionState | Mapping[str, Any],
) -> FactorProjectionState:
    """Return a detached CPU projection state from a dataclass or mapping."""
    if isinstance(state, FactorProjectionState):
        return FactorProjectionState(
            mode=state.mode,
            basis=state.basis.detach().cpu().float(),
            mean=state.mean.detach().cpu().float(),
            n_factors=int(state.n_factors),
            input_dim=int(state.input_dim),
            input_scale=str(state.input_scale),
            is_oracle=bool(state.is_oracle),
            metadata=dict(state.metadata),
        )
    mode = cast(FactorProjectionMode, str(state["mode"]))
    return FactorProjectionState(
        mode=mode,
        basis=torch.as_tensor(state["basis"]).detach().cpu().float(),
        mean=torch.as_tensor(state["mean"]).detach().cpu().float(),
        n_factors=int(state["n_factors"]),
        input_dim=int(state["input_dim"]),
        input_scale=str(state.get("input_scale", "unknown")),
        is_oracle=bool(state.get("is_oracle", mode == "oracle_loadings")),
        metadata=dict(cast(Mapping[str, Any], state.get("metadata", {}))),
    )


def factor_projection_metadata(
    state: FactorProjectionState,
    *,
    state_source: str,
    factor_returns: Tensor,
) -> dict[str, Any]:
    """Return metadata describing the factor projection view."""
    metadata = dict(state.metadata)
    metadata.update(
        {
            "mode": state.mode,
            "state_source": state_source,
            "n_factors": int(state.n_factors),
            "input_dim": int(state.input_dim),
            "input_scale": state.input_scale,
            "is_oracle": bool(state.is_oracle),
            "basis_shape": list(state.basis.shape),
            "mean_shape": list(state.mean.shape),
            "factor_returns_shape": list(factor_returns.shape),
            "basis": state.basis,
            "mean": state.mean,
        }
    )
    return metadata


def _dataset_input_scale(dataset: MultifactorMarketDataset) -> str:
    if dataset.standardize_returns:
        return "standardized_returns"
    return "raw_log_returns"


def _validate_projection_count(n_factors: int, *, n_assets: int) -> None:
    if n_factors <= 0:
        raise ValueError("projection_n_factors must be positive.")
    if n_factors > n_assets:
        raise ValueError("projection_n_factors must be no larger than n_assets.")


def _validate_return_tensor(returns: Tensor) -> Tensor:
    if returns.ndim != 3:
        raise ValueError(f"Expected returns with shape [batch, time, assets]; got {returns.shape}.")
    if returns.shape[-1] < 2:
        raise ValueError("Expected at least two assets.")
    if not bool(torch.isfinite(returns).all()):
        raise ValueError("returns must be finite.")
    return returns.detach().float()


def _validate_factor_tensor(factor_coordinates: Tensor, *, n_factors: int) -> Tensor:
    if factor_coordinates.ndim != 3:
        raise ValueError(
            "Expected factor coordinates with shape [batch, time, factors]; "
            f"got {factor_coordinates.shape}."
        )
    if factor_coordinates.shape[-1] != n_factors:
        raise ValueError(f"Expected {n_factors} factors; got {factor_coordinates.shape[-1]}.")
    if not bool(torch.isfinite(factor_coordinates).all()):
        raise ValueError("factor_coordinates must be finite.")
    return factor_coordinates.detach().float()


def _validate_loading_matrix(loadings: Tensor, *, input_dim: int) -> Tensor:
    if loadings.ndim != 2:
        raise ValueError(f"Expected loadings with shape [assets, factors]; got {loadings.shape}.")
    if loadings.shape[0] != input_dim:
        raise ValueError(f"Expected loadings for {input_dim} assets; got {loadings.shape[0]}.")
    if not bool(torch.isfinite(loadings).all()):
        raise ValueError("loadings must be finite.")
    return loadings.detach().float()


def _canonicalise_basis_signs(basis: Tensor) -> Tensor:
    canonical = basis.clone()
    for column_index in range(canonical.shape[1]):
        column = canonical[:, column_index]
        anchor = int(torch.argmax(column.abs()).item())
        if float(column[anchor].item()) < 0.0:
            canonical[:, column_index] = -column
    return canonical
