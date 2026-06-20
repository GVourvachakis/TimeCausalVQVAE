"""Synthetic multi-asset factor-market benchmark.

The generator produces channel-last log-return tensors for a controlled
cross-sectional benchmark. Oracle simulator metadata is stored on the dataset
object for diagnostics and should not be exposed as a model-visible condition
unless a later experiment explicitly opts into a prefix-safe summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Literal, cast

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

__all__ = [
    "MultifactorMarketDataset",
    "MultifactorMarketSimulation",
    "fit_return_standardization",
    "inverse_standardize_multifactor_returns",
    "simulate_multifactor_market_paths",
    "standardize_multifactor_returns",
]


ConditionMode = Literal["constant", "initial_regime"]


@dataclass(frozen=True)
class MultifactorMarketSimulation:
    """Container for simulated returns and oracle cross-sectional metadata."""

    log_returns: Tensor
    labels: Tensor
    loadings: Tensor
    sector_labels: Tensor
    factor_returns: Tensor
    factor_vol_paths: Tensor
    factor_base_volatilities: Tensor
    idiosyncratic_volatilities: Tensor
    regime_states: Tensor
    true_covariance: Tensor
    true_correlation: Tensor
    common_jump_indicators: Tensor
    sector_jump_indicators: Tensor
    jump_sizes: Tensor
    metadata: dict[str, Any]


def simulate_multifactor_market_paths(
    n_sample: int,
    n_timesteps: int = 60,
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
    log_vol_ar: float = 0.96,
    regime_switch_probability: float = 0.04,
    factor_vol_min: float = 0.006,
    factor_vol_max: float = 0.014,
    idiosyncratic_vol_min: float = 0.004,
    idiosyncratic_vol_max: float = 0.010,
    common_jump_base_probability: float = 0.010,
    sector_jump_base_probability: float = 0.015,
    common_jump_excitation: float = 0.070,
    sector_jump_excitation: float = 0.090,
    jump_probability_decay: float = 0.78,
    max_jump_probability: float = 0.35,
    dtype: torch.dtype = torch.float32,
) -> MultifactorMarketSimulation:
    """Simulate stochastic-covariance multi-asset log-return paths.

    Parameters
    ----------
    n_sample:
        Number of independent paths.
    n_timesteps:
        Number of log-return observations per path.
    n_assets:
        Number of assets in the cross-section. Defaults to the benchmark target
        of 50.
    n_factors:
        Number of low-rank latent factors. Defaults to five.
    n_sectors:
        Number of deterministic sector blocks used to structure loadings.
    seed:
        Backward-compatible CPU generator seed. When ``structure_seed`` and
        ``path_seed`` are omitted, this seed is used for both generators.
    structure_seed:
        Seed for persistent market structure: sector loadings, factor volatility
        baselines, and idiosyncratic volatility baselines.
    path_seed:
        Seed for realised path randomness: regimes, factor shocks,
        idiosyncratic shocks, and jump events.
    condition_mode:
        ``"constant"`` exposes only a constant model condition. ``"initial_regime"``
        exposes the initial factor-volatility regime and remains prefix-safe.
    with_jumps:
        Include jump shocks controlled by ``common_jumps`` and
        ``sector_jumps``. Idiosyncratic jumps are intentionally not implemented
        in the first stress profile.

    Returns
    -------
    MultifactorMarketSimulation
        Log returns with shape ``[n_sample, n_timesteps, n_assets]`` plus oracle
        loadings, regimes, realised covariance summaries, and jump metadata.
    """
    _validate_simulation_inputs(
        n_sample=n_sample,
        n_timesteps=n_timesteps,
        n_assets=n_assets,
        n_factors=n_factors,
        n_sectors=n_sectors,
        log_vol_ar=log_vol_ar,
        regime_switch_probability=regime_switch_probability,
        factor_vol_min=factor_vol_min,
        factor_vol_max=factor_vol_max,
        idiosyncratic_vol_min=idiosyncratic_vol_min,
        idiosyncratic_vol_max=idiosyncratic_vol_max,
        common_jump_base_probability=common_jump_base_probability,
        sector_jump_base_probability=sector_jump_base_probability,
        common_jump_excitation=common_jump_excitation,
        sector_jump_excitation=sector_jump_excitation,
        jump_probability_decay=jump_probability_decay,
        max_jump_probability=max_jump_probability,
    )
    if condition_mode not in {"constant", "initial_regime"}:
        raise ValueError("condition_mode must be 'constant' or 'initial_regime'.")
    if idiosyncratic_jumps:
        raise ValueError("idiosyncratic_jumps is not implemented for the first jump stress pass.")

    resolved_structure_seed, resolved_path_seed = _resolve_simulation_seeds(
        seed=seed,
        structure_seed=structure_seed,
        path_seed=path_seed,
    )
    structure_generator = _make_cpu_generator(resolved_structure_seed)
    path_generator = _make_cpu_generator(resolved_path_seed)

    sector_labels = _build_sector_labels(n_assets=n_assets, n_sectors=n_sectors)
    loadings = _build_sector_loading_matrix(
        n_assets=n_assets,
        n_factors=n_factors,
        n_sectors=n_sectors,
        sector_labels=sector_labels,
        generator=structure_generator,
        dtype=dtype,
    )
    factor_base_volatilities = _build_factor_base_volatilities(
        n_factors=n_factors,
        factor_vol_min=factor_vol_min,
        factor_vol_max=factor_vol_max,
        generator=structure_generator,
        dtype=dtype,
    )
    idiosyncratic_volatilities = _build_idiosyncratic_volatilities(
        n_assets=n_assets,
        n_sectors=n_sectors,
        sector_labels=sector_labels,
        vol_min=idiosyncratic_vol_min,
        vol_max=idiosyncratic_vol_max,
        generator=structure_generator,
        dtype=dtype,
    )
    regime_states = _simulate_regime_states(
        n_sample=n_sample,
        n_timesteps=n_timesteps,
        switch_probability=regime_switch_probability,
        generator=path_generator,
    )
    factor_vol_paths = _simulate_factor_vol_paths(
        regime_states=regime_states,
        base_volatilities=factor_base_volatilities,
        log_vol_ar=log_vol_ar,
        generator=path_generator,
        dtype=dtype,
    )
    factor_returns = factor_vol_paths * torch.randn(
        (n_sample, n_timesteps, n_factors),
        generator=path_generator,
        dtype=dtype,
    )
    idiosyncratic_noise = torch.randn(
        (n_sample, n_timesteps, n_assets),
        generator=path_generator,
        dtype=dtype,
    )
    idiosyncratic_noise = idiosyncratic_noise * idiosyncratic_volatilities.view(1, 1, -1)
    log_returns = torch.einsum("btk,ak->bta", factor_returns, loadings)
    log_returns = log_returns + idiosyncratic_noise

    common_jump_indicators = torch.zeros((n_sample, n_timesteps, 1), dtype=torch.bool)
    sector_jump_indicators = torch.zeros((n_sample, n_timesteps, n_sectors), dtype=torch.bool)
    jump_sizes = torch.zeros((n_sample, n_timesteps, n_assets), dtype=dtype)
    if with_jumps:
        jump_sizes, common_jump_indicators, sector_jump_indicators = _simulate_jump_shocks(
            n_sample=n_sample,
            n_timesteps=n_timesteps,
            n_assets=n_assets,
            n_sectors=n_sectors,
            sector_labels=sector_labels,
            generator=path_generator,
            dtype=dtype,
            common_jumps=common_jumps,
            sector_jumps=sector_jumps,
            common_jump_base_probability=common_jump_base_probability,
            sector_jump_base_probability=sector_jump_base_probability,
            common_jump_excitation=common_jump_excitation,
            sector_jump_excitation=sector_jump_excitation,
            jump_probability_decay=jump_probability_decay,
            max_jump_probability=max_jump_probability,
        )
        log_returns = log_returns + jump_sizes

    labels = _build_labels(
        n_sample=n_sample,
        regime_states=regime_states,
        condition_mode=condition_mode,
        dtype=dtype,
    )
    true_covariance = _empirical_covariance(log_returns)
    true_correlation = _covariance_to_correlation(true_covariance)
    factor_model_covariance = _factor_model_covariance(
        loadings=loadings,
        factor_vol_paths=factor_vol_paths,
        idiosyncratic_volatilities=idiosyncratic_volatilities,
    )
    metadata = _build_metadata(
        n_sample=n_sample,
        n_timesteps=n_timesteps,
        n_assets=n_assets,
        n_factors=n_factors,
        n_sectors=n_sectors,
        seed=seed,
        structure_seed=resolved_structure_seed,
        path_seed=resolved_path_seed,
        condition_mode=condition_mode,
        with_jumps=with_jumps,
        common_jumps=common_jumps,
        sector_jumps=sector_jumps,
        idiosyncratic_jumps=idiosyncratic_jumps,
        loadings=loadings,
        sector_labels=sector_labels,
        factor_vol_paths=factor_vol_paths,
        factor_base_volatilities=factor_base_volatilities,
        true_covariance=true_covariance,
        true_correlation=true_correlation,
        factor_model_covariance=factor_model_covariance,
        common_jump_indicators=common_jump_indicators,
        sector_jump_indicators=sector_jump_indicators,
        jump_sizes=jump_sizes,
    )
    return MultifactorMarketSimulation(
        log_returns=log_returns,
        labels=labels,
        loadings=loadings,
        sector_labels=sector_labels,
        factor_returns=factor_returns,
        factor_vol_paths=factor_vol_paths,
        factor_base_volatilities=factor_base_volatilities,
        idiosyncratic_volatilities=idiosyncratic_volatilities,
        regime_states=regime_states,
        true_covariance=true_covariance,
        true_correlation=true_correlation,
        common_jump_indicators=common_jump_indicators,
        sector_jump_indicators=sector_jump_indicators,
        jump_sizes=jump_sizes,
        metadata=metadata,
    )


class MultifactorMarketDataset(TensorDataset):
    """Torch dataset exposing 50D factor-market log returns and oracle metadata."""

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
        standardize_returns: bool = False,
        standardization_stats: dict[str, Tensor] | None = None,
        standardization_epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """Simulate a multi-factor market dataset.

        Extra keyword arguments are forwarded to
        :func:`simulate_multifactor_market_paths`.
        """
        simulation = simulate_multifactor_market_paths(
            n_sample,
            n_timesteps=n_timestep,
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
            **kwargs,
        )
        self.simulation = simulation
        self.raw_log_returns = simulation.log_returns
        self.standardize_returns = bool(standardize_returns)
        self.standardization_stats: dict[str, Tensor] | None
        if self.standardize_returns:
            if standardization_stats is None:
                stats = fit_return_standardization(
                    self.raw_log_returns,
                    epsilon=standardization_epsilon,
                )
                stats_source = "fit_on_current_sample"
            else:
                stats = coerce_standardization_stats(standardization_stats)
                stats_source = "provided"
            self.standardization_stats = stats
            self.data = standardize_multifactor_returns(self.raw_log_returns, stats)
        else:
            self.standardization_stats = None
            self.data = self.raw_log_returns
            stats_source = "disabled"
        self.labels = simulation.labels
        self.loadings = simulation.loadings
        self.sector_labels = simulation.sector_labels
        self.factor_returns = simulation.factor_returns
        self.factor_vol_paths = simulation.factor_vol_paths
        self.factor_base_volatilities = simulation.factor_base_volatilities
        self.idiosyncratic_volatilities = simulation.idiosyncratic_volatilities
        self.regime_states = simulation.regime_states
        self.true_covariance = simulation.true_covariance
        self.true_correlation = simulation.true_correlation
        self.common_jump_indicators = simulation.common_jump_indicators
        self.sector_jump_indicators = simulation.sector_jump_indicators
        self.jump_sizes = simulation.jump_sizes
        self.metadata = dict(simulation.metadata)
        self.metadata["standardization"] = _standardization_metadata(
            enabled=self.standardize_returns,
            stats=self.standardization_stats,
            stats_source=stats_source,
            epsilon=standardization_epsilon,
            raw_log_returns=self.raw_log_returns,
            model_visible_data=self.data,
        )
        super().__init__(self.data)


def fit_return_standardization(
    log_returns: Tensor,
    *,
    epsilon: float = 1e-6,
) -> dict[str, Tensor]:
    """Fit per-asset mean and standard deviation for signed log returns."""
    _validate_return_tensor(log_returns)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    flat_returns = log_returns.detach().float().reshape(-1, log_returns.shape[-1])
    mean = flat_returns.mean(dim=0)
    std = flat_returns.std(dim=0, unbiased=False).clamp_min(float(epsilon))
    return {"mean": mean, "std": std}


def standardize_multifactor_returns(log_returns: Tensor, stats: dict[str, Tensor]) -> Tensor:
    """Apply per-asset standardization to signed log-return paths."""
    _validate_return_tensor(log_returns)
    coerced = coerce_standardization_stats(stats)
    mean = coerced["mean"].to(device=log_returns.device, dtype=log_returns.dtype)
    std = coerced["std"].to(device=log_returns.device, dtype=log_returns.dtype)
    if mean.numel() != log_returns.shape[-1] or std.numel() != log_returns.shape[-1]:
        raise ValueError("standardization stats must match the asset dimension.")
    return (log_returns - mean.view(1, 1, -1)) / std.view(1, 1, -1)


def inverse_standardize_multifactor_returns(
    standardized_returns: Tensor,
    stats: dict[str, Tensor],
) -> Tensor:
    """Map standardized model outputs back to raw log-return scale."""
    _validate_return_tensor(standardized_returns)
    coerced = coerce_standardization_stats(stats)
    mean = coerced["mean"].to(device=standardized_returns.device, dtype=standardized_returns.dtype)
    std = coerced["std"].to(device=standardized_returns.device, dtype=standardized_returns.dtype)
    if (
        mean.numel() != standardized_returns.shape[-1]
        or std.numel() != standardized_returns.shape[-1]
    ):
        raise ValueError("standardization stats must match the asset dimension.")
    return standardized_returns * std.view(1, 1, -1) + mean.view(1, 1, -1)


def coerce_standardization_stats(stats: dict[str, Tensor]) -> dict[str, Tensor]:
    """Return detached CPU tensor standardization stats."""
    if "mean" not in stats or "std" not in stats:
        raise ValueError("standardization_stats must contain 'mean' and 'std'.")
    mean = torch.as_tensor(stats["mean"]).detach().float().reshape(-1)
    std = torch.as_tensor(stats["std"]).detach().float().reshape(-1)
    if mean.numel() == 0 or std.numel() == 0:
        raise ValueError("standardization stats must be non-empty.")
    if mean.numel() != std.numel():
        raise ValueError("standardization mean and std must have matching lengths.")
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
        raise ValueError("standardization stats must be finite.")
    if not bool((std > 0.0).all()):
        raise ValueError("standardization std entries must be positive.")
    return {"mean": mean, "std": std}


def _validate_return_tensor(returns: Tensor) -> None:
    if returns.ndim != 3:
        raise ValueError(f"Expected returns with shape [batch, time, assets]; got {returns.shape}.")
    if returns.shape[-1] < 2:
        raise ValueError("Expected at least two assets.")
    if not bool(torch.isfinite(returns).all()):
        raise ValueError("returns must be finite.")


def _standardization_metadata(
    *,
    enabled: bool,
    stats: dict[str, Tensor] | None,
    stats_source: str,
    epsilon: float,
    raw_log_returns: Tensor,
    model_visible_data: Tensor,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "enabled": bool(enabled),
        "stats_source": stats_source,
        "epsilon": float(epsilon),
        "raw_log_returns_shape": list(raw_log_returns.shape),
        "model_visible_data_shape": list(model_visible_data.shape),
        "raw_log_returns_available_on_dataset": True,
    }
    if stats is not None:
        metadata["mean"] = stats["mean"]
        metadata["std"] = stats["std"]
        metadata["mean_abs_standardized_asset_mean"] = float(
            model_visible_data.detach().float().mean(dim=(0, 1)).abs().mean().item()
        )
        metadata["mean_standardized_asset_std"] = float(
            model_visible_data.detach().float().std(dim=(0, 1), unbiased=False).mean().item()
        )
    return metadata


def _validate_simulation_inputs(
    *,
    n_sample: int,
    n_timesteps: int,
    n_assets: int,
    n_factors: int,
    n_sectors: int,
    log_vol_ar: float,
    regime_switch_probability: float,
    factor_vol_min: float,
    factor_vol_max: float,
    idiosyncratic_vol_min: float,
    idiosyncratic_vol_max: float,
    common_jump_base_probability: float,
    sector_jump_base_probability: float,
    common_jump_excitation: float,
    sector_jump_excitation: float,
    jump_probability_decay: float,
    max_jump_probability: float,
) -> None:
    if n_sample <= 0:
        raise ValueError("n_sample must be positive.")
    if n_timesteps <= 1:
        raise ValueError("n_timesteps must be greater than one.")
    if n_assets < 2:
        raise ValueError("n_assets must be at least two for cross-sectional diagnostics.")
    if n_factors <= 0:
        raise ValueError("n_factors must be positive.")
    if n_factors > n_assets:
        raise ValueError("n_factors must be no larger than n_assets.")
    if n_sectors <= 0 or n_sectors > n_assets:
        raise ValueError("n_sectors must be in [1, n_assets].")
    if not 0.0 <= log_vol_ar < 1.0:
        raise ValueError("log_vol_ar must lie in [0, 1).")
    if not 0.0 <= regime_switch_probability <= 1.0:
        raise ValueError("regime_switch_probability must lie in [0, 1].")
    if factor_vol_min <= 0.0 or factor_vol_max <= factor_vol_min:
        raise ValueError("factor volatility bounds must be positive and increasing.")
    if idiosyncratic_vol_min <= 0.0 or idiosyncratic_vol_max <= idiosyncratic_vol_min:
        raise ValueError("idiosyncratic volatility bounds must be positive and increasing.")
    _validate_probability(common_jump_base_probability, "common_jump_base_probability")
    _validate_probability(sector_jump_base_probability, "sector_jump_base_probability")
    _validate_probability(common_jump_excitation, "common_jump_excitation")
    _validate_probability(sector_jump_excitation, "sector_jump_excitation")
    if not 0.0 <= jump_probability_decay <= 1.0:
        raise ValueError("jump_probability_decay must lie in [0, 1].")
    if not 0.0 < max_jump_probability <= 1.0:
        raise ValueError("max_jump_probability must lie in (0, 1].")


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")


def _resolve_simulation_seeds(
    *,
    seed: int | None,
    structure_seed: int | None,
    path_seed: int | None,
) -> tuple[int | None, int | None]:
    resolved_structure_seed = seed if structure_seed is None else structure_seed
    resolved_path_seed = seed if path_seed is None else path_seed
    return resolved_structure_seed, resolved_path_seed


def _make_cpu_generator(seed: int | None) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    return generator


def _build_sector_labels(*, n_assets: int, n_sectors: int) -> Tensor:
    labels = torch.div(
        torch.arange(n_assets, dtype=torch.long) * n_sectors,
        n_assets,
        rounding_mode="floor",
    )
    return torch.clamp(labels, max=n_sectors - 1)


def _build_sector_loading_matrix(
    *,
    n_assets: int,
    n_factors: int,
    n_sectors: int,
    sector_labels: Tensor,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> Tensor:
    base_loadings = 0.07 + 0.05 * torch.rand(
        (n_assets, n_factors),
        generator=generator,
        dtype=dtype,
    )
    loadings = base_loadings + 0.025 * torch.randn(
        (n_assets, n_factors),
        generator=generator,
        dtype=dtype,
    )
    for factor_index in range(n_factors):
        sector_index = factor_index % n_sectors
        sector_mask = sector_labels == sector_index
        sector_count = int(sector_mask.sum().item())
        sector_strength = 0.42 + 0.09 * torch.rand(
            sector_count,
            generator=generator,
            dtype=dtype,
        )
        loadings[sector_mask, factor_index] = loadings[sector_mask, factor_index] + sector_strength

    column_norms = torch.linalg.vector_norm(loadings, dim=0).clamp_min(1e-8)
    loadings = loadings / column_norms.view(1, -1) * sqrt(float(n_assets) / float(n_factors))
    return cast(Tensor, loadings)


def _build_factor_base_volatilities(
    *,
    n_factors: int,
    factor_vol_min: float,
    factor_vol_max: float,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> Tensor:
    return factor_vol_min + (factor_vol_max - factor_vol_min) * torch.rand(
        n_factors,
        generator=generator,
        dtype=dtype,
    )


def _simulate_regime_states(
    *,
    n_sample: int,
    n_timesteps: int,
    switch_probability: float,
    generator: torch.Generator,
) -> Tensor:
    current = torch.randint(0, 3, (n_sample,), generator=generator, dtype=torch.long)
    states = torch.empty((n_sample, n_timesteps), dtype=torch.long)
    for step in range(n_timesteps):
        if step > 0:
            switches = torch.rand(n_sample, generator=generator) < switch_probability
            proposed = torch.randint(0, 3, (n_sample,), generator=generator, dtype=torch.long)
            current = torch.where(switches, proposed, current)
        states[:, step] = current
    return states


def _simulate_factor_vol_paths(
    *,
    regime_states: Tensor,
    base_volatilities: Tensor,
    log_vol_ar: float,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> Tensor:
    n_sample, n_timesteps = regime_states.shape
    n_factors = int(base_volatilities.numel())
    base_vol = base_volatilities.to(dtype=dtype)
    regime_adjustments = torch.tensor([-0.35, 0.00, 0.45], dtype=dtype)
    previous = base_vol.log().view(1, -1).repeat(n_sample, 1)
    log_vol = torch.empty((n_sample, n_timesteps, n_factors), dtype=dtype)
    innovation_scale = 0.055

    for step in range(n_timesteps):
        target = base_vol.log().view(1, -1)
        target = target + regime_adjustments[regime_states[:, step]].view(-1, 1)
        innovation = innovation_scale * torch.randn(
            (n_sample, n_factors),
            generator=generator,
            dtype=dtype,
        )
        current = log_vol_ar * previous + (1.0 - log_vol_ar) * target + innovation
        log_vol[:, step, :] = current
        previous = current

    return log_vol.exp().clamp_min(1e-6)


def _build_idiosyncratic_volatilities(
    *,
    n_assets: int,
    n_sectors: int,
    sector_labels: Tensor,
    vol_min: float,
    vol_max: float,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> Tensor:
    sector_vol = vol_min + (vol_max - vol_min) * torch.rand(
        n_sectors,
        generator=generator,
        dtype=dtype,
    )
    asset_scaling = 0.85 + 0.30 * torch.rand(n_assets, generator=generator, dtype=dtype)
    return sector_vol[sector_labels] * asset_scaling


def _simulate_jump_shocks(
    *,
    n_sample: int,
    n_timesteps: int,
    n_assets: int,
    n_sectors: int,
    sector_labels: Tensor,
    generator: torch.Generator,
    dtype: torch.dtype,
    common_jumps: bool,
    sector_jumps: bool,
    common_jump_base_probability: float,
    sector_jump_base_probability: float,
    common_jump_excitation: float,
    sector_jump_excitation: float,
    jump_probability_decay: float,
    max_jump_probability: float,
) -> tuple[Tensor, Tensor, Tensor]:
    jump_sizes = torch.zeros((n_sample, n_timesteps, n_assets), dtype=dtype)
    common_indicators = torch.zeros((n_sample, n_timesteps, 1), dtype=torch.bool)
    sector_indicators = torch.zeros((n_sample, n_timesteps, n_sectors), dtype=torch.bool)
    common_probability = torch.full((n_sample,), common_jump_base_probability, dtype=dtype)
    sector_probability = torch.full(
        (n_sample, n_sectors),
        sector_jump_base_probability,
        dtype=dtype,
    )
    common_sensitivity = 0.85 + 0.30 * torch.rand(n_assets, generator=generator, dtype=dtype)
    sector_sensitivity = 0.85 + 0.30 * torch.rand(n_assets, generator=generator, dtype=dtype)

    for step in range(n_timesteps):
        if common_jumps:
            common_draw = torch.rand(n_sample, generator=generator) < common_probability
        else:
            common_draw = torch.zeros(n_sample, dtype=torch.bool)
        common_indicators[:, step, 0] = common_draw
        if bool(common_draw.any()):
            common_size = _sample_signed_jump_sizes(
                shape=(n_sample,),
                mean=0.030,
                std=0.012,
                negative_probability=0.75,
                generator=generator,
                dtype=dtype,
            )
            common_size = torch.where(common_draw, common_size, torch.zeros_like(common_size))
            jump_sizes[:, step, :] = jump_sizes[:, step, :] + (
                common_size.view(-1, 1) * common_sensitivity.view(1, -1)
            )

        if sector_jumps:
            sector_draws = (
                torch.rand((n_sample, n_sectors), generator=generator) < sector_probability
            )
        else:
            sector_draws = torch.zeros((n_sample, n_sectors), dtype=torch.bool)
        sector_indicators[:, step, :] = sector_draws
        if bool(sector_draws.any()):
            sector_sizes = _sample_signed_jump_sizes(
                shape=(n_sample, n_sectors),
                mean=0.022,
                std=0.009,
                negative_probability=0.70,
                generator=generator,
                dtype=dtype,
            )
            sector_sizes = torch.where(sector_draws, sector_sizes, torch.zeros_like(sector_sizes))
            for sector_index in range(n_sectors):
                asset_mask = sector_labels == sector_index
                if bool(asset_mask.any()):
                    jump_sizes[:, step, asset_mask] = jump_sizes[:, step, asset_mask] + (
                        sector_sizes[:, sector_index].view(-1, 1)
                        * sector_sensitivity[asset_mask].view(1, -1)
                    )

        common_probability = common_jump_base_probability + jump_probability_decay * (
            common_probability - common_jump_base_probability
        )
        common_probability = common_probability + common_jump_excitation * common_draw.to(dtype)
        common_probability = common_probability.clamp(0.0, max_jump_probability)

        sector_probability = sector_jump_base_probability + jump_probability_decay * (
            sector_probability - sector_jump_base_probability
        )
        sector_probability = sector_probability + sector_jump_excitation * sector_draws.to(dtype)
        sector_probability = sector_probability + 0.25 * sector_jump_excitation * common_draw.view(
            -1,
            1,
        ).to(dtype)
        sector_probability = sector_probability.clamp(0.0, max_jump_probability)

    return jump_sizes, common_indicators, sector_indicators


def _sample_signed_jump_sizes(
    *,
    shape: tuple[int, ...],
    mean: float,
    std: float,
    negative_probability: float,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> Tensor:
    magnitudes = (mean + std * torch.randn(shape, generator=generator, dtype=dtype)).abs()
    signs = torch.where(
        torch.rand(shape, generator=generator) < negative_probability,
        torch.full(shape, -1.0, dtype=dtype),
        torch.ones(shape, dtype=dtype),
    )
    return signs * magnitudes


def _build_labels(
    *,
    n_sample: int,
    regime_states: Tensor,
    condition_mode: ConditionMode,
    dtype: torch.dtype,
) -> Tensor:
    if condition_mode == "constant":
        return torch.ones((n_sample, 1), dtype=dtype)
    return regime_states[:, 0].to(dtype=dtype).view(-1, 1) / 2.0


def _empirical_covariance(log_returns: Tensor) -> Tensor:
    flat_returns = log_returns.reshape(-1, log_returns.shape[-1]).float()
    centred = flat_returns - flat_returns.mean(dim=0, keepdim=True)
    denominator = max(flat_returns.shape[0] - 1, 1)
    return centred.T.matmul(centred) / float(denominator)


def _covariance_to_correlation(covariance: Tensor) -> Tensor:
    std = covariance.diag().clamp_min(1e-12).sqrt()
    correlation = covariance / (std.view(-1, 1) * std.view(1, -1))
    correlation = correlation.clamp(-1.0, 1.0)
    correlation.fill_diagonal_(1.0)
    return correlation


def _factor_model_covariance(
    *,
    loadings: Tensor,
    factor_vol_paths: Tensor,
    idiosyncratic_volatilities: Tensor,
) -> Tensor:
    factor_variance = factor_vol_paths.square().mean(dim=(0, 1))
    factor_covariance = loadings.matmul(torch.diag(factor_variance)).matmul(loadings.T)
    return factor_covariance + torch.diag(idiosyncratic_volatilities.square())


def _build_metadata(
    *,
    n_sample: int,
    n_timesteps: int,
    n_assets: int,
    n_factors: int,
    n_sectors: int,
    seed: int | None,
    structure_seed: int | None,
    path_seed: int | None,
    condition_mode: ConditionMode,
    with_jumps: bool,
    common_jumps: bool,
    sector_jumps: bool,
    idiosyncratic_jumps: bool,
    loadings: Tensor,
    sector_labels: Tensor,
    factor_vol_paths: Tensor,
    factor_base_volatilities: Tensor,
    true_covariance: Tensor,
    true_correlation: Tensor,
    factor_model_covariance: Tensor,
    common_jump_indicators: Tensor,
    sector_jump_indicators: Tensor,
    jump_sizes: Tensor,
) -> dict[str, Any]:
    true_eigenvalues = torch.linalg.eigvalsh(true_covariance).flip(0)
    return {
        "n_sample": int(n_sample),
        "n_timesteps": int(n_timesteps),
        "n_assets": int(n_assets),
        "n_factors": int(n_factors),
        "n_sectors": int(n_sectors),
        "seed": None if seed is None else int(seed),
        "structure_seed": None if structure_seed is None else int(structure_seed),
        "path_seed": None if path_seed is None else int(path_seed),
        "seed_convention": (
            "seed aliases both structure_seed and path_seed when explicit split seeds are omitted"
        ),
        "condition_mode": condition_mode,
        "oracle_metadata_model_visible": False,
        "loadings": loadings,
        "sector_labels": sector_labels,
        "factor_base_volatilities": factor_base_volatilities,
        "factor_vol_paths": factor_vol_paths,
        "true_covariance_summaries": {
            "realised_covariance": true_covariance,
            "realised_correlation": true_correlation,
            "factor_model_covariance": factor_model_covariance,
            "realised_covariance_trace": float(torch.trace(true_covariance).item()),
            "factor_model_covariance_trace": float(torch.trace(factor_model_covariance).item()),
            "top5_realised_covariance_eigenvalues": true_eigenvalues[:5].tolist(),
            "mean_abs_offdiagonal_correlation": _mean_abs_offdiagonal(true_correlation),
        },
        "jump_indicators": {
            "enabled": bool(with_jumps),
            "common_jumps_enabled": bool(with_jumps and common_jumps),
            "sector_jumps_enabled": bool(with_jumps and sector_jumps),
            "idiosyncratic_jumps_enabled": bool(with_jumps and idiosyncratic_jumps),
            "common": common_jump_indicators,
            "sector": sector_jump_indicators,
            "common_jump_count": int(common_jump_indicators.sum().item()),
            "sector_jump_count": int(sector_jump_indicators.sum().item()),
            "nonzero_asset_jump_fraction": float((jump_sizes.abs() > 0.0).float().mean().item()),
        },
    }


def _mean_abs_offdiagonal(matrix: Tensor) -> float:
    if matrix.shape[0] <= 1:
        return 0.0
    off_diagonal = matrix[~torch.eye(matrix.shape[0], dtype=torch.bool)]
    return float(off_diagonal.abs().mean().item())
