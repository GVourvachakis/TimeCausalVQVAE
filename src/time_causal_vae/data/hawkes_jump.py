"""Synthetic Hawkes-jump market-path dataset.

The generator is intended for rare-event smoke tests and future discrete-latent
benchmarks. It produces positive one-dimensional normalised price paths using a
Brownian component, self-exciting jump arrivals, asymmetric jump marks, and an
optional jump-excited volatility state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, sqrt
from typing import Literal, cast

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from time_causal_vae.data.ogata_hawkes_jump import (
    OgataHawkesJumpDataset,
    OgataHawkesJumpSimulation,
    simulate_ogata_hawkes_paths,
    simulate_single_ogata_path,
)

__all__ = [
    "HawkesJumpDataset",
    "HawkesJumpSimulation",
    "OgataHawkesJumpDataset",
    "OgataHawkesJumpSimulation",
    "simulate_hawkes_jump_paths",
    "simulate_ogata_hawkes_paths",
    "simulate_single_ogata_path",
]


@dataclass(frozen=True)
class HawkesJumpSimulation:
    """Container for Hawkes-jump simulated paths and oracle event metadata."""

    prices: Tensor
    log_returns: Tensor
    jump_indicators: Tensor
    jump_counts: Tensor
    jump_sizes: Tensor
    intensities: Tensor
    volatilities: Tensor
    metadata: dict[str, float]


def simulate_hawkes_jump_paths(
    n_sample: int,
    n_timestep: int,
    *,
    seed: int | None = None,
    dt: float = 1.0 / 60.0,
    drift: float = 0.0,
    brownian_volatility: float = 0.18,
    baseline_intensity: float = 3.0,
    excitation: float = 2.0,
    decay: float = 12.0,
    mark_excitation: float = 20.0,
    max_intensity: float = 80.0,
    max_jumps_per_step: int = 8,
    negative_jump_probability: float = 0.7,
    positive_jump_mean: float = 0.018,
    positive_jump_std: float = 0.008,
    negative_jump_mean: float = 0.035,
    negative_jump_std: float = 0.018,
    severe_jump_probability: float = 0.08,
    severe_jump_mean: float = 0.12,
    severe_jump_std: float = 0.04,
    volatility_excitation: bool = True,
    volatility_excitation_scale: float = 1.2,
    volatility_decay: float = 18.0,
    max_volatility: float = 1.5,
    dtype: torch.dtype = torch.float32,
) -> HawkesJumpSimulation:
    """Simulate normalised paths from a discrete-time Hawkes-jump approximation.

    The returned tensors use shape ``[n_sample, n_timestep, 1]``. The first time
    index is the initial state; log returns and jump metadata at index zero are
    therefore zero. Jump metadata is oracle simulator information and should not
    be exposed as a model-visible condition unless a later benchmark explicitly
    justifies that choice.
    """
    _validate_simulation_inputs(
        n_sample=n_sample,
        n_timestep=n_timestep,
        dt=dt,
        brownian_volatility=brownian_volatility,
        baseline_intensity=baseline_intensity,
        excitation=excitation,
        decay=decay,
        max_intensity=max_intensity,
        max_jumps_per_step=max_jumps_per_step,
        negative_jump_probability=negative_jump_probability,
        severe_jump_probability=severe_jump_probability,
        volatility_decay=volatility_decay,
        max_volatility=max_volatility,
    )

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))

    prices = torch.ones((n_sample, n_timestep), dtype=dtype)
    log_returns = torch.zeros((n_sample, n_timestep), dtype=dtype)
    jump_counts = torch.zeros((n_sample, n_timestep), dtype=torch.long)
    jump_sizes = torch.zeros((n_sample, n_timestep), dtype=dtype)
    jump_indicators = torch.zeros((n_sample, n_timestep), dtype=torch.bool)
    intensities = torch.zeros((n_sample, n_timestep), dtype=dtype)
    volatilities = torch.zeros((n_sample, n_timestep), dtype=dtype)

    intensity = torch.full((n_sample,), baseline_intensity, dtype=dtype)
    volatility_state = torch.zeros((n_sample,), dtype=dtype)
    sqrt_dt = sqrt(dt)
    intensity_decay = exp(-decay * dt)
    volatility_decay_factor = exp(-volatility_decay * dt)
    intensities[:, 0] = intensity
    volatilities[:, 0] = brownian_volatility

    for step in range(1, n_timestep):
        current_volatility = torch.clamp(
            torch.full((n_sample,), brownian_volatility, dtype=dtype) + volatility_state,
            max=max_volatility,
        )
        intensities[:, step] = intensity
        volatilities[:, step] = current_volatility

        event_rates = torch.clamp(intensity * dt, min=0.0, max=float(max_jumps_per_step))
        step_counts = torch.poisson(event_rates, generator=generator).long()
        step_counts = torch.clamp(step_counts, max=max_jumps_per_step)
        aggregate_marks = _sample_aggregate_marks(
            step_counts,
            generator=generator,
            dtype=dtype,
            negative_jump_probability=negative_jump_probability,
            positive_jump_mean=positive_jump_mean,
            positive_jump_std=positive_jump_std,
            negative_jump_mean=negative_jump_mean,
            negative_jump_std=negative_jump_std,
            severe_jump_probability=severe_jump_probability,
            severe_jump_mean=severe_jump_mean,
            severe_jump_std=severe_jump_std,
        )

        brownian_noise = torch.randn(n_sample, generator=generator, dtype=dtype)
        step_returns = drift * dt + current_volatility * sqrt_dt * brownian_noise
        step_returns = step_returns + aggregate_marks

        log_returns[:, step] = step_returns
        prices[:, step] = prices[:, step - 1] * torch.exp(step_returns)
        jump_counts[:, step] = step_counts
        jump_sizes[:, step] = aggregate_marks
        jump_indicators[:, step] = step_counts > 0

        mark_boost = mark_excitation * aggregate_marks.abs()
        intensity = baseline_intensity + (intensity - baseline_intensity) * intensity_decay
        intensity = intensity + excitation * step_counts.to(dtype=dtype) + mark_boost
        intensity = torch.clamp(intensity, min=0.0, max=max_intensity)

        if volatility_excitation:
            volatility_state = volatility_state * volatility_decay_factor
            volatility_state = (
                volatility_state + volatility_excitation_scale * aggregate_marks.abs()
            )
            volatility_state = torch.clamp(
                volatility_state,
                min=0.0,
                max=max(0.0, max_volatility - brownian_volatility),
            )
        else:
            volatility_state.zero_()

    return HawkesJumpSimulation(
        prices=prices.unsqueeze(-1),
        log_returns=log_returns.unsqueeze(-1),
        jump_indicators=jump_indicators.unsqueeze(-1),
        jump_counts=jump_counts.unsqueeze(-1),
        jump_sizes=jump_sizes.unsqueeze(-1),
        intensities=intensities.unsqueeze(-1),
        volatilities=volatilities.unsqueeze(-1),
        metadata=_simulation_metadata(
            prices=prices,
            jump_counts=jump_counts,
            jump_sizes=jump_sizes,
            intensities=intensities,
            volatilities=volatilities,
            baseline_intensity=baseline_intensity,
            excitation=excitation,
            decay=decay,
            dt=dt,
        ),
    )


class HawkesJumpDataset(TensorDataset):
    """Torch dataset exposing Hawkes-jump paths with oracle metadata attributes."""

    def __init__(
        self,
        n_sample: int,
        n_timestep: int,
        *,
        seed: int | None = None,
        data_output: Literal["price", "log_return"] = "price",
        simulation_scheme: Literal["fixed_grid", "ogata"] = "fixed_grid",
        **kwargs: object,
    ) -> None:
        """Simulate a Hawkes-jump dataset.

        ``simulation_scheme="fixed_grid"`` preserves the original fast
        discrete-time approximation. ``simulation_scheme="ogata"`` uses exact
        continuous-time Hawkes arrivals and then observes the path on the same
        regular grid.
        """
        if data_output not in {"price", "log_return"}:
            raise ValueError("data_output must be 'price' or 'log_return'.")
        if simulation_scheme not in {"fixed_grid", "ogata"}:
            raise ValueError("simulation_scheme must be 'fixed_grid' or 'ogata'.")
        simulation = _simulate_selected_scheme(
            n_sample=n_sample,
            n_timestep=n_timestep,
            seed=seed,
            simulation_scheme=simulation_scheme,
            kwargs=kwargs,
        )
        self.simulation = simulation
        self.prices = simulation.prices
        self.log_returns = simulation.log_returns
        self.jump_indicators = simulation.jump_indicators
        self.jump_counts = simulation.jump_counts
        self.jump_sizes = simulation.jump_sizes
        self.intensities = simulation.intensities
        self.volatilities = simulation.volatilities
        self.metadata = simulation.metadata
        self.data = simulation.log_returns if data_output == "log_return" else simulation.prices
        self.labels = torch.ones((n_sample, 1), dtype=torch.float32)
        super().__init__(self.data)


def _simulate_selected_scheme(
    *,
    n_sample: int,
    n_timestep: int,
    seed: int | None,
    simulation_scheme: Literal["fixed_grid", "ogata"],
    kwargs: Mapping[str, object],
) -> HawkesJumpSimulation | OgataHawkesJumpSimulation:
    dt = _float_kwarg(kwargs, "dt", 1.0 / 60.0)
    drift = _float_kwarg(kwargs, "drift", 0.0)
    brownian_volatility = _float_kwarg(kwargs, "brownian_volatility", 0.18)
    baseline_intensity = _float_kwarg(kwargs, "baseline_intensity", 3.0)
    excitation = _float_kwarg(kwargs, "excitation", 2.0)
    decay = _float_kwarg(kwargs, "decay", 12.0)
    mark_excitation = _float_kwarg(kwargs, "mark_excitation", 20.0)
    max_intensity = _float_kwarg(kwargs, "max_intensity", 80.0)
    negative_jump_probability = _float_kwarg(kwargs, "negative_jump_probability", 0.7)
    positive_jump_mean = _float_kwarg(kwargs, "positive_jump_mean", 0.018)
    positive_jump_std = _float_kwarg(kwargs, "positive_jump_std", 0.008)
    negative_jump_mean = _float_kwarg(kwargs, "negative_jump_mean", 0.035)
    negative_jump_std = _float_kwarg(kwargs, "negative_jump_std", 0.018)
    severe_jump_probability = _float_kwarg(kwargs, "severe_jump_probability", 0.08)
    severe_jump_mean = _float_kwarg(kwargs, "severe_jump_mean", 0.12)
    severe_jump_std = _float_kwarg(kwargs, "severe_jump_std", 0.04)
    volatility_excitation = _bool_kwarg(kwargs, "volatility_excitation", True)
    volatility_excitation_scale = _float_kwarg(kwargs, "volatility_excitation_scale", 1.2)
    volatility_decay = _float_kwarg(kwargs, "volatility_decay", 18.0)
    max_volatility = _float_kwarg(kwargs, "max_volatility", 1.5)
    if simulation_scheme == "ogata":
        return simulate_ogata_hawkes_paths(
            n_sample,
            n_timestep,
            seed=seed,
            dt=dt,
            drift=drift,
            brownian_volatility=brownian_volatility,
            baseline_intensity=baseline_intensity,
            excitation=excitation,
            decay=decay,
            mark_excitation=mark_excitation,
            max_intensity=max_intensity,
            negative_jump_probability=negative_jump_probability,
            positive_jump_mean=positive_jump_mean,
            positive_jump_std=positive_jump_std,
            negative_jump_mean=negative_jump_mean,
            negative_jump_std=negative_jump_std,
            severe_jump_probability=severe_jump_probability,
            severe_jump_mean=severe_jump_mean,
            severe_jump_std=severe_jump_std,
            volatility_excitation=volatility_excitation,
            volatility_excitation_scale=volatility_excitation_scale,
            volatility_decay=volatility_decay,
            max_volatility=max_volatility,
        )
    return simulate_hawkes_jump_paths(
        n_sample,
        n_timestep,
        seed=seed,
        dt=dt,
        drift=drift,
        brownian_volatility=brownian_volatility,
        baseline_intensity=baseline_intensity,
        excitation=excitation,
        decay=decay,
        mark_excitation=mark_excitation,
        max_intensity=max_intensity,
        max_jumps_per_step=_int_kwarg(kwargs, "max_jumps_per_step", 8),
        negative_jump_probability=negative_jump_probability,
        positive_jump_mean=positive_jump_mean,
        positive_jump_std=positive_jump_std,
        negative_jump_mean=negative_jump_mean,
        negative_jump_std=negative_jump_std,
        severe_jump_probability=severe_jump_probability,
        severe_jump_mean=severe_jump_mean,
        severe_jump_std=severe_jump_std,
        volatility_excitation=volatility_excitation,
        volatility_excitation_scale=volatility_excitation_scale,
        volatility_decay=volatility_decay,
        max_volatility=max_volatility,
    )


def _float_kwarg(kwargs: Mapping[str, object], name: str, default: float) -> float:
    value = kwargs.get(name, default)
    if value is None:
        return default
    return float(cast(float | int | str, value))


def _int_kwarg(kwargs: Mapping[str, object], name: str, default: int) -> int:
    value = kwargs.get(name, default)
    if value is None:
        return default
    return int(cast(float | int | str, value))


def _bool_kwarg(kwargs: Mapping[str, object], name: str, default: bool) -> bool:
    value = kwargs.get(name, default)
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _sample_aggregate_marks(
    counts: Tensor,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
    negative_jump_probability: float,
    positive_jump_mean: float,
    positive_jump_std: float,
    negative_jump_mean: float,
    negative_jump_std: float,
    severe_jump_probability: float,
    severe_jump_mean: float,
    severe_jump_std: float,
) -> Tensor:
    marks = torch.zeros(counts.shape[0], dtype=dtype)
    non_zero_indices = torch.nonzero(counts > 0, as_tuple=False).flatten()
    for path_index_tensor in non_zero_indices:
        path_index = int(path_index_tensor.item())
        count = int(counts[path_index].item())
        signs = torch.rand(count, generator=generator)
        severe_draws = torch.rand(count, generator=generator)
        negative_mask = signs < negative_jump_probability
        severe_mask = negative_mask & (severe_draws < severe_jump_probability)

        positive_marks = torch.abs(
            positive_jump_mean + positive_jump_std * torch.randn(count, generator=generator)
        )
        negative_marks = -torch.abs(
            negative_jump_mean + negative_jump_std * torch.randn(count, generator=generator)
        )
        severe_marks = -torch.abs(
            severe_jump_mean + severe_jump_std * torch.randn(count, generator=generator)
        )
        path_marks = torch.where(negative_mask, negative_marks, positive_marks)
        path_marks = torch.where(severe_mask, severe_marks, path_marks)
        marks[path_index] = path_marks.to(dtype=dtype).sum()
    return marks


def _simulation_metadata(
    *,
    prices: Tensor,
    jump_counts: Tensor,
    jump_sizes: Tensor,
    intensities: Tensor,
    volatilities: Tensor,
    baseline_intensity: float,
    excitation: float,
    decay: float,
    dt: float,
) -> dict[str, float]:
    total_jump_steps = int((jump_counts > 0).sum().item())
    total_jumps = int(jump_counts.sum().item())
    nonzero_jumps = jump_sizes[jump_counts > 0]
    negative_jump_fraction = 0.0
    mean_jump_size = 0.0
    mean_abs_jump_size = 0.0
    if nonzero_jumps.numel() > 0:
        negative_jump_fraction = float((nonzero_jumps < 0.0).float().mean().item())
        mean_jump_size = float(nonzero_jumps.mean().item())
        mean_abs_jump_size = float(nonzero_jumps.abs().mean().item())
    jump_counts_per_path = jump_counts.sum(dim=1).float()
    return {
        "n_sample": float(prices.shape[0]),
        "n_timestep": float(prices.shape[1]),
        "dt": float(dt),
        "baseline_intensity": float(baseline_intensity),
        "excitation": float(excitation),
        "decay": float(decay),
        "branching_ratio_proxy": float(excitation / decay),
        "total_jumps": float(total_jumps),
        "total_jump_steps": float(total_jump_steps),
        "mean_jump_count_per_path": float(jump_counts_per_path.mean().item()),
        "paths_with_jump_fraction": float((jump_counts_per_path > 0.0).float().mean().item()),
        "negative_jump_fraction": negative_jump_fraction,
        "mean_jump_size": mean_jump_size,
        "mean_abs_jump_size": mean_abs_jump_size,
        "max_intensity_observed": float(intensities.max().item()),
        "mean_intensity_observed": float(intensities.mean().item()),
        "max_volatility_observed": float(volatilities.max().item()),
        "mean_volatility_observed": float(volatilities.mean().item()),
        "min_price": float(prices.min().item()),
        "max_price": float(prices.max().item()),
    }


def _validate_simulation_inputs(
    *,
    n_sample: int,
    n_timestep: int,
    dt: float,
    brownian_volatility: float,
    baseline_intensity: float,
    excitation: float,
    decay: float,
    max_intensity: float,
    max_jumps_per_step: int,
    negative_jump_probability: float,
    severe_jump_probability: float,
    volatility_decay: float,
    max_volatility: float,
) -> None:
    if n_sample <= 0:
        raise ValueError("n_sample must be positive.")
    if n_timestep <= 1:
        raise ValueError("n_timestep must be greater than one.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if brownian_volatility < 0.0:
        raise ValueError("brownian_volatility must be non-negative.")
    if baseline_intensity < 0.0:
        raise ValueError("baseline_intensity must be non-negative.")
    if excitation < 0.0:
        raise ValueError("excitation must be non-negative.")
    if decay <= 0.0:
        raise ValueError("decay must be positive.")
    if excitation / decay >= 1.0:
        raise ValueError("excitation / decay must stay below one for Hawkes stability.")
    if max_intensity <= 0.0:
        raise ValueError("max_intensity must be positive.")
    if max_jumps_per_step <= 0:
        raise ValueError("max_jumps_per_step must be positive.")
    if not 0.0 <= negative_jump_probability <= 1.0:
        raise ValueError("negative_jump_probability must be in [0, 1].")
    if not 0.0 <= severe_jump_probability <= 1.0:
        raise ValueError("severe_jump_probability must be in [0, 1].")
    if volatility_decay <= 0.0:
        raise ValueError("volatility_decay must be positive.")
    if max_volatility <= 0.0:
        raise ValueError("max_volatility must be positive.")
