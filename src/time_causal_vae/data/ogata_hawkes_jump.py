"""Continuous-time Ogata simulator for Hawkes-jump market paths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, sqrt
from typing import Literal, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class OgataHawkesJumpSimulation:
    """Container for Ogata-simulated paths and oracle event metadata."""

    prices: Tensor
    log_returns: Tensor
    jump_indicators: Tensor
    jump_counts: Tensor
    jump_sizes: Tensor
    intensities: Tensor
    volatilities: Tensor
    metadata: dict[str, float]


@dataclass(frozen=True)
class _OgataParams:
    drift: float
    brownian_volatility: float
    baseline_intensity: float
    excitation: float
    decay: float
    mark_excitation: float
    max_intensity: float
    negative_jump_probability: float
    positive_jump_mean: float
    positive_jump_std: float
    negative_jump_mean: float
    negative_jump_std: float
    severe_jump_probability: float
    severe_jump_mean: float
    severe_jump_std: float
    volatility_excitation: bool
    volatility_excitation_scale: float
    volatility_decay: float
    max_volatility: float


def simulate_single_ogata_path(
    t_max: float,
    params: Mapping[str, object] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[float], list[float]]:
    """Simulate one continuous-time Hawkes event path with Ogata thinning."""
    ogata_params = _params_from_mapping({} if params is None else params)
    event_times, event_marks, _, _ = _simulate_ogata_full(
        t_max,
        ogata_params,
        rng or np.random.default_rng(),
    )
    return event_times, event_marks


def simulate_ogata_hawkes_paths(
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
) -> OgataHawkesJumpSimulation:
    """Simulate SVMHJD paths with exact event times and fixed-grid observations."""
    _validate_ogata_inputs(
        n_sample=n_sample,
        n_timestep=n_timestep,
        dt=dt,
        brownian_volatility=brownian_volatility,
        baseline_intensity=baseline_intensity,
        excitation=excitation,
        decay=decay,
        max_intensity=max_intensity,
        negative_jump_probability=negative_jump_probability,
        severe_jump_probability=severe_jump_probability,
        volatility_decay=volatility_decay,
        max_volatility=max_volatility,
    )
    params = _OgataParams(
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

    horizon = (n_timestep - 1) * dt
    rng_master = np.random.default_rng(seed)
    np_prices = np.ones((n_sample, n_timestep), dtype=np.float64)
    np_log_returns = np.zeros((n_sample, n_timestep), dtype=np.float64)
    np_jump_counts = np.zeros((n_sample, n_timestep), dtype=np.int64)
    np_jump_sizes = np.zeros((n_sample, n_timestep), dtype=np.float64)
    np_jump_indicators = np.zeros((n_sample, n_timestep), dtype=bool)
    np_intensities = np.zeros((n_sample, n_timestep), dtype=np.float64)
    np_volatilities = np.zeros((n_sample, n_timestep), dtype=np.float64)

    for index in range(n_sample):
        path_seed = int(rng_master.integers(0, 2**31))
        rng = np.random.default_rng(path_seed)
        event_times, event_marks, _, _ = _simulate_ogata_full(horizon, params, rng)
        (
            np_prices[index],
            np_log_returns[index],
            np_jump_indicators[index],
            np_jump_counts[index],
            np_jump_sizes[index],
            np_intensities[index],
            np_volatilities[index],
        ) = _project_to_grid(
            event_times=event_times,
            event_marks=event_marks,
            n_timestep=n_timestep,
            dt=dt,
            params=params,
            rng=rng,
        )

    prices = torch.from_numpy(np_prices).to(dtype).unsqueeze(-1)
    log_returns = torch.from_numpy(np_log_returns).to(dtype).unsqueeze(-1)
    jump_indicators = torch.from_numpy(np_jump_indicators).unsqueeze(-1)
    jump_counts = torch.from_numpy(np_jump_counts).long().unsqueeze(-1)
    jump_sizes = torch.from_numpy(np_jump_sizes).to(dtype).unsqueeze(-1)
    intensities = torch.from_numpy(np_intensities).to(dtype).unsqueeze(-1)
    volatilities = torch.from_numpy(np_volatilities).to(dtype).unsqueeze(-1)

    return OgataHawkesJumpSimulation(
        prices=prices,
        log_returns=log_returns,
        jump_indicators=jump_indicators,
        jump_counts=jump_counts,
        jump_sizes=jump_sizes,
        intensities=intensities,
        volatilities=volatilities,
        metadata=_simulation_metadata(
            prices=np_prices,
            jump_counts=np_jump_counts,
            jump_sizes=np_jump_sizes,
            intensities=np_intensities,
            volatilities=np_volatilities,
            baseline_intensity=baseline_intensity,
            excitation=excitation,
            decay=decay,
            dt=dt,
        ),
    )


class OgataHawkesJumpDataset(TensorDataset):
    """Torch dataset exposing exact Ogata paths with fixed-grid tensors."""

    def __init__(
        self,
        n_sample: int,
        n_timestep: int,
        *,
        seed: int | None = None,
        data_output: Literal["price", "log_return"] = "price",
        **kwargs: object,
    ) -> None:
        """Simulate an Ogata Hawkes-jump dataset."""
        if data_output not in {"price", "log_return"}:
            raise ValueError("data_output must be 'price' or 'log_return'.")
        simulation = simulate_ogata_hawkes_paths(
            n_sample,
            n_timestep,
            seed=seed,
            dt=_float_kwarg(kwargs, "dt", 1.0 / 60.0),
            drift=_float_kwarg(kwargs, "drift", 0.0),
            brownian_volatility=_float_kwarg(kwargs, "brownian_volatility", 0.18),
            baseline_intensity=_float_kwarg(kwargs, "baseline_intensity", 3.0),
            excitation=_float_kwarg(kwargs, "excitation", 2.0),
            decay=_float_kwarg(kwargs, "decay", 12.0),
            mark_excitation=_float_kwarg(kwargs, "mark_excitation", 20.0),
            max_intensity=_float_kwarg(kwargs, "max_intensity", 80.0),
            negative_jump_probability=_float_kwarg(kwargs, "negative_jump_probability", 0.7),
            positive_jump_mean=_float_kwarg(kwargs, "positive_jump_mean", 0.018),
            positive_jump_std=_float_kwarg(kwargs, "positive_jump_std", 0.008),
            negative_jump_mean=_float_kwarg(kwargs, "negative_jump_mean", 0.035),
            negative_jump_std=_float_kwarg(kwargs, "negative_jump_std", 0.018),
            severe_jump_probability=_float_kwarg(kwargs, "severe_jump_probability", 0.08),
            severe_jump_mean=_float_kwarg(kwargs, "severe_jump_mean", 0.12),
            severe_jump_std=_float_kwarg(kwargs, "severe_jump_std", 0.04),
            volatility_excitation=_bool_kwarg(kwargs, "volatility_excitation", True),
            volatility_excitation_scale=_float_kwarg(kwargs, "volatility_excitation_scale", 1.2),
            volatility_decay=_float_kwarg(kwargs, "volatility_decay", 18.0),
            max_volatility=_float_kwarg(kwargs, "max_volatility", 1.5),
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


def _simulate_ogata_full(
    t_max: float,
    params: _OgataParams,
    rng: np.random.Generator,
) -> tuple[list[float], list[float], list[float], list[float]]:
    event_times: list[float] = []
    event_marks: list[float] = []
    post_jump_intensities: list[float] = []
    post_jump_vol_states: list[float] = []
    time = 0.0
    intensity = params.baseline_intensity
    vol_state = 0.0
    max_vol_state = max(0.0, params.max_volatility - params.brownian_volatility)

    while time < t_max:
        upper_bound = max(intensity, 1e-12)
        waiting_time = float(rng.exponential(1.0 / upper_bound))
        candidate_time = time + waiting_time
        if candidate_time >= t_max:
            elapsed = t_max - time
            intensity = _decay_intensity(intensity, elapsed, params)
            vol_state *= exp(-params.volatility_decay * elapsed)
            break

        candidate_intensity = _decay_intensity(intensity, waiting_time, params)
        if rng.uniform() <= candidate_intensity / upper_bound:
            time = candidate_time
            mark = _sample_mark(rng, params)
            abs_mark = abs(mark)
            intensity = min(
                candidate_intensity + params.excitation + params.mark_excitation * abs_mark,
                params.max_intensity,
            )
            if params.volatility_excitation:
                vol_state = (
                    vol_state * exp(-params.volatility_decay * waiting_time)
                    + params.volatility_excitation_scale * abs_mark
                )
                vol_state = min(vol_state, max_vol_state)
            else:
                vol_state = 0.0

            event_times.append(time)
            event_marks.append(mark)
            post_jump_intensities.append(intensity)
            post_jump_vol_states.append(vol_state)
        else:
            intensity = candidate_intensity
            vol_state *= exp(-params.volatility_decay * waiting_time)
            time = candidate_time

    return event_times, event_marks, post_jump_intensities, post_jump_vol_states


def _project_to_grid(
    *,
    event_times: list[float],
    event_marks: list[float],
    n_timestep: int,
    dt: float,
    params: _OgataParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prices = np.ones(n_timestep, dtype=np.float64)
    log_returns = np.zeros(n_timestep, dtype=np.float64)
    jump_counts = np.zeros(n_timestep, dtype=np.int64)
    jump_sizes = np.zeros(n_timestep, dtype=np.float64)
    jump_indicators = np.zeros(n_timestep, dtype=bool)
    intensities = np.zeros(n_timestep, dtype=np.float64)
    volatilities = np.zeros(n_timestep, dtype=np.float64)

    times = np.asarray(event_times, dtype=np.float64)
    marks = np.asarray(event_marks, dtype=np.float64)
    intensity = params.baseline_intensity
    vol_state = 0.0
    last_time = 0.0
    event_index = 0
    max_vol_state = max(0.0, params.max_volatility - params.brownian_volatility)
    intensities[0] = intensity
    volatilities[0] = params.brownian_volatility

    for step in range(1, n_timestep):
        step_start = (step - 1) * dt
        step_end = step * dt
        if last_time < step_start:
            elapsed = step_start - last_time
            intensity = _decay_intensity(intensity, elapsed, params)
            vol_state *= exp(-params.volatility_decay * elapsed)
            last_time = step_start

        intensities[step] = intensity
        volatilities[step] = min(
            params.brownian_volatility + vol_state,
            params.max_volatility,
        )
        step_variance = 0.0
        step_mark_sum = 0.0
        step_count = 0

        while event_index < len(times) and times[event_index] < step_end:
            jump_time = float(times[event_index])
            jump_mark = float(marks[event_index])
            gap = jump_time - last_time
            if gap > 1e-15:
                step_variance += _subinterval_diffusion_variance(
                    gap,
                    vol_state,
                    params.brownian_volatility,
                    params.volatility_decay,
                )
                intensity = _decay_intensity(intensity, gap, params)
                vol_state *= exp(-params.volatility_decay * gap)

            abs_mark = abs(jump_mark)
            intensity = min(
                intensity + params.excitation + params.mark_excitation * abs_mark,
                params.max_intensity,
            )
            if params.volatility_excitation:
                vol_state = min(
                    vol_state + params.volatility_excitation_scale * abs_mark,
                    max_vol_state,
                )
            else:
                vol_state = 0.0
            step_mark_sum += jump_mark
            step_count += 1
            last_time = jump_time
            event_index += 1

        final_gap = step_end - last_time
        if final_gap > 1e-15:
            step_variance += _subinterval_diffusion_variance(
                final_gap,
                vol_state,
                params.brownian_volatility,
                params.volatility_decay,
            )
            intensity = _decay_intensity(intensity, final_gap, params)
            vol_state *= exp(-params.volatility_decay * final_gap)
            last_time = step_end

        step_return = (
            params.drift * dt
            + sqrt(max(step_variance, 0.0)) * float(rng.standard_normal())
            + step_mark_sum
        )
        log_returns[step] = step_return
        prices[step] = prices[step - 1] * exp(step_return)
        jump_counts[step] = step_count
        jump_sizes[step] = step_mark_sum
        jump_indicators[step] = step_count > 0

    return prices, log_returns, jump_indicators, jump_counts, jump_sizes, intensities, volatilities


def _subinterval_diffusion_variance(
    gap: float,
    vol_state_at_start: float,
    brownian_volatility: float,
    volatility_decay: float,
) -> float:
    r"""Return exact ``Integral sigma(t)^2 dt`` over a no-jump sub-interval.

    Between jumps, ``sigma(t) = v_base + vs * exp(-kappa_v * (t - a))`` is
    deterministic. Therefore, for ``alpha = exp(-kappa_v * Delta t)``:

    ``v_base^2 * Delta t + 2 v_base vs (1-alpha) / kappa_v
    + vs^2 (1-alpha^2) / (2 kappa_v)``.
    """
    if gap <= 1e-15:
        return 0.0
    alpha = exp(-volatility_decay * gap)
    return (
        brownian_volatility * brownian_volatility * gap
        + 2.0 * brownian_volatility * vol_state_at_start * (1.0 - alpha) / volatility_decay
        + vol_state_at_start * vol_state_at_start * (1.0 - alpha * alpha) / (2.0 * volatility_decay)
    )


def _sample_mark(rng: np.random.Generator, params: _OgataParams) -> float:
    if rng.uniform() < params.negative_jump_probability:
        if rng.uniform() < params.severe_jump_probability:
            return -abs(float(rng.normal(params.severe_jump_mean, params.severe_jump_std)))
        return -abs(float(rng.normal(params.negative_jump_mean, params.negative_jump_std)))
    return abs(float(rng.normal(params.positive_jump_mean, params.positive_jump_std)))


def _decay_intensity(intensity: float, elapsed: float, params: _OgataParams) -> float:
    return params.baseline_intensity + (intensity - params.baseline_intensity) * exp(
        -params.decay * elapsed
    )


def _simulation_metadata(
    *,
    prices: np.ndarray,
    jump_counts: np.ndarray,
    jump_sizes: np.ndarray,
    intensities: np.ndarray,
    volatilities: np.ndarray,
    baseline_intensity: float,
    excitation: float,
    decay: float,
    dt: float,
) -> dict[str, float]:
    total_jump_steps = int((jump_counts > 0).sum())
    total_jumps = int(jump_counts.sum())
    nonzero_jumps = jump_sizes[jump_counts > 0]
    negative_jump_fraction = 0.0
    mean_jump_size = 0.0
    mean_abs_jump_size = 0.0
    if nonzero_jumps.size > 0:
        negative_jump_fraction = float((nonzero_jumps < 0.0).mean())
        mean_jump_size = float(nonzero_jumps.mean())
        mean_abs_jump_size = float(np.abs(nonzero_jumps).mean())
    jump_counts_per_path = jump_counts.sum(axis=1).astype(np.float64)
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
        "mean_jump_count_per_path": float(jump_counts_per_path.mean()),
        "paths_with_jump_fraction": float((jump_counts_per_path > 0.0).mean()),
        "negative_jump_fraction": negative_jump_fraction,
        "mean_jump_size": mean_jump_size,
        "mean_abs_jump_size": mean_abs_jump_size,
        "max_intensity_observed": float(intensities.max()),
        "mean_intensity_observed": float(intensities.mean()),
        "max_volatility_observed": float(volatilities.max()),
        "mean_volatility_observed": float(volatilities.mean()),
        "min_price": float(prices.min()),
        "max_price": float(prices.max()),
    }


def _params_from_mapping(params: Mapping[str, object]) -> _OgataParams:
    return _OgataParams(
        drift=_float_kwarg(params, "drift", 0.0),
        brownian_volatility=_float_kwarg(params, "brownian_volatility", 0.18),
        baseline_intensity=_float_kwarg(params, "baseline_intensity", 3.0),
        excitation=_float_kwarg(params, "excitation", 2.0),
        decay=_float_kwarg(params, "decay", 12.0),
        mark_excitation=_float_kwarg(params, "mark_excitation", 20.0),
        max_intensity=_float_kwarg(params, "max_intensity", 80.0),
        negative_jump_probability=_float_kwarg(params, "negative_jump_probability", 0.7),
        positive_jump_mean=_float_kwarg(params, "positive_jump_mean", 0.018),
        positive_jump_std=_float_kwarg(params, "positive_jump_std", 0.008),
        negative_jump_mean=_float_kwarg(params, "negative_jump_mean", 0.035),
        negative_jump_std=_float_kwarg(params, "negative_jump_std", 0.018),
        severe_jump_probability=_float_kwarg(params, "severe_jump_probability", 0.08),
        severe_jump_mean=_float_kwarg(params, "severe_jump_mean", 0.12),
        severe_jump_std=_float_kwarg(params, "severe_jump_std", 0.04),
        volatility_excitation=_bool_kwarg(params, "volatility_excitation", True),
        volatility_excitation_scale=_float_kwarg(params, "volatility_excitation_scale", 1.2),
        volatility_decay=_float_kwarg(params, "volatility_decay", 18.0),
        max_volatility=_float_kwarg(params, "max_volatility", 1.5),
    )


def _float_kwarg(kwargs: Mapping[str, object], name: str, default: float) -> float:
    value = kwargs.get(name, default)
    if value is None:
        return default
    return float(cast(float | int | str, value))


def _bool_kwarg(kwargs: Mapping[str, object], name: str, default: bool) -> bool:
    value = kwargs.get(name, default)
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _validate_ogata_inputs(
    *,
    n_sample: int,
    n_timestep: int,
    dt: float,
    brownian_volatility: float,
    baseline_intensity: float,
    excitation: float,
    decay: float,
    max_intensity: float,
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
    if brownian_volatility <= 0.0:
        raise ValueError("brownian_volatility must be positive for Ogata simulation.")
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
    if not 0.0 <= negative_jump_probability <= 1.0:
        raise ValueError("negative_jump_probability must be in [0, 1].")
    if not 0.0 <= severe_jump_probability <= 1.0:
        raise ValueError("severe_jump_probability must be in [0, 1].")
    if volatility_decay <= 0.0:
        raise ValueError("volatility_decay must be positive.")
    if max_volatility <= brownian_volatility:
        raise ValueError("max_volatility must exceed brownian_volatility.")
