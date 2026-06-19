"""Local S&P500 50-stock panel dataset.

The dataset reads tensors produced by ``scripts/download_sp500_50_panel.py``.
Downloaded data remains local-only under ``data/raw`` and ``data/processed``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from time_causal_vae.data.factor_projected_market import (
    FactorProjectionState,
    coerce_factor_projection_state,
    factor_projection_metadata,
    fit_train_pca_projection,
    project_returns_to_factors,
    reconstruct_model_visible_returns_from_factors,
)
from time_causal_vae.data.multifactor_market import (
    coerce_standardization_stats,
    fit_return_standardization,
    inverse_standardize_multifactor_returns,
    standardize_multifactor_returns,
)

__all__ = ["SP50050PanelDataset"]

PanelSplit = Literal["train", "eval", "all"]
PanelProjectionMode = Literal["train_pca"]


class SP50050PanelDataset(TensorDataset):
    """Torch dataset exposing 50-stock daily log-return windows and conditions."""

    def __init__(
        self,
        n_sample: int | None,
        n_timestep: int = 60,
        *,
        base_data_dir: str | Path = "data/processed",
        processed_subdir: str = "sp500_50_panel",
        split: PanelSplit = "all",
        condition_mode: str | None = None,
        projection_mode: PanelProjectionMode | None = None,
        projection_n_factors: int | None = None,
        projection_state: FactorProjectionState | Mapping[str, Any] | None = None,
        standardize_returns: bool = False,
        standardization_stats: Mapping[str, Any] | None = None,
        standardization_epsilon: float = 1e-6,
    ) -> None:
        """Load a processed local S&P500 50-stock panel.

        Parameters
        ----------
        n_sample:
            Number of windows to expose from the selected split. ``None`` uses
            every available window in that split.
        n_timestep:
            Required window length. The first empirical profile uses 60.
        base_data_dir:
            Base processed-data directory, normally ``data/processed``.
        processed_subdir:
            Subdirectory containing ``data.pt``, ``labels.pt``, and
            ``metadata.json``.
        split:
            One of ``"train"``, ``"eval"``, or ``"all"``.
        condition_mode:
            Optional guard against loading tensors with an unexpected condition
            convention.
        projection_mode:
            Optional factor-projected view. The empirical convention only
            supports ``"train_pca"`` because no oracle loading matrix exists.
        projection_n_factors:
            Number of train-PCA factors to expose when ``projection_mode`` is
            enabled.
        projection_state:
            Train-fitted projection state reused for eval data.
        standardize_returns:
            Expose train-standardised log returns to the model while retaining
            raw log returns on the dataset for diagnostics.
        standardization_stats:
            Optional train-fitted per-asset mean/std reused for eval data.
        standardization_epsilon:
            Minimum per-asset standard deviation when fitting train-only stats.
        """
        if n_sample is not None and n_sample <= 0:
            raise ValueError("n_sample must be positive when provided.")
        if n_timestep <= 1:
            raise ValueError("n_timestep must be greater than one.")
        if standardization_epsilon <= 0.0:
            raise ValueError("standardization_epsilon must be positive.")
        if split not in {"train", "eval", "all"}:
            raise ValueError("split must be 'train', 'eval', or 'all'.")

        directory = Path(base_data_dir) / processed_subdir
        metadata = _load_metadata(directory / "metadata.json")
        if condition_mode is not None and metadata.get("condition_mode") != condition_mode:
            raise ValueError(
                "Processed panel condition_mode mismatch: "
                f"expected {condition_mode!r}, got {metadata.get('condition_mode')!r}."
            )

        data = _load_raw_return_tensor(directory)
        labels = _load_tensor(directory / "labels.pt")
        if data.ndim != 3:
            raise ValueError(
                f"Expected data tensor [window, time, asset]; got {tuple(data.shape)}."
            )
        if labels.ndim != 2:
            raise ValueError(
                f"Expected labels tensor [window, condition]; got {tuple(labels.shape)}."
            )
        if data.shape[0] != labels.shape[0]:
            raise ValueError("data and labels must contain the same number of windows.")
        if data.shape[1] != n_timestep:
            raise ValueError(f"Expected n_timestep={n_timestep}; got {data.shape[1]}.")
        if data.shape[2] != 50:
            raise ValueError(f"Expected 50 assets; got {data.shape[2]}.")
        if not bool(torch.isfinite(data).all()):
            raise ValueError("data tensor contains non-finite values.")
        if not bool(torch.isfinite(labels).all()):
            raise ValueError("labels tensor contains non-finite values.")

        start, end = _split_bounds(metadata, split=split, total_windows=data.shape[0])
        split_data = data[start:end]
        split_labels = labels[start:end]
        selected_count = split_data.shape[0] if n_sample is None else int(n_sample)
        if selected_count > split_data.shape[0]:
            raise IndexError(
                f"At most {split_data.shape[0]} samples are available for split={split!r}."
            )

        selected_raw_data = split_data[:selected_count].float()
        self.labels = split_labels[:selected_count].float()
        self.metadata = metadata
        self.selected_split = split
        self.processed_dir = directory
        self.tickers = [str(ticker) for ticker in metadata["tickers"]]
        self.sectors = [str(sector) for sector in metadata["sectors"]]
        self.sector_labels = torch.tensor(
            [int(label) for label in metadata["sector_label_ids"]],
            dtype=torch.long,
        )
        self.condition_names = [str(name) for name in metadata["condition_names"]]
        self.date_index = _selected_dates(metadata, start=start, n_sample=selected_count)
        self.raw_log_returns = selected_raw_data
        self.standardize_returns = bool(standardize_returns)
        self.standardization_stats: dict[str, Tensor] | None
        if self.standardize_returns:
            if standardization_stats is None:
                fit_data = _standardization_fit_data(
                    data,
                    metadata=metadata,
                    split=split,
                    selected_count=selected_count,
                )
                stats = fit_return_standardization(
                    fit_data,
                    epsilon=standardization_epsilon,
                )
                stats_source = (
                    "fit_on_selected_train_split"
                    if split == "train"
                    else "fit_on_metadata_train_split"
                )
            else:
                stats = coerce_standardization_stats(dict(standardization_stats))
                stats_source = "provided"
            self.standardization_stats = stats
            self.model_visible_returns = standardize_multifactor_returns(
                selected_raw_data,
                stats,
            )
        else:
            self.standardization_stats = None
            self.model_visible_returns = selected_raw_data
            stats_source = "disabled"
        self.metadata = dict(self.metadata)
        self.metadata["standardization"] = _standardization_metadata(
            enabled=self.standardize_returns,
            stats=self.standardization_stats,
            stats_source=stats_source,
            epsilon=standardization_epsilon,
            raw_log_returns=self.raw_log_returns,
            model_visible_returns=self.model_visible_returns,
        )
        self.projection_mode = projection_mode
        self.projection_state: FactorProjectionState | None = None
        self.factor_returns: Tensor | None = None
        if projection_mode is None:
            self.data = self.model_visible_returns
        else:
            self.data = self._build_factor_projected_view(
                self.model_visible_returns,
                projection_mode=projection_mode,
                projection_n_factors=projection_n_factors,
                projection_state=projection_state,
            )
        super().__init__(self.data)

    def reconstruct_model_visible_returns(self, factor_coordinates: Tensor | None = None) -> Tensor:
        """Map empirical factor coordinates back to 50D log-return space."""
        if self.projection_state is None:
            return self.model_visible_returns if factor_coordinates is None else factor_coordinates
        coordinates = self.factor_returns if factor_coordinates is None else factor_coordinates
        if coordinates is None:
            raise ValueError("factor coordinates are unavailable.")
        return reconstruct_model_visible_returns_from_factors(coordinates, self.projection_state)

    def reconstruct_raw_returns(self, factor_coordinates: Tensor | None = None) -> Tensor:
        """Map empirical factor coordinates back to raw 50D log returns."""
        return self.inverse_transform_returns(
            self.reconstruct_model_visible_returns(factor_coordinates)
        )

    def inverse_transform_returns(self, returns: Tensor) -> Tensor:
        """Map model-visible returns back to raw log-return scale."""
        if self.standardization_stats is None:
            return returns
        return inverse_standardize_multifactor_returns(returns, self.standardization_stats)

    def _build_factor_projected_view(
        self,
        selected_data: Tensor,
        *,
        projection_mode: PanelProjectionMode,
        projection_n_factors: int | None,
        projection_state: FactorProjectionState | Mapping[str, Any] | None,
    ) -> Tensor:
        """Return train-PCA factor coordinates for the empirical panel."""
        if projection_mode != "train_pca":
            raise ValueError("Empirical panel factor projection supports only 'train_pca'.")
        factor_count = int(projection_n_factors if projection_n_factors is not None else 5)
        if projection_state is None:
            state = fit_train_pca_projection(
                selected_data,
                n_factors=factor_count,
                input_scale=(
                    "standardized_log_returns" if self.standardize_returns else "raw_log_returns"
                ),
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
            if state.input_dim != selected_data.shape[-1]:
                raise ValueError(
                    f"projection_state expects input_dim={state.input_dim}; "
                    f"got {selected_data.shape[-1]}."
                )
            state_source = "provided"
        self.projection_state = state
        self.factor_returns = project_returns_to_factors(selected_data, state).float()
        self.metadata = dict(self.metadata)
        self.metadata["factor_projection"] = factor_projection_metadata(
            state,
            state_source=state_source,
            factor_returns=self.factor_returns,
        )
        self.metadata["factor_projection"]["empirical_warning"] = (
            "Train-fitted PCA projection only; no oracle factor loading matrix is used."
        )
        return self.factor_returns


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed panel metadata: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected metadata mapping in {path}.")
    return cast(dict[str, Any], loaded)


def _load_tensor(path: Path) -> Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed panel tensor: {path}")
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, Tensor):
        raise ValueError(f"Expected tensor payload in {path}.")
    return tensor


def _load_raw_return_tensor(directory: Path) -> Tensor:
    raw_path = directory / "raw_data.pt"
    if raw_path.exists():
        return _load_tensor(raw_path)
    return _load_tensor(directory / "data.pt")


def _split_bounds(
    metadata: dict[str, Any],
    *,
    split: PanelSplit,
    total_windows: int,
) -> tuple[int, int]:
    split_metadata = cast(dict[str, Any], metadata.get("split", {}))
    train_count = int(split_metadata.get("train_window_count", total_windows))
    train_count = max(0, min(train_count, total_windows))
    if split == "train":
        return 0, train_count
    if split == "eval":
        return train_count, total_windows
    return 0, total_windows


def _selected_dates(
    metadata: dict[str, Any],
    *,
    start: int,
    n_sample: int,
) -> list[str]:
    dates = cast(list[str], metadata.get("window_start_dates", []))
    if not dates:
        return []
    return [str(date) for date in dates[start : start + n_sample]]


def _standardization_fit_data(
    data: Tensor,
    *,
    metadata: dict[str, Any],
    split: PanelSplit,
    selected_count: int,
) -> Tensor:
    if split == "train":
        return data[:selected_count].float()
    train_start, train_end = _split_bounds(metadata, split="train", total_windows=data.shape[0])
    fit_data = data[train_start:train_end].float()
    if fit_data.shape[0] == 0:
        raise ValueError("Cannot fit return standardization: train split is empty.")
    return fit_data


def _standardization_metadata(
    *,
    enabled: bool,
    stats: dict[str, Tensor] | None,
    stats_source: str,
    epsilon: float,
    raw_log_returns: Tensor,
    model_visible_returns: Tensor,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "enabled": bool(enabled),
        "fit_split": "train",
        "stats_source": stats_source,
        "epsilon": float(epsilon),
        "raw_log_returns_shape": list(raw_log_returns.shape),
        "model_visible_data_shape": list(model_visible_returns.shape),
        "raw_log_returns_available_on_dataset": True,
    }
    if stats is None:
        return metadata
    metadata["mean"] = stats["mean"]
    metadata["std"] = stats["std"]
    metadata["mean_abs_standardized_asset_mean"] = float(
        model_visible_returns.detach().float().mean(dim=(0, 1)).abs().mean().item()
    )
    metadata["mean_standardized_asset_std"] = float(
        model_visible_returns.detach().float().std(dim=(0, 1), unbiased=False).mean().item()
    )
    return metadata
