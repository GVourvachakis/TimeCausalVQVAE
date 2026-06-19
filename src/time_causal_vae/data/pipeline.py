# ruff: noqa
"""Target data factory preserving legacy dataset behaviour.

The factory returns ``BaseDataset`` instances with public ``data`` and
``labels`` tensors. For selected path configs, data has shape
``[n_sample, n_timestep, data_dim]`` and labels has shape
``[n_sample, condition_dim]``. Synthetic structurally conditional datasets keep
constant-one labels to match the legacy conditional model wiring.
"""

from __future__ import annotations

import torch

from time_causal_vae.data.base import BaseDataset
from time_causal_vae.data.black_scholes import BlackScholes2Dataset, BlackScholesDataset
from time_causal_vae.data.factor_projected_market import FactorProjectedMultifactorMarketDataset
from time_causal_vae.data.hawkes_jump import HawkesJumpDataset
from time_causal_vae.data.heston import HestonDataset
from time_causal_vae.data.market import LogrDataset, SP500VIXDataset
from time_causal_vae.data.multifactor_market import MultifactorMarketDataset
from time_causal_vae.data.path_dependent_volatility import PDVPriceFeatureDataset
from time_causal_vae.data.sp500_panel import SP50050PanelDataset
from time_causal_vae.data.toy import ANMDataset, CheckerBoard, MixMultiVariateNormal, Spiral

DATASET_NAME_ALIASES = {
    "black_scholes": "BSprice",
    "BSprice": "BSprice",
    "black_scholes_2d": "BS2price",
    "BS2price": "BS2price",
    "heston": "Hestonprice",
    "Hestonprice": "Hestonprice",
    "hawkes_jump": "HawkesJump",
    "HawkesJump": "HawkesJump",
    "multifactor_market": "MultifactorMarket",
    "MultifactorMarket": "MultifactorMarket",
    "multifactor_market_factor_projected": "MultifactorFactorProjectedMarket",
    "MultifactorFactorProjectedMarket": "MultifactorFactorProjectedMarket",
    "path_dependent_volatility": "PDVPriceConFeature",
    "PDVPriceConFeature": "PDVPriceConFeature",
    "sp500_vix": "SP500VIX",
    "SP500VIX": "SP500VIX",
    "sp500_50_panel": "SP50050Panel",
    "SP50050Panel": "SP50050Panel",
    "Logr": "Logr",
    "ANM": "ANM",
    "GM": "GM",
    "Board": "Board",
    "Spiral": "Spiral",
}


class DataPipeline:
    """Build target train/eval datasets from legacy or modern config names."""

    def __init__(self) -> None:
        """Initialise the pipeline and store the first raw dataset as reference."""
        self.base_dataset = None
        self._multifactor_standardization_stats = None
        self._multifactor_factor_projection_state = None
        self._sp500_standardization_stats = None
        self._sp500_factor_projection_state = None

    def __call__(self, exp_config, **kwargs):
        """Return train and eval ``BaseDataset`` objects."""
        self._multifactor_standardization_stats = None
        self._multifactor_factor_projection_state = None
        self._sp500_standardization_stats = None
        self._sp500_factor_projection_state = None
        train_data, train_labels = self._get_data_label(exp_config, use="train", **kwargs)
        train_dataset = BaseDataset(train_data, train_labels)
        eval_data, eval_labels = self._get_data_label(exp_config, use="eval", **kwargs)
        eval_dataset = BaseDataset(eval_data, eval_labels)
        return train_dataset, eval_dataset

    def _get_data_label(self, exp_config, use=None, **kwargs):
        """Return raw data and labels for one dataset split."""
        dataset_name = DATASET_NAME_ALIASES.get(exp_config.dataset, exp_config.dataset)
        dataset_kwargs = {**dict(exp_config.get("data_params", {})), **kwargs}
        if dataset_name == "BSprice":
            dataset = BlackScholesDataset(
                exp_config.n_sample,
                exp_config.n_timestep - 1,
                **dataset_kwargs,
            )
            data = dataset.data
            labels = torch.ones([data.shape[0], 1])
        elif dataset_name == "BS2price":
            dataset = BlackScholes2Dataset(
                exp_config.n_sample,
                exp_config.n_timestep - 1,
                rho=exp_config.rho,
                **dataset_kwargs,
            )
            data = dataset.data
            labels = torch.ones([data.shape[0], 1])
        elif dataset_name == "Hestonprice":
            dataset = HestonDataset(
                exp_config.n_sample,
                exp_config.n_timestep - 1,
                **dataset_kwargs,
            )
            data = dataset.data
            labels = torch.ones([data.shape[0], 1])
        elif dataset_name == "HawkesJump":
            hawkes_kwargs = dict(dataset_kwargs)
            if use == "eval" and hawkes_kwargs.get("seed") is not None:
                hawkes_kwargs["seed"] = int(hawkes_kwargs["seed"]) + 1
            dataset = HawkesJumpDataset(
                exp_config.n_sample,
                exp_config.n_timestep,
                **hawkes_kwargs,
            )
            data = dataset.data
            labels = dataset.labels
        elif dataset_name == "MultifactorMarket":
            multifactor_kwargs = dict(dataset_kwargs)
            if use == "eval":
                if multifactor_kwargs.get("path_seed") is not None:
                    multifactor_kwargs["path_seed"] = int(multifactor_kwargs["path_seed"]) + 1
                elif multifactor_kwargs.get("seed") is not None:
                    multifactor_kwargs["seed"] = int(multifactor_kwargs["seed"]) + 1
                if (
                    multifactor_kwargs.get("standardize_returns")
                    and self._multifactor_standardization_stats is not None
                ):
                    multifactor_kwargs["standardization_stats"] = (
                        self._multifactor_standardization_stats
                    )
            dataset = MultifactorMarketDataset(
                exp_config.n_sample,
                exp_config.n_timestep,
                **multifactor_kwargs,
            )
            if use == "train" and multifactor_kwargs.get("standardize_returns"):
                self._multifactor_standardization_stats = dataset.standardization_stats
            data = dataset.data
            labels = dataset.labels
        elif dataset_name == "MultifactorFactorProjectedMarket":
            multifactor_kwargs = dict(dataset_kwargs)
            if use == "eval":
                if multifactor_kwargs.get("path_seed") is not None:
                    multifactor_kwargs["path_seed"] = int(multifactor_kwargs["path_seed"]) + 1
                elif multifactor_kwargs.get("seed") is not None:
                    multifactor_kwargs["seed"] = int(multifactor_kwargs["seed"]) + 1
                if (
                    multifactor_kwargs.get("standardize_returns")
                    and self._multifactor_standardization_stats is not None
                ):
                    multifactor_kwargs["standardization_stats"] = (
                        self._multifactor_standardization_stats
                    )
                if self._multifactor_factor_projection_state is not None:
                    multifactor_kwargs["projection_state"] = (
                        self._multifactor_factor_projection_state
                    )
            dataset = FactorProjectedMultifactorMarketDataset(
                exp_config.n_sample,
                exp_config.n_timestep,
                **multifactor_kwargs,
            )
            if use == "train":
                if multifactor_kwargs.get("standardize_returns"):
                    self._multifactor_standardization_stats = dataset.standardization_stats
                self._multifactor_factor_projection_state = dataset.projection_state
            data = dataset.data
            labels = dataset.labels
        elif dataset_name == "PDVPriceConFeature":
            dataset = PDVPriceFeatureDataset(
                exp_config.n_sample,
                exp_config.n_timestep,
                **dataset_kwargs,
            )
            data = dataset.data
            labels = dataset.labels
        elif dataset_name == "SP500VIX":
            dataset = SP500VIXDataset(
                exp_config.n_sample,
                exp_config.n_timestep,
                base_data_dir=exp_config.base_data_dir,
            )
            data = dataset.data
            labels = dataset.labels
        elif dataset_name == "SP50050Panel":
            panel_kwargs = dict(dataset_kwargs)
            if use in {"train", "eval"} and "split" not in panel_kwargs:
                panel_kwargs["split"] = use
            if (
                use == "eval"
                and panel_kwargs.get("standardize_returns")
                and self._sp500_standardization_stats is not None
            ):
                panel_kwargs["standardization_stats"] = self._sp500_standardization_stats
            if (
                use == "eval"
                and panel_kwargs.get("projection_mode") is not None
                and self._sp500_factor_projection_state is not None
            ):
                panel_kwargs["projection_state"] = self._sp500_factor_projection_state
            dataset = SP50050PanelDataset(
                _split_sample_count(exp_config, use=use),
                exp_config.n_timestep,
                base_data_dir=exp_config.base_data_dir,
                **panel_kwargs,
            )
            if use == "train":
                if panel_kwargs.get("standardize_returns"):
                    self._sp500_standardization_stats = dataset.standardization_stats
                if panel_kwargs.get("projection_mode") is not None:
                    self._sp500_factor_projection_state = dataset.projection_state
            data = dataset.data
            labels = dataset.labels
        elif dataset_name == "Logr":
            dataset = LogrDataset(
                exp_config.n_sample,
                base_data_dir=exp_config.base_data_dir,
            )
            data = dataset.data
            labels = dataset.labels
        elif dataset_name == "ANM":
            dataset = ANMDataset(exp_config.n_sample, **dataset_kwargs)
            data = dataset.data
            labels = dataset.labels
        elif dataset_name == "GM":
            dataset = MixMultiVariateNormal(exp_config.n_sample, **dataset_kwargs)
            data = dataset.sample().view(-1, 1, 2)
            labels = torch.ones([data.shape[0], 1])
        elif dataset_name == "Board":
            dataset = CheckerBoard(exp_config.n_sample, **dataset_kwargs)
            data = dataset.sample().view(-1, 1, 2)
            labels = torch.ones([data.shape[0], 1])
        elif dataset_name == "Spiral":
            dataset = Spiral(exp_config.n_sample, **dataset_kwargs)
            data = dataset.sample().view(-1, 1, 2)
            labels = torch.ones([data.shape[0], 1])
        else:
            raise ValueError("No such dataset name")
        if self.base_dataset is None:
            self.base_dataset = dataset
        return data, labels


def _split_sample_count(exp_config, *, use=None) -> int | None:
    """Return split-specific sample count, falling back to legacy ``n_sample``."""
    if use == "train" and "train_n_sample" in exp_config and exp_config.train_n_sample is not None:
        return _optional_positive_int(exp_config.train_n_sample)
    if use == "eval" and "eval_n_sample" in exp_config and exp_config.eval_n_sample is not None:
        return _optional_positive_int(exp_config.eval_n_sample)
    return int(exp_config.n_sample)


def _optional_positive_int(value) -> int | None:
    """Return ``None`` or a validated positive integer."""
    if value is None:
        return None
    count = int(value)
    if count <= 0:
        raise ValueError("split-specific sample counts must be positive when provided.")
    return count
