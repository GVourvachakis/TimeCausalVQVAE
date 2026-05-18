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
from time_causal_vae.data.hawkes_jump import HawkesJumpDataset
from time_causal_vae.data.heston import HestonDataset
from time_causal_vae.data.market import LogrDataset, SP500VIXDataset
from time_causal_vae.data.path_dependent_volatility import PDVPriceFeatureDataset
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
    "path_dependent_volatility": "PDVPriceConFeature",
    "PDVPriceConFeature": "PDVPriceConFeature",
    "sp500_vix": "SP500VIX",
    "SP500VIX": "SP500VIX",
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

    def __call__(self, exp_config, **kwargs):
        """Return train and eval ``BaseDataset`` objects."""
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
