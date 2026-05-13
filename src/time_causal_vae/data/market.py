# ruff: noqa
"""Market-data datasets for S&P500/VIX and multi-asset log returns.

``SP500VIXDataset`` preserves the legacy window normalisation and scalar VIX
label construction. The expected local file is
``<base_data_dir>/sp500vix/sp500vix_normalized.npy``.
"""

import json
import pickle
from os import path as pt

import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset


def load_obj(filepath):
    """Load local arrays or pickles using the legacy file conventions."""
    if filepath.endswith("pkl"):
        with open(filepath, "rb") as handle:
            return pickle.load(handle)
    if filepath.endswith("pt"):
        with open(filepath, "rb") as handle:
            return torch.load(handle, weights_only=True)
    if filepath.endswith("json"):
        with open(filepath, encoding="utf-8") as handle:
            return json.load(handle)
    if filepath.endswith("npy"):
        import numpy as np

        return np.load(filepath)
    raise NotImplementedError(f"No suitable loader for the path: {filepath}")


class SP500VIXDataset(Dataset):
    def __init__(self, n_sample, n_timestep, base_data_dir):
        self.n_timestep = n_timestep
        self.n_sample = n_sample

        self.sp500vix = load_obj(pt.join(base_data_dir, "sp500vix/sp500vix_normalized.npy"))
        self.sp500 = self.sp500vix[:, 0]
        self.vix = self.sp500vix[:, 1]
        self.path = self.sp500

        self.window_shape = n_timestep
        self.paths = sliding_window_view(self.sp500, self.window_shape)
        stride = 1
        self.paths = torch.tensor(self.paths).type(torch.float32)[..., None][::stride]  # stride = 1
        self.data = self.paths / self.paths[:, :1, :]

        n = len(self.data)
        if n_sample > n:
            raise IndexError(f"At most {n} many samples!")
        self.data = self.data[:n_sample]
        self.labels = torch.tensor(self.vix).type(torch.float32)[:n_sample, None]


def logr2price(logr):
    logr_cumsum = logr.cumsum(dim=1)
    price = logr_cumsum.exp()
    return price


def price2logr(price):
    return price.log().diff(dim=1)


class LogrDataset(Dataset):
    def __init__(self, n_sample, base_data_dir):
        self.logr0 = torch.tensor(load_obj(pt.join(base_data_dir, "multiasset/logr.pkl")))
        self.logr = torch.cat([torch.zeros_like(self.logr0[:, :1, :]), self.logr0], dim=1)
        self.price_norm = logr2price(self.logr)

        self.init_price = torch.tensor(
            load_obj(pt.join(base_data_dir, "multiasset/init_price.pkl"))
        )
        init_price_mean = self.init_price.mean(dim=0, keepdim=True)
        self.init_price_norm = self.init_price / init_price_mean

        if n_sample > len(self.price_norm):
            raise IndexError(f"At most {len(self.price_norm)} many samples!")

        self.data = self.price_norm[:n_sample]
        self.labels = self.init_price_norm[:n_sample, 0]
