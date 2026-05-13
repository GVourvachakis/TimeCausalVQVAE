# ruff: noqa
"""Black-Scholes, Bachelier, and Brownian-motion datasets.

The selected Black-Scholes pipeline returns paths with shape
``[n_sample, n_timestep, 1]`` and constant-one labels with shape
``[n_sample, 1]`` through :mod:`time_causal_vae.data.pipeline`.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, StackDataset
import matplotlib.pyplot as plt


def simulate_Bachelier(n_sample, dt, n_timestep, mu, sigma):
    time_grid = torch.linspace(0, dt * n_timestep, n_timestep + 1)
    time_paths = time_grid.expand([n_sample, n_timestep + 1])[..., None]
    BM_paths = simulate_BM(n_sample, dt, n_timestep)
    Bachelier_paths = mu * time_paths + sigma * BM_paths
    return Bachelier_paths


def simulate_BM(n_sample, dt, n_timestep):
    noise = torch.randn(size=(n_sample, n_timestep))
    paths_incr = noise * torch.sqrt(torch.tensor(dt))
    paths = torch.cumsum(paths_incr, axis=1)
    BM_paths = torch.cat([torch.zeros((n_sample, 1)), paths], axis=1)
    BM_paths = BM_paths[..., None]
    return BM_paths


def simulate_BS(n_sample, dt, n_timestep, mu, sigma):
    time_grid = torch.linspace(0, dt * n_timestep, n_timestep + 1)
    time_paths = time_grid.expand([n_sample, n_timestep + 1])[..., None]
    BM_paths = simulate_BM(n_sample, dt, n_timestep)
    BS_paths = torch.exp(sigma * BM_paths + (mu - 0.5 * sigma**2) * time_paths)
    return BS_paths


def simulate_BS2(n_sample, dt, n_timestep, mu, sigma, rho):
    r"""
    Simulate 2 BS paths with same mu, sigma with correlation rho
    """
    time_grid = torch.linspace(0, dt * n_timestep, n_timestep + 1)
    time_paths = time_grid.expand([n_sample, n_timestep + 1])[..., None]
    BM_paths0 = simulate_BM(n_sample, dt, n_timestep)
    BM_paths1 = simulate_BM(n_sample, dt, n_timestep)
    BM_paths2 = rho * BM_paths1 + np.sqrt(1 - rho**2) * BM_paths0
    BS_paths1 = torch.exp(sigma * BM_paths1 + (mu - 0.5 * sigma**2) * time_paths)
    BS_paths2 = torch.exp(sigma * BM_paths2 + (mu - 0.5 * sigma**2) * time_paths)
    BS_paths = torch.cat([BS_paths1, BS_paths2], dim=-1)
    return BS_paths


class BlackScholesDataset(TensorDataset):
    def __init__(self, n_sample, n_timestep, mu=0.1, sigma=0.2, dt=1 / 12, **kwargs):
        self.mu = mu
        self.sigma = kwargs.get("control", sigma)
        self.dt = kwargs.get("dt", dt)
        self.n_sample = n_sample
        self.n_timestep = n_timestep
        self.data = simulate_BS(self.n_sample, self.dt, self.n_timestep, self.mu, self.sigma).type(
            torch.float32
        )
        super(BlackScholesDataset, self).__init__(self.data)


class BlackScholes2Dataset(TensorDataset):
    def __init__(self, n_sample, n_timestep, mu=0.1, sigma=0.2, dt=1 / 12, rho=0, **kwargs):
        self.mu = mu
        self.sigma = kwargs.get("control", sigma)
        self.dt = kwargs.get("dt", dt)
        self.n_sample = n_sample
        self.n_timestep = n_timestep
        self.rho = rho
        self.data = simulate_BS2(
            self.n_sample, self.dt, self.n_timestep, self.mu, self.sigma, self.rho
        ).type(torch.float32)
        super(BlackScholes2Dataset, self).__init__(self.data)


class BachelierDataset(TensorDataset):
    def __init__(self, n_sample, n_timestep, mu=0.1, sigma=0.2, dt=1 / 12):
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.n_sample = n_sample
        self.n_timestep = n_timestep
        self.data = simulate_Bachelier(
            self.n_sample, self.dt, self.n_timestep, self.mu, self.sigma
        ).type(torch.float32)
        super(BachelierDataset, self).__init__(self.data)


class BrownianMotionDataset(TensorDataset):
    def __init__(self, n_sample, n_timestep, dt=1 / 12):
        self.dt = dt
        self.n_sample = n_sample
        self.n_timestep = n_timestep
        self.data = simulate_BM(self.n_sample, self.dt, self.n_timestep).type(torch.float32)
        super(BrownianMotionDataset, self).__init__(self.data)
