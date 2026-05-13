# ruff: noqa
"""Heston stochastic-volatility dataset.

The target pipeline preserves the legacy Euler simulation and exposes only the
price channel as ``data`` with constant-one labels for the structurally
conditional selected model.
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt


def simulate_Heston(
    n_sample, n_timestep, r=0.02, kappa=1, theta=0.2, v_0=0.2, rho=-0.9, xi=0.5, dt=1 / 12
):
    r"""
    Simulate Heston by Euler Scheme, this works under stability assumption. TODO: update to Alfonsi Scheme
    See Theorem 2.8 in <Alfonsi, A. (2010). High order discretization schemes for the CIR process: application to Affine Term Structure and Heston models. Mathematics of Computation. Vol. 79, pp. 209-237>.
    """
    size = (n_sample, n_timestep + 1)
    s_0 = 1
    prices = np.ones(size) * s_0
    sigs = np.ones(size) * v_0
    S_t = s_0
    v_t = v_0
    for t in range(1, n_timestep + 1):
        WT = np.random.multivariate_normal(
            np.array([0, 0]), cov=np.array([[1, rho], [rho, 1]]), size=n_sample
        ) * np.sqrt(dt)

        S_t = S_t * (np.exp((r - 0.5 * v_t) * dt + np.sqrt(v_t) * WT[:, 0]))
        v_t = np.abs(v_t + kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * WT[:, 1])
        prices[:, t] = S_t
        sigs[:, t] = v_t

    heston_paths = torch.tensor(np.concatenate([prices[..., None], sigs[..., None]], axis=-1)).type(
        torch.float32
    )
    return heston_paths


class HestonDataset(TensorDataset):
    def __init__(
        self,
        n_sample,
        n_timestep,
        r=0.02,
        kappa=1,
        theta=0.2,
        v_0=0.2,
        rho=-0.9,
        xi=0.5,
        dt=1 / 12,
        **kwargs,
    ):
        self.r = r
        self.mu = r
        self.kappa = kappa
        theta = kwargs.get("control", theta)
        v_0 = kwargs.get("control", v_0)
        self.theta = theta
        self.v_0 = v_0
        self.rho = rho
        self.xi = xi
        self.dt = dt
        self.n_sample = n_sample
        self.n_timestep = n_timestep
        self.paths = simulate_Heston(n_sample, n_timestep, r, kappa, theta, v_0, rho, xi, dt).type(
            torch.float32
        )
        self.data = self.paths[:, :, :1]
        super(HestonDataset, self).__init__(self.data)
