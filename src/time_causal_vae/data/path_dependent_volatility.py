# ruff: noqa
"""Path-dependent-volatility datasets.

``PDVPriceFeatureDataset`` preserves the legacy path normalisation and the
``n_discard - 1`` label offset used for conditional volatility features.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, StackDataset
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm


def load_feature(path, alpha_1=1.06, delta_1=0.02, alpha_2=1.60, delta_2=0.052, dt=1 / 365):
    N = len(path)
    time = np.arange(N)
    time_inverse = np.arange(N - 1, -1, -1)

    # log_return = sp500[1:]/ sp500[:-1] - 1
    log_return = np.log(path[1:]) - np.log(path[:-1])
    log_return_square = log_return**2

    z_1 = delta_1 ** (1 - alpha_1) / (alpha_1 - 1)
    k_1 = (time_inverse * dt + delta_1) ** (-alpha_1) / z_1
    k_1 = k_1 / k_1.sum() / dt  # normalize

    z_2 = delta_2 ** (1 - alpha_2) / (alpha_2 - 1)
    k_2 = (time_inverse * dt + delta_2) ** (-alpha_2) / z_2
    k_2 = k_2 / k_2.sum() / dt  # normalize

    r1 = np.zeros_like(log_return)
    r2 = np.zeros_like(log_return)
    p = 10
    rp = np.zeros_like(log_return)
    for i in range(0, N - 1):
        r1[i] = (log_return[: i + 1] * k_1[-i - 1 :]).sum()
        r2[i] = np.sqrt((log_return_square[: i + 1] * k_2[-i - 1 :]).sum())
        rp[i] = np.power((log_return[: i + 1] ** p * k_2[-i - 1 :]).sum(), 1 / p)

    return log_return, log_return_square, r1, r2, rp, k_1, k_2


class PDV4:
    def __init__(
        self,
        beta0=0.04,
        beta1=-0.13,
        beta2=0.65,
        lamb10=55,
        lamb11=10,
        theta1=0.25,
        lamb20=20,
        lamb21=3,
        theta2=0.5,
        ddt=1 / 365,
        day_timestep=1,
        mu=0.1,
        *args,
        **kwargs,
    ) -> None:
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.lamb10 = lamb10
        self.lamb11 = lamb11
        self.theta1 = theta1
        self.lamb20 = lamb20
        self.lamb21 = lamb21
        self.theta2 = theta2
        self.ddt = ddt
        self.day_timestep = day_timestep
        self.dt = self.ddt / self.day_timestep
        self.mu = mu

    def simulate(
        self,
        n_sample: int,
        n_timestep: int,
        r10_init=0.078,
        r11_init=0.16,
        r20_init=0.074,
        r21_init=0.016,
    ):
        """
        prices: L+1
        rxx: L+1
        r: L
        sigma: L

        """
        n_timestep = n_timestep * self.day_timestep

        r10 = np.ones(shape=[n_sample, n_timestep + 1, 1]) * r10_init
        r11 = np.ones(shape=[n_sample, n_timestep + 1, 1]) * r11_init
        r20 = np.ones(shape=[n_sample, n_timestep + 1, 1]) * r20_init
        r21 = np.ones(shape=[n_sample, n_timestep + 1, 1]) * r21_init
        log_return = np.zeros(shape=[n_sample, n_timestep + 1, 1])

        sigma = np.ones(shape=[n_sample, n_timestep, 1])
        r1 = np.ones(shape=[n_sample, n_timestep, 1])
        r2 = np.ones(shape=[n_sample, n_timestep, 1])

        for t in range(n_timestep):
            # compute sigma
            r1[:, t] = (1 - self.theta1) * r10[:, t] + self.theta1 * r11[:, t]
            r2[:, t] = (1 - self.theta2) * r20[:, t] + self.theta2 * r21[:, t]
            sigma[:, t] = self.beta0 + self.beta1 * r1[:, t] + self.beta2 * np.sqrt(r2[:, t])
            # update R_{i,j}
            dw = np.random.normal(loc=0, scale=np.sqrt(self.dt), size=[n_sample, 1])
            r10[:, t + 1] = r10[:, t] + self.lamb10 * (sigma[:, t] * dw - r10[:, t] * self.dt)
            r11[:, t + 1] = r11[:, t] + self.lamb11 * (sigma[:, t] * dw - r11[:, t] * self.dt)
            r20[:, t + 1] = r20[:, t] + self.lamb20 * (sigma[:, t] ** 2 - r20[:, t]) * self.dt
            r21[:, t + 1] = r21[:, t] + self.lamb21 * (sigma[:, t] ** 2 - r21[:, t]) * self.dt
            # update logreturn
            log_return[:, t + 1] = sigma[:, t] * dw + self.mu * self.dt

        log_return = log_return[:, :: self.day_timestep]
        sigma = sigma[:, :: self.day_timestep]
        r10 = r10[:, :: self.day_timestep]
        r11 = r11[:, :: self.day_timestep]
        r20 = r20[:, :: self.day_timestep]
        r21 = r21[:, :: self.day_timestep]
        r1 = r1[:, :: self.day_timestep]
        r2 = r2[:, :: self.day_timestep]

        prices = np.cumprod(np.exp(log_return), axis=1)

        return prices, log_return, sigma, r10, r11, r20, r21, r1, r2

    def vix(self, r10_init, r11_init, r20_init, r21_init):
        T_month = 30
        n_sample = 5000
        n_timestep = T_month * self.day_timestep
        log_return, sigma, r10, r11, r20, r21 = self.simulate(
            n_sample, n_timestep, r10_init, r11_init, r20_init, r21_init
        )
        return np.sqrt(np.mean(sigma**2))

    def vix_path(self, r10, r11, r20, r21):
        vix_path = np.zeros_like(r10)
        for t in range(len(r10)):
            r10_init, r11_init, r20_init, r21_init = r10[t], r11[t], r20[t], r21[t]
            vix_path[t] = self.vix(r10_init, r11_init, r20_init, r21_init)
        return vix_path


class PDVDataset(Dataset):
    def __init__(self, n_sample, n_timestep, *args, **kwargs):
        self.n_sample = n_sample
        self.n_timestep = n_timestep
        self.model = PDV4(**kwargs)
        (
            self.prices,
            self.log_return,
            self.sigma,
            self.r10,
            self.r11,
            self.r20,
            self.r21,
            self.r1,
            self.r2,
        ) = self.model.simulate(1, n_timestep + n_sample - 1)

        n = self.sigma.shape[1]
        # From now all data same length
        self.information = np.concatenate(
            [self.sigma, self.r2, self.r20[:, :n], self.r21[:, :n]], axis=-1
        )[0]
        self.path = self.prices[0, :-1, 0]

        self.window_shape = n_timestep
        self.paths = sliding_window_view(self.path, self.window_shape)
        self.paths = torch.tensor(self.paths).type(torch.float32)[..., None]


class PDVPriceFeatureDataset(PDVDataset):
    def __init__(self, n_sample, n_timestep, *args, **kwargs):
        # Since we are using path features, we dont want to use the beginning of simulated paths
        self.n_discard = 100
        super().__init__(n_sample + self.n_discard, n_timestep, *args, **kwargs)

        log_return, log_return_square, r1, r2, rp, k_1, k_2 = load_feature(self.path)
        self.features = r2[:, None]

        self.data = self.paths[
            self.n_discard : self.n_discard + n_sample
        ]  # discard first n_discard data
        self.data = self.data / self.data[:, :1, :]

        self.labels = self.features[self.n_discard - 1 : self.n_discard - 1 + n_sample]
        self.labels = torch.tensor(self.labels).type(torch.float32)
