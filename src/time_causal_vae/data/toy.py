# ruff: noqa
"""Toy two-dimensional and additive-noise datasets."""

import numpy as np
import torch
import torch.distributions as td
from torch.utils.data import Dataset


class MixMultiVariateNormal:
    def __init__(self, batch_size, x_scalar=1.0, y_scalar=1.0, radius=12, num=8, sigmas=None):
        # build mu's and sigma's
        arc = 2 * np.pi / num
        xs = [np.cos(arc * idx) * radius for idx in range(num)]
        ys = [np.sin(arc * idx) * radius for idx in range(num)]
        mus = [torch.Tensor([x, y]) for x, y in zip(xs, ys)]
        dim = len(mus[0])
        sigmas = [torch.eye(dim) for _ in range(num)] if sigmas is None else sigmas

        if batch_size % num != 0:
            raise ValueError("batch size must be devided by number of gaussian")
        self.num = num
        self.batch_size = batch_size
        self.x_scalar = x_scalar
        self.y_scalar = y_scalar
        self.dists = [
            td.multivariate_normal.MultivariateNormal(mu, sigma) for mu, sigma in zip(mus, sigmas)
        ]

    def log_prob(self, x):
        # assume equally-weighted
        densities = [torch.exp(dist.log_prob(x)) for dist in self.dists]
        return torch.log(sum(densities) / len(self.dists))

    def sample(self):
        ind_sample = self.batch_size / self.num
        samples = [dist.sample([int(ind_sample)]) for dist in self.dists]
        samples = torch.cat(samples, dim=0)
        samples[:, 0] = self.x_scalar * samples[:, 0]
        samples[:, 1] = self.y_scalar * samples[:, 1]
        return samples


class CheckerBoard:
    def __init__(self, batch_size, x_scalar=1.0, y_scalar=1.0):
        self.batch_size = batch_size
        self.x_scalar = x_scalar
        self.y_scalar = y_scalar

        print(f"rescale data with {self.x_scalar}x on X-axis  and {self.y_scalar}x Y-axis.")

    def sample(self):
        n = self.batch_size
        n_points = 3 * n
        n_classes = 2
        freq = 5
        x = np.random.uniform(-(freq // 2) * np.pi, (freq // 2) * np.pi, size=(n_points, n_classes))
        mask = np.logical_or(
            np.logical_and(np.sin(x[:, 0]) > 0.0, np.sin(x[:, 1]) > 0.0),
            np.logical_and(np.sin(x[:, 0]) < 0.0, np.sin(x[:, 1]) < 0.0),
        )
        y = np.eye(n_classes)[1 * mask]
        x0 = x[:, 0] * y[:, 0]
        x1 = x[:, 1] * y[:, 0]
        sample = np.concatenate([x0[..., None], x1[..., None]], axis=-1)
        sqr = np.sum(np.square(sample), axis=-1)
        idxs = np.where(sqr == 0)
        sample = np.delete(sample, idxs, axis=0)
        # res=res+np.random.randn(*res.shape)*1
        sample = torch.Tensor(sample)
        sample = sample[0:n, :]

        sample[:, 0] = self.x_scalar * sample[:, 0]
        sample[:, 1] = self.y_scalar * sample[:, 1]
        return sample


class Spiral:
    def __init__(self, batch_size, x_scalar=1.0, y_scalar=1.0):
        self.batch_size = batch_size
        self.x_scalar = x_scalar
        self.y_scalar = y_scalar

        print(f"rescale data with {self.x_scalar}x on X-axis  and {self.y_scalar}x Y-axis.")

    def sample(self):
        n = self.batch_size
        theta = np.sqrt(np.random.rand(n)) * 3 * np.pi - 0.5 * np.pi  # np.linspace(0,2*pi,100)

        r_a = theta + np.pi
        data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
        x_a = data_a + 0.25 * np.random.randn(n, 2)
        samples = np.append(x_a, np.zeros((n, 1)), axis=1)
        samples = samples[:, 0:2] * 0.4  # default 0.4 re-scaled
        samples[:, 0] = self.x_scalar * samples[:, 0]
        samples[:, 1] = self.y_scalar * samples[:, 1]
        return torch.Tensor(samples)


class ANMDataset(Dataset):
    """Additive-noise toy dataset preserving the legacy formula."""

    def __init__(self, n_sample, x_lower=0, x_upper=2, noise_std=1):
        true_function = lambda x: torch.nn.Softplus()(x)
        x = torch.rand(n_sample, 1) * (x_upper - x_lower) + x_lower
        eps = torch.randn(n_sample, 1) * noise_std
        xn = x + eps
        y = true_function(xn)
        self.labels = x
        self.data = y.unsqueeze(1)
