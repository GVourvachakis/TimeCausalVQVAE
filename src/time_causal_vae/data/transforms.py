# ruff: noqa
"""Price/return conversion helpers copied from the legacy data utilities."""

import torch
from torch.nn.functional import relu


def prices2returns(prices):
    returns = prices[:, 1:] / prices[:, :-1] - 1
    return returns


def returns2prices_Cap(returns):
    u = 0.2
    U = 5
    ratio = returns + 1
    ratio = relu(ratio - u) + u
    ratio = U - relu(U - ratio)
    prices = torch.cumprod(ratio, dim=1)
    prices = torch.cat([torch.ones_like(prices[:, :1]), prices], axis=1)
    return prices
