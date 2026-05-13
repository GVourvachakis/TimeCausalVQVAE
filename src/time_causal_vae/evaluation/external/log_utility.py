"""Optional log-utility evaluation copied from upstream downstream evaluations.

This module requires optional `neuralhedge` dependencies and is not required for
selected checkpoint reproduction.
"""

import matplotlib.pyplot as plt
import numpy as np

from time_causal_vae.evaluation.style import apply_source_style
import torch
from neuralhedge.data.base import ManagerDataset
from neuralhedge.data.market import BS_Market
from neuralhedge.nn.blackschole import BlackScholesAlpha
from neuralhedge.nn.datamanager import Manager

from time_causal_vae.evaluation.checkpoints import TargetModelEvaluator as ModelEvaluator

from tqdm import tqdm


def compute_log_utility(manager, manage_ds):
    prices, info = manage_ds.data
    with torch.no_grad():
        wealth = manager.forward(prices, info)
        terminal_wealth = wealth[:, -1]
    n = len(terminal_wealth)
    log_wealth = terminal_wealth.log()
    log_wealth = log_wealth[log_wealth.isfinite()]
    utility = log_wealth.mean().numpy()
    return utility


def log_utility_max_stability(n_sample, n_timestep, dt, mu, sigma, r, n_trial=1):
    bs_market = BS_Market(n_sample, n_timestep, dt, mu, sigma, r)
    model = BlackScholesAlpha(mu, sigma, r)
    manager = Manager(model)
    utility_list = []
    for j in range(n_trial):
        manage_ds = bs_market.get_manage_ds()
        utility = compute_log_utility(manager, manage_ds)
        utility_list.append(utility)
    return utility_list


def log_utility_max_data(n_sample, n_timestep, dt, mu, sigma, r, path, sigma_list):
    bs_market = BS_Market(n_sample, n_timestep, dt, mu, sigma, r)
    manage_ds = bs_market.get_manage_ds()
    prices, info = manage_ds.data

    bond_prices = manage_ds.prices[..., 1:]
    prices = torch.cat([path, bond_prices], dim=-1)
    data = (prices, prices)  # plain information
    manage_ds = ManagerDataset(*data)
    utility_list = []
    for sigma1 in tqdm(sigma_list):
        model = BlackScholesAlpha(mu, sigma1, r)
        manager = Manager(model)
        utility = compute_log_utility(manager, manage_ds)
        utility_list.append(utility)
    return utility_list


def optimal_log_utility(mu, sigma, r, T, alpha=None):
    if alpha is None:
        return (r + (mu - r) ** 2 / 2 / sigma**2) * T
    else:
        return (alpha * mu + (1 - alpha) * r - alpha**2 * sigma**2 / 2) * T


def inv_optimal_log_utility(mu, v, r, T):
    return abs(mu - r) / np.sqrt(2 * (v / T - r))


def compare_log_utility_max(
    model_evaluator: ModelEvaluator,
    file_path=None,
):
    apply_source_style()
    base_ds = model_evaluator.ensure_base_dataset()
    mu, sigma, r, n_sample_small, dt, n_timestep = (
        base_ds.mu,
        base_ds.sigma,
        0.01,
        1000,
        base_ds.dt,
        base_ds.n_timestep,
    )

    n_sample_big = 50000
    print("Loading data")
    real_path, fake_path, _ = model_evaluator.load_data(
        n_sample_test=n_sample_big, seed=99)

    sigma_list = np.linspace(0.15, 0.25, 31)
    real_utility_list = log_utility_max_data(
        n_sample_big, n_timestep, dt, mu, sigma, r, real_path, sigma_list)
    fake_utility_list = log_utility_max_data(
        n_sample_big, n_timestep, dt, mu, sigma, r, fake_path, sigma_list)
    real_utility = max(real_utility_list)
    real_sigma = sigma_list[real_utility_list.index(real_utility)]
    fake_utility = max(fake_utility_list)
    fake_sigma = sigma_list[fake_utility_list.index(fake_utility)]

    real_path_fake_strategy_utility = log_utility_max_data(
        n_sample_big, n_timestep, dt, mu, sigma, r, real_path, [fake_sigma])[0]
    fake_path_real_strategy_utility = log_utility_max_data(
        n_sample_big, n_timestep, dt, mu, sigma, r, fake_path, [real_sigma])[0]

    utility_list = log_utility_max_stability(
        n_sample_small, n_timestep, dt, mu, sigma, r, n_trial=200)
    opt_utility = optimal_log_utility(mu, sigma, r, dt * n_timestep)

    # plt.plot(sigma_list, real_utility_list)
    # plt.plot(sigma_list, fake_utility_list)
    # plt.show()

    fig, ax = plt.subplots(1, 2, figsize=[12, 4])
    ax[0].hist(utility_list, bins=30, label=f"N = {n_sample_small}", alpha=0.4)
    ax[0].vlines(
        opt_utility,
        0,
        9,
        label=r"$v^{*}$" + f"={opt_utility:.2f}",
        color="k",
        linewidths=(5,),
    )

    ax[0].vlines(
        real_utility,
        0,
        9,
        linestyles="solid",
        label=r"$V(\mu_\mathrm{real}, G_{\mathrm{real}})$" +
                   f"={real_utility:.2f}",
        color="C1",
        linewidths=(3,),
    )

    ax[0].vlines(
        real_path_fake_strategy_utility,
        0,
        9,
        linestyles="dashed",
        label=r"$V(\mu_\mathrm{real}, G_{\mathrm{fake}})$" +
                   f"={real_path_fake_strategy_utility:.2f}",
        color="C1",
        linewidths=(3,),
    )

    ax[0].vlines(
        fake_path_real_strategy_utility,
        0,
        9,
        linestyles="dashed",
        label=r"$V(\mu_\mathrm{fake}, G_{\mathrm{real}})$" +
                   f"={fake_path_real_strategy_utility:.2f}",
        color="C2",
        linewidths=(3,),
    )

    ax[0].vlines(
        fake_utility,
        0,
        9,
        linestyles="solid",
        label=r"$V(\mu_\mathrm{fake}, G_{\mathrm{fake}})$" +
                   f"={fake_utility:.2f}",
        color="C2",
        linewidths=(3,),
    )

    ax[0].set_title("Log-utility")
    ax[0].set_xlabel("Log-utility")
    ax[0].set_ylabel("Number of samples")
    ax[0].legend()

    sigma_range = np.linspace(0.1, 0.4, 101)
    utility_list0 = []
    for sigma0 in sigma_range:
        utility = optimal_log_utility(mu, sigma0, r, dt * n_timestep)
        utility_list0.append(utility)
    ax[1].plot(sigma_range, utility_list0, "-",
               label="optimal log-utility (BS)")
    opt_sigma = inv_optimal_log_utility(mu, opt_utility, r, dt * n_timestep)
    fake_sigma = inv_optimal_log_utility(mu, fake_utility, r, dt * n_timestep)
    ax[1].hlines(
        opt_utility,
        0.1,
        opt_sigma,
        color="k",
        linestyles="dashed",
        label=r"$v^{*}$" + f"={opt_utility:.2f}",
    )
    ax[1].hlines(
        fake_utility,
        0.1,
        fake_sigma,
        color="r",
        linestyles="solid",
        label=r"$V(\mu_\mathrm{fake}, G_{\mathrm{fake}})$" +
                   f"={fake_utility:.2f}",
    )
    ax[1].vlines(opt_sigma, 0, opt_utility, color="k", linestyles="dashed")
    ax[1].vlines(fake_sigma, 0, fake_utility, color="r", linestyles="solid")
    ax[1].set_xlabel(r"Volatility $\sigma$")
    ax[1].set_ylabel("Optimal log-utility")
    ax[1].set_title("Black-Scholes log-utility")
    ax[1].legend()

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")
    plt.show()
