"""Optional mean-variance evaluation copied from upstream downstream evaluations.

This module requires optional `neuralhedge` dependencies and is not required for
selected checkpoint reproduction.
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from neuralhedge.data.base import ManagerDataset
from neuralhedge.data.market import BS_Market
from neuralhedge.data.stochastic import simulate_time
from neuralhedge.nn.blackschole import BlackScholesAlpha, BlackScholesMeanVarianceAlpha
from neuralhedge.nn.datamanager import Manager, WealthManager
from neuralhedge.nn.loss import log_utility
from tqdm import tqdm

from time_causal_vae.evaluation.checkpoints import TargetModelEvaluator as ModelEvaluator
from time_causal_vae.evaluation.style import apply_source_style


class BlackScholesMeanVarianceAlphaBiClip(BlackScholesMeanVarianceAlpha):
    def __init__(self, mu, sigma, r, Wstar, clip):
        super().__init__(mu, sigma, r, Wstar)
        self.clip = clip

    def forward(self, x):
        alpha = self.compute_alpha(x)
        alpha_clip = alpha
        prop = torch.cat([alpha_clip, 1 - alpha_clip], dim=-1)
        return prop


def robust_evaluate_mv(
    hedger: Manager,
    ds: ManagerDataset,
):
    prices, info = ds.data
    with torch.no_grad():
        wealth = hedger.forward(prices, info)
        terminal_wealth = wealth[:, -1]
    terminal_wealth = terminal_wealth[torch.isfinite(terminal_wealth)]
    mean = terminal_wealth.mean().item()
    var = terminal_wealth.var().item()
    return mean, var


def compute_efficient_frontier(ds, strategy_name, mu, sigma, r):

    if strategy_name == "constant":
        constant_mean_var = defaultdict(list)
        for alpha in np.linspace(0, 5, 101):
            constant_strategy = BlackScholesAlpha(mu, sigma, r, alpha=alpha)
            constant_manager = WealthManager(constant_strategy, utility_func=log_utility)
            mean, var = robust_evaluate_mv(constant_manager, ds)
            constant_mean_var["mean"].append(mean)
            constant_mean_var["var"].append(var)
        return constant_mean_var
    elif strategy_name == "optimal":
        optimal_mean_var = defaultdict(list)
        for Wstar in np.linspace(1, 15, 101):
            optimal_strategy = BlackScholesMeanVarianceAlpha(mu=mu, sigma=sigma, r=r, Wstar=Wstar)
            optimal_manager = WealthManager(optimal_strategy, utility_func=log_utility)
            mean, var = robust_evaluate_mv(optimal_manager, ds)
            optimal_mean_var["mean"].append(mean)
            optimal_mean_var["var"].append(var)
        return optimal_mean_var


def compute_efficient_frontier_list(n_sample, n_timestep, dt, mu, r, sigma_range, stock_prices=None):
    sigma_strategy_dict = defaultdict(dict)
    for sigma in tqdm(sigma_range):
        bs_market = BS_Market(
            n_sample=n_sample,
            n_timestep=n_timestep,
            dt=dt,
            mu=mu,
            r=r,
            sigma=sigma,
            init_price=1,
        )
        ds = bs_market.get_manage_ds()
        if stock_prices is not None:
            prices = torch.cat([stock_prices, ds.prices[..., 1:]], dim=-1)

            info = torch.cat(
                [
                    torch.log(prices[..., :1]),
                    simulate_time(n_sample, dt, n_timestep, reverse=True),
                ],
                dim=-1,
            )
            data = (prices, info)
            ds = ManagerDataset(*data)

        efficient_frontier_dict = defaultdict(dict)
        for strategy_name in ["constant", "optimal"]:
            efficient_frontier_dict[strategy_name] = compute_efficient_frontier(ds, strategy_name, mu, sigma, r)
        sigma_strategy_dict[sigma] = efficient_frontier_dict
    return sigma_strategy_dict


def compute_mean_variance(model_evaluator: ModelEvaluator, fake_data):
    base_dataset = model_evaluator.ensure_base_dataset()
    mu, real_sigma, r, n_sample, dt, n_timestep = (
        base_dataset.mu,
        base_dataset.sigma,
        0.01,
        len(fake_data),
        base_dataset.dt,
        base_dataset.n_timestep,
    )

    sigma_range = [0.15, 0.175, 0.2, 0.25, 0.3]

    sigma_strategy_dict_data = compute_efficient_frontier_list(n_sample, n_timestep, dt, mu, r, sigma_range, stock_prices=fake_data)

    sigma_strategy_dict = compute_efficient_frontier_list(n_sample, n_timestep, dt, mu, r, sigma_range, stock_prices=None)
    return sigma_strategy_dict, sigma_strategy_dict_data


def plot_mean_variance(
    sigma_strategy_dict,
    sigma_strategy_dict_data,
    real_sigma,
    file_path=None,
):
    apply_source_style()
    fig, ax = plt.subplots(1, 2, figsize=[12, 4], sharey=True, sharex=True)
    name_list = ["constant", "optimal"]
    title_list = ["Constant strategy", "Optimal strategy"]

    for i, name in enumerate(name_list):

        for sigma, strategy_dict in sigma_strategy_dict.items():
            mean_list = strategy_dict[name]["mean"]
            var_list = strategy_dict[name]["var"]
            if real_sigma == sigma:
                ax[i].plot(
                    var_list,
                    mean_list,
                    "-",
                    label=rf"$\sigma = ${sigma :.3f} (real)",
                    linewidth=3,
                    c="r",
                )
            else:
                ax[i].plot(
                    var_list,
                    mean_list,
                    label=rf"$\sigma = ${sigma :.3f}",
                    linestyle="--",
                    linewidth=1,
                )
        max_mean = 0
        for sigma, strategy_dict in sigma_strategy_dict_data.items():
            mean_list = np.array(strategy_dict[name]["mean"])
            var_list = np.array(strategy_dict[name]["var"])
            # print(sigma)
            # print(max(mean_list[var_list < 10]))
            if max(mean_list[var_list < 10]) > max_mean:
                max_mean = max(mean_list[var_list < 10])
                optimal_sigma = sigma
            # ax[i].plot(var_list, mean_list, ":", label=f"$\sigma = ${sigma :.3f} fake", linewidth=3)

        strategy_dict = sigma_strategy_dict_data[optimal_sigma]
        mean_list = strategy_dict[name]["mean"]
        var_list = strategy_dict[name]["var"]
        ax[i].plot(var_list, mean_list, ":", label=f"fake", linewidth=3, c="k")

        ax[i].set_title(title_list[i])
        ax[i].legend(loc="upper left")
        ax[i].set_xlabel("Variance")
        ax[i].set_ylabel("Mean")
        ax[i].set_xlim(-1, 10)
        ax[i].set_ylim(1, 6)

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")
    plt.show()
