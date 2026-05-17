"""Optional conditional evaluation helpers copied from the upstream release.

These helpers depend on optional signature/adapted-Wasserstein routines and are not
required for selected checkpoint reproduction. Standard reproduction uses the target
checkpoint evaluator with Gaussian MMD and SWD.
"""

import os
from os import path as pt

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from scipy import stats
from tqdm import tqdm

from time_causal_vae.models.continuous.distances import GaussianMMD
from time_causal_vae.evaluation.metrics import SAWD, SWD, SignatureMMD
from time_causal_vae.data.black_scholes import simulate_BS
from time_causal_vae.data.path_dependent_volatility import PDV4, load_feature
from time_causal_vae.evaluation.style import apply_source_style
from time_causal_vae.utils.serialization import load_obj, save_obj


def logr2price(logr):
    logr_cumsum = logr.cumsum(dim=1)
    price = logr_cumsum.exp()
    return price


def price2logreturn_normalized(price):
    re = np.log(price[1:]) - np.log(price[:-1])
    return (re - re.mean()) / re.std()


def price2return_normalized(price):
    re = price[1:] / price[:-1] - 1
    return (re - re.mean()) / re.std()


def price2return(price):
    re = price[1:] / price[:-1] - 1
    return re


def price2logreturn(price):
    re = np.log(price[1:]) - np.log(price[:-1])
    return re


def plot_path_condition(
    path,
    condition_path,
):
    apply_source_style()
    fig, ax = plt.subplots(1, 2, figsize=[12, 4])
    ax1 = ax[0]
    ax2 = ax1.twinx()
    ax1.plot(path, "b-")
    ax2.plot(condition_path, "r-")
    ax1.set_ylabel("Paths", color="b")
    ax2.set_ylabel("Condition paths", color="r")

    ax[1].hist(condition_path, bins=50)
    ax[1].set_xlabel("Conditions")
    plt.tight_layout()


def con_gen_plot(model, conditions, file_path=None):
    apply_source_style()
    fig, ax = plt.subplots(2, 4, figsize=[12, 4], sharex=True, sharey=True)
    # fig.suptitle("Generated returns conditional on different conditions")
    axs = ax.reshape(-1)
    for i in range(len(axs)):
        axi = axs[i]
        condition = conditions[i]
        c = cm.viridis(0.9 * (1 - condition / conditions.max()))
        with torch.no_grad():
            n_sample = 1000
            x_sample = model.generation(n_sample, c=condition * torch.ones(size=[n_sample, 1]))
        axi.plot(
            x_sample[..., 0].numpy().T,
            alpha=0.1,
            marker="o",
            linewidth=1,
            markersize=0.1,
            c=c,
        )
        axi.set_title(rf"$\Sigma_t=${condition:.3f}")
    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")
    plt.show()


class Extension:
    def __init__(
        self,
    ) -> None:
        pass

    @classmethod
    def calculate_condition(cls, path):
        log_return, log_return_square, r1, r2_feature, rp, k_1, k_2 = load_feature(path)
        condition = r2_feature[-1]
        return condition

    @classmethod
    def extend(cls, model, path, n_sample):
        condition = cls.calculate_condition(path)
        with torch.no_grad():
            x_sample = model.generation(n_sample, c=condition * torch.ones(size=[n_sample, 1]))
        x_sample[:, 0, :] = 0 * x_sample[:, 0, :] + 1
        post_path = path[-1] * x_sample  # with overlapping !!!
        return post_path, condition


def plot_path_extension(
    path,
    model,
    exp_config,
    n_extend_path=10,
    n_extend_time=2,
    file_path=None,
    label=True,
    plot=True,
    idx=1000,  # the last day we have price information
):

    prices0 = path[: idx + 1]
    prices_list = []
    for j in tqdm(range(n_extend_path)):
        prices = prices0
        for i in range(n_extend_time):
            post_path, condition = Extension().extend(model, prices, 1)
            prices = np.concatenate([prices, post_path[0, 1:, 0].numpy()])
        prices_list.append(prices)

    n_timestep_back = 500  # for plotting
    data_length = model.model_config.data_length
    n_timestep_forward = n_extend_time * (data_length - 1)

    if plot:
        apply_source_style()
        fig = plt.figure(figsize=[12, 4])

        path2plot = path[idx + 1 - n_timestep_back : idx + 1]
        plt.plot(
            np.arange(-n_timestep_back + 1, len(path2plot) - n_timestep_back + 1),
            path2plot,
            label="History (S&P500)",
        )

        for prices in prices_list:
            plt.plot(prices[idx:], alpha=0.7, label="Generated")
        # plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Prices")
        plt.title("Path Extension")
        if file_path is not None:
            plt.savefig(file_path, bbox_inches="tight")
        plt.show()

    return [prices[idx:] for prices in prices_list]


def generate_pdv4_from_history(
    idx,
    n_sample,
    n_timestep,
    ds,
):
    prices_post2 = []
    for i in range(2):
        prices_post, log_return, sigma, r10, r11, r20, r21, r1, r2 = PDV4().simulate(
            n_sample,
            n_timestep - 1,
            r10_init=ds.r10[0, idx, 0],
            r11_init=ds.r11[0, idx, 0],
            r20_init=ds.r20[0, idx, 0],
            r21_init=ds.r21[0, idx, 0],
        )
        prices_post *= ds.path[idx]
        prices_post2.append(prices_post)

    return prices_post2


def generate_bs_from_history(
    idx,
    n_sample,
    n_timestep,
    ds,
):
    log_return = price2logreturn(ds.path[idx - 100 : idx + 1])
    log_return_square = log_return**2

    bs_sigma = np.sqrt(log_return_square.mean() / ds.model.dt)
    bs_mu = log_return.mean() / ds.model.dt + bs_sigma**2 / 2

    bs_path = simulate_BS(n_sample, ds.model.dt, n_timestep - 1, bs_mu, bs_sigma).numpy()
    prices_bs = bs_path * ds.path[idx]
    return prices_bs


def load_data_eval_dist_con(n_sample, ds, model, output_dir, plot=False):
    n_timestep = ds.n_timestep
    file_path = pt.join(output_dir, "conditional_paths_dict.pkl")
    try:
        con_paths_dict = load_obj(file_path)
        print("Already exits")
    except:
        base_file_path = pt.join(output_dir, "path_generation/")
        os.makedirs(base_file_path, exist_ok=True)
        con_paths_dict = {}
        for idx in tqdm(np.arange(100, 1000, 20)):
            prices_post2 = generate_pdv4_from_history(idx, n_sample, n_timestep, ds)

            prices_gen, condition = Extension().extend(model, ds.path[: idx + 1], n_sample)
            prices_gen = prices_gen.numpy()

            prices_bs = generate_bs_from_history(idx, n_sample, n_timestep, ds)

            if plot:
                plot_data_eval_dist_con(
                    idx,
                    ds,
                    condition,
                    prices_post2[0],
                    prices_gen,
                    prices_bs,
                    base_file_path,
                )

            transform = lambda x: x / ds.path[idx]

            paths_dict = {
                condition: {
                    "real": transform(prices_post2[0]),
                    "real2": transform(prices_post2[1]),
                    "fake": transform(prices_gen),
                    "bs": transform(prices_bs),
                }
            }

            con_paths_dict.update(paths_dict)
        save_obj(con_paths_dict, file_path)

    return con_paths_dict


def plot_data_eval_dist_con(idx, ds, condition, prices_post, prices_gen, prices_bs, base_file_path):
    apply_source_style()
    n_timestep_back = 10
    fig, ax = plt.subplots(1, 3, figsize=[16, 4], sharex=True, sharey=True)
    ax = ax.flatten()
    ax[0].plot(
        np.arange(-n_timestep_back + 1, 1),
        ds.path[idx + 1 - n_timestep_back : idx + 1],
    )
    ax[0].plot(prices_post[..., 0].T, alpha=0.3, linewidth=1)
    ax[0].legend(["History", "PDV4"])
    ax[0].set_xlabel("Time step")
    ax[0].set_ylabel("Prices")
    ax[0].set_title(f"condition: {condition:.2f}")

    ax[1].plot(
        np.arange(-n_timestep_back + 1, 1),
        ds.path[idx + 1 - n_timestep_back : idx + 1],
    )
    ax[1].plot(prices_gen[..., 0].T, alpha=0.3, linewidth=1)
    ax[1].legend(["History", "Generator"])
    ax[1].set_xlabel("Time step")
    ax[1].set_ylabel("Prices")
    ax[1].set_title(f"condition: {condition:.2f}")

    ax[2].plot(
        np.arange(-n_timestep_back + 1, 1),
        ds.path[idx + 1 - n_timestep_back : idx + 1],
    )
    ax[2].plot(prices_bs[..., 0].T, alpha=0.3, linewidth=1)
    ax[2].legend(["History", "BS_benchmark"])
    ax[2].set_xlabel("Time step")
    ax[2].set_ylabel("Prices")
    ax[2].set_title(f"condition: {condition:.2f}")

    if base_file_path is None:
        plt.show()
    else:
        file_path = base_file_path + f"idx_{idx}_condition_{condition:.2f}.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close(fig)


def compute_eval_dist_con(con_paths_dict, dist_name, output_dir):
    con_metric_dict = {}
    if dist_name == "mmd":
        dist_func = GaussianMMD()
    elif dist_name == "swd":
        dist_func = SWD()
    elif dist_name == "esig":
        dist_func = SignatureMMD()

    file_path = pt.join(output_dir, dist_name + "_conditional_metric_dict.pkl")
    try:
        con_metric_dict = load_obj(file_path)
        print("Load metrics")
    except:
        print("Compute metrics")
        for condition, paths_dict in tqdm(con_paths_dict.items()):
            real1 = torch.tensor(paths_dict["real"], dtype=torch.float32)
            real2 = torch.tensor(paths_dict["real2"], dtype=torch.float32)
            fake = torch.tensor(paths_dict["fake"], dtype=torch.float32)
            bs = torch.tensor(paths_dict["bs"], dtype=torch.float32)

            metric = {
                condition: {
                    "realreal": dist_func(real1, real2),
                    "realfake": dist_func(real1, fake),
                    "realcontrol": dist_func(real1, bs),
                }
            }

            con_metric_dict.update(metric)
        save_obj(con_metric_dict, file_path)
    return con_metric_dict


def plot_eval_dist_con(mmd_con_metric_dict, swd_con_metric_dict, esig_con_metric_dict, file_path=None):
    apply_source_style()
    fig, ax = plt.subplots(1, 3, figsize=[16, 4])

    plot_labels = ["real-real", "real-fake", "real-control"]
    keys = ["realreal", "realfake", "realcontrol"]
    for i, label in enumerate(keys):
        swds = np.array([values[label] for values in swd_con_metric_dict.values()])
        mmds = np.array([values[label] for values in mmd_con_metric_dict.values()])
        esigs = np.array([values[label] for values in esig_con_metric_dict.values()])

        swd_labels = np.array(list(swd_con_metric_dict.keys()))
        mmd_labels = np.array(list(mmd_con_metric_dict.keys()))
        esig_labels = np.array(list(esig_con_metric_dict.keys()))

        def poly_fit_plot(labels, dists, axi, order=2):
            coef = np.polyfit(labels, dists, order)
            p = np.poly1d(coef)
            x = np.linspace(mmd_labels.min(), mmd_labels.max(), 100)
            axi.plot(x, p(x), "--")

        poly_fit_plot(swd_labels, swds, ax[0])
        poly_fit_plot(mmd_labels, mmds, ax[1])
        poly_fit_plot(esig_labels, esigs, ax[2])

        plot_label = plot_labels[i]
        ax[0].scatter(swd_con_metric_dict.keys(), swds, label=plot_label)
        ax[1].scatter(mmd_con_metric_dict.keys(), mmds, label=plot_label)
        ax[2].scatter(esig_con_metric_dict.keys(), esigs, label=plot_label)

    ax[0].set_title("Sliced Wasserstein distance")
    ax[0].set_xlabel("Condition")
    ax[0].legend(loc="upper left")

    ax[1].set_title("Gaussian MMD")
    ax[1].set_xlabel("Condition")
    # ax[1].set_ylabel("Gaussian MMD")
    ax[1].legend(loc="upper left")

    ax[2].set_title("Signature MMD")
    ax[2].set_xlabel("Condition")
    # ax[2].set_ylabel("Signature MMD")
    ax[2].legend(loc="upper left")

    ax[0].set_ylabel("Distance")

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")
    plt.show()


def compute_eval_awd_dist_con(con_paths_dict, output_dir, n_compute_awd=500, n_slices=10, len_slices=3):
    sawd = SAWD(n_compute_awd=n_compute_awd, n_slices=n_slices, len_slices=len_slices)
    condition_awd_dist_dict = {}
    for condition, paths_dict in tqdm(con_paths_dict.items()):
        real_path = paths_dict["real"]
        awd_dist_dict = {}
        for key, value in paths_dict.items():
            if key != "real":
                path = value
                dist = sawd(real_path, path)
                awd_dist_dict[key] = dist

        condition_awd_dist_dict.update({condition: awd_dist_dict})

    file_path = pt.join(
        output_dir,
        f"conditional_awd_metric_dict_{n_compute_awd}_{n_slices}_{len_slices}.pkl",
    )
    save_obj(condition_awd_dist_dict, file_path)
    return condition_awd_dist_dict


def plot_eval_awd_dist_con(metric_dict, file_path=None):
    apply_source_style()
    fig, ax = plt.subplots(1, 1, figsize=[6, 4])

    plot_labels = ["real-real", "real-fake", "real-control"]

    for i, label in enumerate(["real2", "fake", "bs"]):
        awd_array = np.array([values[label] for values in metric_dict.values()])
        awd_mean = awd_array.mean()

        plot_label = plot_labels[i]
        ax.scatter(metric_dict.keys(), awd_array, label=plot_label)
        ax.hlines(awd_mean, 0, 1, color="C" + str(i), alpha=0.5)

    ax.set_title("Sliced adapted Wasserstein distance")
    ax.set_xlabel("Condition")
    ax.set_ylabel("SAWD")
    ax.legend()

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")
    plt.show()


def plot_eval_dist_con_allin1(
    mmd_con_metric_dict,
    swd_con_metric_dict,
    esig_con_metric_dict,
    sawd_con_metric_dict,
    file_path=None,
):
    apply_source_style()
    fig, ax = plt.subplots(1, 4, figsize=[12, 4])

    c_all = np.array(list(swd_con_metric_dict.keys()))  # all have same labels
    idx = c_all < 0.5
    c = c_all[idx]

    plot_labels = ["real-real", "real-fake", "real-control"]
    keys = ["realreal", "realfake", "realcontrol"]

    for i, label in enumerate(keys):

        def poly_fit_plot(labels, dists, axi, order=2):
            coef = np.polyfit(labels, dists, order)
            p = np.poly1d(coef)
            x = np.linspace(labels.min(), labels.max(), 100)
            axi.plot(x, p(x), "--")

        def plot_helper(metric_dict, axi):
            dists = np.array([values[label] for values in metric_dict.values()])[idx]
            poly_fit_plot(c, dists, axi)
            axi.scatter(c, dists, label=plot_labels[i])
            axi.legend(loc="upper left")
            axi.set_xlabel("Condition")

        plot_helper(swd_con_metric_dict, ax[0])
        plot_helper(mmd_con_metric_dict, ax[1])
        plot_helper(esig_con_metric_dict, ax[2])
        plot_helper(sawd_con_metric_dict, ax[3])

    ax[0].set_title("Sliced Wasserstein distance")
    ax[1].set_title("Gaussian MMD")
    ax[2].set_title("Signature MMD")
    ax[3].set_title("Sliced adapted Wasserstein distance")
    ax[0].set_ylabel("Distance")

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")
    plt.show()


def plot_returns(real_prices, fake_prices, file_path=None, real_data_name="S&P 500"):

    apply_source_style()
    fig = plt.figure(figsize=[12, 4])
    ax11 = fig.add_subplot(222)  # add subplot into first position in a 2x2 grid (upper left)
    ax12 = fig.add_subplot(224, sharex=ax11, sharey=ax11)  # add to third position in 2x2 grid (lower left) and sharex with ax11
    ax13 = fig.add_subplot(121)  # add subplot to cover both upper and lower right, in a 2x2 grid. This is the same

    plt.sca(ax13)

    mean = np.mean(price2return(real_prices))
    std = np.std(price2return(real_prices))
    gau = np.random.normal(size=real_prices.size - 1) * std + mean
    # _, bins, _ = plt.hist(gau, bins=100, alpha=0.5, label="Black Scholes", density=True)
    bins = np.linspace(-0.05, 0.05, 100)
    plt.hist(price2return(real_prices), bins=bins, alpha=0.5, label=real_data_name, density=True)
    plt.hist(price2return(fake_prices), bins=bins, alpha=0.5, label="Fake", density=True)
    plt.hist(gau, bins=bins, alpha=0.5, label="Black Scholes", density=True)

    x = np.linspace(-0.04, 0.04, 100)
    plt.plot(x, stats.norm.pdf(x, mean, std), label="Gaussian density")
    plt.legend()
    plt.title("Returns")
    plt.xlim(-0.05, 0.05)

    plt.sca(ax11)
    plt.plot(price2return(real_prices), c="tab:blue", label=real_data_name)
    # plt.ylabel('Returns (S&P500)')
    plt.title("Returns")
    plt.legend(loc="upper left")

    plt.sca(ax12)
    plt.plot(price2return(fake_prices), c="tab:orange", label="Fake")
    plt.xlabel("Time")
    plt.legend(loc="upper left")

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")
    plt.show()


import pandas as pd


def plot_autocorrelation(real_prices, fake_prices, file_path=None, real_data_name="S&P 500"):
    apply_source_style()
    fig, ax = plt.subplots(1, 3, figsize=[12, 4], sharey=True)
    n_lag = 100

    def func_ac(log_return_path):
        pd_log_return = pd.Series(log_return_path)
        autocorrelation = [pd_log_return.autocorr(lag) for lag in range(n_lag)]
        return autocorrelation

    def plot_ac(func=lambda x: x):
        autocorrelation = func_ac(func(price2return(real_prices)))
        plt.plot(np.arange(n_lag), autocorrelation, marker=".", label=real_data_name)
        autocorrelation = func_ac(func(price2return(fake_prices)))
        plt.plot(np.arange(n_lag), autocorrelation, marker=".", label="Fake")
        plt.xlabel("Time lag")
        plt.legend()

    plt.sca(ax[0])
    plt.title("Returns")
    plt.ylabel("Correlation")
    plot_ac()

    plt.sca(ax[1])
    plt.title("Square Returns")
    plot_ac(func=lambda x: x**2)

    plt.sca(ax[2])
    plt.title("Absolute Returns")
    plot_ac(func=lambda x: np.abs(x))

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")
    plt.show()


def plot_skew_kurtosis(real_prices, fake_prices_list, file_path=None, real_data_name="S&P 500"):
    apply_source_style()
    fig, ax = plt.subplots(1, 2, figsize=[12, 4], sharey=True)

    def plot_stats(stats_func):
        stats_features = np.array([stats_func(price2return(path)) for path in fake_prices_list])
        plt.hist(stats_features, bins=50, alpha=0.5, label="Fake")
        plt.vlines(stats_features.mean(), 0, 100, "tab:blue", linewidth=3, label="Fake (mean)")
        plt.vlines(stats_func(price2return(real_prices)), 0, 100, "r", linewidth=3, label=real_data_name)
        plt.legend()

    plt.sca(ax[0])
    plt.title("Skewness of Returns")
    plot_stats(stats.skew)

    plt.sca(ax[1])
    plt.title("Kurtosis of Returns")
    plot_stats(stats.kurtosis)

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")
    plt.show()
