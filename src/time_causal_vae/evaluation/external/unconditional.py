"""Optional unconditional evaluation helpers copied from the upstream release.

These helpers expose broader paper-analysis routines and optional metrics. They are not
required for selected checkpoint reproduction.
"""

from collections import defaultdict
from os import path as pt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from time_causal_vae.models.distances import GaussianMMD, GaussianMMD2
from time_causal_vae.evaluation.metrics import SAWD, SWD, SignatureMMD
from time_causal_vae.evaluation.checkpoints import TargetModelEvaluator as ModelEvaluator
from time_causal_vae.evaluation.style import apply_source_style
from time_causal_vae.utils.serialization import load_obj, save_obj


def load_data_eval_dist_uncon(model_evaluator: ModelEvaluator):
    fake_data_list = []
    for i in range(10):
        test_data, gen_data, _ = model_evaluator.load_data()
        fake_data = gen_data
        fake_data_list.append(fake_data)

    control_data_dict = {}
    for sigma in np.linspace(0.1, 0.3, 21):
        data, label = model_evaluator.data_ppl._get_data_label(model_evaluator.exp_config, control=sigma)
        control_data_dict[sigma] = data
    return fake_data_list, control_data_dict


def compute_eval_dist_uncon(real_data, fake_data_list, control_data_dict, dist_name, output_dir, comment=""):
    if dist_name == "mmd":
        dist_func = GaussianMMD()
    elif dist_name == "swd":
        dist_func = SWD()
    elif dist_name == "esig":
        dist_func = SignatureMMD()

    p = Path(output_dir)
    file_path = pt.join(p.parent, dist_name + f"{len(control_data_dict)}_control.pkl")
    try:
        control_dist = load_obj(file_path)
    except:
        control_dist = {}
        for control_key, control_data in tqdm(control_data_dict.items()):
            control_dist[control_key] = dist_func(real_data, control_data)
        save_obj(control_dist, file_path)

    file_path = pt.join(output_dir, dist_name + f"{len(fake_data_list)}_fake{comment}.pt")
    try:
        fake_dist = load_obj(file_path)
    except:
        fake_dist_list = []
        for fake_data in tqdm(fake_data_list):
            fake_dist_list.append(dist_func(real_data, fake_data))
        fake_dist = torch.tensor(fake_dist_list)
        save_obj(fake_dist, file_path)
    return control_dist, fake_dist


def plot_eval_dist_uncon(
    mmd_control_dist,
    mmd_fake_dist,
    swd_control_dist,
    swd_fake_dist,
    esig_control_dist,
    esig_fake_dist,
    plot_file_path=None,
):
    apply_source_style()
    fig, ax = plt.subplots(1, 3, figsize=[12, 4])

    ax[0].scatter(swd_control_dist.keys(), swd_control_dist.values(), label="control")
    for swd in swd_fake_dist:
        ax[0].hlines(swd, 0.1, 0.3, color="r", alpha=0.2)
    mean_swd_fake = swd_fake_dist.mean()
    ax[0].hlines(mean_swd_fake, 0.1, 0.3, color="r", alpha=0.8, label="fake")
    ax[0].set_title("Sliced Wasserstein distance")
    ax[0].set_xlabel("Volatility")
    # ax[0].set_ylabel("SWD")
    ax[0].legend(loc="upper right")

    ax[1].scatter(mmd_control_dist.keys(), mmd_control_dist.values(), label="control")
    for mmd in mmd_fake_dist:
        ax[1].hlines(mmd, 0.1, 0.3, color="r", alpha=0.2)
    mean_mmd_fake = mmd_fake_dist.mean()
    ax[1].hlines(mean_mmd_fake, 0.1, 0.3, color="r", alpha=0.8, label="fake")
    ax[1].set_title("Gaussian MMD")
    ax[1].set_xlabel("Volatility")
    # ax[1].set_ylabel("MMD")
    ax[1].legend()
    ax[1].legend(loc="upper right")

    ax[2].scatter(esig_control_dist.keys(), esig_control_dist.values(), label="control")
    for esig in esig_fake_dist:
        ax[2].hlines(esig, 0.1, 0.3, color="r", alpha=0.2)
    mean_esig_fake = esig_fake_dist.mean()
    ax[2].hlines(mean_esig_fake, 0.1, 0.3, color="r", alpha=0.8, label="fake")
    ax[2].set_title("Signature MMD")
    ax[2].set_xlabel("Volatility")
    # ax[2].set_ylabel("SigMMD")
    ax[2].legend(loc="upper right")

    ax[0].set_ylabel("Distance")

    if plot_file_path is not None:
        plt.savefig(plot_file_path, bbox_inches="tight")
    plt.show()


def compute_eval_awd_dist_uncon(
    real_data,
    fake_data,
    control_data,
    output_dir,
    n_compute_awd=500,
    n_slices=10,
    len_slices=3,
    n_seed=10,
):
    sawd = SAWD(n_compute_awd=n_compute_awd, n_slices=n_slices, len_slices=len_slices)

    file_path = pt.join(output_dir, f"sawd_{n_seed}_{n_compute_awd}_{n_slices}_{len_slices}.pkl")
    try:
        sawd_dist_dict = load_obj(file_path)
    except:
        sawd_dist_dict = defaultdict(list)
        for i in tqdm(range(n_seed)):
            idx = np.random.choice(len(real_data), n_compute_awd)
            idx2 = np.random.choice(len(real_data), n_compute_awd)
            real1 = real_data[idx]
            real2 = real_data[idx2]
            fake = fake_data[idx2]
            control = control_data[idx2]

            dist_real = sawd(real1, real2)
            dist_fake = sawd(real1, fake)
            dist_control = sawd(real1, control)

            sawd_dist_dict["realreal"].append(dist_real)
            sawd_dist_dict["realfake"].append(dist_fake)
            sawd_dist_dict["realcontrol"].append(dist_control)

            print(f"Real-Real: {dist_real:.2f}")
            print(f"Real-Fake: {dist_fake:.2f}")
            print(f"Real-Control: {dist_control:.2f}")

        save_obj(sawd_dist_dict, file_path)
        sawd_dist_dict = load_obj(file_path)
    return sawd_dist_dict


def plot_eval_awd_dist_uncon(sawd_dist_dict, output_dir):
    apply_source_style()
    for label in ["realreal", "realfake", "realcontrol"]:
        dist_array = np.array(sawd_dist_dict[label])
        print(label)
        print(f"mean: {dist_array.mean()}")
        print(f"std: {dist_array.std()}")

    plt.hist(sawd_dist_dict["realreal"])
    plt.hist(sawd_dist_dict["realfake"])
    plt.hist(sawd_dist_dict["realcontrol"])
    file_path = pt.join(output_dir, "sawd.png")
    plt.savefig(file_path, bbox_inches="tight")
