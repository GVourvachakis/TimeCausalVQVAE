"""Optional expected-signature helpers copied from the upstream evaluation code.

This module contains borrowed/adapted signature metric utilities referenced by the
upstream Time-Causal VAE release. It is optional and not required for selected
checkpoint reproduction or the promoted public S&P500/VIX workflow. Calling
``compute_exp_sig`` may require the optional ``signatory`` package. Relevant
upstream methods include Sig-Wasserstein-GANs by Ni et al. (2021) and
Randomised-Signature-TimeSeries-Generation by Biagini, Gonon, and Walter.

New signature-conditioning and signature-kernel work should be implemented
outside ``evaluation.external`` so this upstream compatibility area remains
isolated from project-native metrics.
"""

import math
import torch

# THE SIGNATURE COMPUTATION ON CPU BY DEFAULT

DEVICE = torch.device("cpu")

# l2 loss


def l2_dist(x, y: float) -> float:
    return (x - y).pow(2).sum().sqrt()


# Time augmentation of input path


def apply_time_augmentations(x: torch.tensor, device=DEVICE) -> torch.tensor:
    y = x.clone().to(device)
    t = torch.linspace(0, 1, y.shape[1]).reshape(
        1, -1, 1).repeat(y.shape[0], 1, 1).to(device)
    return torch.cat([t, x], dim=1)


# Basepoint augmentation of input path


def apply_bp_augmentation(x: torch.tensor, device=DEVICE) -> torch.tensor:
    y = x.clone().to(device)
    basepoint = torch.zeros(y.shape[0], 1, y.shape[2]).to(device)
    return torch.cat([basepoint, x], dim=1)


# Visibility augmentation of input path


def apply_ivisi_augmentation(x, device=DEVICE):
    y = x.clone().to(device)

    init_tworows_ = torch.zeros_like(y[:, :1, :]).to(device)
    init_tworows = torch.cat((init_tworows_, y[:, :1, :]), axis=1)

    temp = torch.cat((init_tworows, y), axis=1)

    last_col1 = torch.zeros_like(y[:, :2, :1]).to(device)
    last_col2 = torch.cat((last_col1, torch.ones_like(y[:, :, :1])), axis=1)

    output = torch.cat((temp, last_col2), axis=-1)
    return output


# Lead-lag transformation of input path


def apply_lead_lag_augmentation(x: torch.tensor, device=DEVICE):
    y = x.clone().to(device)
    y_rep = torch.repeat_interleave(y, repeats=2, dim=1).to(device)
    y_lead_lag = torch.cat([y_rep[:, :-1], y_rep[:, 1:]], dim=2)
    return y_lead_lag


# Applying augmentations/transformations to input path


def apply_augmentations(
    x: torch.tensor,
    time=True,
    lead_lag=True,
    ivisi=True,
    basepoint=False,
    device=DEVICE,
) -> torch.tensor:
    y = x.clone().to(device)
    if time:
        y = apply_time_augmentations(y, device)
    if lead_lag:
        y = apply_lead_lag_augmentation(y, device)
    if ivisi:
        y = apply_ivisi_augmentation(y, device)
    if basepoint:
        y = apply_bp_augmentation(y, device)
    return y


# Computing the expected signature of input paths


def compute_exp_sig(x: torch.tensor, trunc: int, augmented=True, normalise=True) -> torch.tensor:
    try:
        import signatory  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Expected-signature evaluation requires the optional dependency 'signatory'."
        ) from exc

    if augmented:
        x = apply_augmentations(x.clone()).to(DEVICE)
    exp_sig = signatory.signature(x, depth=trunc).mean(0).to(DEVICE)
    dim = x.shape[2]
    count = 0
    if normalise:
        for i in range(trunc):
            exp_sig[count: count + dim **
                    (i + 1)] = exp_sig[count: count + dim ** (i + 1)] * math.factorial(i + 1)
            count = count + dim ** (i + 1)
    return exp_sig.to(DEVICE)
