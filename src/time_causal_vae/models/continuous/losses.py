# mypy: ignore-errors
# ruff: noqa
import torch


def get_loss(loss_name):
    if loss_name == "l1":
        return l1_loss
    if loss_name == "l2":
        return l2_loss


def l1_loss(x, recon_x):
    recon_loss = torch.abs(recon_x - x).mean(dim=0).sum()
    return recon_loss


def l2_loss(x, recon_x):
    recon_loss = torch.square(recon_x - x).mean(dim=0).sum()
    return recon_loss
