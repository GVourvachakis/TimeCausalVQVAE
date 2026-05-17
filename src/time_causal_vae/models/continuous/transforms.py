# mypy: ignore-errors
# ruff: noqa
import torch
from torch import Tensor


def get_transform(transform_name):
    if transform_name == "" or transform_name == "id":
        return Id_transform()
    elif transform_name == "exp":
        return Exp_transform()
    elif transform_name == "log":
        return Log_transform()
    else:
        raise NameError("No such transform name: {transform_name}")


class Id_transform(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return x


class Log_transform(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return x.log()


class Exp_transform(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return x.exp()
