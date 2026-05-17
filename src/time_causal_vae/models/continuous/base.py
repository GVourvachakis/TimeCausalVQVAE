# mypy: ignore-errors
# ruff: noqa
import os
from copy import deepcopy
from typing import Any

import torch
from torch.nn import Module

from time_causal_vae.models.continuous.config import BaseConfig
from time_causal_vae.utils.output import ModelOutput


class BaseModel(Module):
    """Base class for target Time-Causal VAE models.

    The save and load methods intentionally mirror the legacy checkpoint
    structure: ``model_config.json`` plus ``model.pt`` containing a
    ``model_state_dict`` entry.
    """

    model_config: BaseConfig

    def __init__(self, model_config: BaseConfig):
        """Initialise the model with a serialisable config."""
        super().__init__()
        self.model_config = model_config
        self.model_name = "BaseModel"
        self.device = None  # Do we want to enforce that all parameter in the same device?

    def forward(self, inputs: Any, **kwargs) -> ModelOutput:
        """Delegate to ``torch.nn.Module`` for the default error behaviour."""
        return super().forward()

    def generation(self, n_sample: int, device, **kwargs):
        """Generate samples from the model."""
        raise NotImplementedError()

    def load_from_folder(self, dir_path: str) -> None:
        """Load ``model.pt`` weights from a legacy-compatible folder."""
        model_weights = BaseModel._load_model_weights_from_folder(dir_path)
        self.load_state_dict(model_weights)

    def save(self, dir_path: str) -> None:
        """Save config and weights using the legacy checkpoint layout."""
        model_dict = {"model_state_dict": deepcopy(self.state_dict())}
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # TODO: unify this two saving to saving obj
        self.model_config.save_json(dir_path, "model_config")
        torch.save(model_dict, os.path.join(dir_path, "model.pt"))

    def update(self) -> None:
        """Run any per-epoch model update; no-op by default."""
        pass

    @classmethod
    def _load_model_weights_from_folder(cls, dir_path: str):
        """Load a state dict from ``model.pt`` inside ``dir_path``."""
        file_list = os.listdir(dir_path)

        if "model.pt" not in file_list:
            raise FileNotFoundError(
                f"Missing model weights file ('model.pt') file in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_weights = os.path.join(dir_path, "model.pt")

        try:
            model_weights = torch.load(path_to_model_weights, map_location="cpu", weights_only=True)

        except RuntimeError:
            RuntimeError("Enable to load model weights. Ensure they are saves in a '.pt' format.")

        if "model_state_dict" not in model_weights.keys():
            raise KeyError(
                "Model state dict is not available in 'model.pt' file. Got keys:"
                f"{model_weights.keys()}"
            )

        model_weights = model_weights["model_state_dict"]

        return model_weights
