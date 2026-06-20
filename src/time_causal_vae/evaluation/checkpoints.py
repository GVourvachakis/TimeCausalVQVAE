# mypy: ignore-errors
# ruff: noqa
"""Target checkpoint-backed evaluator for selected Time-Causal VAE runs."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from time_causal_vae.data.base import DatasetOutput
from time_causal_vae.data.pipeline import DataPipeline
from time_causal_vae.evaluation.checkpoint_compatibility import (
    compatibility_summary,
    load_legacy_exp_config,
)
from time_causal_vae.evaluation.metrics import SWD, GaussianMMD
from time_causal_vae.models.continuous.factory import ModelFactory
from time_causal_vae.utils.random import set_seed
from time_causal_vae.utils.serialization import load_obj, save_obj


def base2final_model_dirs(base_model_dir: str) -> list[str]:
    """Return legacy ``final_model`` folders below a base model directory."""
    hyper_model_dirs = [f.path for f in os.scandir(base_model_dir) if f.is_dir()]
    return [
        checkpoint_dir.path
        for model_dir in hyper_model_dirs
        for checkpoint_dir in os.scandir(model_dir)
        if checkpoint_dir.is_dir() and "final_model" in checkpoint_dir.path
    ]


class TargetModelEvaluator:
    """Evaluate a target model from a legacy-compatible checkpoint folder.

    Parameters
    ----------
    model_dir:
        Path to the ``final_model`` directory containing ``model.pt``.
    exp_config:
        Optional already-loaded legacy experiment config.
    model:
        Optional pre-built target model.
    base_data_dir:
        Base directory for local datasets. This is injected into the loaded
        legacy config, matching the migrated legacy evaluator behaviour.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        exp_config: Any | None = None,
        model: Any | None = None,
        base_data_dir: str | None = None,
        device: torch.device | str | None = None,
        *args,
        **kwargs,
    ) -> None:
        self.device = None if device is None else torch.device(device)
        if model_dir:
            self.exp_config = self.load_from_folder(Path(model_dir), base_data_dir)
            self.load_model()
        else:
            self.exp_config = deepcopy(exp_config)
            self.model = model
            if self.device is not None and self.model is not None:
                self._move_model_to_device()

        self.data_ppl = DataPipeline()
        self.compatibility = compatibility_summary(self.exp_config)
        self.autoload_hyper_label()

    def load_from_folder(self, model_dir: Path, base_data_dir: str | None) -> Any:
        """Load ``exp_config.yaml`` from the parent of ``final_model``."""
        self.model_dir = str(model_dir)
        self.hyper_model_dir = str(model_dir.parent)
        self.exp_config_path = str(model_dir.parent / "exp_config.yaml")
        return load_legacy_exp_config(Path(self.exp_config_path), base_data_dir=base_data_dir)

    def load_model(self) -> None:
        """Build the target model from legacy config fields and load weights."""
        self.network_ppl = ModelFactory()
        self.model = self.network_ppl(self.exp_config)
        self.model.load_from_folder(self.model_dir)
        if self.device is not None:
            self._move_model_to_device()

    def _move_model_to_device(self) -> None:
        """Move the evaluator model to the configured device."""
        self.model = self.model.to(self.device)
        self.model.device = self.device

    def load_data(
        self,
        n_sample_test: int = 5000,
        seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load real data, generated data, and reconstructions."""
        if seed > 0:
            set_seed(seed)
        exp_config = deepcopy(self.exp_config)
        exp_config.n_sample = n_sample_test
        _, test_dataset = self.data_ppl(exp_config)

        data = test_dataset.data
        labels = test_dataset.labels
        if self.device is not None:
            data = data.to(self.device)
            labels = labels.to(self.device)
        dataset_output = DatasetOutput(data=data, labels=labels)

        with torch.no_grad():
            model_output = self.model(dataset_output)
            test_data = dataset_output["data"]
            recon_data = model_output["recon_x"]
            gen_data = self.model.generation(
                n_sample_test,
                c=dataset_output["labels"][:n_sample_test],
            )

        return test_data, gen_data, recon_data

    def ensure_base_dataset(self) -> Any:
        """Initialise and return the raw dataset used by optional diagnostics."""
        if self.data_ppl.base_dataset is None:
            exp_config = deepcopy(self.exp_config)
            self.data_ppl._get_data_label(exp_config, use="eval")
        return self.data_ppl.base_dataset

    def compute_hyper_metric(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the legacy hyper metrics: Gaussian MMD and SWD."""
        return {
            "mmd": GaussianMMD()(real_data, fake_data),
            "swd": SWD()(real_data, fake_data),
        }

    def save_hyper_metric(self, hyper_metric: dict[str, Any]) -> None:
        """Save ``hyper_metric.pkl`` inside the checkpoint directory."""
        self.hyper_metric_path = os.path.join(self.model_dir, "hyper_metric.pkl")
        save_obj(hyper_metric, self.hyper_metric_path)

    def load_hyper_metric(self) -> Any:
        """Load ``hyper_metric.pkl`` from the checkpoint directory."""
        self.hyper_metric_path = os.path.join(self.model_dir, "hyper_metric.pkl")
        return load_obj(self.hyper_metric_path)

    def autoload_hyper_label(self) -> None:
        """Construct the same short hyper-parameter label as the legacy evaluator."""
        if self.exp_config.model == "BetaCVAE":
            self.hyper_label = f"{self.model.beta:.4f}"
        elif self.exp_config.model == "InfoCVAE":
            self.hyper_label = f"({self.model.alpha:.4f},{self.model.beta:.4f})"
        else:
            self.hyper_label = self.exp_config.model


ModelEvaluator = TargetModelEvaluator
