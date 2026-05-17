# mypy: ignore-errors
# ruff: noqa
from typing import List, Optional, Union

import numpy as np
import torch

from time_causal_vae.models.continuous.config import BasePipeline
from time_causal_vae.models.continuous.objectives.vae import VAE
from time_causal_vae.training.config import BaseTrainerConfig
from time_causal_vae.training.trainer import BaseTrainer
from time_causal_vae.utils.logging import get_console_logger

logger = get_console_logger(__name__)


class TrainingPipeline(BasePipeline):
    def __init__(
        self,
        model: Optional[VAE],
        training_config: Optional[BaseTrainerConfig] = None,
        exp_config=None,
    ):
        if not isinstance(training_config, BaseTrainerConfig):
            raise AssertionError("A 'BaseTrainerConfig' is expected for the pipeline")

        self.model = model
        self.training_config = training_config
        self.exp_config = exp_config

    def __call__(
        self,
        train_dataset: Union[np.ndarray, torch.Tensor, torch.utils.data.Dataset],
        eval_dataset: Union[np.ndarray, torch.Tensor, torch.utils.data.Dataset] = None,
        device_name=None,
    ):
        if self.training_config.wandb_callback:
            from time_causal_vae.training.callbacks import WandbCallback

            callbacks = []  # the TrainingPipeline expects a list of callbacks
            wandb_cb = WandbCallback()  # Build the callback
            wandb_cb.setup(
                exp_config=self.exp_config,
                training_config=self.training_config,  # training config
                model_config=self.model.model_config,  # model config
                project_name=getattr(self.training_config, "wandb_project", "time-causal-vae"),
                entity_name=getattr(self.training_config, "wandb_entity", None),
                run_name=getattr(self.training_config, "wandb_run_name", None),
                wandb_mode=getattr(self.training_config, "wandb_mode", None),
            )
            callbacks.append(wandb_cb)  # Add it to the callbacks list
        else:
            callbacks = None

        trainer = BaseTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=self.training_config,
            exp_config=self.exp_config,
            callbacks=callbacks,
            device_name=device_name,
        )

        self.trainer = trainer
        return trainer

    def train(self, log_output):
        self.trainer.train(log_output=log_output)
