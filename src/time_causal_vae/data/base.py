# ruff: noqa
"""Dataset containers shared by the target data layer.

All target pipelines return ``BaseDataset`` objects whose items expose ``data``
and ``labels`` fields. For selected path datasets, ``data`` has shape
``[n_sample, n_timestep, data_dim]`` and ``labels`` has shape
``[n_sample, condition_dim]``.
"""

# We could simplified the datasetoutput later

import logging
from collections import OrderedDict
from typing import Any, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class DatasetOutput(OrderedDict):
    r"""Same dataset output as in pythae library, inspired from
    the ``ModelOutput`` class from hugginface transformers library.

    This works with our BaseDataset, which uses DatasetOutput as output
    """

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


def collate_dataset_output(batch):
    """Collate function that treats the `DatasetOutput` class correctly."""
    if isinstance(batch[0], DatasetOutput):
        # `default_collate` returns a dict for older versions of PyTorch.
        return DatasetOutput(**default_collate(batch))
    else:
        return default_collate(batch)


class BaseDataset(Dataset):
    """This class is the Base class copied from pythae's dataset

    A ``__getitem__`` is redefined and outputs a python dictionary
    with the keys corresponding to `data` and `labels`.
    This Class should be used for any new data sets.

    The goal of datasetput is just:
    datasetoutput['data'] = data
    datasetoutput['labels'] = labels


    """

    def __init__(self, data: Tensor, labels: Tensor):
        self.labels = labels.type(torch.float)
        self.data = data.type(torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data

        Args:
            index (int): The index of the data in the Dataset

        Returns:
            (dict): A dictionary with the keys 'data' and 'labels' and corresponding
            torch.Tensor
        """
        # Select sample
        X = self.data[index]
        y = self.labels[index]

        return DatasetOutput(data=X, labels=y)
