"""Optional optimal-stopping evaluation copied from upstream downstream evaluations.

The numerical optimal-stopping algorithms are isolated under
`time_causal_vae.evaluation.external.optimal_stopping`. They are optional and not
required for selected checkpoint reproduction.
"""

from copy import deepcopy
from os import path as pt

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from time_causal_vae.evaluation.checkpoints import TargetModelEvaluator as ModelEvaluator
from time_causal_vae.evaluation.external.optimal_stopping.algorithms.backward_induction.DOS import DeepOptimalStopping
from time_causal_vae.evaluation.external.optimal_stopping.data.stock_model import Model as OSModel
from time_causal_vae.evaluation.external.optimal_stopping.payoffs.payoff import MaxPut
from time_causal_vae.evaluation.style import apply_source_style


class DatasetModel(OSModel):
    def __init__(
        self,
        data,
        drift,
        volatility,
        nb_paths,
        nb_stocks,
        nb_dates,
        spot,
        maturity,
        dividend=0,
        **keywords,
    ):
        """
        paths: (nb_paths, nb_dates+1, nb_stocks)
        nparray
        """
        self.nb_paths, _, self.nb_stocks = data.shape
        assert self.nb_paths == nb_paths and self.nb_stocks == nb_stocks, "Should be the same"
        self.data = np.swapaxes(data, 1, 2)

        super(DatasetModel, self).__init__(
            drift=drift,
            dividend=dividend,
            volatility=volatility,
            nb_stocks=nb_stocks,
            nb_paths=nb_paths,
            nb_dates=nb_dates,
            spot=spot,
            maturity=maturity,
            name="DatasetModel",
        )
        # Volatility is no longer relevant in generating paths

    def generate_paths(
        self,
    ):
        """Returns a nparray (nb_paths * nb_stocks * nb_dates) with prices."""
        return self.data, None


def load_os_data(model_evaluator: ModelEvaluator, s0, n_maturity_timestep, n_sample_test):
    exp_config = deepcopy(model_evaluator.exp_config)
    exp_config.n_sample = n_sample_test

    real_data_list = []
    fake_data_list = []
    for i in tqdm(range(10)):
        test_data, gen_data, _ = model_evaluator.load_data(n_sample_test)
        real_data = test_data[:, : n_maturity_timestep + 1] * s0
        gen_data[:, 0] = 0 * gen_data[:, 0] + 1
        fake_data = gen_data[:, : n_maturity_timestep + 1] * s0
        real_data_list.append(real_data)
        fake_data_list.append(fake_data)

    control_data_dict = {}
    for sigma in tqdm(np.linspace(0.1, 0.3, 21)):
        data, label = model_evaluator.data_ppl._get_data_label(exp_config, control=sigma)
        control_data_dict[sigma] = data[:, : n_maturity_timestep + 1] * s0

    return real_data_list, fake_data_list, control_data_dict


def compare_os(
    r,
    s0,
    n_maturity_timestep,
    dt,
    strike,
    real_data_list,
    fake_data_list,
    control_data_dict,
    file_path=None,
):
    apply_source_style()
    control_data_list = list(control_data_dict.values())

    def data_to_price(data_list):
        price_list = []
        for data in tqdm(data_list):
            data = data.numpy()
            stock_model_ = DatasetModel(
                data=data,
                drift=r,
                volatility=0,
                nb_paths=len(data),
                nb_stocks=1,
                nb_dates=n_maturity_timestep,
                spot=s0,
                maturity=n_maturity_timestep * dt,
                dividend=0,
            )
            payoff_ = MaxPut(strike=strike)
            pricer = DeepOptimalStopping(stock_model_, payoff_)
            # pricer = LeastSquaresPricer(stock_model_, payoff_)
            price, gen_time = pricer.price()
            price_list.append(price)
        return price_list

    real_price_list = data_to_price(real_data_list)
    fake_price_list = data_to_price(fake_data_list)
    control_price_list = data_to_price(control_data_list)

    fig, ax = plt.subplots(1, 1, figsize=[12, 4])
    price_dict = {"real": real_price_list, "fake": fake_price_list}
    for i, key in enumerate(price_dict.keys()):
        for value in price_dict[key]:
            plt.hlines(value, 0.1, 0.3, colors="C" + str(i + 2), alpha=0.2)
        plt.hlines(
            np.array(price_dict[key]).mean(),
            0.1,
            0.3,
            colors="C" + str(i + 2),
            label=key,
            alpha=0.8,
        )

    plt.scatter(list(control_data_dict.keys()), control_price_list, label="control")

    plt.legend()
    plt.xlabel("Volatility")
    plt.ylabel("Optimal Stopping Values")
    plt.title("Optimal Stopping Values")

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")
    plt.show()
