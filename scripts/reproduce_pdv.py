"""Reproduce the selected path-dependent volatility paper workflow."""

from reproduction_common import ExperimentSpec, main_for_experiment

SPEC = ExperimentSpec(
    name="pdv",
    config_path="configs/experiments/pdv_info_cvae.yaml",
)


if __name__ == "__main__":
    main_for_experiment(SPEC)
