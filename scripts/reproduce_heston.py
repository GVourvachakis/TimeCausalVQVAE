"""Reproduce the selected Heston paper workflow."""

from reproduction_common import ExperimentSpec, main_for_experiment

SPEC = ExperimentSpec(
    name="heston",
    config_path="configs/experiments/heston_info_cvae.yaml",
)


if __name__ == "__main__":
    main_for_experiment(SPEC)
