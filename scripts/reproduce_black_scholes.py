"""Reproduce the selected Black-Scholes paper workflow."""

from reproduction_common import ExperimentSpec, main_for_experiment

SPEC = ExperimentSpec(
    name="black_scholes",
    config_path="configs/experiments/black_scholes_beta_cvae.yaml",
)


if __name__ == "__main__":
    main_for_experiment(SPEC)
