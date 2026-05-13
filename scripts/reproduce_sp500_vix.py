"""Reproduce the selected S&P500/VIX paper workflow."""

from reproduction_common import ExperimentSpec, main_for_experiment

SPEC = ExperimentSpec(
    name="sp500_vix",
    config_path="configs/experiments/sp500_vix_beta_cvae.yaml",
)


if __name__ == "__main__":
    main_for_experiment(SPEC)
