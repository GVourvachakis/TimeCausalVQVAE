# Notebook Guidance

Notebooks are demonstrative only. The canonical workflow is the CLI/script route in the
repository README.

Kept notebooks:

- `black_scholes.ipynb`
- `discrete_latent_geometry_demo.ipynb`
- `heston.ipynb`
- `pdv.ipynb`
- `sp500_vix.ipynb`

These notebooks should stay output-stripped and should not contain generated figures,
checkpoints, or local data. Generated artefacts belong under ignored `outputs/` paths.

`discrete_latent_geometry_demo.ipynb` is a lightweight inspection notebook for the CLI-generated
latent-geometry summaries and plots. It defaults to local ignored `outputs/` paths and prints
setup instructions when tokenizer or token-data artifacts are absent.
