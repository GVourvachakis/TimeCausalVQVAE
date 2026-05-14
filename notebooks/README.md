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

`sp500_vix.ipynb` is the final promoted-method notebook for the S&P500/VIX report workflow. It
centres the standard causal VQ tokenizer and additive scalar-conditioned causal AR prior, with
continuous TC-VAE outputs as optional reference artefacts only.

`discrete_latent_geometry_demo.ipynb` is a lightweight inspection notebook for the CLI-generated
latent-geometry summaries and plots. It defaults to the promoted standard VQ preset, includes an
RVQ q2 ablation preset, and prints setup instructions when tokenizer or token-data artefacts are
absent.
