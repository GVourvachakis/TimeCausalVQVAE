# Continuous And Discrete Model Namespace Refactor Plan

## Scope

This document plans a public-safe namespace refactor only. It does not move files, change imports,
train models, change checkpoint layouts, change model behaviour, change config schemas, or merge
into `main`.

The goal is to place continuous and discrete latent model families in parallel under
`time_causal_vae.models`, while keeping all current public imports available through lightweight
compatibility wrappers.

## Naming Note

- The project title can be `TimeCausalVAE`.
- The repository may remain `TimeCausalVQVAE`.
- The distribution and import package remain `time-causal-vae` and `time_causal_vae`.
- Do not rename the package import path.

## Target Layout

```text
src/time_causal_vae/
  models/
    continuous/
      encoders/
      decoders/
      conditioners/
      priors/
      objectives/
      factory.py
      config.py
      transforms.py
      losses.py
      distances.py
      initialisation.py
      base.py
    discrete/
      tokenizers/
        causal_vq_tokenizer.py
        quantizers.py
        config.py
      priors/
        causal_transformer.py
        config.py
        data.py
        masks.py
    layers/
      causal_conv.py
```

`models/layers/causal_conv.py` should remain a shared model layer namespace because the causal
convolution stack is used by discrete tokenizers and may also be useful for continuous components.

## Import Audit

The audit searched for imports or references to these current public namespaces:

- `time_causal_vae.models.encoders`
- `time_causal_vae.models.decoders`
- `time_causal_vae.models.priors`
- `time_causal_vae.models.objectives`
- `time_causal_vae.tokenization`
- `time_causal_vae.token_prior`

### Continuous Model Imports

Current continuous modules import one another through the existing `models.*` namespace. The main
clusters are:

- `src/time_causal_vae/models/factory.py`: imports encoders, decoders, priors, and objectives.
- `src/time_causal_vae/models/objectives/*.py`: imports encoders, decoders, priors, losses,
  transforms, and distance helpers.
- `src/time_causal_vae/models/encoders/*.py` and `src/time_causal_vae/models/decoders/*.py`:
  import local base classes and initialisation helpers.
- `src/time_causal_vae/models/priors/*.py`: import prior base classes and Gaussian helpers.
- `src/time_causal_vae/training/pipeline.py`: imports `models.objectives.vae.VAE`.

Additional old `models.*` imports that should move to `models.continuous.*` internally:

- `src/time_causal_vae/cli/train.py`
- `src/time_causal_vae/evaluation/checkpoints.py`
- `src/time_causal_vae/evaluation/external/conditional.py`
- `src/time_causal_vae/evaluation/external/unconditional.py`
- `src/time_causal_vae/evaluation/metrics.py`
- `src/time_causal_vae/evaluation/token_prior.py`
- `src/time_causal_vae/training/callbacks.py`
- `src/time_causal_vae/training/config.py`
- `src/time_causal_vae/training/trainer.py`

The current public notebooks do not directly import the audited continuous model namespaces.

### Discrete Model Imports

Current discrete tokenizer imports are concentrated in:

- `src/time_causal_vae/cli/train_tokenizer.py`
- `src/time_causal_vae/cli/evaluate_tokenizer.py`
- `src/time_causal_vae/cli/evaluate_token_prior.py`
- `src/time_causal_vae/evaluation/tokenizer.py`
- `src/time_causal_vae/evaluation/token_prior.py`
- `src/time_causal_vae/token_prior/data.py`
- `src/time_causal_vae/tokenization/__init__.py`
- `src/time_causal_vae/tokenization/causal_vq_tokenizer.py`

Current discrete prior imports are concentrated in:

- `src/time_causal_vae/cli/train_token_prior.py`
- `src/time_causal_vae/evaluation/token_prior.py`
- `src/time_causal_vae/token_prior/__init__.py`
- `src/time_causal_vae/token_prior/causal_transformer.py`

`README.md` currently documents `time_causal_vae.tokenization` and `time_causal_vae.token_prior`
as top-level package components. Public documentation should prefer the new
`time_causal_vae.models.discrete.*` namespaces after implementation, while mentioning that the old
imports remain available temporarily.

The current public notebooks do not directly import `time_causal_vae.tokenization` or
`time_causal_vae.token_prior`.

### Config And Metadata Audit

No current YAML config or package metadata audit result requires an import-path schema change.
Config names such as `objective`, `encoder`, `decoder`, `prior`, tokenizer settings, and token
prior settings should remain unchanged.

## Files To Move In The Implementation Prompt

Use `git mv` so history remains easy to inspect.

### Continuous Family

| Current path | New path |
| --- | --- |
| `src/time_causal_vae/models/base.py` | `src/time_causal_vae/models/continuous/base.py` |
| `src/time_causal_vae/models/config.py` | `src/time_causal_vae/models/continuous/config.py` |
| `src/time_causal_vae/models/factory.py` | `src/time_causal_vae/models/continuous/factory.py` |
| `src/time_causal_vae/models/transforms.py` | `src/time_causal_vae/models/continuous/transforms.py` |
| `src/time_causal_vae/models/losses.py` | `src/time_causal_vae/models/continuous/losses.py` |
| `src/time_causal_vae/models/distances.py` | `src/time_causal_vae/models/continuous/distances.py` |
| `src/time_causal_vae/models/initialisation.py` | `src/time_causal_vae/models/continuous/initialisation.py` |
| `src/time_causal_vae/models/conditioners/*.py` | `src/time_causal_vae/models/continuous/conditioners/*.py` |
| `src/time_causal_vae/models/encoders/*.py` | `src/time_causal_vae/models/continuous/encoders/*.py` |
| `src/time_causal_vae/models/decoders/*.py` | `src/time_causal_vae/models/continuous/decoders/*.py` |
| `src/time_causal_vae/models/priors/*.py` | `src/time_causal_vae/models/continuous/priors/*.py` |
| `src/time_causal_vae/models/objectives/*.py` | `src/time_causal_vae/models/continuous/objectives/*.py` |

Create `src/time_causal_vae/models/continuous/__init__.py` and package `__init__.py` files for
each moved subpackage.

### Discrete Family

| Current path | New path |
| --- | --- |
| `src/time_causal_vae/tokenization/causal_vq_tokenizer.py` | `src/time_causal_vae/models/discrete/tokenizers/causal_vq_tokenizer.py` |
| `src/time_causal_vae/tokenization/quantizers.py` | `src/time_causal_vae/models/discrete/tokenizers/quantizers.py` |
| `src/time_causal_vae/tokenization/config.py` | `src/time_causal_vae/models/discrete/tokenizers/config.py` |
| `src/time_causal_vae/token_prior/causal_transformer.py` | `src/time_causal_vae/models/discrete/priors/causal_transformer.py` |
| `src/time_causal_vae/token_prior/config.py` | `src/time_causal_vae/models/discrete/priors/config.py` |
| `src/time_causal_vae/token_prior/data.py` | `src/time_causal_vae/models/discrete/priors/data.py` |
| `src/time_causal_vae/token_prior/masks.py` | `src/time_causal_vae/models/discrete/priors/masks.py` |

Create:

- `src/time_causal_vae/models/discrete/__init__.py`
- `src/time_causal_vae/models/discrete/tokenizers/__init__.py`
- `src/time_causal_vae/models/discrete/priors/__init__.py`

Leave `src/time_causal_vae/models/layers/causal_conv.py` in place.

## Imports To Update Internally

After moving files, update internal imports to prefer the new namespaces.

### Continuous Import Targets

- Replace `time_causal_vae.models.factory` with `time_causal_vae.models.continuous.factory`.
- Replace `time_causal_vae.models.config` with `time_causal_vae.models.continuous.config`.
- Replace `time_causal_vae.models.base` with `time_causal_vae.models.continuous.base`.
- Replace `time_causal_vae.models.encoders.*` with
  `time_causal_vae.models.continuous.encoders.*`.
- Replace `time_causal_vae.models.decoders.*` with
  `time_causal_vae.models.continuous.decoders.*`.
- Replace `time_causal_vae.models.conditioners.*` with
  `time_causal_vae.models.continuous.conditioners.*`.
- Replace `time_causal_vae.models.priors.*` with `time_causal_vae.models.continuous.priors.*`.
- Replace `time_causal_vae.models.objectives.*` with
  `time_causal_vae.models.continuous.objectives.*`.
- Replace `time_causal_vae.models.losses`, `transforms`, `distances`, and `initialisation` with
  the matching `time_causal_vae.models.continuous.*` modules.

`src/time_causal_vae/models/__init__.py` should continue to expose `ModelFactory` and
`NetworkPipeline`, re-exported from `models.continuous.factory`.

### Discrete Import Targets

- Replace `time_causal_vae.tokenization` with `time_causal_vae.models.discrete.tokenizers`.
- Replace `time_causal_vae.tokenization.causal_vq_tokenizer` with
  `time_causal_vae.models.discrete.tokenizers.causal_vq_tokenizer`.
- Replace `time_causal_vae.tokenization.config` with
  `time_causal_vae.models.discrete.tokenizers.config`.
- Replace `time_causal_vae.tokenization.quantizers` with
  `time_causal_vae.models.discrete.tokenizers.quantizers`.
- Replace `time_causal_vae.token_prior` with `time_causal_vae.models.discrete.priors`.
- Replace `time_causal_vae.token_prior.causal_transformer` with
  `time_causal_vae.models.discrete.priors.causal_transformer`.
- Replace `time_causal_vae.token_prior.config` with
  `time_causal_vae.models.discrete.priors.config`.
- Replace `time_causal_vae.token_prior.data` with
  `time_causal_vae.models.discrete.priors.data`.
- Replace `time_causal_vae.token_prior.masks` with
  `time_causal_vae.models.discrete.priors.masks`.

Public CLI modules and notebooks should prefer these new imports after the implementation. Since
the notebooks do not currently import the audited model namespaces directly, the likely notebook
change is explanatory text only.

## Compatibility Shims To Keep

Keep the old import paths as small wrappers during the public transition. Wrappers should import
symbols from the new location and define `__all__` consistently. They should not implement logic.

### Continuous Wrappers

Retain wrappers for:

- `time_causal_vae.models.base`
- `time_causal_vae.models.config`
- `time_causal_vae.models.factory`
- `time_causal_vae.models.transforms`
- `time_causal_vae.models.losses`
- `time_causal_vae.models.distances`
- `time_causal_vae.models.initialisation`
- `time_causal_vae.models.conditioners`
- `time_causal_vae.models.conditioners.base`
- `time_causal_vae.models.conditioners.identity`
- `time_causal_vae.models.encoders`
- `time_causal_vae.models.encoders.base`
- `time_causal_vae.models.encoders.lstm`
- `time_causal_vae.models.encoders.mlp`
- `time_causal_vae.models.decoders`
- `time_causal_vae.models.decoders.base`
- `time_causal_vae.models.decoders.lstm`
- `time_causal_vae.models.decoders.mlp`
- `time_causal_vae.models.decoders.neural_sde`
- `time_causal_vae.models.priors`
- `time_causal_vae.models.priors.base`
- `time_causal_vae.models.priors.gaussian`
- `time_causal_vae.models.priors.realnvp`
- `time_causal_vae.models.objectives`
- `time_causal_vae.models.objectives.beta_cvae`
- `time_causal_vae.models.objectives.info_cvae`
- `time_causal_vae.models.objectives.vae`

### Discrete Wrappers

Retain wrappers for:

- `time_causal_vae.tokenization`
- `time_causal_vae.tokenization.causal_vq_tokenizer`
- `time_causal_vae.tokenization.config`
- `time_causal_vae.tokenization.quantizers`
- `time_causal_vae.token_prior`
- `time_causal_vae.token_prior.causal_transformer`
- `time_causal_vae.token_prior.config`
- `time_causal_vae.token_prior.data`
- `time_causal_vae.token_prior.masks`

These compatibility modules should be documented as transitional. Do not add deprecation warnings
in the first migration commit because warnings could make notebooks or tests noisy. A later public
release can add warnings if the compatibility period has an agreed end date.

## Behaviour And Schema Constraints

- Do not change model classes, method names, tensor shapes, sampling logic, no-leakage logic, loss
  functions, or evaluation metrics.
- Do not change checkpoint file names such as `model.pt`, `model_config.json`,
  `tokenizer.pt`, or `token_prior.pt`.
- Do not change checkpoint state-dict keys.
- Do not change YAML config schemas or registry schemas.
- Do not change public command-line flags.
- If future metadata contains import paths, update only those import-path metadata strings and keep
  old-path loading through compatibility wrappers.

## Risks

- Wrapper drift: wrapper `__all__` values can fall out of sync with the moved modules.
- Cyclic imports: moving `factory.py`, objective modules, and base classes together can expose
  cycles hidden by the current flat namespace.
- Checkpoint loading expectations: current continuous checkpoints store state dicts, but any
  external user who pickled whole modules could depend on old module paths.
- Type-checking surface: mypy may follow wrappers and moved modules differently, especially for
  modules that already use `# mypy: ignore-errors`.
- Documentation ambiguity: keeping old imports working while recommending new imports can confuse
  users unless README and notebooks are clear.
- Distance helpers: `models.distances` is used by evaluation code, including token-prior
  evaluation. Moving it under `models.continuous` matches the target layout but may read as less
  shared than its actual use.
- Local caches: existing `__pycache__` directories should stay ignored and must not be staged.

## Validation Commands

Run the planning-only checks before this document is committed:

```bash
poetry check
poetry run ruff format docs
poetry run ruff check docs --fix
```

For the future implementation commit, run:

```bash
poetry check
poetry run ruff check src scripts configs docs --fix
poetry run ruff format --preview src scripts configs README.md CONTRIBUTING.md trained_models docs
poetry run mypy src/time_causal_vae
poetry run python -c "from time_causal_vae.models import ModelFactory; from time_causal_vae.models.encoders import MLPEncoder; from time_causal_vae.models.continuous.encoders import MLPEncoder as NewMLPEncoder; assert MLPEncoder is NewMLPEncoder"
poetry run python -c "from time_causal_vae.tokenization import CausalVQTokenizer; from time_causal_vae.models.discrete.tokenizers import CausalVQTokenizer as NewTokenizer; assert CausalVQTokenizer is NewTokenizer"
poetry run python -c "from time_causal_vae.token_prior import CausalTokenPriorConfig; from time_causal_vae.models.discrete.priors import CausalTokenPriorConfig as NewConfig; assert CausalTokenPriorConfig is NewConfig"
poetry run python scripts/select_registered_model.py --experiment sp500_vix --family discrete
poetry run python scripts/select_registered_model.py --experiment black_scholes --family continuous
```

Recommended smoke checks for the future implementation commit:

```bash
poetry run tcvae-train --config configs/experiments/sp500_vix_beta_cvae.yaml --output-dir outputs/sp500_vix_continuous_namespace_smoke --epochs 1 --no-wandb --dry-run
poetry run python scripts/check_causal_conv_no_leakage.py
poetry run python scripts/check_conditional_vq_tokenizer_no_leakage.py
poetry run python scripts/check_conditional_token_prior_no_leakage.py
```

The smoke training command must keep `--dry-run` unless a later prompt explicitly requests
training.

## Rollback Plan

Because the implementation should use `git mv` plus wrappers, rollback should be straightforward:

1. Revert the implementation commit with `git revert <commit>`.
2. If partial changes are present before commit, restore the working tree with targeted
   `git restore` commands for moved paths and wrappers.
3. Rerun the old-path import checks for `models.encoders`, `models.decoders`, `models.priors`,
   `models.objectives`, `tokenization`, and `token_prior`.
4. Rerun `poetry check`, Ruff, mypy, and the registry selector commands.
5. Confirm that no generated artefacts or cache files were staged.

Do not merge the namespace refactor into `main` until compatibility imports, public notebooks,
CLI commands, and registry selector checks all pass on the public branch.
