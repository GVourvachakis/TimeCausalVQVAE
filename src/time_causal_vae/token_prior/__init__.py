"""Compatibility aliases for the discrete token-prior namespace."""

from __future__ import annotations

import importlib
import sys

_CANONICAL_PACKAGE = "time_causal_vae.models.discrete.priors"
_COMPAT_ALIASES = {
    "causal_transformer": f"{_CANONICAL_PACKAGE}.causal_transformer",
    "config": f"{_CANONICAL_PACKAGE}.config",
    "data": f"{_CANONICAL_PACKAGE}.data",
    "masks": f"{_CANONICAL_PACKAGE}.masks",
}

_canonical_package = importlib.import_module(_CANONICAL_PACKAGE)
__all__ = list(getattr(_canonical_package, "__all__", ()))

for _name in __all__:
    globals()[_name] = getattr(_canonical_package, _name)

for _old_suffix, _new_name in _COMPAT_ALIASES.items():
    _module = importlib.import_module(_new_name)
    sys.modules.setdefault(f"{__name__}.{_old_suffix}", _module)
    setattr(sys.modules[__name__], _old_suffix, _module)

del _canonical_package, _module, _name, _new_name, _old_suffix
