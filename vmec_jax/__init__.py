"""Compatibility shim: ``vmec_jax`` was renamed to :mod:`vmex`.

Importing ``vmec_jax`` re-exports everything from :mod:`vmex` and emits a
``DeprecationWarning``.  Submodule access (``vmec_jax.core.solver`` etc.) is
forwarded by aliasing the already-imported ``vmex`` submodules into
``sys.modules`` under the old names.  This shim is a one-release courtesy;
update imports to ``vmex``.
"""
from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "the 'vmec_jax' package has been renamed to 'vmex'; import 'vmex' instead "
    "(this compatibility shim will be removed in a future release)",
    DeprecationWarning,
    stacklevel=2,
)

import vmex as _vmex

# Re-export the public API.
from vmex import *  # noqa: F401,F403
__version__ = _vmex.__version__
if hasattr(_vmex, "__all__"):
    __all__ = list(_vmex.__all__)

# Forward submodule access: alias every imported vmex submodule under the old
# top-level name so `import vmec_jax.core.X` and `from vmec_jax.core import X`
# resolve to the identical vmex module object.
for _name, _mod in list(sys.modules.items()):
    if _name == "vmex" or _name.startswith("vmex."):
        sys.modules["vmec_jax" + _name[len("vmex"):]] = _mod


def __getattr__(name: str):
    """Lazily import and forward any vmex submodule not yet loaded."""
    try:
        mod = importlib.import_module(f"vmex.{name}")
    except ModuleNotFoundError as exc:  # pragma: no cover - mirror vmex
        raise AttributeError(name) from exc
    sys.modules[f"vmec_jax.{name}"] = mod
    return mod
