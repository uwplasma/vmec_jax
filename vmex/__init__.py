"""vmex: a JAX implementation of VMEC2000 for fixed and free-boundary equilibria.

Public API (lazily imported; ``import vmex as vj``):

- :class:`~vmex.core.input.VmecInput` — INDATA / VMEC++-JSON input pytree
- :func:`~vmex.core.solver.solve` — single-grid fixed-boundary solve
- :func:`~vmex.core.multigrid.solve_multigrid` — NS_ARRAY ladder (runvmec.f)
- :func:`~vmex.core.freeboundary.solve_free_boundary` — NESTOR free boundary
- :func:`~vmex.core.wout.read_wout` / :func:`~vmex.core.wout.write_wout`
  / :func:`~vmex.core.wout.wout_from_state` / :class:`~vmex.core.wout.WoutData`
- :func:`~vmex.core.plotting.plot_wout` / :func:`~vmex.core.plotting.plot_boozmn`
- :func:`~vmex.core.boozer.run_booz_xform` — Boozer transform (booz_xform_jax)
- :func:`~vmex.core.mgrid.read_mgrid` / :func:`~vmex.core.mgrid.write_mgrid`
  / :class:`~vmex.core.mgrid.MgridField` (external field is an mgrid or any
  ``xyz->B`` callable; coils live in ESSOS, ``essos.coils.Coils``)
- ``vmex.optimize`` — objectives + least-squares driver (module)
- ``vmex.implicit`` — implicit differentiation of the equilibrium (module)
- ``vmex.parallel`` — concurrent ensembles of independent solves (module)
- ``vmex.errors`` — typed zero-crash exceptions (also exported directly)

The ``vmec`` console entry point lives in :mod:`vmex.core.cli`.
"""

from importlib import import_module as _import_module
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _package_version
import os as _os
from pathlib import Path as _Path

from ._compat import _default_compilation_cache_dir as _default_jax_cache_dir


def _source_tree_version() -> str | None:
    pyproject = _Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.exists():
        return None
    in_project = False
    for raw_line in pyproject.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "[project]":
            in_project = True
            continue
        if in_project and line.startswith("["):
            return None
        if in_project and line.startswith("version"):
            return line.split("=", 1)[1].strip().strip('"')
    return None


try:
    __version__ = _source_tree_version() or _package_version("vmex")
except _PackageNotFoundError:  # pragma: no cover - source tree without installed metadata.
    __version__ = "0+unknown"

# Suppress noisy C++ warnings from XLA/PjRt backend (e.g. repeated
# "Assume version compatibility. PjRt-IFRT does not track XLA executable
# versions." on persistent-cache hits). Must be set before *any* ``import
# jax`` in the process. Uses setdefault so the user can still override via the
# environment.
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
_os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
_os.environ.setdefault("GLOG_minloglevel", "2")

# Enable the JAX persistent XLA compilation cache in a machine-scoped
# directory when requested by the backend/env policy in _compat. Accelerator
# runs use the cache by default; CPU runs are opt-in to avoid XLA:CPU AOT
# feature-mismatch warnings on shared or changing runtime environments.
# ``core.solver._harden_compilation_cache`` re-applies this policy on every
# solve path in case this module never ran (namespace-package shadowing).
import jax as _jax

_jax_cache_dir = _default_jax_cache_dir()
if _jax_cache_dir is not None:
    _os.makedirs(_jax_cache_dir, exist_ok=True)
    _jax.config.update("jax_enable_compilation_cache", True)
    _jax.config.update("jax_compilation_cache_dir", _jax_cache_dir)

# Lazy public exports: name -> (module, attribute).  ``attribute=None``
# exports the module itself.
_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    # input
    "VmecInput": (".core.input", "VmecInput"),
    # solvers
    "solve": (".core.solver", "solve"),
    "solve_multigrid": (".core.multigrid", "solve_multigrid"),
    "solve_free_boundary": (".core.freeboundary", "solve_free_boundary"),
    # wout IO
    "WoutData": (".core.wout", "WoutData"),
    "read_wout": (".core.wout", "read_wout"),
    "write_wout": (".core.wout", "write_wout"),
    "wout_from_state": (".core.wout", "wout_from_state"),
    # plotting + Boozer
    "plot_wout": (".core.plotting", "plot_wout"),
    "plot_boozmn": (".core.plotting", "plot_boozmn"),
    "run_booz_xform": (".core.boozer", "run_booz_xform"),
    # external fields
    "MgridData": (".core.mgrid", "MgridData"),
    "MgridField": (".core.mgrid", "MgridField"),
    "read_mgrid": (".core.mgrid", "read_mgrid"),
    "write_mgrid": (".core.mgrid", "write_mgrid"),
    # errors
    "VmecError": (".core.errors", "VmecError"),
    "VmecInputError": (".core.errors", "VmecInputError"),
    "VmecJacobianError": (".core.errors", "VmecJacobianError"),
    "VmecConvergenceError": (".core.errors", "VmecConvergenceError"),
    "MgridNotFoundError": (".core.errors", "MgridNotFoundError"),
    # modules
    "core": (".core", None),
    "errors": (".core.errors", None),
    "optimize": (".core.optimize", None),
    "implicit": (".core.implicit", None),
    "parallel": (".core.parallel", None),
    "doctor": (".doctor", None),
}

__all__ = ["__version__", *sorted(_LAZY_ATTRS)]


def __getattr__(name: str):
    entry = _LAZY_ATTRS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = entry
    module = _import_module(module_name, __name__)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
