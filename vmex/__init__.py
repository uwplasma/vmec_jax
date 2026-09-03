"""vmex: a JAX implementation of VMEC2000 for fixed and free-boundary equilibria.

Every name below is a lazy attribute of the top-level package
(``import vmex as vj``): the owning module is imported on first access, and
``vmex.__all__`` lists exactly these names plus ``__version__``.  Each entry
links to the module that documents it.

**Inputs and run controls**

- :class:`~vmex.core.input.VmecInput` — INDATA / VMEC++-JSON input pytree
- :class:`~vmex.core.run_options.RunOptions` /
  :class:`~vmex.core.run_options.InputRequest` /
  :func:`~vmex.core.run_options.read_input_request` — ``!@VMEX`` directives
  and the JSON ``_vmex`` section (execution metadata, never physics)

**Solvers**

- :func:`~vmex.core.solver.solve` — single-grid fixed-boundary solve
- :func:`~vmex.core.multigrid.solve_multigrid` — NS_ARRAY ladder (runvmec.f)
- :func:`~vmex.core.multigrid.solve_file` — run a deck the way the CLI does
  (directives honored, ``wout_<case>.nc`` written)
- :func:`~vmex.core.freeboundary.solve_free_boundary` — NESTOR free boundary
- :func:`~vmex.core.multigrid.solve_free_boundary_multigrid` — free-boundary ladder
- :func:`~vmex.core.freeboundary_implicit.make_free_boundary_config` /
  :func:`~vmex.core.freeboundary_implicit.solve_free_boundary_implicit` /
  :func:`~vmex.core.freeboundary_implicit.solve_free_boundary_implicit_status`
  — coupled NESTOR/VMEC implicit derivative
- :func:`~vmex.core.restart.state_from_wout` /
  :func:`~vmex.core.restart.restart_state` — hot restart from any wout
  (also ``solve*(..., restart_from=...)``)

**Outputs and scaling**

- :class:`~vmex.core.wout.WoutData` / :func:`~vmex.core.wout.read_wout` /
  :func:`~vmex.core.wout.write_wout` / :func:`~vmex.core.wout.wout_from_state`
- :func:`~vmex.core.turbulence.gk_fieldline_geometry_from_wout` — GK
  field-line geometry from any compatible wout, without a solve
- :func:`~vmex.core.scaling.scale_input` / :func:`~vmex.core.scaling.scale_wout`
  / :func:`~vmex.core.scaling.scale_mgrid` — dimensional similarity transforms

**High-order reconstruction and strong-force certificate**

- :class:`~vmex.core.strong_force.HighOrderEquilibriumState` /
  :func:`~vmex.core.strong_force.high_order_state_from_wout` /
  :func:`~vmex.core.strong_force.lift_high_order_state` — axis-regular
  continuous reconstruction
- :func:`~vmex.core.strong_force.evaluate_high_order_fields` →
  :class:`~vmex.core.strong_force.HighOrderFieldSamples`;
  :func:`~vmex.core.strong_force.evaluate_high_order_surface` →
  :class:`~vmex.core.strong_force.HighOrderSurfaceSamples`
- :func:`~vmex.core.strong_force.evaluate_strong_force` →
  :class:`~vmex.core.strong_force.StrongForceSamples`;
  :func:`~vmex.core.strong_force.certify_strong_force` →
  :class:`~vmex.core.strong_force.StrongForceReport` /
  :func:`~vmex.core.strong_force.plot_strong_force_report` — independent
  strong-force certificate
- :func:`~vmex.core.omnigenity.boozer_spectrum_state` /
  :func:`~vmex.core.omnigenity.boozer_spectrum_high_order` — Boozer ``|B|``
  spectrum without a sampled radial mesh
  (:func:`~vmex.core.omnigenity.boozer_bmnc_high_order` is a deprecated
  alias that warns on call)

**Force-balance polishing**

- :class:`~vmex.core.polish_driver.PolishConfig` /
  :class:`~vmex.core.polish_driver.PolishContext` /
  :class:`~vmex.core.polish_driver.PolishResult` /
  :class:`~vmex.core.polish_driver.PolishReport` — strong-root correction
  (``solve*(..., polish=...)``)
- :class:`~vmex.core.polish_implicit.PolishLinearConfig` /
  :func:`~vmex.core.polish_implicit.collocation_polish_tangent` /
  :func:`~vmex.core.polish_implicit.collocation_polish_adjoint` /
  :func:`~vmex.core.polish_implicit.implicit_collocation_polished_state`
  — derivatives through a polished root

**Optimization**

- ``vmex.optimize`` — objectives + least-squares driver (module)
- ``vmex.implicit`` — implicit differentiation of the equilibrium (module)
- ``vmex.parallel`` — concurrent ensembles of independent solves (module)
- :class:`~vmex.core.problem.VmecProblem` /
  :class:`~vmex.core.problem.FunctionProblem` /
  :class:`~vmex.core.problem.Evaluation` — optimizer-neutral value,
  residual, and derivative callables
- :class:`~vmex.core.monitoring.OptimizationMonitor` /
  :class:`~vmex.core.monitoring.OptimizationRecord` — accepted iterations;
  :class:`~vmex.core.monitoring.EquilibriumReporter` — compact diagnostics

**Post-processing and plotting**

- :func:`~vmex.core.boozer.run_booz_xform` — Boozer transform (booz_xform_jax)
- :func:`~vmex.core.neoclassical.epsilon_effective_from_wout` /
  :func:`~vmex.core.neoclassical.epsilon_effective_from_boozer` — optional
  NEO_JAX effective-ripple profile
- :func:`~vmex.core.gammac.gamma_c_from_wout` — fast-ion ``Gamma_c`` profile
  from any compatible wout, without a solve
- :func:`~vmex.core.tracing.essos_vmec_field` — hand a solved equilibrium to
  ESSOS as an ``essos.fields.Vmec`` (optional ESSOS dependency)
- :func:`~vmex.core.tracing.trace_alphas` →
  :class:`~vmex.core.tracing.AlphaTracingResult` /
  :func:`~vmex.core.plotting.plot_tracing` — optional ESSOS alpha-particle
  tracing (exact loss fraction; also ``vmex --trace``)
- :func:`~vmex.core.plotting.plot_wout` / :func:`~vmex.core.plotting.plot_boozmn`
  / :func:`~vmex.core.plotting.plot_bootstrap_current`
  / :func:`~vmex.core.plotting.plot_optimization_movie`
  / :func:`~vmex.core.plotting.plot_optimization_objects` — wout, boozmn,
  bootstrap, optimization-history, and surfaces-and-coils plots

**External fields**

- :class:`~vmex.core.mgrid.MgridData` / :class:`~vmex.core.mgrid.MgridField`
  / :func:`~vmex.core.mgrid.read_mgrid` / :func:`~vmex.core.mgrid.write_mgrid`
  / :func:`~vmex.core.mgrid.tabulate_cartesian_field` — mgrid or tabulated
  direct field (``MgridField.from_coils`` tabulates an ESSOS coil set)
- :class:`~vmex.core.extender.MagneticField` — base JAX field with explicit
  and stored-point evaluation;
  :class:`~vmex.core.extender.VmecInteriorField` — field inside the plasma;
  :class:`~vmex.core.extender.VmecExtender` — field outside the plasma surface
- :class:`~vmex.core.virtual_casing.PlasmaVacuumInterface` /
  :func:`~vmex.core.virtual_casing.surface_field_data_from_state` /
  :func:`~vmex.core.virtual_casing.surface_field_data_from_high_order` /
  :func:`~vmex.core.virtual_casing.surface_field_data_from_wout` —
  virtual-casing diagnostics on a prescribed plasma-vacuum interface

**Errors, diagnostics, and modules**

- ``vmex.errors`` — typed zero-crash exceptions (module); also exported
  directly: :class:`~vmex.core.errors.VmecError`,
  :class:`~vmex.core.errors.VmecInputError`,
  :class:`~vmex.core.errors.VmecJacobianError`,
  :class:`~vmex.core.errors.VmecConvergenceError`,
  :class:`~vmex.core.errors.VmecNumericalError`,
  :class:`~vmex.core.errors.MgridNotFoundError`,
  :class:`~vmex.core.errors.StrongForceContinuationError`,
  :class:`~vmex.core.errors.StrongForceCertificationError`,
  :class:`~vmex.core.errors.StrongForceLinearSolveError`
- ``vmex.doctor`` — installation diagnostics behind ``vmex --doctor`` (module)
- ``vmex.core`` — the solver internals (module)

The ``vmex`` console entry point (``vmec`` is an alias) lives in
:mod:`vmex.core.cli`.
"""

from importlib import import_module as _import_module
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _package_version
import os as _os
from pathlib import Path as _Path
import warnings as _warnings

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

# Suppress noisy XLA/PjRt C++ logs (see _compat._configure_jax_environment).
# Must be set before *any* ``import jax`` in the process; setdefault keeps
# user overrides working.
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
_os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
_os.environ.setdefault("GLOG_minloglevel", "2")

# Enable the JAX persistent XLA compilation cache in a machine-scoped
# directory per the _compat policy (see _default_compilation_cache_dir).
# ``core.solver._harden_compilation_cache`` re-applies this policy on every
# solve path in case this module never ran (namespace-package shadowing).
import jax as _jax


def _configure_jax_logging(jax_module) -> None:
    """Quiet JAX by default, with explicit overrides and an old-JAX notice."""
    if not hasattr(jax_module.config, "jax_logging_level"):
        _warnings.warn(
            f"JAX {getattr(jax_module, '__version__', 'unknown')} does not "
            "provide jax_logging_level (available since JAX 0.4.36). VMEX "
            "will use environment-level log suppression, but repeated "
            "XLA/PjRt warnings may still appear. Upgrade JAX to silence them "
            "reliably.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    level = _os.environ.get("VMEX_JAX_LOGGING_LEVEL")
    if level is None:
        level = _os.environ.get("JAX_LOGGING_LEVEL", "ERROR")
    level = level.strip().upper()
    if level not in ("", "INHERIT"):
        jax_module.config.update("jax_logging_level", level)


_configure_jax_logging(_jax)

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
    "solve_free_boundary_multigrid": (
        ".core.multigrid", "solve_free_boundary_multigrid"),
    "solve_free_boundary": (".core.freeboundary", "solve_free_boundary"),
    "make_free_boundary_config": (
        ".core.freeboundary_implicit", "make_free_boundary_config"),
    "solve_free_boundary_implicit": (
        ".core.freeboundary_implicit", "solve_free_boundary_implicit"),
    "solve_free_boundary_implicit_status": (
        ".core.freeboundary_implicit", "solve_free_boundary_implicit_status"),
    # wout IO
    "WoutData": (".core.wout", "WoutData"),
    "read_wout": (".core.wout", "read_wout"),
    "write_wout": (".core.wout", "write_wout"),
    "wout_from_state": (".core.wout", "wout_from_state"),
    "gk_fieldline_geometry_from_wout": (
        ".core.turbulence", "gk_fieldline_geometry_from_wout"),
    # hot restart
    "restart_state": (".core.restart", "restart_state"),
    "state_from_wout": (".core.restart", "state_from_wout"),
    # high-order reconstruction and independent strong-force certificate
    "HighOrderEquilibriumState": (
        ".core.strong_force", "HighOrderEquilibriumState"),
    "HighOrderFieldSamples": (".core.strong_force", "HighOrderFieldSamples"),
    "HighOrderSurfaceSamples": (
        ".core.strong_force", "HighOrderSurfaceSamples"),
    "ForceErrorNormalizations": (
        ".core.strong_force", "ForceErrorNormalizations"),
    "StrongForceReport": (".core.strong_force", "StrongForceReport"),
    "StrongForceSamples": (".core.strong_force", "StrongForceSamples"),
    "certify_strong_force": (".core.strong_force", "certify_strong_force"),
    "evaluate_high_order_fields": (
        ".core.strong_force", "evaluate_high_order_fields"),
    "evaluate_high_order_surface": (
        ".core.strong_force", "evaluate_high_order_surface"),
    "boozer_spectrum_high_order": (
        ".core.omnigenity", "boozer_spectrum_high_order"),
    "boozer_spectrum_state": (
        ".core.omnigenity", "boozer_spectrum_state"),
    # deprecated alias (warns on call, dispatches to the canonical name)
    "boozer_bmnc_high_order": (
        ".core.omnigenity", "boozer_bmnc_high_order"),
    "evaluate_strong_force": (".core.strong_force", "evaluate_strong_force"),
    "high_order_state_from_wout": (
        ".core.strong_force", "high_order_state_from_wout"),
    "lift_high_order_state": (".core.strong_force", "lift_high_order_state"),
    "plot_strong_force_report": (
        ".core.strong_force", "plot_strong_force_report"),
    # plotting + Boozer
    "plot_wout": (".core.plotting", "plot_wout"),
    "plot_boozmn": (".core.plotting", "plot_boozmn"),
    "plot_bootstrap_current": (".core.plotting", "plot_bootstrap_current"),
    "plot_optimization_movie": (".core.plotting", "plot_optimization_movie"),
    "plot_optimization_objects": (".core.plotting", "plot_optimization_objects"),
    "run_booz_xform": (".core.boozer", "run_booz_xform"),
    "epsilon_effective_from_boozer": (
        ".core.neoclassical", "epsilon_effective_from_boozer"),
    "epsilon_effective_from_wout": (
        ".core.neoclassical", "epsilon_effective_from_wout"),
    "gamma_c_from_wout": (".core.gammac", "gamma_c_from_wout"),
    # alpha-particle tracing (ESSOS)
    "AlphaTracingResult": (".core.tracing", "AlphaTracingResult"),
    "essos_vmec_field": (".core.tracing", "essos_vmec_field"),
    "trace_alphas": (".core.tracing", "trace_alphas"),
    "plot_tracing": (".core.plotting", "plot_tracing"),
    # optimizer-neutral problem callables
    "Evaluation": (".core.problem", "Evaluation"),
    "FunctionProblem": (".core.problem", "FunctionProblem"),
    "VmecProblem": (".core.problem", "VmecProblem"),
    "EquilibriumReporter": (".core.monitoring", "EquilibriumReporter"),
    "OptimizationMonitor": (".core.monitoring", "OptimizationMonitor"),
    "OptimizationRecord": (".core.monitoring", "OptimizationRecord"),
    # high-order strong-force polishing
    "PolishConfig": (".core.polish_driver", "PolishConfig"),
    "PolishContext": (".core.polish_driver", "PolishContext"),
    "PolishReport": (".core.polish_driver", "PolishReport"),
    "PolishResult": (".core.polish_driver", "PolishResult"),
    "PolishLinearConfig": (".core.polish_implicit", "PolishLinearConfig"),
    "InputRequest": (".core.run_options", "InputRequest"),
    "RunOptions": (".core.run_options", "RunOptions"),
    "read_input_request": (".core.run_options", "read_input_request"),
    "solve_file": (".core.multigrid", "solve_file"),
    "collocation_polish_adjoint": (
        ".core.polish_implicit", "collocation_polish_adjoint"),
    "collocation_polish_tangent": (
        ".core.polish_implicit", "collocation_polish_tangent"),
    "implicit_collocation_polished_state": (
        ".core.polish_implicit", "implicit_collocation_polished_state"),
    # external fields
    "MgridData": (".core.mgrid", "MgridData"),
    "MgridField": (".core.mgrid", "MgridField"),
    "read_mgrid": (".core.mgrid", "read_mgrid"),
    "tabulate_cartesian_field": (".core.mgrid", "tabulate_cartesian_field"),
    "write_mgrid": (".core.mgrid", "write_mgrid"),
    "MagneticField": (".core.extender", "MagneticField"),
    "VmecInteriorField": (".core.extender", "VmecInteriorField"),
    "VmecExtender": (".core.extender", "VmecExtender"),
    "PlasmaVacuumInterface": (
        ".core.virtual_casing", "PlasmaVacuumInterface"),
    "surface_field_data_from_state": (
        ".core.virtual_casing", "surface_field_data_from_state"),
    "surface_field_data_from_high_order": (
        ".core.virtual_casing", "surface_field_data_from_high_order"),
    "surface_field_data_from_wout": (
        ".core.virtual_casing", "surface_field_data_from_wout"),
    # dimensional scaling
    "scale_input": (".core.scaling", "scale_input"),
    "scale_mgrid": (".core.scaling", "scale_mgrid"),
    "scale_wout": (".core.scaling", "scale_wout"),
    # errors
    "VmecError": (".core.errors", "VmecError"),
    "VmecInputError": (".core.errors", "VmecInputError"),
    "VmecJacobianError": (".core.errors", "VmecJacobianError"),
    "VmecConvergenceError": (".core.errors", "VmecConvergenceError"),
    "VmecNumericalError": (".core.errors", "VmecNumericalError"),
    "StrongForceContinuationError": (
        ".core.errors", "StrongForceContinuationError"),
    "StrongForceCertificationError": (
        ".core.errors", "StrongForceCertificationError"),
    "StrongForceLinearSolveError": (
        ".core.errors", "StrongForceLinearSolveError"),
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
