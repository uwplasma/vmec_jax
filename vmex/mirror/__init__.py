"""Public API for open and closed magnetic-mirror equilibria.

Open mirrors use a nonperiodic straight axis; stellarator-mirror hybrids use a
periodic B-spline axis. Its public surface is intentionally small: inputs and
state, solves, continuation, implicit derivatives, MOUT I/O, and plotting.
Numerical kernels remain available from their owning submodules.
"""

from importlib import import_module as _import_module


# Public names are lazy so importing ``vmex.mirror`` does not initialize
# every exterior-vacuum and solver dependency.
_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    # Model and configuration.
    "MirrorBoundary": (".model", "MirrorBoundary"),
    "MirrorConfig": (".model", "MirrorConfig"),
    "MirrorResolution": (".model", "MirrorResolution"),
    "MirrorState": (".model", "MirrorState"),
    # Coefficient-native fixed- and free-boundary solves.
    "SplineMirrorBoundary": (".splines", "SplineMirrorBoundary"),
    "SplineMirrorDiscretization": (".splines", "SplineMirrorDiscretization"),
    "SplineMirrorState": (".splines", "SplineMirrorState"),
    "build_stellarator_mirror_hybrid": (
        ".splines",
        "build_stellarator_mirror_hybrid",
    ),
    "trace_closed_field_line": (".splines", "trace_closed_field_line"),
    "solve_fixed_boundary": (".splines", "solve_fixed_boundary"),
    "solve_fixed_boundary_from_radius": (
        ".splines",
        "solve_fixed_boundary_from_radius",
    ),
    "solve_free_boundary": (".free_boundary", "solve_free_boundary"),
    # Continuation.
    "solve_beta_scan": (".free_boundary", "solve_beta_scan"),
    # Implicit differentiation.
    "spline_fixed_boundary_adjoint": (
        ".implicit",
        "spline_fixed_boundary_adjoint",
    ),
    "spline_fixed_boundary_tangent": (
        ".implicit",
        "spline_fixed_boundary_tangent",
    ),
    "free_boundary_adjoint": (
        ".implicit",
        "free_boundary_adjoint",
    ),
    # MOUT and plots.
    "mout_from_result": (".output", "mout_from_result"),
    "read_mout": (".output", "read_mout"),
    "write_mout": (".output", "write_mout"),
    "plot_mout": (".output", "plot_mout"),
    "plot_stellarator_mirror_hybrid": (
        ".output",
        "plot_stellarator_mirror_hybrid",
    ),
    # Free-boundary restart I/O and beta-scan summary.
    "FreeBoundaryRestart": (".output", "FreeBoundaryRestart"),
    "save_free_boundary_restart": (".output", "save_free_boundary_restart"),
    "load_free_boundary_restart": (".output", "load_free_boundary_restart"),
    "summarize_axisymmetric_beta_scan": (
        ".output",
        "summarize_axisymmetric_beta_scan",
    ),
}

__all__ = sorted(_LAZY_ATTRS)


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
