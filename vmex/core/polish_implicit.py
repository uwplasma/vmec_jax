"""Implicit derivatives of certified collocation least-squares polishing.

The nonlinear solve is deliberately not differentiated. Once its correction
is stationary, this module applies the implicit-function theorem to
``J(c, native).T @ r(c, native) = 0`` with matrix-free JVPs/VJPs. A gradient
therefore costs one Krylov solve rather than a replay of Gauss--Newton steps.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from solvax import gmres

from .errors import StrongForceLinearSolveError
from .polish import (
    HighOrderCorrection,
    StrongPhysicalChart,
    StrongRootRuntime,
    strong_collocation_residual_at_native,
)
from .polish_driver import PolishContext
from .strong_force import HighOrderEquilibriumState


@dataclass(frozen=True)
class PolishLinearConfig:
    """Krylov controls and failure policy for polished-root derivatives."""

    rtol: float = 1.0e-8
    atol: float = 1.0e-11
    restart: int = 30
    max_restarts: int = 30
    fail_policy: Literal["raise", "nan"] = "raise"

    def __post_init__(self) -> None:
        if not np.isfinite(self.rtol) or self.rtol <= 0.0:
            raise ValueError("rtol must be finite and positive")
        if not np.isfinite(self.atol) or self.atol < 0.0:
            raise ValueError("atol must be finite and non-negative")
        if self.restart < 1:
            raise ValueError("restart must be positive")
        if self.max_restarts < 1:
            raise ValueError("max_restarts must be positive")
        if self.fail_policy not in ("raise", "nan"):
            raise ValueError("fail_policy must be 'raise' or 'nan'")


class PolishLinearReport(NamedTuple):
    """True residual certificate for one tangent or adjoint solve."""

    residual_norm: jax.Array
    tolerance: jax.Array
    iterations: jax.Array
    converged: jax.Array


class PolishTangentResult(NamedTuple):
    """Total polished-state tangent and its reduced correction response."""

    native_tangent: HighOrderEquilibriumState
    correction_tangent: jax.Array
    report: PolishLinearReport


class PolishAdjointResult(NamedTuple):
    """Native-state cotangent and the strong-equation adjoint variable."""

    native_cotangent: HighOrderEquilibriumState
    equation_adjoint: jax.Array
    report: PolishLinearReport


def _tree_norm(value) -> jax.Array:
    return jnp.sqrt(
        sum(
            (jnp.vdot(leaf, leaf).real for leaf in jax.tree.leaves(value)),
            jnp.asarray(0.0),
        )
    )


def _add_correction(
    native: HighOrderEquilibriumState,
    correction: HighOrderCorrection,
) -> HighOrderEquilibriumState:
    """Add correction data while preserving the native PyTree metadata."""

    return replace(
        native,
        R_cos=native.R_cos + correction.R_cos,
        R_sin=native.R_sin + correction.R_sin,
        Z_cos=native.Z_cos + correction.Z_cos,
        Z_sin=native.Z_sin + correction.Z_sin,
        L_cos=native.L_cos + correction.L_cos,
        L_sin=native.L_sin + correction.L_sin,
    )


def _linear_report(operator, rhs, solution, config: PolishLinearConfig):
    residual_norm = jnp.linalg.norm(rhs - operator(solution.x))
    tolerance = jnp.maximum(config.atol, config.rtol * jnp.linalg.norm(rhs))
    converged = jnp.logical_or(solution.converged, residual_norm <= tolerance)
    return PolishLinearReport(
        residual_norm=residual_norm,
        tolerance=tolerance,
        iterations=solution.iterations,
        converged=converged,
    )


def _checked_solution(
    value: jax.Array,
    report: PolishLinearReport,
    config: PolishLinearConfig,
    solve_kind: str,
) -> jax.Array:
    traced = any(
        isinstance(item, jax.core.Tracer)
        for item in (value, report.residual_norm, report.tolerance, report.converged)
    )
    if not traced and not bool(np.asarray(report.converged)):
        if config.fail_policy == "raise":
            raise StrongForceLinearSolveError(
                message=(
                    f"strong-root {solve_kind} solve did not converge: residual "
                    f"{float(report.residual_norm):.3e} > tolerance "
                    f"{float(report.tolerance):.3e} after "
                    f"{int(report.iterations)} Krylov iterations"
                ),
                hint=(
                    "increase max_restarts/restart, loosen the derivative "
                    "tolerance, or refresh the polish preconditioner"
                ),
                solve_kind=solve_kind,
                iterations=int(report.iterations),
                residual_norm=float(report.residual_norm),
                tolerance=float(report.tolerance),
            )
        return jnp.full_like(value, jnp.nan)
    return jnp.where(report.converged, value, jnp.full_like(value, jnp.nan))


def _solve_linear(operator, rhs, preconditioner, config, solve_kind):
    size = int(rhs.shape[0])
    solution = gmres(
        operator,
        rhs,
        precond=preconditioner,
        restart=min(config.restart, size),
        rtol=config.rtol,
        atol=config.atol,
        max_restarts=config.max_restarts,
    )
    report = _linear_report(operator, rhs, solution, config)
    return _checked_solution(solution.x, report, config, solve_kind), report


def _collocation_corrected_state(
    native: HighOrderEquilibriumState,
    correction: jax.Array,
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart,
) -> HighOrderEquilibriumState:
    full = chart.lift(correction)
    high = runtime.layout.unpack(jnp.asarray(runtime.coordinate_scale) * full)
    return _add_correction(native, high)


def _collocation_stationarity(
    correction: jax.Array,
    native: HighOrderEquilibriumState,
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart,
) -> jax.Array:
    """Return the exact gradient of one-half the collocation residual norm."""

    from solvax import least_squares_stationarity

    return least_squares_stationarity(
        lambda value: strong_collocation_residual_at_native(
            value, native, runtime, chart
        ),
        correction,
    )


def collocation_polish_tangent(
    context: PolishContext,
    native_tangent: HighOrderEquilibriumState,
    *,
    config: PolishLinearConfig = PolishLinearConfig(),
) -> PolishTangentResult:
    """Apply the IFT tangent of the rectangular polish stationarity equation."""

    runtime, chart = context.runtime, context.chart
    correction = jnp.asarray(context.correction)
    if correction.shape != (chart.size,):
        raise ValueError(
            f"correction has shape {correction.shape}; expected {(chart.size,)}"
        )
    if jax.tree.structure(native_tangent) != jax.tree.structure(runtime.native):
        raise ValueError("native_tangent must have the runtime native-state structure")
    stationarity = lambda value: _collocation_stationarity(  # noqa: E731
        value, runtime.native, runtime, chart
    )
    _, operator = jax.linearize(stationarity, correction)
    _, parameter_direction = jax.jvp(
        lambda native: _collocation_stationarity(
            correction, native, runtime, chart
        ),
        (runtime.native,),
        (native_tangent,),
    )
    diagonal_inverse = jnp.asarray(context.variable_scale) ** 2
    response, report = _solve_linear(
        operator,
        -parameter_direction,
        lambda rhs: diagonal_inverse * rhs,
        config,
        "least-squares tangent",
    )
    _, correction_tangent = jax.jvp(
        lambda value: _collocation_corrected_state(
            runtime.native, value, runtime, chart
        ),
        (correction,),
        (response,),
    )
    return PolishTangentResult(
        native_tangent=jax.tree.map(jnp.add, native_tangent, correction_tangent),
        correction_tangent=response,
        report=report,
    )


def collocation_polish_adjoint(
    context: PolishContext,
    polished_cotangent: HighOrderEquilibriumState,
    *,
    config: PolishLinearConfig = PolishLinearConfig(),
) -> PolishAdjointResult:
    """Apply the IFT pullback of the rectangular polish stationarity equation."""

    runtime, chart = context.runtime, context.chart
    correction = jnp.asarray(context.correction)
    if correction.shape != (chart.size,):
        raise ValueError(
            f"correction has shape {correction.shape}; expected {(chart.size,)}"
        )
    if jax.tree.structure(polished_cotangent) != jax.tree.structure(runtime.native):
        raise ValueError(
            "polished_cotangent must have the runtime native-state structure"
        )
    stationarity = lambda value: _collocation_stationarity(  # noqa: E731
        value, runtime.native, runtime, chart
    )
    _, stationarity_pullback = jax.vjp(stationarity, correction)
    transpose_operator = lambda value: stationarity_pullback(value)[0]  # noqa: E731
    _, correction_pullback = jax.vjp(
        lambda value: _collocation_corrected_state(
            runtime.native, value, runtime, chart
        ),
        correction,
    )
    correction_cotangent = correction_pullback(polished_cotangent)[0]
    diagonal_inverse = jnp.asarray(context.variable_scale) ** 2
    equation_adjoint, report = _solve_linear(
        transpose_operator,
        correction_cotangent,
        lambda rhs: diagonal_inverse * rhs,
        config,
        "least-squares adjoint",
    )
    _, stationarity_native_pullback = jax.vjp(
        lambda native: _collocation_stationarity(
            correction, native, runtime, chart
        ),
        runtime.native,
    )
    force_cotangent = stationarity_native_pullback(equation_adjoint)[0]
    _, direct_native_pullback = jax.vjp(
        lambda native: _collocation_corrected_state(
            native, correction, runtime, chart
        ),
        runtime.native,
    )
    direct_cotangent = direct_native_pullback(polished_cotangent)[0]
    native_cotangent = jax.tree.map(
        jnp.subtract, direct_cotangent, force_cotangent
    )
    return PolishAdjointResult(native_cotangent, equation_adjoint, report)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _implicit_collocation_leaves(
    native_leaves: tuple[jax.Array, ...],
    correction: jax.Array,
    variable_scale: jax.Array,
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart,
    config: PolishLinearConfig,
) -> tuple[jax.Array, ...]:
    del variable_scale, config
    native = jax.tree.unflatten(jax.tree.structure(runtime.native), native_leaves)
    correction = jax.lax.stop_gradient(jnp.asarray(correction))
    return tuple(
        jax.tree.leaves(
            _collocation_corrected_state(native, correction, runtime, chart)
        )
    )


def _implicit_collocation_leaves_fwd(
    native_leaves, correction, variable_scale, runtime, chart, config
):
    output = _implicit_collocation_leaves(
        native_leaves,
        correction,
        variable_scale,
        runtime,
        chart,
        config,
    )
    return output, (correction, variable_scale)


def _implicit_collocation_leaves_bwd(
    runtime,
    chart,
    config,
    saved,
    output_cotangent_leaves,
):
    correction, variable_scale = saved
    output_cotangent = jax.tree.unflatten(
        jax.tree.structure(runtime.native), output_cotangent_leaves
    )
    result = collocation_polish_adjoint(
        PolishContext(runtime, chart, correction, variable_scale),
        output_cotangent,
        config=config,
    )
    return (
        tuple(jax.tree.leaves(result.native_cotangent)),
        jnp.zeros_like(correction),
        jnp.zeros_like(variable_scale),
    )


_implicit_collocation_leaves.defvjp(
    _implicit_collocation_leaves_fwd,
    _implicit_collocation_leaves_bwd,
)


def implicit_collocation_polished_state(
    native: HighOrderEquilibriumState,
    context: PolishContext,
    config: PolishLinearConfig = PolishLinearConfig(),
) -> HighOrderEquilibriumState:
    """Return a certified polished state with a stationarity-equation VJP.

    The primal correction comes from :func:`polish_collocation_least_squares`.
    Reverse mode solves the exact transposed least-squares stationarity
    equation once; it never differentiates through the nonlinear iterations.
    Use :func:`collocation_polish_tangent` for forward sensitivities.
    """

    if jax.tree.structure(native) != jax.tree.structure(context.runtime.native):
        raise ValueError("native must have the polish context native-state structure")
    leaves = tuple(jax.tree.leaves(native))
    polished_leaves = _implicit_collocation_leaves(
        leaves,
        context.correction,
        context.variable_scale,
        context.runtime,
        context.chart,
        config,
    )
    return jax.tree.unflatten(jax.tree.structure(native), polished_leaves)


__all__ = [
    "PolishAdjointResult",
    "PolishLinearConfig",
    "PolishLinearReport",
    "PolishTangentResult",
    "collocation_polish_adjoint",
    "collocation_polish_tangent",
    "implicit_collocation_polished_state",
]
