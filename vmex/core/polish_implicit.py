"""Implicit tangents and adjoints of a converged strong-force polish root.

The nonlinear continuation is deliberately not differentiated.  Once its
correction ``c`` satisfies the alpha=1 equation ``F(c, native) = 0``, this
module applies the implicit-function theorem with matrix-free JVPs/VJPs and
the stored low-order block factors.  A gradient therefore costs one Krylov
solve rather than a replay of continuation and pseudo-time steps.
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
    StrongRootRuntime,
    strong_root_residual_at_native,
)
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


def _reduced_low_inverse(rhs: jax.Array, runtime: StrongRootRuntime) -> jax.Array:
    low_rhs = runtime.layout.unpack(rhs)
    solved = runtime.low_preconditioner.solve_scaled(low_rhs)
    return runtime.layout.pack(solved)


def _reduced_low_inverse_transpose(
    rhs: jax.Array,
    runtime: StrongRootRuntime,
) -> jax.Array:
    low_rhs = runtime.layout.unpack(rhs)
    solved = runtime.low_preconditioner.solve_scaled_transpose(low_rhs)
    return runtime.layout.pack(solved)


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


def strong_root_tangent(
    runtime: StrongRootRuntime,
    correction: jax.Array,
    native_tangent: HighOrderEquilibriumState,
    *,
    config: PolishLinearConfig = PolishLinearConfig(),
) -> PolishTangentResult:
    """Apply the IFT tangent of a converged alpha=1 strong root.

    ``runtime`` freezes the local collocation chart and positive residual
    scaling.  ``correction`` must be the converged reduced vector returned by
    the polish driver.  The result differentiates geometry, lambda, and all
    three continuous profiles exposed by :class:`HighOrderEquilibriumState`.
    """

    correction = jnp.asarray(correction)
    if correction.shape != (runtime.layout.size,):
        raise ValueError(
            f"correction has shape {correction.shape}; expected {(runtime.layout.size,)}"
        )
    if jax.tree.structure(native_tangent) != jax.tree.structure(runtime.native):
        raise ValueError("native_tangent must have the runtime native-state structure")
    residual = lambda vector: strong_root_residual_at_native(  # noqa: E731
        vector, runtime.native, runtime
    )
    _, operator = jax.linearize(residual, correction)
    _, parameter_direction = jax.jvp(
        lambda native: strong_root_residual_at_native(correction, native, runtime),
        (runtime.native,),
        (native_tangent,),
    )
    response, report = _solve_linear(
        operator,
        -parameter_direction,
        lambda rhs: _reduced_low_inverse(rhs, runtime),
        config,
        "tangent",
    )
    high_response = runtime.transfer.prolong(runtime.layout.unpack(response))
    return PolishTangentResult(
        native_tangent=_add_correction(native_tangent, high_response),
        correction_tangent=response,
        report=report,
    )


def strong_root_adjoint(
    runtime: StrongRootRuntime,
    correction: jax.Array,
    polished_cotangent: HighOrderEquilibriumState,
    *,
    config: PolishLinearConfig = PolishLinearConfig(),
) -> PolishAdjointResult:
    """Apply the IFT pullback of a converged alpha=1 strong root."""

    correction = jnp.asarray(correction)
    if correction.shape != (runtime.layout.size,):
        raise ValueError(
            f"correction has shape {correction.shape}; expected {(runtime.layout.size,)}"
        )
    if jax.tree.structure(polished_cotangent) != jax.tree.structure(runtime.native):
        raise ValueError("polished_cotangent must have the runtime native-state structure")
    residual = lambda vector: strong_root_residual_at_native(  # noqa: E731
        vector, runtime.native, runtime
    )
    _, correction_pullback = jax.vjp(residual, correction)
    transpose_operator = lambda value: correction_pullback(value)[0]  # noqa: E731
    high_cotangent = HighOrderCorrection(
        **{
            name: getattr(polished_cotangent, name)
            for name in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")
        }
    )
    reduced_cotangent = runtime.layout.pack(
        runtime.transfer.prolong_transpose(high_cotangent)
    )
    equation_adjoint, report = _solve_linear(
        transpose_operator,
        reduced_cotangent,
        lambda rhs: _reduced_low_inverse_transpose(rhs, runtime),
        config,
        "adjoint",
    )
    _, native_pullback = jax.vjp(
        lambda native: strong_root_residual_at_native(correction, native, runtime),
        runtime.native,
    )
    force_cotangent = native_pullback(equation_adjoint)[0]
    native_cotangent = jax.tree.map(
        jnp.subtract, polished_cotangent, force_cotangent
    )
    return PolishAdjointResult(native_cotangent, equation_adjoint, report)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _implicit_polished_leaves(
    native_leaves: tuple[jax.Array, ...],
    correction: jax.Array,
    runtime: StrongRootRuntime,
    config: PolishLinearConfig,
) -> tuple[jax.Array, ...]:
    del config
    native = jax.tree.unflatten(jax.tree.structure(runtime.native), native_leaves)
    correction = jax.lax.stop_gradient(jnp.asarray(correction))
    high = runtime.transfer.prolong(runtime.layout.unpack(correction))
    return tuple(jax.tree.leaves(_add_correction(native, high)))


def _implicit_polished_leaves_fwd(
    native_leaves, correction, runtime, config
):
    output = _implicit_polished_leaves(
        native_leaves, correction, runtime, config
    )
    return output, correction


def _implicit_polished_leaves_bwd(
    runtime, config, correction, output_cotangent_leaves
):
    output_cotangent = jax.tree.unflatten(
        jax.tree.structure(runtime.native), output_cotangent_leaves
    )
    result = strong_root_adjoint(
        runtime, correction, output_cotangent, config=config
    )
    return (
        tuple(jax.tree.leaves(result.native_cotangent)),
        jnp.zeros_like(correction),
    )


_implicit_polished_leaves.defvjp(
    _implicit_polished_leaves_fwd,
    _implicit_polished_leaves_bwd,
)


def implicit_polished_state(
    native: HighOrderEquilibriumState,
    correction: jax.Array,
    runtime: StrongRootRuntime,
    config: PolishLinearConfig = PolishLinearConfig(),
) -> HighOrderEquilibriumState:
    """Return a polished state whose reverse pass uses one implicit solve.

    The correction is treated as the converged solution of the strong root,
    not as an independent differentiable input.  Validate/certify the primal
    with the polish driver before using this optimization-facing wrapper.
    Use :func:`strong_root_tangent` when a forward-mode response is needed.

    The custom primitive operates on the state's array leaves rather than on
    the validated state object itself.  This keeps JAX's internal sentinel
    values out of :class:`HighOrderEquilibriumState.__post_init__`.
    """

    if jax.tree.structure(native) != jax.tree.structure(runtime.native):
        raise ValueError("native must have the runtime native-state structure")
    leaves = tuple(jax.tree.leaves(native))
    polished_leaves = _implicit_polished_leaves(
        leaves, correction, runtime, config
    )
    return jax.tree.unflatten(jax.tree.structure(native), polished_leaves)


__all__ = [
    "PolishAdjointResult",
    "PolishLinearConfig",
    "PolishLinearReport",
    "PolishTangentResult",
    "implicit_polished_state",
    "strong_root_adjoint",
    "strong_root_tangent",
]
