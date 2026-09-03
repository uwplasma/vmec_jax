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
    """Krylov controls and failure policy for polished-root derivatives.

    One tangent or one adjoint costs exactly one preconditioned GMRES solve of
    the ``(chart.size, chart.size)`` stationarity operator, so these five
    values set the entire linear cost and the entire accuracy of a polished
    derivative.  Convergence is not taken on the solver's word: the true
    residual ``||b - A x||`` is recomputed on the returned iterate and
    compared against ``max(atol, rtol * ||b||)``.  Every range below is
    enforced in ``__post_init__``.

    ``fail_policy`` decides what a non-converged Krylov solve produces:
    ``"raise"`` (the default) raises :class:`StrongForceLinearSolveError`,
    ``"nan"`` returns NaN. **The raise only happens outside tracing.** Under
    ``jax.jit`` -- which is where these derivatives are normally taken -- the
    convergence flag is a traced value, so no Python exception can be raised
    from it and both policies return NaN through :func:`jnp.where`. A jitted
    gradient that fails therefore comes back as NaN, not as an error; check
    for it rather than relying on an exception. Making it raise would need a
    host callback in the traced graph, which would stop those programs being
    written to the persistent compilation cache.

    Attributes
    ----------
    rtol:
        Relative tolerance on that true residual, dimensionless.  Must be
        finite and strictly positive.  The default is tight because a
        polished-root gradient is normally consumed by an optimizer that
        assumes exact derivatives.
    atol:
        Absolute floor on the same residual, in the units of the stationarity
        right-hand side (a cotangent of the collocation residual, so scaled
        force per chart coordinate).  Must be finite and nonnegative; ``0.0``
        is allowed and makes the test purely relative.
    restart:
        GMRES Arnoldi cycle length, i.e. how many Krylov vectors are kept
        before restarting.  Clamped down to the system size for small charts.
        Must be at least 1.
    max_restarts:
        Maximum restart cycles, so at most ``restart * max_restarts``
        iterations in total.  Must be at least 1.
    fail_policy:
        What a non-converged solve does.  ``"raise"`` raises
        :class:`~vmex.core.errors.StrongForceLinearSolveError` carrying the
        solve kind, iteration count, achieved residual, and tolerance;
        ``"nan"`` returns a NaN-filled result instead.  The check is a host
        check: under :func:`jax.jit` the values are tracers, so neither
        branch can run and the solve returns NaN where it did not converge
        regardless of the policy.  The report always comes back either way,
        so a traced caller can inspect ``report.converged`` itself.
    """

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
    """Finite unpreconditioned residual certificate for a tangent/adjoint solve.

    ``converged`` requires finite operands, solution, norms and tolerance, with
    ``residual_norm <= tolerance``; an internal Krylov success flag is insufficient.
    

    Recomputed from the returned iterate rather than taken from the Krylov
    solver's internal estimate, so a restarted GMRES cannot report success on
    a stale recurrence.
    """

    #: ``||b - A x||`` of the returned solution, a scalar array.
    residual_norm: jax.Array
    #: The bar it was compared against, ``max(atol, rtol * ||b||)``.
    tolerance: jax.Array
    #: Krylov iterations consumed, summed over restart cycles.
    iterations: jax.Array
    #: Boolean scalar array, true when either the solver's own flag or the
    #: recomputed residual clears ``tolerance``.  The returned solution is
    #: filled with NaN wherever this is false.
    converged: jax.Array


class PolishTangentResult(NamedTuple):
    """Total polished-state tangent and its reduced correction response."""

    #: Directional derivative of the polished continuous state, carrying the
    #: same pytree structure as the input ``native_tangent`` and equal to that
    #: input plus the response of the correction.
    native_tangent: HighOrderEquilibriumState
    #: Derivative of the accepted correction itself, shape ``(chart.size,)``
    #: in the chart's gauge-free physical coordinates.
    correction_tangent: jax.Array
    #: Certificate of the single Krylov solve; see :class:`PolishLinearReport`.
    report: PolishLinearReport


class PolishAdjointResult(NamedTuple):
    """Native-state cotangent and the strong-equation adjoint variable."""

    #: Pullback of the output cotangent to the unpolished native state, with
    #: the same pytree structure as ``context.runtime.native``.
    native_cotangent: HighOrderEquilibriumState
    #: Adjoint variable of the stationarity equation, shape
    #: ``(chart.size,)``: the solution of the transposed stationarity operator
    #: against the correction cotangent, i.e. the multiplier the parameter
    #: pullback is contracted with.  Exposed as a diagnostic; the finished
    #: gradient is ``native_cotangent``.
    equation_adjoint: jax.Array
    #: Certificate of the single Krylov solve; see :class:`PolishLinearReport`.
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
    applied = operator(solution.x)
    residual_norm = jnp.linalg.norm(rhs - applied)
    rhs_norm = jnp.linalg.norm(rhs)
    tolerance = jnp.maximum(config.atol, config.rtol * rhs_norm)
    # Krylov flags can describe a preconditioned estimate. Only the finite,
    # recomputed unpreconditioned residual certifies a derivative here.
    finite = (
        jnp.all(jnp.isfinite(solution.x))
        & jnp.all(jnp.isfinite(rhs))
        & jnp.all(jnp.isfinite(applied))
        & jnp.isfinite(rhs_norm)
        & jnp.isfinite(residual_norm)
        & jnp.isfinite(tolerance)
    )
    converged = finite & (residual_norm <= tolerance)
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
                    f"strong-root {solve_kind} solve did not converge: finite "
                    f"true-residual certificate failed (residual "
                    f"{float(report.residual_norm):.3e}, tolerance "
                    f"{float(report.tolerance):.3e}) after "
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
    """Apply the IFT tangent of the rectangular polish stationarity equation.

    Forward-mode sensitivity of the polished equilibrium with respect to the
    unpolished one.  Write ``P`` for the native state, ``c`` for the
    correction in chart coordinates, ``r(c, P)`` for the rectangular
    collocation residual, and

    ``g(c, P) = J(c, P).T @ r(c, P)``

    for its least-squares stationarity, which the accepted correction ``c*``
    zeroes.  The polished state is ``S(c, P) = P + L(c)`` with ``L`` the linear
    lift of a chart vector into spline coefficients.  Differentiating ``g = 0``
    at ``(c*, P)`` gives ``dc = -g_c^{-1} g_P dP``, and the returned total
    tangent is ``dP + L(dc)``.

    ``g_c`` is the exact Hessian of ``0.5 ||r||^2``, obtained by linearizing
    ``g`` itself, so the nonzero-residual second-order term is retained and
    this is not a Gauss--Newton approximation.  Both ``g_c`` and ``g_P`` are
    applied matrix-free through JVPs; nothing is assembled and the nonlinear
    iterations are never replayed.  Cost: one preconditioned GMRES solve of
    size ``chart.size``, preconditioned by ``diag(context.variable_scale)**2``
    — the primal column-norm estimate, which approximates the inverse diagonal
    of the normal operator up to one positive constant.

    Parameters
    ----------
    context:
        The :class:`~vmex.core.polish_driver.PolishContext` of a certified
        polish, i.e. ``PolishResult.context``, surfaced by the solver as
        ``result.polish_context``.  It supplies the frozen runtime, chart, and
        stationary correction; the derivative is taken at exactly that point.
        A ``correction`` whose shape is not ``(chart.size,)`` is rejected.
    native_tangent:
        Perturbation of the unpolished native state, a pytree with the same
        structure as ``context.runtime.native`` (checked) and matching leaf
        shapes: the six ``(mnmax, nbasis)`` spline coefficient tables
        (``R``/``Z`` in metres, lambda in radians), the three ``(nbasis,)``
        profile leaves ``phipf``, ``chipf`` (flux derivatives with respect to
        normalized toroidal flux ``s``) and ``pressure`` (Pa), plus the
        boundary tables when the state carries them.  Typically the seed of an
        outer forward-mode derivative with respect to a shape or profile
        parameter.
    config:
        Krylov controls for the single linear solve.

    Returns
    -------
    A :class:`PolishTangentResult` whose ``native_tangent`` is the total
    tangent of the polished state, with the input's pytree structure and leaf
    shapes; whose ``correction_tangent`` is ``dc`` with shape
    ``(chart.size,)``; and whose ``report`` certifies the solve.  A
    non-converged solve follows ``fail_policy`` — it raises, or fills the
    response with NaN so the returned tangent is NaN as well.
    """

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
    """Apply the IFT pullback of the rectangular polish stationarity equation.

    The reverse-mode counterpart of :func:`collocation_polish_tangent`, and
    the routine :func:`implicit_collocation_polished_state` installs as its
    custom VJP.  With the same notation — stationarity
    ``g(c, P) = J(c, P).T @ r(c, P)`` zeroed at the accepted ``c*``, polished
    state ``S(c, P) = P + L(c)`` — a cotangent ``w`` on the polished state
    pulls back as

    ``lambda = g_c^{-T} (L.T w)``  and  ``dP = w - g_P.T lambda``

    since ``S`` depends on ``P`` only through the additive identity.  The
    transposes are taken with VJPs of the exact stationarity map, so the
    nonzero-residual Hessian term is retained here too; the single Krylov
    solve uses the same ``diag(context.variable_scale)**2`` Jacobi
    preconditioner as the tangent.  One gradient therefore costs one linear
    solve, independent of how many Gauss--Newton steps the primal polish took.

    Parameters
    ----------
    context:
        The :class:`~vmex.core.polish_driver.PolishContext` of a certified
        polish; see :func:`collocation_polish_tangent`.  A ``correction``
        whose shape is not ``(chart.size,)`` is rejected.
    polished_cotangent:
        Cotangent of the *polished* state — the gradient of a downstream
        scalar objective with respect to the polished equilibrium.  Same
        pytree structure as ``context.runtime.native`` (checked) and the same
        leaf shapes, since the polished state has the native state's
        structure.  Typically produced by ``jax.grad`` of an objective
        evaluated on the polished state.
    config:
        Krylov controls for the single linear solve.

    Returns
    -------
    A :class:`PolishAdjointResult` whose ``native_cotangent`` is the gradient
    with respect to the unpolished native state, carrying that state's pytree
    structure and leaf shapes; whose ``equation_adjoint`` is the stationarity
    multiplier of shape ``(chart.size,)``; and whose ``report`` certifies the
    solve.  A non-converged solve follows ``fail_policy`` — it raises, or NaN
    propagates from the multiplier into the returned cotangent.
    """

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
    return output, (native_leaves, correction, variable_scale)


def _implicit_collocation_leaves_bwd(
    runtime,
    chart,
    config,
    saved,
    output_cotangent_leaves,
):
    native_leaves, correction, variable_scale = saved
    # The frozen discretization may be reused, but the adjoint must linearize
    # at the native inputs of this forward call, not the runtime's old state.
    native = jax.tree.unflatten(jax.tree.structure(runtime.native), native_leaves)
    runtime = replace(runtime, native=native)
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

    The primal value is simply ``native`` plus the frozen correction lifted
    into spline coefficients; the correction is held under
    :func:`jax.lax.stop_gradient`, and the registered VJP replaces the naive
    derivative of that sum with :func:`collocation_polish_adjoint`.  The
    context's ``correction`` and ``variable_scale`` are treated as constants
    and receive zero cotangents.  Because the rule is a
    :func:`jax.custom_vjp`, only reverse mode is defined: differentiating this
    function with :func:`jax.jvp` raises rather than silently returning the
    wrong tangent.

    Parameters
    ----------
    native:
        The unpolished continuous state to polish and differentiate through.
        Must have the same pytree structure as ``context.runtime.native``
        (checked); it does not have to be the identical object, which is what
        lets an outer :func:`jax.grad` trace through it.
    context:
        The :class:`~vmex.core.polish_driver.PolishContext` of a certified
        polish, pinning the runtime, chart, correction, and variable scaling.
    config:
        Krylov controls for the adjoint solve performed in the backward pass;
        the forward pass does no linear solve at all.

    Returns
    -------
    A :class:`~vmex.core.strong_force.HighOrderEquilibriumState` with
    ``native``'s pytree structure and leaf shapes, carrying the polished
    coefficients, and differentiable in reverse mode with respect to
    ``native``.
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
