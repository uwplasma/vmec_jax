"""Fixed-boundary strong-force polishing and independent certification.

The production path solves the overdetermined physical collocation residual
with SOLVAX's matrix-free Gauss--Newton method.  The earlier square
continuation driver remains available for rank and branch diagnostics, but is
not the public polishing route.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from inspect import signature
from time import perf_counter
from typing import Any, Literal, NamedTuple

import functools

import jax
import jax.numpy as jnp
import numpy as np
from solvax import gmres

from .errors import StrongForceCertificationError, StrongForceContinuationError
from .printing import (
    POLISH_AUTO_DECLINED,
    POLISH_SCREEN_HEADER,
    compile_notice,
    emit_flushed,
    force_error_rows,
    polish_certificate_summary,
    polish_cost_decline,
    polish_progress_line,
    polish_screen_line,
)
from .polish import (
    StrongModeBlockPreconditioner,
    StrongPhysicalChart,
    StrongRootRuntime,
    apply_high_order_correction,
    build_low_order_preconditioner,
    build_strong_physical_block_preconditioner,
    build_strong_mode_block_preconditioner,
    make_strong_root_runtime,
    make_strong_structured_chart,
    strong_collocation_residual,
    strong_physical_residual,
    strong_root_residual,
)
from .strong_force import (
    HighOrderEquilibriumState,
    StrongForceReport,
    certify_strong_force,
    evaluate_strong_force,
    force_error_measures,
)

Array = object


def _solvax_continuation_api() -> tuple[Any, ...]:
    """Load the continuation extension supplied by the companion SOLVAX PR."""

    try:
        from solvax import (
            ContinuationConfig,
            PseudoTransientConfig,
            adaptive_continuation,
            pseudo_arclength_corrector,
            pseudo_transient_continuation,
        )
    except ImportError as error:
        raise RuntimeError(
            "strong-force polishing requires a SOLVAX release containing "
            "adaptive continuation, pseudo-transient continuation, and "
            "pseudo-arclength correction (uwplasma/SOLVAX#87)"
        ) from error
    return (
        ContinuationConfig,
        PseudoTransientConfig,
        adaptive_continuation,
        pseudo_arclength_corrector,
        pseudo_transient_continuation,
    )


def _supports_keyword(function: Any, keyword: str) -> bool:
    """Return whether an installed SOLVAX callable exposes a new keyword."""

    try:
        return keyword in signature(function).parameters
    except (TypeError, ValueError):
        return False


def _residual_evaluations(result: Any) -> int:
    """Read exact work accounting, with a conservative pre-0.19 fallback."""

    nonlinear_steps = getattr(result, "nonlinear_steps", getattr(result, "steps", 0))
    return int(getattr(result, "residual_evaluations", nonlinear_steps + 1))


#: Wall-clock ceiling ``POLISH = AUTO`` commits to before it declines.
#:
#: Chosen from the decks polishing is shipped on, not as a round number:
#: ``benchmarks/polish_cost.py`` records the measured cost of each, and the
#: axisymmetric cases that carry the published polish claim predict far
#: under an hour even at the driver's full iteration limits.
#:
#: What is compared against it is the *worst case* those limits allow, not a
#: forecast of what the solve will use — Gauss--Newton usually stops earlier.
#: That is deliberate.  The iteration limits are what the caller authorized,
#: so they are the honest upper bound on what AUTO would be committing their
#: machine to, and the decline names ``POLISH_MAX_ITER`` precisely because
#: lowering the authorization is one of the ways to fit the budget.
#:
#: The consequence is that a 3-D deck at production resolution is declined
#: by default.  That matches the documented scope rather than contradicting
#: it: no 3-D deck has passed the independent certificate, and the one that
#: improved substantially needed hours.  ``POLISH = ON`` never consults the
#: budget, and ``POLISH_BUDGET`` raises it, so neither case is unreachable —
#: only unrequestable by accident.
_DEFAULT_AUTO_BUDGET_SECONDS = 3600.0

#: Normal-equation products the timing lane applies.  Enough to amortize the
#: single linearization it also pays -- so the per-product number is an upper
#: bound, and the predicted wall time errs toward declining rather than
#: toward an overnight surprise -- and few enough to stay a rounding error
#: against the budget being tested.
_POLISH_COST_PROBE_PRODUCTS = 4


class PolishCostEstimate(NamedTuple):
    """What one Gauss--Newton product costs, and what the budget implies.

    ``products`` is the worst case the configured limits allow
    (``max_nonlinear_iterations * linear_restart * linear_max_restarts``),
    not a prediction that the solve will use them: Gauss--Newton stops early
    when its stationarity tolerance is met.  ``predicted_seconds`` is
    therefore an upper bound on the Gauss--Newton phase, which is the number
    a user needs before committing a machine to it.
    """

    seconds_per_product: float
    products: int
    predicted_seconds: float
    chart_size: int
    residual_rows: int


@dataclass(frozen=True)
class PolishConfig:
    """Conservative controls for a fixed-boundary strong-root correction."""

    tolerance: float = 1.0e-3
    validation_tolerance: float | None = 1.0e-2
    radial_degree: int = 3
    radial_spans: int | None = None
    radial_quadrature_order: int | None = None
    radial_refinement_tolerance: float = 1.0e-3
    collocation_scale_probes: int = 8
    least_squares_initial_damping: float = 1.0e-3
    max_continuation_stages: int = 32
    alpha_initial_step: float = 1.0e-3
    alpha_min_step: float = 1.0e-5
    alpha_max_step: float = 0.1
    ptc_initial_dtau: float = 1.0e6
    ptc_max_dtau: float = 1.0e12
    max_nonlinear_iterations: int = 80
    max_backtracks: int = 12
    linear_restart: int = 30
    linear_max_restarts: int = 20
    preconditioner: Literal["none", "legacy", "mode-block"] = "mode-block"
    minimum_jacobian_ratio: float = 0.1
    minimum_jacobian_floor: float = 1.0e-8
    use_pseudo_arclength: bool = True
    max_arclength_steps: int = 16
    arclength_step: float = 1.0e-2
    fail_policy: Literal["raise", "return_unpolished"] = "raise"
    auto_budget_seconds: float = _DEFAULT_AUTO_BUDGET_SECONDS

    def __post_init__(self) -> None:
        finite = (
            self.tolerance,
            self.validation_tolerance
            if self.validation_tolerance is not None
            else self.tolerance,
            self.alpha_initial_step,
            self.alpha_min_step,
            self.alpha_max_step,
            self.ptc_initial_dtau,
            self.ptc_max_dtau,
            self.minimum_jacobian_ratio,
            self.minimum_jacobian_floor,
            self.arclength_step,
            self.radial_refinement_tolerance,
            self.least_squares_initial_damping,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValueError("polish controls must be finite")
        if self.tolerance <= 0.0 or (
            self.validation_tolerance is not None
            and self.validation_tolerance <= 0.0
        ):
            raise ValueError("polish tolerances must be positive")
        if self.radial_degree not in (3, 5, 7):
            raise ValueError("radial_degree must be 3, 5, or 7")
        if self.radial_spans is not None and self.radial_spans < 1:
            raise ValueError("radial_spans must be positive")
        if (
            self.radial_quadrature_order is not None
            and self.radial_quadrature_order < 2
        ):
            raise ValueError("radial_quadrature_order must be at least 2")
        if self.radial_refinement_tolerance <= 0.0:
            raise ValueError("radial_refinement_tolerance must be positive")
        if self.collocation_scale_probes < 0:
            raise ValueError("collocation_scale_probes must be nonnegative")
        if self.least_squares_initial_damping <= 0.0:
            raise ValueError("least_squares_initial_damping must be positive")
        if not 0.0 < self.alpha_min_step <= self.alpha_initial_step <= self.alpha_max_step:
            raise ValueError("require alpha_min_step <= alpha_initial_step <= alpha_max_step")
        if not 0.0 < self.ptc_initial_dtau <= self.ptc_max_dtau:
            raise ValueError("require 0 < ptc_initial_dtau <= ptc_max_dtau")
        if self.max_continuation_stages < 1 or self.max_nonlinear_iterations < 1:
            raise ValueError("polish iteration limits must be positive")
        if self.max_backtracks < 0 or self.linear_restart < 1 or self.linear_max_restarts < 1:
            raise ValueError("polish linear/backtracking limits are invalid")
        if self.preconditioner not in ("none", "legacy", "mode-block"):
            raise ValueError(
                "preconditioner must be 'none', 'legacy', or 'mode-block'"
            )
        if not 0.0 < self.minimum_jacobian_ratio <= 1.0:
            raise ValueError("minimum_jacobian_ratio must lie in (0, 1]")
        if self.minimum_jacobian_floor <= 0.0:
            raise ValueError("minimum_jacobian_floor must be positive")
        if self.max_arclength_steps < 0 or self.arclength_step <= 0.0:
            raise ValueError("pseudo-arclength controls are invalid")
        if self.fail_policy not in ("raise", "return_unpolished"):
            raise ValueError("fail_policy must be 'raise' or 'return_unpolished'")
        if not self.auto_budget_seconds > 0.0:
            raise ValueError("auto_budget_seconds must be positive")

    @property
    def certificate_tolerance(self) -> float:
        """Independent validation threshold used after the solve."""

        return (
            self.tolerance
            if self.validation_tolerance is None
            else self.validation_tolerance
        )


@dataclass(frozen=True)
class PolishReport:
    """Compact machine-readable summary of one correction attempt.

    ``initial_normalized_l2`` and ``final_normalized_l2`` are the pointwise
    ``eps_F`` volume L2, which is the acceptance criterion and is bounded
    above by 2 by construction; on a low-beta or vacuum state both ends of
    that pair sit at the ceiling and the pair says nothing.  The
    ``*_volume_average_force``, ``*_relative_force_error`` and
    ``*_magnetic_relative_force_error`` fields are the non-saturating
    companions taken from the certificates' ``window_normalizations`` over
    ``normalization_window``; quote one of those, not the ``eps_F`` pair,
    whenever a polish gain is being reported.
    """

    converged: bool
    termination_reason: str
    final_alpha: float
    initial_normalized_l2: float
    final_normalized_l2: float
    continuation_accepted: int
    continuation_rejected: int
    nonlinear_iterations: int
    linear_iterations: int
    residual_evaluations: int
    arclength_steps: int
    minimum_signed_jacobian: float
    factor_build_seconds: float
    solve_seconds: float
    least_squares_cost: float | None = None
    least_squares_optimality: float | None = None
    least_squares_initial_optimality: float | None = None
    least_squares_relative_optimality: float | None = None
    least_squares_success: bool | None = None
    least_squares_damping: float | None = None
    radial_refinement_tolerance: float | None = None
    variable_scale_min: float | None = None
    variable_scale_max: float | None = None
    variable_scale_probes: int = 0
    initial_volume_average_force: float | None = None
    final_volume_average_force: float | None = None
    initial_relative_force_error: float | None = None
    final_relative_force_error: float | None = None
    initial_magnetic_relative_force_error: float | None = None
    final_magnetic_relative_force_error: float | None = None
    normalization_window: tuple[float, float] | None = None
    seconds_per_linear_product: float | None = None
    predicted_solve_seconds: float | None = None
    auto_budget_seconds: float | None = None


def _normalization_fields(
    initial: StrongForceReport, final: StrongForceReport | None = None
) -> dict[str, Any]:
    """Non-saturating certificate fields for a :class:`PolishReport`.

    ``final`` defaults to ``initial`` so an attempt that never produced a
    corrected state still reports where it started rather than leaving the
    fields empty, exactly as the ``eps_F`` pair already does.
    """

    final = initial if final is None else final
    window = initial.window_normalizations
    return {
        "initial_volume_average_force": float(window.volume_average_force),
        "final_volume_average_force": float(
            final.window_normalizations.volume_average_force
        ),
        "initial_relative_force_error": float(window.relative_force_error),
        "final_relative_force_error": float(
            final.window_normalizations.relative_force_error
        ),
        "initial_magnetic_relative_force_error": float(
            window.magnetic_relative_force_error
        ),
        "final_magnetic_relative_force_error": float(
            final.window_normalizations.magnetic_relative_force_error
        ),
        "normalization_window": (float(window.s_min), float(window.s_max)),
    }


class PolishContext(NamedTuple):
    """Frozen chart and converged coordinates for implicit differentiation."""

    runtime: StrongRootRuntime
    chart: StrongPhysicalChart
    correction: jax.Array
    variable_scale: jax.Array


class PolishResult(NamedTuple):
    """Certified native state, report, full correction, and derivative context."""

    native_equilibrium: HighOrderEquilibriumState
    strong_force: StrongForceReport
    polish_report: PolishReport
    correction: jax.Array
    context: PolishContext | None = None
    compatibility_state: Any = None


def polished_compatibility_state(legacy_state, result: PolishResult):
    """Project a certified native correction onto the sampled WOUT mesh."""

    if result.context is None or not np.asarray(result.correction).size:
        return legacy_state
    runtime = result.context.runtime
    high = runtime.layout.unpack(
        jnp.asarray(runtime.coordinate_scale) * jnp.asarray(result.correction)
    )
    low = runtime.transfer.restrict(high)
    return jax.tree.map(jnp.add, legacy_state, low)


#: Minimum radial surfaces for a polished WOUT export.  The stable wout
#: reconstruction (:func:`~vmex.core.strong_force.lift_high_order_state`'s
#: default) fits ``min(32, (ns - degree + 1) // 2)`` uniform spans, which
#: locks every pre-cap mesh to ~2 samples per span — a near-interpolatory
#: fit whose curvature ringing swamps the polish correction.  Only beyond
#: the 32-span cap do additional samples turn that fit into an
#: overdetermined L2 projection, so the export mesh provides four samples
#: per capped span: ``4 * 32 + 1``.
_POLISHED_WOUT_MIN_NS = 129


def polished_wout_ns(
    native: HighOrderEquilibriumState, *, solve_ns: int
) -> int:
    """Radial export mesh on which sampling ``native`` stays certifiable."""

    determined = 2 * int(native.radial_basis.size) + 1
    return max(int(solve_ns), _POLISHED_WOUT_MIN_NS, determined)


def polished_wout_state(
    native: HighOrderEquilibriumState, source, *, solve_ns: int
):
    """Sample the certified native state on the WOUT export mesh.

    ``polished_state`` keeps its API contract of matching the solve mesh;
    this is the file-export companion.  The WOUT is the only carrier of the
    polish gain for downstream consumers, and on a coarse solve mesh the
    stable reconstruction cannot resolve the between-node correction: on the
    bundled shaped tokamak (``ns = 31``) the native certificate is 1.8e-3
    while the solve-mesh export certifies at 3.3e-2 — worse than an
    unrefined VMEC2000 wout — even though the samples still determine the
    native state exactly.  Sampling on :func:`polished_wout_ns` surfaces
    instead certifies within a few percent of the dense-sampling floor of
    the default reconstruction (1.9e-3 at ``ns = 129`` against a 1.88e-3
    floor measured at ``ns = 401``).
    """

    from .polish import sample_high_order_state
    from .solver import prepare_runtime, resolution_from_input

    ns = polished_wout_ns(native, solve_ns=solve_ns)
    runtime = prepare_runtime(source, resolution_from_input(source, ns=ns))
    return sample_high_order_state(native, runtime)


@dataclass(frozen=True)
class _IdentityPreconditioner:
    """Explicit identity action used to benchmark unpreconditioned JFNK."""

    build_seconds: float = 0.0

    def apply(self, rhs, alpha=1.0, dtau=jnp.inf):
        del alpha, dtau
        return rhs


def _build_mode_block_preconditioner(
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart | None = None,
) -> StrongModeBlockPreconditioner:
    """Backward-compatible private seam for the shared block builder."""

    if chart is None:
        return build_strong_mode_block_preconditioner(runtime)
    return build_strong_physical_block_preconditioner(runtime, chart)


def _solve_residual(
    vector: jax.Array,
    runtime: StrongRootRuntime,
    alpha: jax.Array,
    chart: StrongPhysicalChart | None = None,
) -> jax.Array:
    """Evaluate either the legacy full chart or the structured physical chart."""

    if chart is None:
        return strong_root_residual(vector, runtime, alpha)
    return strong_physical_residual(vector, runtime, chart, alpha)


def _full_solve_vector(
    vector: jax.Array,
    chart: StrongPhysicalChart | None,
) -> jax.Array:
    """Lift physical solve coordinates into the existing full root layout."""

    return jnp.asarray(vector) if chart is None else chart.lift(vector)


def _continuation_precondition(
    rhs: jax.Array,
    alpha: jax.Array,
    dtau: jax.Array,
    runtime: StrongRootRuntime,
    block_preconditioner: StrongModeBlockPreconditioner,
    chart: StrongPhysicalChart | None = None,
) -> jax.Array:
    """Use the exact legacy inverse early and mode bands near strong force."""

    if isinstance(block_preconditioner, _IdentityPreconditioner):
        return rhs
    if chart is not None:
        return block_preconditioner.apply(rhs, alpha, dtau)
    return jax.lax.cond(
        jnp.asarray(alpha) < 0.5,
        lambda value: _low_inverse(value, runtime),
        lambda value: block_preconditioner.apply(value, alpha, dtau),
        rhs,
    )


def _corrected_state(
    vector: jax.Array,
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart | None = None,
):
    vector = _full_solve_vector(vector, chart)
    correction = runtime.layout.unpack(
        jnp.asarray(runtime.coordinate_scale) * jnp.asarray(vector)
    )
    return apply_high_order_correction(runtime.native, correction)


def _minimum_signed_jacobian(
    vector: jax.Array,
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart | None = None,
) -> jax.Array:
    state = _corrected_state(vector, runtime, chart)
    rr, tt, zz = jnp.meshgrid(
        jnp.asarray(runtime.radial_nodes),
        jnp.asarray(runtime.theta),
        jnp.asarray(runtime.zeta),
        indexing="ij",
    )
    samples = evaluate_strong_force(state, rr, tt, zz)
    signed = float(state.jacobian_sign) * samples.sqrt_g / jnp.maximum(rr, 1.0e-14)
    return jnp.min(signed)


def _low_inverse(rhs: jax.Array, runtime: StrongRootRuntime) -> jax.Array:
    """Invert the row-scaled low endpoint in reduced vector coordinates."""

    high_rhs = runtime.layout.unpack(
        jnp.asarray(rhs) / jnp.asarray(runtime.equation_scale)
    )
    low_rhs = runtime.transfer.restrict(high_rhs)
    solution = runtime.low_preconditioner.solve_scaled(low_rhs)
    return runtime.layout.pack(runtime.transfer.prolong(solution)) / jnp.asarray(
        runtime.coordinate_scale
    )


def _solve_low_inverse(
    rhs: jax.Array,
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart | None = None,
) -> jax.Array:
    """Apply the legacy inverse in the active solve-coordinate chart."""

    if chart is None:
        return _low_inverse(rhs, runtime)
    full_rhs = jnp.asarray(chart.equation_basis) @ (
        jnp.asarray(rhs) / jnp.asarray(chart.equation_scale)
    )
    full_solution = _low_inverse(full_rhs, runtime)
    return (
        jnp.asarray(chart.coordinate_basis).T @ full_solution
    ) / jnp.asarray(chart.coordinate_scale)


def _normalized_low_residual_norm(
    residual: jax.Array,
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart | None = None,
) -> jax.Array:
    """Return the low-endpoint RMS before numerical row equilibration."""

    equation_scale = (
        runtime.equation_scale if chart is None else chart.equation_scale
    )
    size = runtime.layout.size if chart is None else chart.size
    unscaled = jnp.asarray(residual) / jnp.asarray(equation_scale)
    return jnp.linalg.norm(unscaled) / np.sqrt(float(size))


def _ptc_config(config: PolishConfig, *, residual_scale: float) -> Any:
    _, PseudoTransientConfig, _, _, _ = _solvax_continuation_api()
    return PseudoTransientConfig(
        rtol=config.tolerance,
        # Couple the roundoff floor to a representative residual norm.  Both
        # tolerances then transform with a harmless positive equation scaling:
        # unlike a fixed absolute tolerance this cannot accept an unsolved
        # stage, while unlike ``atol=0`` it does not chase alpha-zero roundoff.
        # The independent dimensional certificate remains the final gate.
        atol=config.tolerance * float(residual_scale),
        max_steps=config.max_nonlinear_iterations,
        initial_dt=config.ptc_initial_dtau,
        max_dt=config.ptc_max_dtau,
        max_backtracks=config.max_backtracks,
        linear_restart=config.linear_restart,
        linear_max_restarts=config.linear_max_restarts,
    )


def _continuation_config(config: PolishConfig) -> Any:
    ContinuationConfig, _, _, _, _ = _solvax_continuation_api()
    return ContinuationConfig(
        target=1.0,
        initial_step=config.alpha_initial_step,
        min_step=config.alpha_min_step,
        max_step=config.alpha_max_step,
        max_stages=config.max_continuation_stages,
    )


def _branch_tangent(
    vector: jax.Array,
    alpha: float,
    runtime: StrongRootRuntime,
    config: PolishConfig,
    previous: tuple[jax.Array, jax.Array] | None,
    block_preconditioner: StrongModeBlockPreconditioner | None = None,
    chart: StrongPhysicalChart | None = None,
) -> tuple[jax.Array, jax.Array]:
    residual = lambda value: _solve_residual(  # noqa: E731
        value, runtime, alpha, chart
    )
    _, jvp = jax.linearize(residual, vector)
    parameter_direction = _solve_residual(
        vector, runtime, 1.0, chart
    ) - _solve_residual(vector, runtime, 0.0, chart)
    if previous is None:
        linear = gmres(
            jvp,
            -parameter_direction,
            precond=(
                (lambda value: _solve_low_inverse(value, runtime, chart))
                if block_preconditioner is None
                else lambda value: block_preconditioner.apply(value, alpha)
            ),
            restart=config.linear_restart,
            rtol=min(1.0e-6, config.tolerance),
            atol=config.tolerance,
            max_restarts=config.linear_max_restarts,
        )
        linear_x = linear.x
        linear_alpha = jnp.asarray(1.0, dtype=jnp.asarray(linear.x).dtype)
    else:
        previous_x, previous_alpha = previous

        def bordered(value):
            tangent_x, tangent_alpha = value
            physical = jax.tree.map(
                jnp.add,
                jvp(tangent_x),
                jax.tree.map(lambda item: item * tangent_alpha, parameter_direction),
            )
            normalization = (
                jnp.vdot(previous_x, tangent_x).real
                + previous_alpha * tangent_alpha
            )
            return physical, normalization

        linear = gmres(
            bordered,
            (jnp.zeros_like(vector), jnp.asarray(1.0, dtype=vector.dtype)),
            precond=lambda rhs: _bordered_preconditioner(
                runtime, previous, block_preconditioner, chart
            )((vector, jnp.asarray(alpha)), rhs, jnp.inf),
            restart=config.linear_restart,
            rtol=min(1.0e-6, config.tolerance),
            atol=config.tolerance,
            max_restarts=config.linear_max_restarts,
        )
        linear_x, linear_alpha = linear.x
    if not bool(linear.converged):
        raise StrongForceContinuationError(
            "pseudo-arclength tangent solve did not converge",
            hint="refine the radial representation or increase the linear budget",
            alpha=float(alpha),
            residual_norm=float(linear.residual_norm),
            linear_iterations=int(linear.iterations),
        )
    tangent_x = jnp.asarray(linear_x)
    tangent_alpha = jnp.asarray(linear_alpha, dtype=tangent_x.dtype)
    norm = jnp.sqrt(jnp.vdot(tangent_x, tangent_x).real + tangent_alpha**2)
    tangent_x, tangent_alpha = tangent_x / norm, tangent_alpha / norm
    if previous is not None:
        orientation = jnp.vdot(tangent_x, previous[0]).real + tangent_alpha * previous[1]
        sign = jnp.where(orientation < 0.0, -1.0, 1.0)
        tangent_x, tangent_alpha = sign * tangent_x, sign * tangent_alpha
    return tangent_x, tangent_alpha


def _bordered_preconditioner(
    runtime: StrongRootRuntime,
    tangent: tuple[jax.Array, jax.Array],
    block_preconditioner: StrongModeBlockPreconditioner | None = None,
    chart: StrongPhysicalChart | None = None,
):
    """Return a low-order block-elimination preconditioner for a bordered root."""

    def apply(state, rhs, dtau):
        return _apply_bordered_preconditioner(
            state,
            rhs,
            dtau,
            tangent,
            runtime,
            block_preconditioner,
            chart,
        )

    return apply


def _apply_bordered_preconditioner(
    state,
    rhs,
    dtau,
    tangent,
    runtime: StrongRootRuntime,
    block_preconditioner: StrongModeBlockPreconditioner | None = None,
    chart: StrongPhysicalChart | None = None,
):
    """Apply bordered block elimination with dynamic branch data."""

    vector, alpha = state
    rhs_x, rhs_alpha = rhs
    tangent_x, tangent_alpha = tangent
    parameter_direction = _solve_residual(
        vector, runtime, 1.0, chart
    ) - _solve_residual(vector, runtime, 0.0, chart)
    inverse = (
        (lambda value: _solve_low_inverse(value, runtime, chart))
        if block_preconditioner is None
        else lambda value: block_preconditioner.apply(value, alpha, dtau)
    )
    q_rhs = inverse(rhs_x)
    q_parameter = inverse(parameter_direction)
    schur = tangent_alpha - jnp.vdot(tangent_x, q_parameter).real
    tiny = jnp.sqrt(jnp.finfo(jnp.asarray(schur).dtype).eps)
    safe_schur = jnp.where(
        jnp.abs(schur) > tiny,
        schur,
        jnp.where(schur < 0.0, -tiny, tiny),
    )
    delta_alpha = (
        rhs_alpha - jnp.vdot(tangent_x, q_rhs).real
    ) / safe_schur
    return q_rhs - q_parameter * delta_alpha, delta_alpha


def _arclength_to_target(
    vector: jax.Array,
    alpha: float,
    runtime: StrongRootRuntime,
    config: PolishConfig,
    admissible,
    block_preconditioner: StrongModeBlockPreconditioner | None,
    initial_tangent: tuple[jax.Array, jax.Array] | None,
    chart: StrongPhysicalChart | None = None,
):
    _, _, _, pseudo_arclength_corrector, pseudo_transient_continuation = (
        _solvax_continuation_api()
    )
    tangent = (
        _branch_tangent(
            vector,
            alpha,
            runtime,
            config,
            None,
            block_preconditioner,
            chart,
        )
        if initial_tangent is None
        else initial_tangent
    )
    solve_size = runtime.layout.size if chart is None else chart.size
    residual_scale = np.sqrt(float(solve_size)) / float(
        runtime.operator_balance
    )
    nonlinear = _ptc_config(config, residual_scale=residual_scale)
    total_nonlinear = total_linear = total_evaluations = 0
    arclength_residual = lambda value, parameter: _solve_residual(  # noqa: E731
        value, runtime, parameter, chart
    )
    arclength_admissible = lambda value, parameter: admissible(  # noqa: E731
        value, parameter
    )
    parameterized_precondition = (  # noqa: E731
        lambda state, rhs, dtau, branch_tangent, predictor: (
            _apply_bordered_preconditioner(
                state,
                rhs,
                dtau,
                branch_tangent,
                runtime,
                block_preconditioner,
                chart,
            )
        )
    )
    for step in range(config.max_arclength_steps):
        predictor = (
            vector + config.arclength_step * tangent[0],
            jnp.asarray(alpha) + config.arclength_step * tangent[1],
        )
        if _supports_keyword(pseudo_arclength_corrector, "parameterized_precond"):
            corrector_kwargs = {
                "parameterized_precond": parameterized_precondition
            }
        elif _supports_keyword(pseudo_arclength_corrector, "precond"):
            corrector_kwargs = {
                "precond": _bordered_preconditioner(
                    runtime, tangent, block_preconditioner, chart
                )
            }
        else:
            corrector_kwargs = {}
        corrected = pseudo_arclength_corrector(
            arclength_residual,
            predictor,
            tangent=tangent,
            predictor=predictor,
            config=nonlinear,
            admissible=arclength_admissible,
            **corrector_kwargs,
        )
        total_nonlinear += int(corrected.steps)
        total_linear += int(corrected.linear_iterations)
        total_evaluations += _residual_evaluations(corrected)
        if not bool(corrected.converged) or not bool(corrected.linear_converged):
            return vector, alpha, step, total_nonlinear, total_linear, total_evaluations
        previous_alpha = alpha
        vector, alpha_array = corrected.x
        alpha = float(alpha_array)
        if (previous_alpha - 1.0) * (alpha - 1.0) <= 0.0:
            target = pseudo_transient_continuation(
                lambda value: _solve_residual(value, runtime, 1.0, chart),
                vector,
                precond=(
                    (
                        lambda state, rhs, dtau: _solve_low_inverse(
                            rhs, runtime, chart
                        )
                    )
                    if block_preconditioner is None
                    else lambda state, rhs, dtau: block_preconditioner.apply(
                        rhs, 1.0, dtau
                    )
                ),
                admissible=lambda value: admissible(value, 1.0),
                config=nonlinear,
            )
            total_nonlinear += int(target.steps)
            total_linear += int(target.linear_iterations)
            total_evaluations += _residual_evaluations(target)
            if bool(target.converged) and bool(target.linear_converged):
                return (
                    target.x,
                    1.0,
                    step + 1,
                    total_nonlinear,
                    total_linear,
                    total_evaluations,
                )
        tangent = _branch_tangent(
            vector,
            alpha,
            runtime,
            config,
            tangent,
            block_preconditioner,
            chart,
        )
    return (
        vector,
        alpha,
        config.max_arclength_steps,
        total_nonlinear,
        total_linear,
        total_evaluations,
    )


def polish_strong_root(
    runtime: StrongRootRuntime,
    *,
    config: PolishConfig | None = None,
    initial_certificate: StrongForceReport | None = None,
    chart: StrongPhysicalChart | None = None,
) -> PolishResult:
    """Follow the legacy-connected fixed-boundary branch to strong force."""

    config = PolishConfig() if config is None else config
    started = perf_counter()
    initial_certificate = (
        certify_strong_force(runtime.native)
        if initial_certificate is None
        else initial_certificate
    )
    solve_size = runtime.layout.size if chart is None else chart.size
    zero = jnp.zeros(
        (solve_size,), dtype=jnp.asarray(runtime.native.R_cos).dtype
    )
    full_zero = jnp.zeros((runtime.layout.size,), dtype=zero.dtype)
    if not _failed_certificate_checks(initial_certificate, config):
        report = PolishReport(
            converged=True,
            termination_reason="already-certified",
            final_alpha=1.0,
            initial_normalized_l2=float(initial_certificate.normalized_l2),
            final_normalized_l2=float(initial_certificate.normalized_l2),
            continuation_accepted=0,
            continuation_rejected=0,
            nonlinear_iterations=0,
            linear_iterations=0,
            residual_evaluations=0,
            arclength_steps=0,
            minimum_signed_jacobian=float(initial_certificate.minimum_signed_jacobian),
            factor_build_seconds=runtime.low_preconditioner.factor_build_seconds,
            solve_seconds=perf_counter() - started,
            radial_refinement_tolerance=config.radial_refinement_tolerance,
            **_normalization_fields(initial_certificate),
        )
        return PolishResult(runtime.native, initial_certificate, report, full_zero)
    _, _, adaptive_continuation, _, pseudo_transient_continuation = (
        _solvax_continuation_api()
    )
    initial_margin = float(_minimum_signed_jacobian(zero, runtime, chart))
    if config.preconditioner == "mode-block":
        block_preconditioner = _build_mode_block_preconditioner(runtime, chart)
    elif config.preconditioner == "none":
        block_preconditioner = _IdentityPreconditioner()
    else:
        block_preconditioner = None
    factor_build_seconds = (
        runtime.low_preconditioner.factor_build_seconds
        + (0.0 if block_preconditioner is None else block_preconditioner.build_seconds)
    )
    margin_floor = max(
        config.minimum_jacobian_floor,
        config.minimum_jacobian_ratio * initial_margin,
    )

    def admissible(vector, alpha):
        del alpha
        residual = _solve_residual(vector, runtime, 1.0, chart)
        return (
            jnp.all(jnp.isfinite(vector))
            & jnp.all(jnp.isfinite(residual))
            & (_minimum_signed_jacobian(vector, runtime, chart) >= margin_floor)
        )

    residual_scale = np.sqrt(float(solve_size)) / float(
        runtime.operator_balance
    )
    nonlinear = _ptc_config(config, residual_scale=residual_scale)
    precondition = (  # noqa: E731
        (lambda state, rhs, dtau: rhs)
        if config.preconditioner == "none"
        else lambda state, rhs, dtau: _solve_low_inverse(rhs, runtime, chart)
    )
    # The low homotopy endpoint subtracts the stored legacy defect, so zero is
    # its mathematical root.  Roundoff from restrict/project/prolong may leave
    # a tiny row-equilibrated remainder.  Check that remainder before row
    # scaling and avoid asking PTC to reduce it below floating-point noise.
    # A genuinely inconsistent endpoint still takes the globalized solve.
    endpoint_residual = _solve_residual(zero, runtime, 0.0, chart)
    endpoint_solved = float(
        _normalized_low_residual_norm(endpoint_residual, runtime, chart)
    ) <= config.tolerance
    if endpoint_solved:
        vector = zero
        nonlinear_iterations = 0
        linear_iterations = 0
        residual_evaluations = 1
        converged = True
    else:
        endpoint = pseudo_transient_continuation(
            lambda vector: _solve_residual(vector, runtime, 0.0, chart),
            zero,
            precond=precondition,
            admissible=lambda vector: admissible(vector, 0.0),
            config=nonlinear,
        )
        vector = endpoint.x
        nonlinear_iterations = int(endpoint.steps)
        linear_iterations = int(endpoint.linear_iterations)
        residual_evaluations = 1 + _residual_evaluations(endpoint)
        converged = bool(endpoint.converged) and bool(endpoint.linear_converged)
    steps: tuple[Any, ...] = ()
    arclength_steps = 0
    alpha = 0.0
    reason = "alpha-zero-failed"
    if converged:
        accepted_states: list[tuple[jax.Array, float]] = [(vector, 0.0)]

        def record_accepted_state(candidate, parameter, solution):
            del solution
            accepted_states.append((candidate, float(parameter)))
            return True

        # Kwargs for adaptive_continuation: the two spellings carry
        # different callable arities, so the annotation is deliberately Any.
        continuation_preconditioners: dict[str, Any] = (
            {"precond": precondition}
            if block_preconditioner is None
            or not _supports_keyword(
                adaptive_continuation, "parameterized_precond"
            )
            else {
                "parameterized_precond": (
                    lambda state, rhs, dtau, parameter: _continuation_precondition(
                        rhs,
                        parameter,
                        dtau,
                        runtime,
                        block_preconditioner,
                        chart,
                    )
                )
            }
        )
        continuation = adaptive_continuation(
            lambda value, parameter: _solve_residual(
                value, runtime, parameter, chart
            ),
            vector,
            alpha0=0.0,
            nonlinear_config=nonlinear,
            continuation_config=_continuation_config(config),
            admissible=admissible,
            accept_stage=record_accepted_state,
            **continuation_preconditioners,
        )
        steps = continuation.steps
        vector, alpha = continuation.x, continuation.alpha
        nonlinear_iterations += sum(stage.nonlinear_steps for stage in steps)
        linear_iterations += sum(stage.linear_iterations for stage in steps)
        residual_evaluations += sum(_residual_evaluations(stage) for stage in steps)
        converged = continuation.converged
        reason = "strong-root" if converged else "continuation-stalled"
        if not converged and config.use_pseudo_arclength:
            initial_tangent = None
            if len(accepted_states) >= 2:
                previous_vector, previous_alpha = accepted_states[-2]
                delta_vector = vector - previous_vector
                delta_alpha = jnp.asarray(alpha - previous_alpha)
                tangent_norm = jnp.sqrt(
                    jnp.vdot(delta_vector, delta_vector).real + delta_alpha**2
                )
                initial_tangent = (
                    delta_vector / tangent_norm,
                    delta_alpha / tangent_norm,
                )
            try:
                (
                    vector,
                    alpha,
                    arclength_steps,
                    arc_nonlinear,
                    arc_linear,
                    arc_evaluations,
                ) = _arclength_to_target(
                    vector,
                    alpha,
                    runtime,
                    config,
                    admissible,
                    block_preconditioner,
                    initial_tangent,
                    chart,
                )
            except StrongForceContinuationError:
                reason = "pseudo-arclength-tangent-failed"
            else:
                nonlinear_iterations += arc_nonlinear
                linear_iterations += arc_linear
                residual_evaluations += arc_evaluations
                converged = alpha == 1.0
                reason = (
                    "pseudo-arclength" if converged else "pseudo-arclength-stalled"
                )

    accepted = sum(stage.accepted for stage in steps)
    rejected = len(steps) - accepted
    if not converged:
        report = PolishReport(
            converged=False,
            termination_reason=reason,
            final_alpha=float(alpha),
            initial_normalized_l2=float(initial_certificate.normalized_l2),
            final_normalized_l2=float(initial_certificate.normalized_l2),
            continuation_accepted=accepted,
            continuation_rejected=rejected,
            nonlinear_iterations=nonlinear_iterations,
            linear_iterations=linear_iterations,
            residual_evaluations=residual_evaluations,
            arclength_steps=arclength_steps,
            minimum_signed_jacobian=float(
                _minimum_signed_jacobian(vector, runtime, chart)
            ),
            factor_build_seconds=factor_build_seconds,
            solve_seconds=perf_counter() - started,
            **_normalization_fields(initial_certificate),
        )
        if config.fail_policy == "raise":
            raise StrongForceContinuationError(
                "strong-force continuation did not reach alpha=1",
                hint="inspect the continuation report and refine the radial representation",
                alpha=float(alpha),
                residual_norm=float(
                    jnp.linalg.norm(
                        _solve_residual(vector, runtime, alpha, chart)
                    )
                ),
                nonlinear_iterations=nonlinear_iterations,
                linear_iterations=linear_iterations,
                accepted_stages=accepted,
                rejected_stages=rejected,
            )
        return PolishResult(runtime.native, initial_certificate, report, full_zero)

    state = _corrected_state(vector, runtime, chart)
    certificate = certify_strong_force(state)
    failed_checks = _failed_certificate_checks(certificate, config)
    certified = not failed_checks
    report = PolishReport(
        converged=certified,
        termination_reason="certified" if certified else "certification-failed",
        final_alpha=1.0,
        initial_normalized_l2=float(initial_certificate.normalized_l2),
        final_normalized_l2=float(certificate.normalized_l2),
        continuation_accepted=accepted,
        continuation_rejected=rejected,
        nonlinear_iterations=nonlinear_iterations,
        linear_iterations=linear_iterations,
        residual_evaluations=residual_evaluations,
        arclength_steps=arclength_steps,
        minimum_signed_jacobian=float(certificate.minimum_signed_jacobian),
        factor_build_seconds=factor_build_seconds,
        solve_seconds=perf_counter() - started,
        radial_refinement_tolerance=config.radial_refinement_tolerance,
        **_normalization_fields(initial_certificate, certificate),
    )
    if not certified and config.fail_policy == "raise":
        raise StrongForceCertificationError(
            "strong root failed the independent force certificate: "
            + "; ".join(failed_checks),
            hint="increase radial degree/resolution and retry once",
            solver_converged=True,
            normalized_l2=float(certificate.normalized_l2),
            tolerance=config.certificate_tolerance,
            radial_refinement=float(certificate.radial_refinement_difference),
            radial_refinement_tolerance=config.radial_refinement_tolerance,
        )
    if not certified:
        return PolishResult(runtime.native, initial_certificate, report, full_zero)
    return PolishResult(
        state,
        certificate,
        report,
        _full_solve_vector(vector, chart),
    )


#: Shortest gap between two live polish heartbeats.  The callbacks that feed
#: them fire far more often than this on small problems and far less often
#: than this at production resolution, so the throttle only ever suppresses
#: output; it never delays a line past the next callback.
_POLISH_PROGRESS_INTERVAL_SECONDS = 30.0

#: The reporter the staged device callbacks below deliver to, or ``None``
#: when no verbose polish is running.  It cannot be captured in the callback
#: closures: ``jax.debug.callback`` bakes the Python callable into the
#: compiled executable, and that executable is reused by every later polish
#: call of the same shape, which would keep reporting to the first call's
#: console long after it finished.
_ACTIVE_POLISH_PROGRESS: Any = None


def _polish_progress_product(_unused: Any) -> None:
    """One matrix-free normal-equation product was applied."""

    reporter = _ACTIVE_POLISH_PROGRESS
    if reporter is not None:
        reporter.product()


def _polish_progress_cost(cost: Any) -> None:
    """The Gauss--Newton loop evaluated the residual at a new point."""

    reporter = _ACTIVE_POLISH_PROGRESS
    if reporter is not None:
        reporter.cost(float(cost))


@jax.custom_jvp
def _progress_probe(vector):
    """Identity whose linearization announces every application."""

    return vector


@_progress_probe.defjvp
def _progress_probe_jvp(primals, tangents):
    (vector,), (tangent,) = primals, tangents
    # The reported value must depend on the tangent.  A tangent-independent
    # callback is hoisted into the known half by partial evaluation, so it
    # would fire once per linearization -- once per Gauss-Newton step, the
    # very granularity that leaves production runs silent for hours -- rather
    # than once per inner product.  The sum costs one reduction over the
    # chart against a full force sweep per product.
    jax.debug.callback(_polish_progress_product, jnp.sum(tangent))
    return vector, tangent


class _PolishProgress:
    """Throttled console view of a running Gauss--Newton polish.

    Printing happens on the host from device callbacks, so it neither
    enters the traced program's arithmetic nor forces a synchronization
    barrier; the solve's numbers are bit-identical with and without it.
    """

    def __init__(self, emit: Any, *, product_budget: int,
                 interval: float | None = None) -> None:
        self._emit = emit
        self._budget = int(product_budget)
        # Read at construction, not as a default argument, so the throttle
        # stays patchable for tests that need every callback to print.
        self._interval = float(
            _POLISH_PROGRESS_INTERVAL_SECONDS if interval is None else interval)
        self._started = perf_counter()
        # The first product always reports: a run that shows one line and
        # then goes quiet for its throttle interval is still attributable,
        # whereas a solve that prints nothing at all is the bug being fixed.
        self._last = float("-inf")
        self._products = 0
        self._cost = float("nan")
        self.lines = 0

    def product(self) -> None:
        self._products += 1
        self._maybe_emit()

    def cost(self, value: float) -> None:
        self._cost = value

    def _maybe_emit(self) -> None:
        now = perf_counter()
        if now - self._last < self._interval:
            return
        self._last = now
        self.lines += 1
        self._emit(polish_progress_line(
            elapsed_seconds=now - self._started,
            products=self._products,
            product_budget=self._budget,
            cost=self._cost,
        ), end="")


@contextmanager
def _polish_progress(reporter: Any) -> Iterator[Any]:
    """Route the staged callbacks to ``reporter`` for the duration."""

    global _ACTIVE_POLISH_PROGRESS
    previous = _ACTIVE_POLISH_PROGRESS
    _ACTIVE_POLISH_PROGRESS = reporter
    try:
        yield reporter
    finally:
        _ACTIVE_POLISH_PROGRESS = previous


# The polish hot path used to jit fresh per-call lambdas closing over the
# runtime and chart, baking every per-solve array in as XLA constants: each
# polish call recompiled the residual, its linearization, and the whole
# Gauss-Newton program, and the changed constants defeated the persistent
# compilation cache as well (different constants, different HLO). These
# module lanes take the pytrees as arguments, so equal-structure polish
# calls share one compiled program in memory and on disk.
@functools.partial(jax.jit, static_argnames=("config", "progress"))
def _gauss_newton_polish_lane(value, runtime, chart, variable_scale,
                              collocation_scale, config, progress=False):
    from solvax import gauss_newton_least_squares

    def residual(vector):
        scaled = variable_scale * vector
        if progress:
            scaled = _progress_probe(scaled)
        residuals = strong_collocation_residual(
            scaled, runtime, chart) / collocation_scale
        if progress:
            jax.debug.callback(
                _polish_progress_cost, 0.5 * jnp.vdot(residuals, residuals))
        return residuals

    # `progress` is static, so the heartbeat compiles a second program rather
    # than perturbing the quiet one.  That program carries host callbacks, and
    # JAX never writes a program with host callbacks to the persistent
    # compilation cache, so a verbose run recompiles this lane on every
    # process while a quiet one reloads it.  The trade is deliberate: the
    # runs that need watching are the long interactive ones, where one lane
    # compile is noise against hours of Gauss-Newton, and the Python API is
    # quiet by default.
    #
    # SOLVAX also takes a `precond` for the inner CG.  Nothing is passed:
    # the only preconditioner this module has is the low-order block
    # inverse, and building its symmetric normal-equation companion from
    # those factors was measured to make the solve worse, not faster --
    # benchmarks/polish3d_tuning.md records the numbers.  The residual's
    # variable_scale change of variables is the diagonal equilibration the
    # CG does get.
    return gauss_newton_least_squares(residual, value, config=config)


@functools.partial(jax.jit, static_argnames=("products",))
def _normal_product_lane(vector, runtime, chart, variable_scale,
                         collocation_scale, products):
    """Apply the Gauss--Newton normal operator ``products`` times.

    This is the loop body SOLVAX's inner PCG runs, and nothing else: one
    linearization of the scaled collocation residual, then repeated
    ``J^T J`` products against it.  Timing it is how the driver learns what
    a Gauss--Newton iteration costs on *this* problem and this machine
    rather than guessing from a size heuristic.  Renormalizing between
    products keeps the probe direction bounded over the repeats without
    changing what is being measured.
    """

    def residual(value):
        return strong_collocation_residual(
            variable_scale * value, runtime, chart) / collocation_scale

    _, jvp = jax.linearize(residual, vector)
    transpose = jax.linear_transpose(jvp, vector)

    def body(_index, result):
        product = transpose(jvp(result))[0]
        return product / jnp.maximum(jnp.linalg.norm(product), 1.0e-300)

    # A rolled loop, not an unrolled one: at production resolution each
    # product is a full linearized force sweep, and unrolling four of them
    # would spend more time compiling the price tag than the price tag
    # saves.
    return jax.lax.fori_loop(0, int(products), body, vector)


def _measure_polish_cost(
    zero: Any,
    runtime: StrongRootRuntime,
    chart: Any,
    variable_scale: Any,
    collocation_scale: Any,
    config: PolishConfig,
) -> PolishCostEstimate:
    """Time the Gauss--Newton inner product and extrapolate to the budget.

    The lane is called twice and only the second call is timed: the first
    pays compilation, which is a one-off the Gauss--Newton phase does not
    repeat and must not be charged to every one of its products.
    """

    probe = jnp.ones_like(zero)
    arguments = (probe, runtime, chart, variable_scale, collocation_scale)
    jax.block_until_ready(
        _normal_product_lane(*arguments, products=_POLISH_COST_PROBE_PRODUCTS))
    started = perf_counter()
    jax.block_until_ready(
        _normal_product_lane(*arguments, products=_POLISH_COST_PROBE_PRODUCTS))
    elapsed = perf_counter() - started
    seconds_per_product = elapsed / float(_POLISH_COST_PROBE_PRODUCTS)
    products = (
        int(config.max_nonlinear_iterations)
        * max(int(config.linear_restart) * int(config.linear_max_restarts), 1)
    )
    return PolishCostEstimate(
        seconds_per_product=seconds_per_product,
        products=products,
        predicted_seconds=seconds_per_product * float(products),
        chart_size=int(chart.size),
        residual_rows=int(np.asarray(runtime.radial_nodes).size)
        * int(np.asarray(runtime.theta).size)
        * int(np.asarray(runtime.zeta).size)
        * 2,
    )


@jax.jit
def _collocation_probe_lane(probes, zero, runtime, chart, collocation_scale):
    """Stacked transpose-JVP responses of the scaled collocation residual.

    The former host-eager linearize/linear_transpose pair rebuilt and re-ran
    a fresh linear program per polish call; like the Ruiz lanes in
    :mod:`.polish`, linearizing inside one module jit with the pytrees as
    traced operands bakes no constants and lets equal-structure polish calls
    share the executable. The primal re-runs on each call — one extra force
    evaluation at setup.
    """

    def residual(value):
        return strong_collocation_residual(
            value, runtime, chart) / collocation_scale

    _, jvp = jax.linearize(residual, zero)
    transpose = jax.linear_transpose(jvp, zero)
    return jax.vmap(lambda probe: transpose(probe)[0])(probes)


def _collocation_variable_scale(
    runtime: StrongRootRuntime,
    chart,
    collocation_scale: jax.Array,
    zero: jax.Array,
    row_count: int,
    probes: int,
) -> np.ndarray:
    """Estimate inverse column norms with deterministic transpose probes.

    The probe draws keep the pre-lane sequence: one generator, one draw per
    probe in order, so the sampled directions are bit-identical to the
    retired per-call implementation.
    """

    if probes == 0:
        return np.ones(np.asarray(zero).shape, dtype=float)
    generator = np.random.default_rng(0)
    stacked = np.stack([
        generator.choice(np.asarray([-1.0, 1.0]), size=row_count)
        for _ in range(probes)
    ])
    responses = np.asarray(_collocation_probe_lane(
        jnp.asarray(stacked), jnp.asarray(zero), runtime, chart,
        jnp.asarray(collocation_scale)))
    column_squared = np.zeros(np.asarray(zero).shape, dtype=float)
    for response in responses:
        column_squared += response * response
    column_norm = np.sqrt(column_squared / float(probes))
    column_floor = max(1.0e-8 * float(np.max(column_norm)), 1.0e-12)
    return 1.0 / np.maximum(column_norm, column_floor)


#: Lane structures whose first-use compile notice was already attributable
#: in this process — the polish analogue of the solver's used-lane keys.
_POLISH_LANE_NOTICED: set[Any] = set()


def _failed_certificate_checks(
    certificate: StrongForceReport, config: PolishConfig
) -> tuple[str, ...]:
    """Name each independent acceptance check the certificate failed."""

    failed = []
    for name, value, lower, upper in (
        ("independent force L2", certificate.normalized_l2,
         0.0, config.certificate_tolerance),
        ("radial refinement difference", certificate.radial_refinement_difference,
         0.0, config.radial_refinement_tolerance),
        ("minimum signed Jacobian", certificate.minimum_signed_jacobian,
         0.0, None),
    ):
        value = float(value)
        if not np.isfinite(value):
            failed.append(f"{name} is nonfinite ({value})")
        elif upper is None:
            if value <= lower:
                failed.append(f"{name} {value:.3E} <= {lower:g}")
        elif not lower <= value <= upper:
            failed.append(f"{name} {value:.3E} outside [{lower:g}, {upper:.3E}]")
    return tuple(failed)


def _emit_gauss_newton_rows(solution: Any, emit: Any) -> None:
    """Print the fixed-shape SOLVAX history as a screen table.

    The Gauss--Newton loop runs inside one jitted ``lax.while_loop`` with no
    host callbacks, so rows appear when the solve returns rather than live;
    the banner and compile notice bracket that silent window on screen.
    Everything printed already exists in ``solution.history`` — no extra
    computation.
    """

    history = solution.history
    cost = np.asarray(history.cost)
    gradient = np.asarray(history.gradient_norm)
    damping = np.asarray(history.damping)
    ratio = np.asarray(history.ratio)
    accepted = np.asarray(history.accepted)
    linear = np.asarray(history.linear_iterations)
    emit(POLISH_SCREEN_HEADER, end="")
    emit(polish_screen_line(0, float(cost[0]), float(gradient[0]),
                            float(damping[0])), end="")
    for step in range(int(solution.steps)):
        emit(polish_screen_line(
            step + 1, float(cost[step + 1]), float(gradient[step + 1]),
            float(damping[step + 1]), ratio=float(ratio[step]),
            linear_iterations=int(linear[step]),
            accepted=bool(accepted[step])), end="")


def polish_collocation_least_squares(
    runtime: StrongRootRuntime,
    *,
    config: PolishConfig | None = None,
    chart: StrongPhysicalChart | None = None,
    initial_certificate: StrongForceReport | None = None,
    auto: bool = False,
    verbose: bool = False,
    emit: Any = emit_flushed,
) -> PolishResult:
    """Solve and certify the overdetermined physical force residual.

    The residual exposes both independent force channels at every collocation
    point. SOLVAX applies matrix-free JVP/VJP normal products; no dense
    Jacobian is formed. The returned correction uses the full constrained
    layout so it composes with the existing native-state utilities.

    ``verbose=True`` prints the CLI progress lines (compile notice,
    Gauss--Newton rows, certificate summary) through ``emit``; the default
    keeps the Python API silent, and printing never changes the numerics.

    ``auto=True`` is ``POLISH = AUTO``: before committing to the
    Gauss--Newton phase the driver times one inner product on this problem
    and this machine and declines if the configured iteration limits could
    run past ``config.auto_budget_seconds``.  ``auto=False`` never measures
    and never declines, so an explicit polish request costs exactly what it
    did before.
    """

    from solvax import LeastSquaresConfig

    config = PolishConfig() if config is None else config
    if chart is None:
        if verbose:
            emit(" building the polish chart (scaling probes compile once)...")
        chart = make_strong_structured_chart(runtime)
    if initial_certificate is None:
        if verbose:
            emit(" evaluating the initial force certificate "
                 "(independent oracle, compiles once)...")
        initial_certificate = certify_strong_force(runtime.native)
    zero = jnp.zeros((chart.size,), dtype=jnp.asarray(runtime.native.R_cos).dtype)
    initial_collocation = strong_collocation_residual(zero, runtime, chart)
    collocation_scale = max(
        float(jnp.linalg.norm(initial_collocation))
        / np.sqrt(float(initial_collocation.size)),
        1.0e-12,
    )
    collocation_scale_array = jnp.asarray(collocation_scale)

    if verbose:
        emit(f" collocation: {int(initial_collocation.size)} residual rows, "
             f"{int(chart.size)} unknowns")
    # First use of this lane structure in the process compiles the probe and
    # Gauss-Newton executables — the pause users previously read as a hang.
    if not jax.config.jax_disable_jit:
        lane_key = (int(chart.size), int(initial_collocation.size), config)
        first_use = lane_key not in _POLISH_LANE_NOTICED
        _POLISH_LANE_NOTICED.add(lane_key)
        if verbose and first_use:
            emit(compile_notice(int(np.asarray(runtime.radial_nodes).size),
                                lane="polish"), end="")

    variable_scale = _collocation_variable_scale(
        runtime,
        chart,
        collocation_scale_array,
        zero,
        int(initial_collocation.size),
        config.collocation_scale_probes,
    )
    variable_scale_array = jnp.asarray(variable_scale)
    least_squares_config = LeastSquaresConfig(
        rtol=config.tolerance,
        max_steps=config.max_nonlinear_iterations,
        initial_damping=config.least_squares_initial_damping,
        linear_rtol=1.0e-3,
        linear_max_steps=max(
            config.linear_restart * config.linear_max_restarts,
            1,
        ),
    )
    started = perf_counter()
    estimate = None
    if auto:
        if verbose:
            emit(" timing one Gauss-Newton product to price the solve...")
        estimate = _measure_polish_cost(
            zero, runtime, chart, variable_scale_array,
            collocation_scale_array, config)
        if estimate.predicted_seconds > float(config.auto_budget_seconds):
            report = PolishReport(
                converged=False,
                termination_reason=POLISH_AUTO_DECLINED,
                final_alpha=1.0,
                initial_normalized_l2=float(initial_certificate.normalized_l2),
                final_normalized_l2=float(initial_certificate.normalized_l2),
                continuation_accepted=0,
                continuation_rejected=0,
                nonlinear_iterations=0,
                linear_iterations=0,
                residual_evaluations=0,
                arclength_steps=0,
                minimum_signed_jacobian=float(
                    initial_certificate.minimum_signed_jacobian),
                factor_build_seconds=(
                    runtime.low_preconditioner.factor_build_seconds),
                solve_seconds=perf_counter() - started,
                radial_refinement_tolerance=config.radial_refinement_tolerance,
                variable_scale_min=float(np.min(variable_scale)),
                variable_scale_max=float(np.max(variable_scale)),
                variable_scale_probes=config.collocation_scale_probes,
                seconds_per_linear_product=estimate.seconds_per_product,
                predicted_solve_seconds=estimate.predicted_seconds,
                auto_budget_seconds=float(config.auto_budget_seconds),
            )
            if verbose:
                emit(polish_cost_decline(
                    seconds_per_product=estimate.seconds_per_product,
                    products=estimate.products,
                    predicted_seconds=estimate.predicted_seconds,
                    budget_seconds=float(config.auto_budget_seconds),
                    chart_size=estimate.chart_size,
                    residual_rows=estimate.residual_rows,
                ), end="")
            # A declined AUTO is a decision, not a certification failure:
            # it never raises, whatever fail_policy says, because nothing
            # was attempted and nothing failed.
            #
            # It skips rather than running a truncated solve on purpose.
            # Acceptance is the independent certificate, so a solve stopped
            # short of it returns the unpolished state anyway -- a bounded
            # attempt would spend the whole budget to arrive exactly where
            # skipping arrives, minus the hours.  A solve that *would*
            # certify inside the budget is not declined in the first place.
            return PolishResult(
                runtime.native,
                initial_certificate,
                report,
                jnp.zeros_like(chart.lift(zero)),
            )

    reporter = (
        _PolishProgress(
            emit,
            product_budget=int(least_squares_config.max_steps)
            * int(least_squares_config.linear_max_steps),
        )
        if verbose
        else None
    )
    with _polish_progress(reporter):
        solution = _gauss_newton_polish_lane(
            zero, runtime, chart, variable_scale_array,
            collocation_scale_array, least_squares_config,
            progress=reporter is not None)
        jax.block_until_ready(solution)
    if verbose:
        _emit_gauss_newton_rows(solution, emit)
    vector = variable_scale_array * solution.x
    state = _corrected_state(vector, runtime, chart)
    certificate = certify_strong_force(state)
    failed_checks = _failed_certificate_checks(certificate, config)
    independently_certified = not failed_checks
    # Acceptance is the independent certificate - the volume-L2 bar, radial
    # refinement stability, and a positive signed Jacobian - exactly as the
    # published contract states. The Gauss-Newton solver's internal relative
    # tolerance is a diagnostic; a certified state whose solver merely ran
    # out its step budget is still an accepted polish.
    converged = independently_certified
    report = PolishReport(
        converged=converged,
        termination_reason=(
            "independently-certified"
            if converged
            else "solvax-collocation-least-squares"
        ),
        final_alpha=1.0,
        initial_normalized_l2=float(initial_certificate.normalized_l2),
        final_normalized_l2=float(certificate.normalized_l2),
        continuation_accepted=int(solution.accepted_steps),
        continuation_rejected=int(solution.rejected_steps),
        nonlinear_iterations=int(solution.steps),
        linear_iterations=int(solution.linear_iterations),
        residual_evaluations=int(solution.steps) + 1,
        arclength_steps=0,
        minimum_signed_jacobian=float(certificate.minimum_signed_jacobian),
        factor_build_seconds=runtime.low_preconditioner.factor_build_seconds,
        solve_seconds=perf_counter() - started,
        least_squares_cost=float(solution.cost),
        least_squares_optimality=float(solution.gradient_norm),
        least_squares_initial_optimality=float(solution.history.gradient_norm[0]),
        least_squares_relative_optimality=float(
            solution.gradient_norm
            / jnp.maximum(solution.history.gradient_norm[0], 1.0e-300)
        ),
        least_squares_success=bool(solution.converged),
        least_squares_damping=float(solution.damping),
        radial_refinement_tolerance=config.radial_refinement_tolerance,
        variable_scale_min=float(np.min(variable_scale)),
        variable_scale_max=float(np.max(variable_scale)),
        variable_scale_probes=config.collocation_scale_probes,
        **_normalization_fields(initial_certificate, certificate),
        # Present whenever AUTO priced the solve, so a run that was allowed
        # through records the prediction it was allowed on, not only the
        # runs that were turned away.
        seconds_per_linear_product=(
            None if estimate is None else estimate.seconds_per_product),
        predicted_solve_seconds=(
            None if estimate is None else estimate.predicted_seconds),
        auto_budget_seconds=(
            None if estimate is None else float(config.auto_budget_seconds)),
    )
    if verbose:
        emit(polish_certificate_summary(
            report.initial_normalized_l2, report.final_normalized_l2,
            config.certificate_tolerance,
            verdict="CERTIFIED" if converged else "FAILED",
            failed_checks=failed_checks,
            measures=force_error_measures(initial_certificate, certificate),
            window=report.normalization_window),
            end="")
    if converged:
        return PolishResult(
            state,
            certificate,
            report,
            chart.lift(vector),
            PolishContext(runtime, chart, vector, variable_scale_array),
        )
    if config.fail_policy == "raise":
        raise StrongForceCertificationError(
            "collocation polish failed its force certificate: " + "; ".join(failed_checks),
            hint="inspect the polish report and refine the radial representation",
            solver_converged=bool(solution.converged),
            normalized_l2=float(certificate.normalized_l2),
            tolerance=config.certificate_tolerance,
            radial_refinement=float(certificate.radial_refinement_difference),
            radial_refinement_tolerance=config.radial_refinement_tolerance,
        )
    return PolishResult(
        runtime.native, initial_certificate, report, jnp.zeros_like(chart.lift(zero))
    )


def polish_legacy_solution(
    source,
    resolution,
    legacy_state,
    *,
    config: PolishConfig | None = None,
    lconm1: bool = True,
    auto: bool = False,
    verbose: bool = False,
    emit: Any = emit_flushed,
) -> PolishResult:
    """Refine and lift one converged legacy solve, then run the strong driver.

    ``verbose=True`` routes the CLI progress lines through ``emit`` (the
    solver prints the phase banner before calling here); the default keeps
    the Python API silent and printing never changes the numerics.
    ``auto=True`` additionally lets the driver decline a solve whose
    measured cost would exceed ``config.auto_budget_seconds``.
    """

    started = perf_counter()
    from . import implicit
    from .input import VmecInput
    from .radial_basis import BSplineBasis
    from .strong_force import certify_strong_force, lift_high_order_state

    if not isinstance(source, VmecInput):
        raise TypeError("strong-force polishing requires a VmecInput source")
    config = PolishConfig() if config is None else config
    implicit_config = implicit.make_config(
        source,
        ns=int(resolution.ns),
        lconm1=bool(lconm1),
        multigrid=False,
    )
    params = implicit.params_from_input(source)
    legacy_runtime = implicit.runtime_from_params(params, implicit_config)
    # Each phase below can run minutes at high resolution with no output of
    # its own, so the CLI announces every one before it starts — a silent
    # console must always be attributable to a named phase.
    if verbose:
        emit(" refining the converged state (Newton anchor)...")
    dof_mask = implicit._dof_mask(legacy_state, legacy_runtime, implicit_config)
    refined_state = implicit._refined_state(
        implicit_config,
        params,
        legacy_state,
        dof_mask,
    )
    # Certification is a reconstruction problem, not a requirement to retain
    # one spline coefficient per legacy sample. Evaluate the stable,
    # overdetermined lift first; an already-certified result needs neither the
    # compatibility chart nor a factorization. The empty correction records
    # that no root coordinates were constructed or applied.
    radial_basis = (
        None
        if config.radial_spans is None
        else BSplineBasis.clamped(
            np.linspace(0.0, 1.0, config.radial_spans + 1),
            degree=config.radial_degree,
            quadrature_order=(
                config.radial_degree + 3
                if config.radial_quadrature_order is None
                else config.radial_quadrature_order
            ),
        )
    )
    if verbose:
        emit(" evaluating the initial force certificate "
             "(independent oracle, compiles once)...")
    certified_native = lift_high_order_state(
        refined_state,
        legacy_runtime,
        radial_basis=radial_basis,
        degree=config.radial_degree,
    )
    initial_certificate = certify_strong_force(certified_native)
    if verbose:
        emit(" initial certificate: EPS-F = "
             f"{float(initial_certificate.normalized_l2):.3E}"
             f"  (tolerance {config.certificate_tolerance:.3E};"
             " EPS-F is bounded by 2 by construction)")
        emit("".join(
            f"{row}\n" for row in force_error_rows(
                force_error_measures(initial_certificate),
                window=(
                    float(initial_certificate.window_normalizations.s_min),
                    float(initial_certificate.window_normalizations.s_max),
                ))),
            end="")
    if not _failed_certificate_checks(initial_certificate, config):
        report = PolishReport(
            converged=True,
            termination_reason="already-certified",
            final_alpha=1.0,
            initial_normalized_l2=float(initial_certificate.normalized_l2),
            final_normalized_l2=float(initial_certificate.normalized_l2),
            continuation_accepted=0,
            continuation_rejected=0,
            nonlinear_iterations=0,
            linear_iterations=0,
            residual_evaluations=0,
            arclength_steps=0,
            minimum_signed_jacobian=float(initial_certificate.minimum_signed_jacobian),
            factor_build_seconds=0.0,
            solve_seconds=perf_counter() - started,
            radial_refinement_tolerance=config.radial_refinement_tolerance,
            **_normalization_fields(initial_certificate),
        )
        if verbose:
            emit(polish_certificate_summary(
                report.initial_normalized_l2, report.final_normalized_l2,
                config.certificate_tolerance,
                verdict="ALREADY CERTIFIED (no correction applied)",
                measures=force_error_measures(initial_certificate),
                window=report.normalization_window), end="")
        return PolishResult(
            certified_native,
            initial_certificate,
            report,
            jnp.zeros((0,), dtype=jnp.asarray(refined_state.R_cos).dtype),
            None,
            refined_state,
        )
    native = certified_native
    if verbose:
        emit(" building the polish preconditioner and root runtime...")
    low_preconditioner = build_low_order_preconditioner(
        native,
        params,
        implicit_config,
        refined_state,
        dof_mask,
        probe_chunk_size=4,
    )
    runtime = make_strong_root_runtime(
        native,
        low_preconditioner,
        dof_mask,
        balance_full_root=False,
        radial_quadrature_order=config.radial_quadrature_order,
    )
    result = polish_collocation_least_squares(
        runtime,
        config=config,
        initial_certificate=initial_certificate,
        auto=auto,
        verbose=verbose,
        emit=emit,
    )
    return result._replace(
        compatibility_state=polished_compatibility_state(refined_state, result)
    )


__all__ = [
    "PolishConfig",
    "PolishContext",
    "PolishReport",
    "PolishResult",
    "polish_collocation_least_squares",
    "polish_legacy_solution",
    "polished_compatibility_state",
    "polished_wout_ns",
    "polished_wout_state",
    "polish_strong_root",
]
