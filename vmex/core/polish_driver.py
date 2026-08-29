"""Branch-preserving fixed-boundary strong-force polishing.

This host orchestrator follows the square residual defined in
``vmex.core.polish``.  Each nonlinear stage remains JIT-compatible in SOLVAX;
host code records continuation decisions and evaluates the independent final
certificate.  No optimizer or residual-norm minimization is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from time import perf_counter
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from solvax import gmres

from .errors import StrongForceCertificationError, StrongForceContinuationError
from .polish import (
    StrongRootRuntime,
    apply_high_order_correction,
    build_low_order_preconditioner,
    make_strong_root_runtime,
    strong_root_residual,
)
from .strong_force import (
    HighOrderEquilibriumState,
    StrongForceReport,
    certify_strong_force,
    evaluate_strong_force,
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


@dataclass(frozen=True)
class PolishConfig:
    """Conservative controls for a fixed-boundary strong-root correction."""

    tolerance: float = 1.0e-8
    validation_tolerance: float | None = None
    radial_degree: int = 5
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
    preconditioner: Literal["legacy", "mode-block"] = "legacy"
    minimum_jacobian_ratio: float = 0.1
    minimum_jacobian_floor: float = 1.0e-8
    use_pseudo_arclength: bool = True
    max_arclength_steps: int = 16
    arclength_step: float = 1.0e-2
    fail_policy: Literal["raise", "return_unpolished"] = "raise"

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
        if not 0.0 < self.alpha_min_step <= self.alpha_initial_step <= self.alpha_max_step:
            raise ValueError("require alpha_min_step <= alpha_initial_step <= alpha_max_step")
        if not 0.0 < self.ptc_initial_dtau <= self.ptc_max_dtau:
            raise ValueError("require 0 < ptc_initial_dtau <= ptc_max_dtau")
        if self.max_continuation_stages < 1 or self.max_nonlinear_iterations < 1:
            raise ValueError("polish iteration limits must be positive")
        if self.max_backtracks < 0 or self.linear_restart < 1 or self.linear_max_restarts < 1:
            raise ValueError("polish linear/backtracking limits are invalid")
        if self.preconditioner not in ("legacy", "mode-block"):
            raise ValueError("preconditioner must be 'legacy' or 'mode-block'")
        if not 0.0 < self.minimum_jacobian_ratio <= 1.0:
            raise ValueError("minimum_jacobian_ratio must lie in (0, 1]")
        if self.minimum_jacobian_floor <= 0.0:
            raise ValueError("minimum_jacobian_floor must be positive")
        if self.max_arclength_steps < 0 or self.arclength_step <= 0.0:
            raise ValueError("pseudo-arclength controls are invalid")
        if self.fail_policy not in ("raise", "return_unpolished"):
            raise ValueError("fail_policy must be 'raise' or 'return_unpolished'")

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
    """Compact machine-readable summary of one correction attempt."""

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


class PolishResult(NamedTuple):
    """Native state, independent certificate, report, and free correction."""

    native_equilibrium: HighOrderEquilibriumState
    strong_force: StrongForceReport
    polish_report: PolishReport
    correction: jax.Array


@dataclass(frozen=True, eq=False)
class _ModeBlockPreconditioner:
    """Independent Fourier-mode blocks for the low/strong Jacobian pencil."""

    indices: tuple[jax.Array, ...]
    low_blocks: tuple[jax.Array, ...]
    strong_blocks: tuple[jax.Array, ...]
    build_seconds: float

    def apply(
        self,
        rhs: jax.Array,
        alpha: jax.Array,
        dtau: jax.Array | float = jnp.inf,
    ) -> jax.Array:
        """Apply regularized block solves without forming a global Jacobian."""

        rhs = jnp.asarray(rhs)
        alpha = jnp.asarray(alpha, dtype=rhs.dtype)
        inverse_dtau = jnp.where(
            jnp.isfinite(jnp.asarray(dtau)),
            1.0 / jnp.asarray(dtau, dtype=rhs.dtype),
            jnp.asarray(0.0, dtype=rhs.dtype),
        )
        result = jnp.zeros_like(rhs)
        for indices, low, strong in zip(
            self.indices, self.low_blocks, self.strong_blocks, strict=True
        ):
            matrix = (1.0 - alpha) * low + alpha * strong
            scale = jnp.maximum(jnp.linalg.norm(matrix, ord=jnp.inf), 1.0)
            regularization = 32.0 * jnp.finfo(rhs.dtype).eps * scale
            shifted = matrix + (
                inverse_dtau + regularization
            ) * jnp.eye(matrix.shape[0], dtype=rhs.dtype)
            result = result.at[indices].set(jnp.linalg.solve(shifted, rhs[indices]))
        return result


def _mode_block_indices(
    runtime: StrongRootRuntime,
    *,
    poloidal_bandwidth: int = 3,
) -> tuple[jax.Array, ...]:
    """Group reduced coordinates into bounded neighboring-mode bands."""

    layout = runtime.layout
    m = np.asarray(runtime.native.m, dtype=int)
    n = np.asarray(runtime.native.n, dtype=int)
    mnmax = int(layout.mnmax)
    mode_columns = [
        *(np.asarray(layout.r_indices, dtype=int) % mnmax),
        *(np.asarray(layout.z_indices, dtype=int)[:, 0] % mnmax),
        *(np.asarray(layout.l_indices, dtype=int) % mnmax),
    ]
    groups: dict[tuple[int, int], list[int]] = {}
    for position, mode in enumerate(mode_columns):
        key = (abs(int(n[mode])), int(m[mode]) // int(poloidal_bandwidth))
        groups.setdefault(key, []).append(position)
    return tuple(
        jnp.asarray(groups[key], dtype=jnp.int32)
        for key in sorted(groups)
    )


def _build_mode_block_preconditioner(
    runtime: StrongRootRuntime,
) -> _ModeBlockPreconditioner:
    """Probe only same-mode Jacobian blocks at the lifted branch point."""

    started = perf_counter()
    zero = jnp.zeros(
        (runtime.layout.size,), dtype=jnp.asarray(runtime.native.R_cos).dtype
    )
    indices = _mode_block_indices(runtime)
    low_blocks: list[jax.Array] = []
    strong_blocks: list[jax.Array] = []
    for block_indices in indices:
        local_zero = jnp.zeros((block_indices.size,), dtype=zero.dtype)

        def block_residual(local, alpha):
            vector = zero.at[block_indices].set(local)
            return strong_root_residual(vector, runtime, alpha)[block_indices]

        low_blocks.append(
            jax.jacfwd(lambda local: block_residual(local, 0.0))(local_zero)
        )
        strong_blocks.append(
            jax.jacfwd(lambda local: block_residual(local, 1.0))(local_zero)
        )
    jax.block_until_ready((low_blocks, strong_blocks))
    return _ModeBlockPreconditioner(
        indices,
        tuple(low_blocks),
        tuple(strong_blocks),
        perf_counter() - started,
    )


def _continuation_precondition(
    rhs: jax.Array,
    alpha: jax.Array,
    dtau: jax.Array,
    runtime: StrongRootRuntime,
    block_preconditioner: _ModeBlockPreconditioner,
) -> jax.Array:
    """Use the exact legacy inverse early and mode bands near strong force."""

    return jax.lax.cond(
        jnp.asarray(alpha) < 0.5,
        lambda value: _low_inverse(value, runtime),
        lambda value: block_preconditioner.apply(value, alpha, dtau),
        rhs,
    )


def _corrected_state(vector: jax.Array, runtime: StrongRootRuntime):
    low = runtime.layout.unpack(vector)
    correction = runtime.transfer.prolong(low)
    return apply_high_order_correction(runtime.native, correction)


def _minimum_signed_jacobian(vector: jax.Array, runtime: StrongRootRuntime) -> jax.Array:
    state = _corrected_state(vector, runtime)
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

    low_rhs = runtime.layout.unpack(rhs)
    solution = runtime.low_preconditioner.solve_scaled(low_rhs)
    return runtime.layout.pack(solution)


def _ptc_config(config: PolishConfig) -> Any:
    _, PseudoTransientConfig, _, _, _ = _solvax_continuation_api()
    return PseudoTransientConfig(
        rtol=config.tolerance,
        atol=config.tolerance,
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
    block_preconditioner: _ModeBlockPreconditioner | None = None,
) -> tuple[jax.Array, jax.Array]:
    residual = lambda value: strong_root_residual(value, runtime, alpha)  # noqa: E731
    _, jvp = jax.linearize(residual, vector)
    parameter_direction = strong_root_residual(
        vector, runtime, 1.0
    ) - strong_root_residual(vector, runtime, 0.0)
    if previous is None:
        linear = gmres(
            jvp,
            -parameter_direction,
            precond=(
                (lambda value: _low_inverse(value, runtime))
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
                runtime, previous, block_preconditioner
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
    block_preconditioner: _ModeBlockPreconditioner | None = None,
):
    """Return a low-order block-elimination preconditioner for a bordered root."""

    tangent_x, tangent_alpha = tangent

    def apply(state, rhs, dtau):
        vector, alpha = state
        rhs_x, rhs_alpha = rhs
        parameter_direction = strong_root_residual(
            vector, runtime, 1.0
        ) - strong_root_residual(vector, runtime, 0.0)
        inverse = (
            (lambda rhs: _low_inverse(rhs, runtime))
            if block_preconditioner is None
            else lambda rhs: block_preconditioner.apply(rhs, alpha, dtau)
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

    return apply


def _arclength_to_target(
    vector: jax.Array,
    alpha: float,
    runtime: StrongRootRuntime,
    config: PolishConfig,
    admissible,
    block_preconditioner: _ModeBlockPreconditioner | None,
    initial_tangent: tuple[jax.Array, jax.Array] | None,
):
    _, _, _, pseudo_arclength_corrector, pseudo_transient_continuation = (
        _solvax_continuation_api()
    )
    tangent = (
        _branch_tangent(vector, alpha, runtime, config, None, block_preconditioner)
        if initial_tangent is None
        else initial_tangent
    )
    nonlinear = _ptc_config(config)
    total_nonlinear = total_linear = total_evaluations = 0
    for step in range(config.max_arclength_steps):
        predictor = (
            vector + config.arclength_step * tangent[0],
            jnp.asarray(alpha) + config.arclength_step * tangent[1],
        )
        corrector_kwargs = (
            {
                "precond": _bordered_preconditioner(
                    runtime, tangent, block_preconditioner
                )
            }
            if _supports_keyword(pseudo_arclength_corrector, "precond")
            else {}
        )
        corrected = pseudo_arclength_corrector(
            lambda value, parameter: strong_root_residual(
                value, runtime, parameter
            ),
            predictor,
            tangent=tangent,
            predictor=predictor,
            config=nonlinear,
            admissible=lambda value, parameter: admissible(value, parameter),
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
                lambda value: strong_root_residual(value, runtime, 1.0),
                vector,
                precond=(
                    (lambda state, rhs, dtau: _low_inverse(rhs, runtime))
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
) -> PolishResult:
    """Follow the legacy-connected fixed-boundary branch to strong force."""

    config = PolishConfig() if config is None else config
    started = perf_counter()
    initial_certificate = (
        certify_strong_force(runtime.native)
        if initial_certificate is None
        else initial_certificate
    )
    zero = jnp.zeros((runtime.layout.size,), dtype=jnp.asarray(runtime.native.R_cos).dtype)
    if float(initial_certificate.normalized_l2) <= config.certificate_tolerance:
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
        )
        return PolishResult(runtime.native, initial_certificate, report, zero)
    _, _, adaptive_continuation, _, pseudo_transient_continuation = (
        _solvax_continuation_api()
    )
    initial_margin = float(_minimum_signed_jacobian(zero, runtime))
    block_preconditioner = (
        _build_mode_block_preconditioner(runtime)
        if config.preconditioner == "mode-block"
        else None
    )
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
        residual = strong_root_residual(vector, runtime, 1.0)
        return (
            jnp.all(jnp.isfinite(vector))
            & jnp.all(jnp.isfinite(residual))
            & (_minimum_signed_jacobian(vector, runtime) >= margin_floor)
        )

    nonlinear = _ptc_config(config)
    precondition = lambda state, rhs, dtau: _low_inverse(rhs, runtime)  # noqa: E731
    endpoint = pseudo_transient_continuation(
        lambda vector: strong_root_residual(vector, runtime, 0.0),
        zero,
        precond=precondition,
        admissible=lambda vector: admissible(vector, 0.0),
        config=nonlinear,
    )
    nonlinear_iterations = int(endpoint.steps)
    linear_iterations = int(endpoint.linear_iterations)
    residual_evaluations = _residual_evaluations(endpoint)
    steps: tuple[Any, ...] = ()
    arclength_steps = 0
    vector = endpoint.x
    alpha = 0.0
    converged = bool(endpoint.converged) and bool(endpoint.linear_converged)
    reason = "alpha-zero-failed"
    if converged:
        accepted_states: list[tuple[jax.Array, float]] = [(vector, 0.0)]

        def record_accepted_state(candidate, parameter, solution):
            del solution
            accepted_states.append((candidate, float(parameter)))
            return True

        continuation_preconditioners = (
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
                    )
                )
            }
        )
        continuation = adaptive_continuation(
            lambda value, parameter: strong_root_residual(
                value, runtime, parameter
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
            minimum_signed_jacobian=float(_minimum_signed_jacobian(vector, runtime)),
            factor_build_seconds=factor_build_seconds,
            solve_seconds=perf_counter() - started,
        )
        if config.fail_policy == "raise":
            raise StrongForceContinuationError(
                "strong-force continuation did not reach alpha=1",
                hint="inspect the continuation report and refine the radial representation",
                alpha=float(alpha),
                residual_norm=float(jnp.linalg.norm(strong_root_residual(vector, runtime, alpha))),
                nonlinear_iterations=nonlinear_iterations,
                linear_iterations=linear_iterations,
                accepted_stages=accepted,
                rejected_stages=rejected,
            )
        return PolishResult(runtime.native, initial_certificate, report, zero)

    state = _corrected_state(vector, runtime)
    certificate = certify_strong_force(state)
    certified = float(certificate.normalized_l2) <= config.certificate_tolerance
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
    )
    if not certified and config.fail_policy == "raise":
        raise StrongForceCertificationError(
            "strong root failed the independent force certificate",
            hint="increase radial degree/resolution and retry once",
            normalized_l2=float(certificate.normalized_l2),
            tolerance=config.certificate_tolerance,
        )
    if not certified:
        return PolishResult(runtime.native, initial_certificate, report, zero)
    return PolishResult(state, certificate, report, vector)


def polish_legacy_solution(
    source,
    resolution,
    legacy_state,
    *,
    config: PolishConfig | None = None,
    lconm1: bool = True,
) -> PolishResult:
    """Refine and lift one converged legacy solve, then run the strong driver."""

    from . import implicit
    from .input import VmecInput
    from .strong_force import lift_high_order_state

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
    dof_mask = implicit._dof_mask(legacy_state, legacy_runtime, implicit_config)
    refined_state = implicit._refined_state(
        implicit_config,
        params,
        legacy_state,
        dof_mask,
    )
    native = lift_high_order_state(
        refined_state,
        legacy_runtime,
        degree=config.radial_degree,
    )
    low_preconditioner = build_low_order_preconditioner(
        native,
        params,
        implicit_config,
        refined_state,
        dof_mask,
    )
    runtime = make_strong_root_runtime(native, low_preconditioner, dof_mask)
    return polish_strong_root(runtime, config=config)


__all__ = [
    "PolishConfig",
    "PolishReport",
    "PolishResult",
    "polish_legacy_solution",
    "polish_strong_root",
]
