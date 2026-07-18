"""Shared host optimization and preconditioning for mirror equilibria.

The coefficient fixed-boundary solver and the nodal free-boundary solver use
the same convergence contract: optimizer status never substitutes for a
normalized variational residual below ``MirrorConfig.ftol``. Open systems use
matrix-free Newton-GMRES at every size, with a bounded dense rescue for small
stalled systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares, minimize
from scipy.sparse.linalg import LinearOperator, gmres

from vmex.core.errors import MORE_ITER_FLAG, VmecError

from .forces import (
    IsotropicForceResidual,
    MirrorEnergy,
    VariationalResidual,
)
from .model import MirrorConfig, MirrorState

Array = Any


def _valid_energy_objective(energy: MirrorEnergy, energy_scale: float) -> Array:
    """Normalize energy and reject states with crossed flux surfaces."""

    value = energy.total / float(energy_scale)
    return jnp.where(energy.geometry.jacobian_sign_changed, jnp.inf, value)


@dataclass(frozen=True)
class MirrorSolveResult:
    """Solved state, diagnostics, and dense iteration history.

    History columns are ``iteration, total_energy, radius_variational_rms,
    lambda_variational_rms, variational_max``. The independently reconstructed
    pointwise force is stored in ``force`` and evaluated for the final state.
    ``optimizer_success`` is recorded separately and never substitutes for
    ``converged``.
    """

    state: MirrorState
    energy: MirrorEnergy
    variational: VariationalResidual
    force: IsotropicForceResidual
    staggered_weak_force: VariationalResidual | None
    normalized_divergence_rms: Array
    history: Array
    iterations: int
    converged: bool
    optimizer_success: bool
    linear_iterations: int
    final_linear_residual: float
    message: str


jax.tree_util.register_dataclass(
    MirrorSolveResult,
    data_fields=[
        "state",
        "energy",
        "variational",
        "force",
        "staggered_weak_force",
        "normalized_divergence_rms",
        "history",
    ],
    meta_fields=[
        "iterations",
        "converged",
        "optimizer_success",
        "linear_iterations",
        "final_linear_residual",
        "message",
    ],
)


class MirrorConvergenceError(VmecError):
    """Raised when a caller requires a physically converged mirror solve.

    Subclasses :class:`vmex.core.errors.VmecError` so mirror solve failures
    flow through the same zero-crash typed-error taxonomy as the core solver.
    """

    def __init__(self, result: MirrorSolveResult):
        self.result = result
        super().__init__(
            f"mirror solve did not reach ftol: variational force "
            f"{float(result.variational.maximum):.3e} after {result.iterations} iterations",
            ier_flag=MORE_ITER_FLAG,
        )


@dataclass(frozen=True)
class SeparableMirrorPreconditioner:
    """Tensor-product inverse for radial, poloidal, and axial stiffness.

    The model is a shifted radial/Fourier/CGL stiffness sum. Its inverse uses
    two small eigendecompositions and FFTs, not a dense 3D factorization.
    """

    radial_vectors: np.ndarray
    axial_vectors: np.ndarray
    denominator: np.ndarray
    active_shape: tuple[int, int, int]

    @classmethod
    def build(
        cls,
        grid: "MirrorGrid",
        *,
        radial_nodes: int | None = None,
        shift: float = 1.0e-3,
        radial_strength: float = 1.0,
        poloidal_strength: float = 1.0,
        axial_strength: float = 1.0,
    ) -> "SeparableMirrorPreconditioner":
        """Build the normalized separable stiffness inverse for ``grid``."""

        derivative = np.asarray(grid.axial_basis.derivative_matrix, dtype=np.float64)
        weights = np.asarray(grid.axial_basis.weights, dtype=np.float64)
        interior_derivative = derivative[:, 1:-1] / float(grid.dz_dxi)
        axial = interior_derivative.T @ (weights[:, None] * interior_derivative)
        return cls.build_from_axial_stiffness(
            grid,
            axial,
            radial_nodes=radial_nodes,
            shift=shift,
            radial_strength=radial_strength,
            poloidal_strength=poloidal_strength,
            axial_strength=axial_strength,
        )

    @classmethod
    def build_from_axial_stiffness(
        cls,
        grid: "MirrorGrid",
        axial_stiffness: Array,
        *,
        radial_nodes: int | None = None,
        poloidal_nodes: int | None = None,
        shift: float = 1.0e-3,
        radial_strength: float = 1.0,
        poloidal_strength: float = 1.0,
        axial_strength: float = 1.0,
    ) -> "SeparableMirrorPreconditioner":
        """Build the tensor inverse from a representation-specific axial block."""

        strengths = (radial_strength, poloidal_strength, axial_strength)
        if shift <= 0.0 or min(strengths) < 0.0:
            raise ValueError("shift must be positive and stiffness strengths nonnegative")
        nr = grid.ns - 2 if radial_nodes is None else int(radial_nodes)
        ntheta = grid.ntheta if poloidal_nodes is None else int(poloidal_nodes)
        axial = np.asarray(axial_stiffness, dtype=float)
        if axial.ndim != 2 or axial.shape[0] != axial.shape[1]:
            raise ValueError("axial stiffness must be a square matrix")
        nx = int(axial.shape[0])
        if nr < 1 or ntheta < 1 or nx < 1:
            raise ValueError("preconditioning requires interior radial and axial values")

        ds = float(grid.s[1] - grid.s[0])
        radial = np.diag(np.full(nr, 2.0 / ds**2))
        if nr > 1:
            off_diagonal = np.full(nr - 1, -1.0 / ds**2)
            radial += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)

        radial_values, radial_vectors = eigh(radial, check_finite=True)
        axial_values, axial_vectors = eigh(axial, check_finite=True)
        radial_values /= max(float(radial_values[-1]), np.finfo(float).tiny)
        axial_values /= max(float(axial_values[-1]), np.finfo(float).tiny)
        poloidal_values = np.fft.fftfreq(ntheta, d=1.0 / ntheta) ** 2
        if np.max(poloidal_values) > 0.0:
            poloidal_values /= np.max(poloidal_values)
        denominator = (
            float(shift)
            + float(radial_strength) * radial_values[:, None, None]
            + float(poloidal_strength) * poloidal_values[None, :, None]
            + float(axial_strength) * axial_values[None, None, :]
        )
        return cls(
            radial_vectors=radial_vectors,
            axial_vectors=axial_vectors,
            denominator=denominator,
            active_shape=(nr, ntheta, nx),
        )

    @property
    def size(self) -> int:
        """Number of active geometry unknowns."""
        return int(np.prod(self.active_shape))

    def _transform(self, vector: Array, *, inverse: bool) -> Array:
        xp = np if isinstance(vector, np.ndarray) else jnp
        values = xp.asarray(vector)
        if values.size != self.size:
            raise ValueError(f"vector has {values.size} values; expected {self.size}")
        blocks = values.reshape(self.active_shape)
        denominator = xp.asarray(self.denominator, dtype=values.dtype)
        multiplier = 1.0 / denominator if inverse else denominator
        radial = xp.asarray(self.radial_vectors, dtype=values.dtype)
        axial = xp.asarray(self.axial_vectors, dtype=values.dtype)
        coefficients = xp.einsum("ri,rtx,xj->itj", radial, blocks, axial)
        coefficients = xp.fft.fft(coefficients, axis=1, norm="ortho")
        coefficients = xp.fft.ifft(multiplier * coefficients, axis=1, norm="ortho").real
        result = xp.einsum("ri,itj,xj->rtx", radial, coefficients, axial)
        return result.reshape(values.shape)

    def apply(self, vector: Array) -> Array:
        """Apply the inverse model operator to a flattened residual."""
        return self._transform(vector, inverse=True)

    def operator(self, vector: Array) -> Array:
        """Apply the model operator, primarily for verification tests."""
        return self._transform(vector, inverse=False)

    def apply_gauge_free(
        self,
        vector: Array,
        *,
        free_indices: np.ndarray,
        pivot: int,
        weights: np.ndarray,
    ) -> Array:
        """Apply the inverse after eliminating one weighted-mean gauge node.

        The lift and orthogonal projection keep the reduced operator symmetric.
        """

        xp = np if isinstance(vector, np.ndarray) else jnp
        values = xp.asarray(vector)
        free_indices = xp.asarray(free_indices, dtype=int)
        weights = xp.asarray(weights, dtype=values.dtype)
        reduced = values.reshape(self.active_shape[0], free_indices.size)
        ratio = weights[free_indices] / weights[int(pivot)]
        gram = 1.0 + ratio @ ratio
        lifted_free = reduced - ((reduced @ ratio) / gram)[:, None] * ratio[None, :]
        lifted = xp.zeros((self.active_shape[0], weights.size), dtype=values.dtype)
        if xp is np:
            lifted[:, free_indices] = lifted_free
            lifted[:, int(pivot)] = -(lifted_free @ ratio)
        else:
            lifted = lifted.at[:, free_indices].set(lifted_free)
            lifted = lifted.at[:, int(pivot)].set(-(lifted_free @ ratio))

        solved = self.apply(lifted.reshape(-1)).reshape(lifted.shape)
        projected = solved[:, free_indices] - solved[:, int(pivot), None] * ratio[None, :]
        projected -= ((projected @ ratio) / gram)[:, None] * ratio[None, :]
        return projected.reshape(values.shape)


def _solve_krylov_system(
    matrix_vector: Any,
    right_hand_side: np.ndarray,
    apply_preconditioner: Any,
    *,
    rtol: float,
    restart: int,
    max_restarts: int,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, int, float, int]:
    """Solve one host GMRES system and verify its true relative residual."""

    right_hand_side = np.asarray(right_hand_side, dtype=float)
    size = right_hand_side.size
    operator = LinearOperator(
        (size, size),
        matvec=lambda vector: np.array(matrix_vector(vector), dtype=float, copy=True),
        dtype=float,
    )
    inverse = LinearOperator(
        (size, size),
        matvec=lambda vector: np.array(apply_preconditioner(vector), dtype=float, copy=True),
        dtype=float,
    )
    iterations = 0

    def count_iteration(_residual: float) -> None:
        nonlocal iterations
        iterations += 1

    solution, info = gmres(
        operator,
        right_hand_side,
        x0=initial,
        M=inverse,
        restart=min(int(restart), size),
        maxiter=int(max_restarts),
        rtol=float(rtol),
        atol=0.0,
        callback=count_iteration,
        callback_type="pr_norm",
    )
    error = np.asarray(matrix_vector(solution), dtype=float) - right_hand_side
    denominator = max(np.linalg.norm(right_hand_side), np.finfo(float).tiny)
    return np.asarray(solution), iterations, float(np.linalg.norm(error) / denominator), int(info)


def _bounded_newton_krylov(
    x0: np.ndarray,
    residual_function: Any,
    linear_model: Any,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    ftol: float,
    max_steps: int,
    record_step: Any,
    restart: int,
    max_restarts: int,
    linear_rtol: Any,
    objective_function: Any | None = None,
    fallback_direction: Any | None = None,
    after_step: Any | None = None,
) -> tuple[np.ndarray, int, int, float, bool, str]:
    """Run the shared bounded Newton-GMRES iteration for mirror residuals."""

    x = np.asarray(x0, dtype=float)
    lower, upper = (np.asarray(bound, dtype=float) for bound in bounds)
    linear_iterations = 0
    final_linear_residual = np.inf
    for step in range(max(0, int(max_steps))):
        residual = np.asarray(residual_function(x), dtype=float)
        residual_max = float(np.max(np.abs(residual)))
        if residual_max <= float(ftol):
            return x, step, linear_iterations, final_linear_residual, True, "Newton-GMRES converged"

        matrix_vector, apply_preconditioner = linear_model(x, residual)
        direction, iterations, final_linear_residual, info = _solve_krylov_system(
            matrix_vector,
            -residual,
            apply_preconditioner,
            rtol=float(linear_rtol(residual_max)),
            restart=restart,
            max_restarts=max_restarts,
        )
        linear_iterations += iterations
        if info < 0 or not np.all(np.isfinite(direction)):
            return x, step, linear_iterations, final_linear_residual, False, "GMRES breakdown"

        slope = float(np.dot(residual, direction))
        if objective_function is not None and slope >= 0.0 and fallback_direction is not None:
            direction = np.asarray(fallback_direction(residual), dtype=float)
            slope = float(np.dot(residual, direction))
        current_merit = (
            float(objective_function(x))
            if objective_function is not None
            else 0.5 * float(np.dot(residual, residual))
        )
        accepted = False
        step_length = 1.0
        for _ in range(24):
            candidate = np.clip(x + step_length * direction, lower, upper)
            if objective_function is None:
                candidate_residual = np.asarray(residual_function(candidate), dtype=float)
                candidate_merit = 0.5 * float(np.dot(candidate_residual, candidate_residual))
                accepted = np.isfinite(candidate_merit) and candidate_merit < current_merit
            else:
                candidate_merit = float(objective_function(candidate))
                accepted = np.isfinite(candidate_merit) and (
                    candidate_merit <= current_merit + 1.0e-4 * step_length * slope
                )
            if accepted:
                x = candidate
                record_step(x)
                break
            step_length *= 0.5
        retry_rejected = bool(after_step(accepted)) if after_step is not None else False
        if not accepted:
            if retry_rejected:
                continue
            return x, step + 1, linear_iterations, final_linear_residual, False, "Newton line search stalled"

    residual_max = float(np.max(np.abs(np.asarray(residual_function(x), dtype=float))))
    converged = residual_max <= float(ftol)
    message = "Newton-GMRES converged" if converged else "Newton-GMRES iteration limit"
    return x, int(max_steps), linear_iterations, final_linear_residual, converged, message


def _polish_fixed_coefficients(
    x0: np.ndarray,
    gradient_function: Any,
    objective_function: Any,
    vectorizer: Any,
    preconditioner: tuple[Any, np.ndarray, Any],
    *,
    ftol: float,
    max_steps: int,
    record_step: Any,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, int, int, float, bool, str]:
    """Damped Newton-GMRES polish using exact JAX Hessian products."""

    apply_preconditioner, block_scales, build_local = preconditioner
    hessian_vector = jax.jit(lambda point, direction: jax.jvp(gradient_function, (point,), (direction,))[1])
    hessian_columns = jax.jit(
        lambda point, directions: jax.vmap(
            lambda direction: jax.jvp(gradient_function, (point,), (direction,))[1]
        )(directions)
    )
    damping = [1.0e-8]
    local_preconditioner = None
    active_preconditioner = apply_preconditioner

    def residual(vector: np.ndarray) -> np.ndarray:
        return np.asarray(gradient_function(jnp.asarray(vector)), dtype=float)

    def linear_model(vector: np.ndarray, _residual: np.ndarray) -> tuple[Any, Any]:
        nonlocal active_preconditioner, local_preconditioner

        def matrix_vector(direction: Array) -> Array:
            product = hessian_vector(jnp.asarray(vector), jnp.asarray(direction))
            return product + damping[0] * jnp.asarray(direction)

        if local_preconditioner is None and build_local is not None:
            def matrix_columns(directions: np.ndarray) -> np.ndarray:
                products = hessian_columns(jnp.asarray(vector), jnp.asarray(directions))
                return np.asarray(products, dtype=float) + damping[0] * directions

            try:
                local_preconditioner = build_local(matrix_columns)
            except RuntimeError:
                local_preconditioner = False
        elif build_local is None:
            local_preconditioner = False

        if local_preconditioner is False:
            block_scales[:] = 1.0
            probe = np.random.default_rng(0).choice((-1.0, 1.0), size=vector.size)
            for block, active in enumerate(vectorizer.block_slices):
                direction = np.zeros_like(probe)
                direction[active] = probe[active]
                response = apply_preconditioner(matrix_vector(direction))
                denominator = abs(float(np.dot(direction, response)))
                if denominator > np.finfo(float).tiny:
                    block_scales[block] = np.clip(
                        np.dot(direction, direction) / denominator,
                        1.0e-8,
                        1.0e8,
                    )
            active_preconditioner = apply_preconditioner
        else:
            active_preconditioner = local_preconditioner
        return matrix_vector, active_preconditioner

    def after_step(accepted: bool) -> bool:
        damping[0] = max(1.0e-12, 0.3 * damping[0]) if accepted else 10.0 * damping[0]
        return not accepted and damping[0] <= 1.0

    return _bounded_newton_krylov(
        x0,
        residual,
        linear_model,
        (lower_bounds, upper_bounds),
        ftol=ftol,
        max_steps=max_steps,
        record_step=record_step,
        restart=200,
        max_restarts=10,
        linear_rtol=lambda residual_max: min(1.0e-8, max(1.0e-10, 0.1 * residual_max)),
        objective_function=lambda vector: objective_function(jnp.asarray(vector)),
        fallback_direction=lambda residual_value: -active_preconditioner(residual_value),
        after_step=after_step,
    )


@dataclass(frozen=True)
class _OptimizationOutcome:
    """Representation-independent result from the host nonlinear driver."""

    vector: np.ndarray
    iterations: int
    optimizer_success: bool
    linear_iterations: int
    final_linear_residual: float
    message: str


def _optimize_fixed_boundary(
    x0: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    objective: Any,
    evaluate: Any,
    packed_variational: Any,
    unpack: Any,
    record: Any,
    config: MirrorConfig,
    gradient_tolerance: float,
    matrix_free_context: tuple[Any, tuple[Any, np.ndarray, Any]] | None,
    start_with_newton: bool = False,
) -> _OptimizationOutcome:
    """Run the common host L-BFGS and residual-Newton solve policy.

    L-BFGS globalizes production-size states before the exact Newton polish.
    The dense residual solve is only a bounded rescue when matrix-free Newton
    misses ``ftol``.
    """

    callback_iterations = 0
    use_matrix_free = matrix_free_context is not None
    history_stride = 10 if use_matrix_free else 1

    def callback(x: np.ndarray) -> None:
        nonlocal callback_iterations
        callback_iterations += 1
        if callback_iterations == 1 or callback_iterations % history_stride == 0:
            record(callback_iterations, x)

    # Reserve iterations for Newton: L-BFGS can stall on relative energy while
    # physical forces are still large.
    polish_cap = 100 if x0.size > 2048 else 50
    if not use_matrix_free:
        polish_cap = 200
    polish_reserve = min(polish_cap, max(1, int(config.max_iterations) // 4))
    available = int(config.max_iterations) - polish_reserve
    lbfgs_budget = max(10, available // 2) if use_matrix_free else max(1, available)
    if use_matrix_free and bool(getattr(matrix_free_context[1][2], "reuse_linearization", False)):
        lbfgs_budget = min(50, lbfgs_budget)
    if start_with_newton:
        final_x = np.asarray(x0)
        optimizer_success = False
        optimizer_message = "local-basin Newton start"
        lbfgs_iterations = 0
    else:
        optimization = minimize(
            fun=lambda x: evaluate(x)[0],
            x0=x0,
            jac=lambda x: evaluate(x)[1],
            method="L-BFGS-B",
            bounds=list(zip(lower_bounds, upper_bounds, strict=True)),
            callback=callback,
            options={
                "maxiter": lbfgs_budget,
                "gtol": float(gradient_tolerance),
                "ftol": np.finfo(float).eps,
                "maxls": 50,
                "maxcor": 20,
            },
        )
        final_x = np.asarray(optimization.x)
        optimizer_success = bool(optimization.success)
        optimizer_message = str(optimization.message)
        lbfgs_iterations = int(optimization.nit)
    polish_evaluations = 0
    newton_steps = 0
    linear_iterations = 0
    final_linear_residual = 0.0

    candidate_variational = packed_variational(final_x, unpack(jnp.asarray(final_x)))
    gradient_function = jax.jit(jax.grad(objective))

    def run_matrix_free_polish() -> None:
        nonlocal final_x, optimizer_success, optimizer_message
        nonlocal newton_steps, linear_iterations, final_linear_residual
        assert matrix_free_context is not None
        remaining = max(
            1,
            int(config.max_iterations) - lbfgs_iterations - polish_evaluations - newton_steps,
        )

        def record_newton(x: np.ndarray) -> None:
            nonlocal newton_steps
            newton_steps += 1
            if newton_steps == 1 or newton_steps % history_stride == 0:
                record(callback_iterations + polish_evaluations + newton_steps, x)

        vectorizer, preconditioner = matrix_free_context
        (
            final_x,
            _attempted_newton_steps,
            linear_iterations,
            final_linear_residual,
            newton_success,
            newton_message,
        ) = _polish_fixed_coefficients(
            final_x,
            gradient_function,
            objective,
            vectorizer,
            preconditioner,
            ftol=config.ftol,
            max_steps=min(polish_reserve, remaining),
            record_step=record_newton,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        optimizer_success = bool(newton_success)
        optimizer_message += f"; {newton_message}"

    if float(candidate_variational.maximum) > config.ftol and use_matrix_free and matrix_free_context is not None:
        run_matrix_free_polish()
        candidate_variational = packed_variational(final_x, unpack(jnp.asarray(final_x)))

    # A bounded dense fallback is the robust reference lane up to 2048 dofs.
    if float(candidate_variational.maximum) > config.ftol and final_x.size <= 2048:
        hessian_function = jax.jit(jax.jacfwd(jax.grad(objective)))
        remaining = max(1, int(config.max_iterations) - lbfgs_iterations - newton_steps)
        polish = least_squares(
            fun=lambda x: np.asarray(gradient_function(jnp.asarray(x)), dtype=float),
            x0=final_x,
            jac=lambda x: np.asarray(hessian_function(jnp.asarray(x)), dtype=float),
            bounds=(lower_bounds, upper_bounds),
            method="trf",
            ftol=1.0e-14,
            xtol=1.0e-14,
            gtol=1.0e-14,
            x_scale="jac",
            max_nfev=remaining,
        )
        final_x = np.asarray(polish.x)
        polish_evaluations = int(polish.nfev)
        optimizer_success = bool(polish.success)
        optimizer_message += f"; residual-Newton: {polish.message}"
        candidate_variational = packed_variational(final_x, unpack(jnp.asarray(final_x)))

    return _OptimizationOutcome(
        vector=final_x,
        iterations=lbfgs_iterations + newton_steps + polish_evaluations,
        optimizer_success=optimizer_success,
        linear_iterations=linear_iterations,
        final_linear_residual=final_linear_residual,
        message=optimizer_message,
    )


if TYPE_CHECKING:
    from .basis import MirrorGrid
