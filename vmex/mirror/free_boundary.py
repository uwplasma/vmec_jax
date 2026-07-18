"""Coupled plasma-boundary-vacuum solves for straight-axis mirrors."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.sparse.linalg import LinearOperator

from .forces import (
    MU0,
    InterfaceResidual,
    IsotropicForceResidual,
    MirrorEnergy,
    VariationalResidual,
    interface_residual,
    isotropic_force_residual,
    isotropic_staggered_fixed_boundary_gradient,
    isotropic_staggered_weak_residual,
    mass_profile_from_pressure,
    mirror_energy,
)
from .geometry import normalized_divergence_rms
from .exterior import (
    ExteriorVacuum,
    solve_axisymmetric_exterior_vacuum,
)
from .model import MirrorBoundary, MirrorConfig, MirrorState
from .output import FreeBoundaryRestart
from .solver import _bounded_newton_krylov
from .splines import (
    SplineMirrorBoundary,
    SplineMirrorDiscretization,
    SplineMirrorState,
    _SplineStateVectorizer,
    _packed_spline_preconditioner,
)

Array = Any
_DENSE_JACOBIAN_MAX_SIZE = 32


@dataclass(frozen=True)
class FreeBoundaryParameters:
    """Differentiable physical controls for a free-boundary equilibrium."""

    external_field: Any
    axial_flux_derivative: Array
    mass_profile: Array
    current_derivative: Array


jax.tree_util.register_dataclass(
    FreeBoundaryParameters,
    data_fields=["external_field", "axial_flux_derivative", "mass_profile", "current_derivative"],
    meta_fields=[],
)


@dataclass(frozen=True, eq=False)
class _SplineFreeBoundaryVectorizer:
    """Compose free boundary coefficients with the shared spline state map."""

    base_boundary: SplineMirrorBoundary
    state_vectorizer: _SplineStateVectorizer
    discretization: SplineMirrorDiscretization
    boundary_indices: tuple[np.ndarray, np.ndarray]
    boundary_scale: float
    calibrate_pressure: bool
    initial_mass_scale: float

    @classmethod
    def build(
        cls,
        boundary: SplineMirrorBoundary,
        state: SplineMirrorState,
        discretization: SplineMirrorDiscretization,
        *,
        axial_flux_derivative: Array,
        solve_lambda: bool,
        calibrate_pressure: bool = False,
        initial_mass_scale: float = 1.0,
    ) -> "_SplineFreeBoundaryVectorizer":
        """Build the square free-equilibrium coefficient layout."""

        coefficients = np.asarray(boundary.radius_coefficients, dtype=float)
        expected = (discretization.grid.ntheta, discretization.coefficient_count)
        if coefficients.shape != expected:
            raise ValueError(f"boundary coefficient shape {coefficients.shape} must be {expected}")
        evaluated = np.asarray(discretization.evaluate_boundary(boundary).radius_scale)
        boundary_scale = float(np.mean(evaluated))
        if not np.isfinite(boundary_scale) or boundary_scale <= 0.0:
            raise ValueError("mean boundary radius must be positive and finite")
        initial_mass_scale = float(initial_mass_scale)
        if not np.isfinite(initial_mass_scale) or initial_mass_scale <= 0.0:
            raise ValueError("initial_mass_scale must be positive and finite")

        mask = np.zeros(expected, dtype=bool)
        mask[:, 1:-1] = True
        state_vectorizer = _SplineStateVectorizer.build(
            state,
            boundary,
            discretization,
            axial_flux_derivative=axial_flux_derivative,
            solve_lambda=solve_lambda,
        )
        return cls(
            base_boundary=boundary,
            state_vectorizer=state_vectorizer,
            discretization=discretization,
            boundary_indices=tuple(np.asarray(index) for index in np.nonzero(mask)),
            boundary_scale=boundary_scale,
            calibrate_pressure=bool(calibrate_pressure),
            initial_mass_scale=initial_mass_scale,
        )

    @property
    def boundary_size(self) -> int:
        """Number of free lateral-boundary coefficients."""

        return int(self.boundary_indices[0].size)

    @property
    def state_size(self) -> int:
        """Number of active interior geometry and stream coefficients."""

        return self.state_vectorizer.radius_size + self.state_vectorizer.lambda_size

    @property
    def size(self) -> int:
        """Total nonlinear unknown count, including optional pressure scale."""

        return self.boundary_size + self.state_size + int(self.calibrate_pressure)

    def pack(self) -> np.ndarray:
        """Pack boundary, plasma, and optional mass scale."""

        boundary = np.asarray(self.base_boundary.radius_coefficients)[self.boundary_indices]
        parts = [boundary / self.boundary_scale, self.state_vectorizer.pack()]
        if self.calibrate_pressure:
            parts.append(np.asarray([self.initial_mass_scale]))
        return np.concatenate(parts)

    def unpack(self, vector: Array) -> tuple[SplineMirrorBoundary, SplineMirrorState, Array]:
        """Reconstruct a constrained coefficient boundary and plasma state."""

        vector = jnp.asarray(vector)
        if vector.ndim != 1 or vector.shape[0] != self.size:
            raise ValueError(f"free-boundary vector shape {vector.shape} must be ({self.size},)")
        indices = tuple(jnp.asarray(index) for index in self.boundary_indices)
        boundary_coefficients = (
            jnp.asarray(self.base_boundary.radius_coefficients)
            .at[indices]
            .set(vector[: self.boundary_size] * self.boundary_scale)
        )
        boundary = SplineMirrorBoundary(boundary_coefficients)
        start, stop = self.boundary_size, self.boundary_size + self.state_size
        state = self.state_vectorizer.unpack(vector[start:stop])
        state = self.discretization.project_fixed_boundary(state, boundary)
        mass_scale = vector[-1] if self.calibrate_pressure else jnp.asarray(1.0, dtype=vector.dtype)
        return boundary, state, mass_scale

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return normalized positivity and stream-function bounds."""

        state_lower, state_upper = self.state_vectorizer.bounds()
        lower = [np.full(self.boundary_size, 0.2), state_lower]
        upper = [np.full(self.boundary_size, np.inf), state_upper]
        if self.calibrate_pressure:
            lower.append(np.asarray([np.finfo(float).tiny]))
            upper.append(np.asarray([np.inf]))
        return np.concatenate(lower), np.concatenate(upper)

    def boundary_work_residual(
        self,
        boundary: SplineMirrorBoundary,
        jump: Array,
        stress_scale: Array,
    ) -> Array:
        """Pull normalized lateral virtual work to free boundary coefficients."""

        normalized = _spline_boundary_work_residual(
            boundary,
            jump,
            stress_scale,
            self.discretization,
        )
        return normalized[self.boundary_indices]


@dataclass(frozen=True, eq=False)
class _FreeEquilibriumProblem:
    """One coefficient residual shared by the free-boundary solver."""

    vectorizer: _SplineFreeBoundaryVectorizer
    residual_function: Callable[[Array], Array]
    components_function: Callable[[Array], tuple[Any, ...]]
    parameterized_residual_function: Callable[[Array, FreeBoundaryParameters], Array]
    parameterized_components_function: Callable[[Array, FreeBoundaryParameters], tuple[Any, ...]]
    plasma_scale: float

    @property
    def size(self) -> int:
        return self.vectorizer.size

    def residual(self, vector: Array) -> Array:
        return self.residual_function(vector)

    def preconditioner(self) -> Callable[[Array], Array]:
        """Return the shared boundary/interior block preconditioner."""

        boundary_size = self.vectorizer.boundary_size
        state_size = self.vectorizer.state_size
        state_preconditioner, _, _ = _packed_spline_preconditioner(
            self.vectorizer.discretization,
            self.vectorizer.state_vectorizer,
        )

        def apply(vector: Array) -> Array:
            host = isinstance(vector, np.ndarray)
            output = np.array(vector, copy=True) if host else jnp.asarray(vector)
            active = slice(boundary_size, boundary_size + state_size)
            state = state_preconditioner(output[active])
            if host:
                output[active] = state
                return output
            return output.at[active].set(state)

        return apply

    def linear_operator(self, vector: np.ndarray) -> LinearOperator:
        """Return repeated exact JVP and VJP residual actions."""

        point = jnp.asarray(vector)
        residual = self.residual_function
        residual_size = int(residual(point).size)
        if residual_size != self.size:
            raise ValueError(f"free residual size {residual_size} must equal state size {self.size}")

        def matvec(direction: np.ndarray) -> np.ndarray:
            tangent = jnp.asarray(direction).reshape(-1)
            return np.asarray(jax.jvp(residual, (point,), (tangent,))[1], dtype=float)

        def rmatvec(cotangent: np.ndarray) -> np.ndarray:
            cotangent = jnp.asarray(cotangent).reshape(-1)
            return np.asarray(jax.vjp(residual, point)[1](cotangent)[0], dtype=float)

        return LinearOperator(
            (residual_size, self.size),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=np.dtype(vector.dtype),
        )

    def linear_action(self, vector: np.ndarray) -> Callable[[Array], Array]:
        """Return the JAX-native exact residual Jacobian action."""

        point = jnp.asarray(vector)
        return jax.jit(
            lambda direction: jax.jvp(
                self.residual_function,
                (point,),
                (direction,),
            )[1]
        )


def _spline_boundary_work(
    boundary: SplineMirrorBoundary,
    jump: Array,
    discretization: SplineMirrorDiscretization,
) -> Array:
    """Return physical interface work for every boundary coefficient.

    A radial coefficient variation moves the side wall along ``e_r``. For
    ``x=(a cos(theta), a sin(theta), z)``, its virtual-work measure is
    ``e_r dot (n dA) = a * dz/dxi * dtheta * dxi``. The endpoint coefficients
    are returned as well; the free map removes them from the solve.
    """

    evaluated = discretization.evaluate_boundary(boundary).radius_scale
    jump = jnp.asarray(jump)
    expected = (discretization.grid.ntheta, discretization.grid.nxi)
    if jump.shape != expected:
        raise ValueError(f"interface jump shape {jump.shape} must be {expected}")
    weights = (
        jnp.asarray(discretization.grid.theta_basis.weights)[:, None]
        * jnp.asarray(discretization.grid.axial_basis.weights)[None, :]
    )
    radial_measure = evaluated * abs(float(discretization.grid.dz_dxi)) * weights
    return jnp.einsum(
        "jk,kc->jc",
        jump * radial_measure,
        jnp.asarray(discretization.evaluation_matrix),
    )


def _spline_boundary_work_residual(
    boundary: SplineMirrorBoundary,
    jump: Array,
    stress_scale: Array,
    discretization: SplineMirrorDiscretization,
) -> Array:
    """Normalize coefficient boundary work without changing its zeros."""

    jump = jnp.asarray(jump)
    stress_scale = jnp.broadcast_to(jnp.asarray(stress_scale, dtype=jump.dtype), jump.shape)
    physical = _spline_boundary_work(boundary, jump, discretization)
    scale = _spline_boundary_work(boundary, jnp.abs(stress_scale), discretization)
    tiny = jnp.finfo(physical.dtype).tiny
    return physical / jnp.maximum(scale, tiny)


@dataclass(frozen=True)
class FreeBoundaryMirrorResult:
    """Joint plasma-boundary-vacuum equilibrium result."""

    coefficient_boundary: SplineMirrorBoundary
    coefficient_state: SplineMirrorState
    boundary: MirrorBoundary
    plasma_state: MirrorState
    plasma_energy: MirrorEnergy
    plasma_force: IsotropicForceResidual
    plasma_staggered_weak_force: VariationalResidual | None
    normalized_divergence_rms: Array
    plasma_b_squared: Array
    pressure: Array
    vacuum_geometry: "ClosedMirrorSurface"
    vacuum_field: ExteriorVacuum
    mass_scale: Array
    plasma_scale: float
    target_central_pressure: float | None
    interface: InterfaceResidual
    history: Array
    variational_max: Array
    iterations: int
    linear_iterations: int
    final_linear_residual: float
    converged: bool
    optimizer_success: bool
    message: str


jax.tree_util.register_dataclass(
    FreeBoundaryMirrorResult,
    data_fields=[
        field.name
        for field in fields(FreeBoundaryMirrorResult)
        if field.name
        not in {
            "iterations",
            "linear_iterations",
            "plasma_scale",
            "target_central_pressure",
            "converged",
            "message",
        }
    ],
    meta_fields=[
        field.name
        for field in fields(FreeBoundaryMirrorResult)
        if field.name
        in {
            "iterations",
            "linear_iterations",
            "plasma_scale",
            "target_central_pressure",
            "converged",
            "message",
        }
    ],
)


def _build_free_equilibrium_problem(
    initial_boundary: SplineMirrorBoundary,
    initial_state: SplineMirrorState,
    discretization: SplineMirrorDiscretization,
    external_field: Any,
    *,
    axial_flux_derivative: Array,
    mass_profile: Array,
    current_derivative: Array,
    solve_lambda: bool,
    gamma: float,
    target_central_pressure: float | None,
    initial_mass_scale: float,
    exterior_ntheta: int,
    exterior_order: int,
    exterior_spectral_side_density: bool,
    plasma_scale: float | None = None,
) -> _FreeEquilibriumProblem:
    """Assemble the square coefficient residual and its physical components."""

    grid = discretization.grid
    if grid.ntheta != 1:
        raise ValueError("free-boundary mirrors currently support only axisymmetric geometry")
    calibrate_pressure = target_central_pressure is not None
    vectorizer = _SplineFreeBoundaryVectorizer.build(
        initial_boundary,
        initial_state,
        discretization,
        axial_flux_derivative=axial_flux_derivative,
        solve_lambda=solve_lambda,
        calibrate_pressure=calibrate_pressure,
        initial_mass_scale=initial_mass_scale,
    )

    parameters = FreeBoundaryParameters(
        external_field=external_field,
        axial_flux_derivative=jnp.asarray(axial_flux_derivative),
        mass_profile=jnp.asarray(mass_profile),
        current_derivative=jnp.asarray(current_derivative),
    )

    def plasma_components(vector: Array, controls: FreeBoundaryParameters):
        coefficient_boundary, coefficient_state, mass_scale = vectorizer.unpack(vector)
        boundary = discretization.evaluate_boundary(coefficient_boundary)
        state = discretization.evaluate_state(coefficient_state)
        plasma = mirror_energy(
            state,
            grid,
            axial_flux_derivative=controls.axial_flux_derivative,
            mass_profile=controls.mass_profile * mass_scale,
            current_derivative=controls.current_derivative,
            gamma=gamma,
        )
        pressure = jnp.broadcast_to(plasma.pressure[:, None, None], plasma.b_squared.shape)
        return coefficient_boundary, coefficient_state, mass_scale, boundary, state, plasma, pressure

    initial_vector = jnp.asarray(vectorizer.pack())
    if plasma_scale is None:
        plasma_scale = max(abs(float(plasma_components(initial_vector, parameters)[5].total)), 1.0)
    else:
        plasma_scale = float(plasma_scale)
        if not np.isfinite(plasma_scale) or plasma_scale <= 0.0:
            raise ValueError("plasma_scale must be positive and finite")

    def normalized_plasma_energy(vector: Array, controls: FreeBoundaryParameters) -> Array:
        return plasma_components(vector, controls)[5].total / plasma_scale

    def parameterized_components(vector: Array, controls: FreeBoundaryParameters):
        coefficient_boundary, coefficient_state, mass_scale, boundary, state, plasma, pressure = plasma_components(
            vector, controls
        )
        vacuum_field = solve_axisymmetric_exterior_vacuum(
            boundary,
            plasma.field,
            grid,
            controls.external_field,
            axisymmetric_ntheta=exterior_ntheta,
            order=exterior_order,
            spectral_side_density=exterior_spectral_side_density,
        )
        return (
            coefficient_boundary,
            coefficient_state,
            mass_scale,
            boundary,
            state,
            plasma,
            pressure,
            vacuum_field.surface,
            vacuum_field,
        )

    def parameterized_residual(vector: Array, controls: FreeBoundaryParameters) -> Array:
        (
            coefficient_boundary,
            _,
            _,
            _,
            _,
            plasma,
            pressure,
            _,
            vacuum_field,
        ) = parameterized_components(vector, controls)
        start = vectorizer.boundary_size
        stop = start + vectorizer.state_size
        plasma_gradient = jax.grad(normalized_plasma_energy, argnums=0)(vector, controls)[start:stop]
        plasma_b_squared = plasma.b_squared[-1]
        vacuum_xyz = vacuum_field.lateral_field_xyz
        if grid.ntheta == 1:
            vacuum_xyz = vacuum_xyz[None]
        vacuum_b_squared = jnp.sum(vacuum_xyz**2, axis=-1)
        pressure = pressure[-1]
        jump = pressure + plasma_b_squared / (2.0 * MU0) - vacuum_b_squared / (2.0 * MU0)
        stress_scale = jnp.abs(pressure) + plasma_b_squared / (2.0 * MU0) + vacuum_b_squared / (2.0 * MU0)
        boundary_work = vectorizer.boundary_work_residual(coefficient_boundary, jump, stress_scale)
        residuals = [boundary_work, plasma_gradient]
        if calibrate_pressure:
            target = float(target_central_pressure)
            residuals.append(jnp.asarray([(plasma.pressure[0] - target) / target]))
        return jnp.concatenate(residuals)

    def components(vector: Array):
        return parameterized_components(vector, parameters)

    def residual_function(vector: Array) -> Array:
        return parameterized_residual(vector, parameters)

    residual = jax.jit(residual_function)
    problem = _FreeEquilibriumProblem(
        vectorizer=vectorizer,
        residual_function=residual,
        components_function=components,
        parameterized_residual_function=parameterized_residual,
        parameterized_components_function=parameterized_components,
        plasma_scale=plasma_scale,
    )
    if int(problem.residual(initial_vector).size) != problem.size:
        raise ValueError("free-boundary coefficient residual must be square")
    return problem


def solve_free_boundary(
    initial_boundary: SplineMirrorBoundary,
    discretization: SplineMirrorDiscretization,
    config: MirrorConfig,
    external_field: Any,
    *,
    axial_flux_derivative: Array,
    mass_profile: Array = 0.0,
    current_derivative: Array = 0.0,
    solve_lambda: bool | None = None,
    gamma: float = 5.0 / 3.0,
    initial_state: SplineMirrorState | None = None,
    target_central_pressure: float | None = None,
    initial_mass_scale: float = 1.0,
    exterior_ntheta: int = 40,
    exterior_order: int = 8,
    exterior_spectral_side_density: bool = False,
    require_convergence: bool = False,
) -> FreeBoundaryMirrorResult:
    """Solve an open free-boundary mirror in longitudinal B-spline coefficients."""

    grid = discretization.grid
    if grid.ntheta != 1:
        raise ValueError("free-boundary mirrors currently support only axisymmetric geometry")
    if grid.ns != config.resolution.ns or grid.ntheta != config.resolution.ntheta:
        raise ValueError("spline radial and poloidal resolution must match MirrorConfig")
    if not np.allclose(np.asarray(grid.xi), np.asarray(config.build_grid().xi), rtol=0.0, atol=2.0e-14):
        raise ValueError("free-boundary exterior panels require SplineMirrorDiscretization.build_cgl")
    if solve_lambda is None:
        solve_lambda = False
    if target_central_pressure is not None and target_central_pressure <= 0.0:
        raise ValueError("target_central_pressure must be positive")
    if initial_mass_scale <= 0.0:
        raise ValueError("initial_mass_scale must be positive")
    expected_boundary = (grid.ntheta, discretization.coefficient_count)
    if tuple(jnp.shape(initial_boundary.radius_coefficients)) != expected_boundary:
        raise ValueError(f"initial boundary coefficient shape must be {expected_boundary}")
    if initial_state is None:
        radius = jnp.broadcast_to(
            jnp.asarray(initial_boundary.radius_coefficients),
            (grid.ns,) + expected_boundary,
        )
        initial_state = SplineMirrorState(radius, jnp.zeros_like(radius))
    expected_state = (grid.ns,) + expected_boundary
    if (
        tuple(jnp.shape(initial_state.radius_coefficients)) != expected_state
        or tuple(jnp.shape(initial_state.lambda_coefficients)) != expected_state
    ):
        raise ValueError(f"initial state coefficient arrays must have shape {expected_state}")
    if not np.allclose(
        np.asarray(initial_state.radius_coefficients[-1]),
        np.asarray(initial_boundary.radius_coefficients),
    ):
        raise ValueError("initial_state boundary must match initial_boundary")

    problem = _build_free_equilibrium_problem(
        initial_boundary,
        initial_state,
        discretization,
        external_field,
        axial_flux_derivative=axial_flux_derivative,
        mass_profile=mass_profile,
        current_derivative=current_derivative,
        solve_lambda=bool(solve_lambda),
        gamma=gamma,
        target_central_pressure=target_central_pressure,
        initial_mass_scale=initial_mass_scale,
        exterior_ntheta=exterior_ntheta,
        exterior_order=exterior_order,
        exterior_spectral_side_density=exterior_spectral_side_density,
    )
    vectorizer = problem.vectorizer
    nb = vectorizer.boundary_size
    np_state = vectorizer.state_size
    x0 = vectorizer.pack()
    lower, upper = vectorizer.bounds()
    matrix_free = problem.size > _DENSE_JACOBIAN_MAX_SIZE
    if matrix_free:
        solver_jacobian = problem.linear_operator
    else:
        dense_jacobian = jax.jit(jax.jacfwd(problem.residual_function))

        def solver_jacobian(vector: np.ndarray) -> np.ndarray:
            return np.asarray(dense_jacobian(jnp.asarray(vector)), dtype=float)

    history: list[tuple[float, float, float, float, float]] = []
    last_recorded: np.ndarray | None = None

    def residual_host(vector: np.ndarray) -> np.ndarray:
        nonlocal last_recorded
        residual = np.asarray(problem.residual(jnp.asarray(vector)), dtype=float)
        if last_recorded is None or not np.array_equal(vector, last_recorded):
            history.append(
                (
                    float(len(history)),
                    float(np.sqrt(np.mean(residual[:nb] ** 2))),
                    float(np.sqrt(np.mean(residual[nb : nb + np_state] ** 2))),
                    0.0,
                    float(np.max(np.abs(residual))),
                )
            )
            last_recorded = np.array(vector, copy=True)
        return residual

    def jacobian_host(vector: np.ndarray) -> np.ndarray | LinearOperator:
        return solver_jacobian(vector)

    trust_region_limit = min(40, config.max_iterations) if matrix_free else config.max_iterations
    solve = least_squares(
        fun=residual_host,
        x0=x0,
        jac=jacobian_host,
        bounds=(lower, upper),
        method="trf",
        ftol=1.0e-14,
        xtol=1.0e-14,
        gtol=1.0e-14,
        x_scale=np.maximum(np.abs(x0), 1.0),
        tr_solver="lsmr" if matrix_free else None,
        tr_options=({"atol": 1.0e-6, "btol": 1.0e-6, "maxiter": min(12, problem.size)} if matrix_free else {}),
        max_nfev=trust_region_limit,
    )
    solution = np.asarray(solve.x)
    polish_steps = 0
    linear_iterations = 0
    final_linear_residual = np.nan
    polish_success = False
    polish_message = ""
    if matrix_free and np.max(np.abs(residual_host(solution))) > config.ftol:
        free_preconditioner = problem.preconditioner()
        (
            solution,
            polish_steps,
            linear_iterations,
            final_linear_residual,
            polish_success,
            polish_message,
        ) = _bounded_newton_krylov(
            solution,
            lambda vector: np.asarray(problem.residual(jnp.asarray(vector)), dtype=float),
            lambda vector, _residual: (
                problem.linear_action(vector),
                free_preconditioner,
            ),
            (lower, upper),
            ftol=config.ftol,
            max_steps=min(30, max(0, config.max_iterations - int(solve.nfev))),
            record_step=residual_host,
            restart=24,
            max_restarts=3,
            linear_rtol=lambda residual_max: min(
                1.0e-5,
                max(1.0e-10, 0.1 * residual_max),
            ),
        )

    (
        coefficient_boundary,
        coefficient_state,
        mass_scale,
        boundary,
        state,
        plasma,
        pressure,
        vacuum_geometry,
        vacuum_field,
    ) = problem.components_function(jnp.asarray(solution))
    plasma_b_squared_full = plasma.b_squared
    plasma_b_squared = plasma_b_squared_full[-1]
    if grid.ntheta == 1:
        vacuum_b_squared = jnp.sum(vacuum_field.lateral_field_xyz**2, axis=-1)[None, :]
        vacuum_b_normal = vacuum_field.lateral_b_normal[None, :]
    else:
        vacuum_b_squared = jnp.sum(vacuum_field.lateral_field_xyz**2, axis=-1)
        vacuum_b_normal = vacuum_field.lateral_b_normal
    compatibility_limit = 1.0e-6 if grid.ntheta == 1 else 2.0e-3
    vacuum_valid = (vacuum_field.neumann_result.compatibility_error <= compatibility_limit) & (
        vacuum_field.neumann_result.condition_number <= 1.0e8
    )
    active_axial_weights = jnp.asarray(grid.axial_basis.weights).at[jnp.asarray([0, grid.nxi - 1])].set(0.0)
    interface = interface_residual(
        pressure=pressure[-1],
        plasma_b_squared=plasma_b_squared,
        vacuum_b_squared=vacuum_b_squared,
        plasma_b_normal=jnp.zeros_like(plasma_b_squared),
        vacuum_b_normal=vacuum_b_normal,
        theta_weights=jnp.asarray(grid.theta_basis.weights),
        axial_weights=active_axial_weights,
    )
    final_residual = np.asarray(problem.residual(jnp.asarray(solution)), dtype=float)
    variational_max = float(np.max(np.abs(final_residual)))
    energy_kwargs = {
        "axial_flux_derivative": axial_flux_derivative,
        "mass_profile": jnp.asarray(mass_profile) * mass_scale,
        "current_derivative": current_derivative,
        "gamma": gamma,
    }
    plasma_force = isotropic_force_residual(
        plasma,
        grid,
        state=state,
        **energy_kwargs,
    )
    full_weak_force = isotropic_staggered_weak_residual(
        state,
        boundary,
        grid,
        **energy_kwargs,
    )
    weak_gradient = isotropic_staggered_fixed_boundary_gradient(
        state,
        boundary,
        grid,
        **energy_kwargs,
    )
    active_weak = vectorizer.state_vectorizer.pullback_evaluated_gradient(weak_gradient) / problem.plasma_scale
    radius_weak = active_weak[: vectorizer.state_vectorizer.radius_size]
    lambda_weak = active_weak[vectorizer.state_vectorizer.radius_size :]
    plasma_staggered_weak_force = VariationalResidual(
        radius_gradient=full_weak_force.radius_gradient,
        lambda_gradient=full_weak_force.lambda_gradient,
        radius_rms=jnp.asarray(np.sqrt(np.mean(radius_weak**2))),
        lambda_rms=jnp.asarray(np.sqrt(np.mean(lambda_weak**2)) if lambda_weak.size else 0.0),
        maximum=jnp.asarray(np.max(np.abs(active_weak))),
    )
    divergence_rms = normalized_divergence_rms(plasma.field, plasma.geometry, grid)
    converged = bool(
        variational_max <= config.ftol and not bool(plasma.geometry.jacobian_sign_changed) and bool(vacuum_valid)
    )
    message = str(solve.message)
    if polish_message:
        message += f"; {polish_message}"
    if not converged:
        message += (
            f"; variational force={variational_max:.3e}"
            f"; crossed surfaces={bool(plasma.geometry.jacobian_sign_changed)}"
            f"; exterior compatibility="
            f"{float(vacuum_field.neumann_result.compatibility_error):.3e}"
            f"; raw compatibility="
            f"{float(vacuum_field.neumann_result.raw_compatibility_error):.3e}"
            f"; exterior condition="
            f"{float(vacuum_field.neumann_result.condition_number):.3e}"
        )
    result = FreeBoundaryMirrorResult(
        coefficient_boundary=coefficient_boundary,
        coefficient_state=coefficient_state,
        boundary=boundary,
        plasma_state=state,
        plasma_energy=plasma,
        plasma_force=plasma_force,
        plasma_staggered_weak_force=plasma_staggered_weak_force,
        normalized_divergence_rms=divergence_rms,
        plasma_b_squared=plasma_b_squared_full,
        pressure=pressure,
        vacuum_geometry=vacuum_geometry,
        vacuum_field=vacuum_field,
        mass_scale=mass_scale,
        plasma_scale=problem.plasma_scale,
        target_central_pressure=target_central_pressure,
        interface=interface,
        history=jnp.asarray(history),
        variational_max=jnp.asarray(variational_max),
        iterations=int(solve.nfev) + polish_steps,
        linear_iterations=linear_iterations,
        final_linear_residual=final_linear_residual,
        converged=converged,
        optimizer_success=bool(solve.success) or polish_success,
        message=message,
    )
    if require_convergence and not converged:
        raise RuntimeError(message)
    return result


def _axisymmetric_flux_initialization(
    boundary: SplineMirrorBoundary,
    discretization: SplineMirrorDiscretization,
    external_field: Callable[[Array], Array],
    axial_flux_derivative: Array,
) -> tuple[SplineMirrorBoundary, SplineMirrorState]:
    """Trace nested axisymmetric vacuum-flux surfaces for a robust CLI start."""

    grid, spline = discretization.grid, discretization.spline
    flux = float(np.asarray(axial_flux_derivative))
    nodes, weights = np.polynomial.legendre.leggauss(16)
    radial_nodes, radial_weights = 0.5 * (nodes + 1.0), 0.5 * weights
    xi = np.asarray(spline.collocation_nodes)
    z = 0.5 * (grid.z[0] + grid.z[-1]) + grid.dz_dxi * xi
    outer = np.asarray(spline.evaluate(boundary.radius_coefficients[0], xi))
    scales = np.empty((grid.ns, xi.size))
    scales[0] = outer

    def enclosed(radius: float, axial: float) -> float:
        points = np.column_stack((radius * radial_nodes, np.zeros(nodes.size), np.full(nodes.size, axial)))
        field = np.asarray(external_field(jnp.asarray(points)))[:, 2]
        return float(radius**2 * np.sum(radial_weights * radial_nodes * field))

    for radial_index, s in enumerate(np.asarray(grid.s)[1:], start=1):
        root_s = np.sqrt(s)
        for axial_index, (axial, outer_radius) in enumerate(zip(z, outer, strict=True)):
            guess = root_s * outer_radius
            upper = 2.0 * guess
            while enclosed(upper, axial) < s * flux:
                upper *= 2.0
                if upper > 16.0 * guess:
                    raise ValueError("failed to bracket the supplied-field flux surface")
            scales[radial_index, axial_index] = brentq(
                lambda radius: enclosed(radius, axial) - s * flux,
                0.0,
                upper,
                xtol=1.0e-13,
                rtol=1.0e-13,
            ) / root_s
    coefficients = jnp.asarray(spline.fit(scales, axis=-1))[:, None]
    return SplineMirrorBoundary(coefficients[-1]), SplineMirrorState(coefficients, jnp.zeros_like(coefficients))


def solve_beta_scan(
    initial_boundary: SplineMirrorBoundary,
    discretization: SplineMirrorDiscretization,
    config: MirrorConfig,
    external_field: Any,
    beta_values: Array,
    *,
    axial_flux_derivative: Array,
    reference_field: float,
    current_derivative: Array = 0.0,
    gamma: float = 5.0 / 3.0,
    beta_rtol: float = 1.0e-8,
    initial_state: SplineMirrorState | None = None,
    initial_restart: FreeBoundaryRestart | None = None,
    exterior_ntheta: int = 40,
    exterior_order: int = 8,
    exterior_spectral_side_density: bool = False,
) -> tuple[FreeBoundaryMirrorResult, ...]:
    """Continue one coefficient-native free-boundary state through increasing beta."""

    beta_values = np.asarray(beta_values, dtype=float)
    if beta_values.ndim != 1 or beta_values.size < 1:
        raise ValueError("beta_values must be a nonempty one-dimensional array")
    if np.any(beta_values < 0.0) or np.any(np.diff(beta_values) < 0.0):
        raise ValueError("beta_values must be nonnegative and increasing")
    if beta_rtol <= 0.0:
        raise ValueError("beta_rtol must be positive")
    grid = discretization.grid
    if initial_state is not None and initial_restart is not None:
        raise ValueError("initial_state and initial_restart are mutually exclusive")
    if initial_state is None and callable(external_field):
        initial_boundary, reference_coefficients = _axisymmetric_flux_initialization(
            initial_boundary,
            discretization,
            external_field,
            axial_flux_derivative,
        )
    elif initial_state is None:
        reference_radius = jnp.broadcast_to(
            jnp.asarray(initial_boundary.radius_coefficients),
            (grid.ns, grid.ntheta, discretization.coefficient_count),
        )
        reference_coefficients = SplineMirrorState(reference_radius, jnp.zeros_like(reference_radius))
    else:
        reference_coefficients = discretization.project_fixed_boundary(initial_state, initial_boundary)
    reference_state = discretization.evaluate_state(reference_coefficients)
    reference_energy = mirror_energy(
        reference_state,
        grid,
        axial_flux_derivative=axial_flux_derivative,
        current_derivative=current_derivative,
    )
    pressure_shape = 1.0 - jnp.asarray(grid.s)
    boundary = initial_boundary if initial_restart is None else initial_restart.boundary
    state = reference_coefficients if initial_restart is None else initial_restart.plasma_state
    mass_scale = 1.0 if initial_restart is None else initial_restart.mass_scale
    results = []
    for beta in beta_values:
        central_pressure = float(beta) * float(reference_field) ** 2 / (2.0 * MU0)
        mass = mass_profile_from_pressure(
            central_pressure * pressure_shape,
            reference_energy.volume_derivative,
            gamma=gamma,
        )
        result = solve_free_boundary(
            boundary,
            discretization,
            config,
            external_field,
            axial_flux_derivative=axial_flux_derivative,
            current_derivative=current_derivative,
            mass_profile=mass,
            gamma=gamma,
            initial_state=state,
            initial_mass_scale=mass_scale,
            exterior_ntheta=exterior_ntheta,
            exterior_order=exterior_order,
            exterior_spectral_side_density=exterior_spectral_side_density,
            target_central_pressure=None if beta == 0.0 else central_pressure,
            require_convergence=True,
        )
        if beta > 0.0:
            center = int(np.argmin(np.abs(grid.z)))
            achieved_beta = 2.0 * MU0 * float(result.pressure[0, 0, center]) / float(reference_field) ** 2
            if abs(achieved_beta - float(beta)) / float(beta) > beta_rtol:
                raise RuntimeError(f"central beta did not reach rtol={beta_rtol:.3e}")
        results.append(result)
        boundary = result.coefficient_boundary
        state = result.coefficient_state
        mass_scale = float(result.mass_scale)
    return tuple(results)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .exterior import ClosedMirrorSurface
