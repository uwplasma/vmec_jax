"""Implicit coefficient-fixed and free-boundary mirror derivatives.

Derivatives use converged equilibrium equations, not host iteration histories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .forces import MirrorEnergy, mirror_energy
from .free_boundary import FreeBoundaryParameters, _build_free_equilibrium_problem
from .geometry import evaluate_closed_spline_axis, regularize_axis_stream_function
from .model import MirrorBoundary, MirrorState
from .solver import _solve_krylov_system

Array = Any
MirrorQuantity = Callable[[MirrorState, MirrorEnergy], Array]


@dataclass(frozen=True)
class SplineFixedBoundaryParameters:
    """Differentiable controls for a coefficient-native spline equilibrium."""

    boundary_coefficients: Array
    axis_coefficients: Array
    axial_flux_derivative: Array
    mass_profile: Array
    current_derivative: Array


@dataclass(frozen=True)
class MirrorAdjointResult:
    """Scalar value, total parameter gradient, and linear-solve diagnostics."""

    value: Array
    gradient: Any
    iterations: int
    relative_residual: float
    converged: bool
    linear_solver: str


@dataclass(frozen=True)
class MirrorTangentResult:
    """Equilibrium state tangent and linear-solve diagnostics."""

    tangent: MirrorState
    iterations: int
    relative_residual: float
    converged: bool


jax.tree_util.register_dataclass(
    SplineFixedBoundaryParameters,
    data_fields=[
        "boundary_coefficients",
        "axis_coefficients",
        "axial_flux_derivative",
        "mass_profile",
        "current_derivative",
    ],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    MirrorAdjointResult,
    data_fields=["value", "gradient"],
    meta_fields=["iterations", "relative_residual", "converged", "linear_solver"],
)
jax.tree_util.register_dataclass(
    MirrorTangentResult,
    data_fields=["tangent"],
    meta_fields=["iterations", "relative_residual", "converged"],
)


def spline_fixed_boundary_parameters(
    boundary: Any,
    *,
    axial_flux_derivative: Array,
    mass_profile: Array = 0.0,
    current_derivative: Array = 0.0,
    axis_coefficients: Array | None = None,
) -> SplineFixedBoundaryParameters:
    """Collect native boundary, axis, flux, pressure, and current controls."""

    return SplineFixedBoundaryParameters(
        boundary_coefficients=jnp.asarray(boundary.radius_coefficients),
        axis_coefficients=jnp.empty((0, 3))
        if axis_coefficients is None
        else jnp.asarray(axis_coefficients),
        axial_flux_derivative=jnp.asarray(axial_flux_derivative),
        mass_profile=jnp.asarray(mass_profile),
        current_derivative=jnp.asarray(current_derivative),
    )


def _solve_implicit_system(
    matrix_vector: Callable[[Array], Array],
    right_hand_side: np.ndarray,
    apply_preconditioner: Callable[[Array], Array],
    scales: np.ndarray,
    block_slices: tuple[slice, ...],
    *,
    rtol: float,
    max_restarts: int,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, int, float, bool]:
    """Solve one preconditioned implicit linear system on the host."""

    size = right_hand_side.size
    probe = np.random.default_rng(0).choice((-1.0, 1.0), size=size)
    for block, active in enumerate(block_slices):
        direction = np.zeros_like(probe)
        direction[active] = probe[active]
        response = apply_preconditioner(matrix_vector(jnp.asarray(direction)))
        denominator = abs(float(np.dot(direction, response)))
        if denominator > np.finfo(float).tiny:
            scales[block] = np.clip(np.dot(direction, direction) / denominator, 1.0e-8, 1.0e8)

    if initial is None:
        initial_relative_residual = np.inf
    else:
        initial_error = matrix_vector(initial) - right_hand_side
        initial_relative_residual = float(
            np.linalg.norm(initial_error) / max(np.linalg.norm(right_hand_side), np.finfo(float).tiny)
        )
    if initial_relative_residual <= rtol:
        solution = np.asarray(initial)
        iterations = 0
        relative_residual = initial_relative_residual
        info = 0
    else:
        solution, iterations, relative_residual, info = _solve_krylov_system(
            matrix_vector,
            right_hand_side,
            apply_preconditioner,
            initial=initial,
            restart=min(100, size),
            max_restarts=min(3, max_restarts) if initial is not None else max_restarts,
            rtol=rtol,
        )
    converged = bool(info == 0 and relative_residual <= max(10.0 * rtol, 1.0e-12))
    return solution, iterations, relative_residual, converged


def _implicit_adjoint(
    x_star: Array,
    parameters: Any,
    residual: Callable[[Array, Any], Array],
    evaluate_quantity: Callable[[Array, Any], Array],
    apply_preconditioner: Callable[[Array], Array],
    scales: np.ndarray,
    block_slices: tuple[slice, ...],
    *,
    rtol: float,
    max_restarts: int,
    initializer: Callable[[Callable[[Array], Array], np.ndarray], tuple[np.ndarray, str]] | None = None,
) -> MirrorAdjointResult:
    """Solve one transpose system shared by spline-fixed and free states."""

    value, (quantity_x, quantity_parameters) = jax.value_and_grad(evaluate_quantity, argnums=(0, 1))(x_star, parameters)
    _, transpose = jax.vjp(lambda x: residual(x, parameters), x_star)
    transpose_action = jax.jit(lambda vector: transpose(vector)[0])

    def matrix_vector(vector: Array) -> Array:
        return transpose_action(vector)

    right_hand_side = np.asarray(quantity_x, dtype=float)
    initial_adjoint, solver_used = (
        (None, "scipy-gmres") if initializer is None else initializer(transpose_action, right_hand_side)
    )
    adjoint, iterations, relative_residual, converged = _solve_implicit_system(
        matrix_vector,
        right_hand_side,
        apply_preconditioner,
        scales,
        block_slices,
        rtol=rtol,
        max_restarts=max_restarts,
        initial=initial_adjoint,
    )
    _, parameter_pullback = jax.vjp(lambda p: residual(x_star, p), parameters)
    residual_parameter_gradient = parameter_pullback(jnp.asarray(adjoint))[0]
    total_gradient = jax.tree.map(
        lambda direct, implicit: direct - implicit,
        quantity_parameters,
        residual_parameter_gradient,
    )
    return MirrorAdjointResult(
        value=value,
        gradient=total_gradient,
        iterations=iterations,
        relative_residual=relative_residual,
        converged=converged,
        linear_solver=solver_used,
    )


def _spline_implicit_problem(
    result: Any,
    parameters: SplineFixedBoundaryParameters,
    discretization: Any,
    *,
    gamma: float,
    solve_lambda: bool,
):
    """Build the converged spline residual and its packed linear model."""

    from .splines import (
        SplineMirrorBoundary,
        _SplineStateVectorizer,
        _packed_spline_preconditioner,
    )

    evaluated = result.evaluated
    if not bool(evaluated.converged):
        raise ValueError("implicit differentiation requires a converged mirror result")
    if not isinstance(evaluated.energy, MirrorEnergy):
        raise ValueError("spline implicit derivatives require the isotropic energy")
    boundary = SplineMirrorBoundary(jnp.asarray(parameters.boundary_coefficients))
    axis_shape = tuple(jnp.shape(parameters.axis_coefficients))
    if discretization.closed:
        expected = (discretization.coefficient_count, 3)
        if axis_shape != expected:
            raise ValueError(f"closed spline axis coefficients must have shape {expected}")
    elif np.prod(axis_shape) != 0:
        raise ValueError("open spline derivatives do not use axis coefficients")
    vectorizer = _SplineStateVectorizer.build(
        result.coefficient_state,
        boundary,
        discretization,
        axial_flux_derivative=parameters.axial_flux_derivative,
        solve_lambda=solve_lambda,
    )
    x_star = jnp.asarray(vectorizer.pack())
    energy_scale = max(abs(float(evaluated.energy.total)), np.finfo(float).tiny)

    def state_at(x: Array, controls: SplineFixedBoundaryParameters) -> MirrorState:
        coefficients = discretization.project_fixed_boundary(
            vectorizer.unpack(x),
            SplineMirrorBoundary(controls.boundary_coefficients),
        )
        return regularize_axis_stream_function(
            discretization.evaluate_state(coefficients),
            discretization.grid,
            controls.axial_flux_derivative,
        )

    def energy_at(x: Array, controls: SplineFixedBoundaryParameters) -> MirrorEnergy:
        axis = None
        if discretization.closed:
            axis = evaluate_closed_spline_axis(
                controls.axis_coefficients,
                discretization.spline,
                discretization.grid.z,
                initial_normal=jnp.asarray([0.0, 1.0, 0.0]),
            )
        return mirror_energy(
            state_at(x, controls),
            discretization.grid,
            axial_flux_derivative=controls.axial_flux_derivative,
            mass_profile=controls.mass_profile,
            current_derivative=controls.current_derivative,
            gamma=gamma,
            axis=axis,
        )

    def residual(x: Array, controls: SplineFixedBoundaryParameters) -> Array:
        return jax.grad(lambda vector: energy_at(vector, controls).total / energy_scale)(x)

    reconstructed = np.asarray(residual(x_star, parameters), dtype=float)
    tolerance = max(10.0 * float(evaluated.variational.maximum), 1.0e-12)
    if np.max(np.abs(reconstructed)) > tolerance:
        raise ValueError("fixed-boundary controls do not reconstruct the converged result")
    apply_preconditioner, scales, _ = _packed_spline_preconditioner(discretization, vectorizer)
    return (
        x_star,
        state_at,
        energy_at,
        residual,
        apply_preconditioner,
        scales,
        vectorizer.block_slices,
    )


def spline_fixed_boundary_adjoint(
    result: Any,
    parameters: SplineFixedBoundaryParameters,
    discretization: Any,
    quantity: MirrorQuantity,
    *,
    gamma: float = 5.0 / 3.0,
    solve_lambda: bool = False,
    rtol: float = 1.0e-10,
    max_restarts: int = 20,
) -> MirrorAdjointResult:
    """Differentiate a scalar through a converged spline equilibrium."""

    if rtol <= 0.0 or max_restarts < 1:
        raise ValueError("rtol and max_restarts must be positive")
    (
        x_star,
        state_at,
        energy_at,
        residual,
        apply_preconditioner,
        scales,
        split,
    ) = _spline_implicit_problem(
        result,
        parameters,
        discretization,
        gamma=gamma,
        solve_lambda=solve_lambda,
    )

    def evaluate_quantity(x: Array, controls: SplineFixedBoundaryParameters) -> Array:
        state = state_at(x, controls)
        value = jnp.asarray(quantity(state, energy_at(x, controls)))
        if value.ndim != 0:
            raise ValueError("mirror adjoint quantity must return a scalar")
        return value

    return _implicit_adjoint(
        x_star,
        parameters,
        residual,
        evaluate_quantity,
        apply_preconditioner,
        scales,
        split,
        rtol=rtol,
        max_restarts=max_restarts,
    )


def spline_fixed_boundary_tangent(
    result: Any,
    parameters: SplineFixedBoundaryParameters,
    parameter_tangent: SplineFixedBoundaryParameters,
    discretization: Any,
    *,
    gamma: float = 5.0 / 3.0,
    solve_lambda: bool = False,
    rtol: float = 1.0e-10,
    max_restarts: int = 20,
) -> MirrorTangentResult:
    """Differentiate a converged spline state in one control direction."""

    if rtol <= 0.0 or max_restarts < 1:
        raise ValueError("rtol and max_restarts must be positive")
    (
        x_star,
        state_at,
        _,
        residual,
        apply_preconditioner,
        scales,
        split,
    ) = _spline_implicit_problem(
        result,
        parameters,
        discretization,
        gamma=gamma,
        solve_lambda=solve_lambda,
    )
    parameter_tangent = jax.tree.map(jnp.asarray, parameter_tangent)
    residual_tangent = jax.jvp(
        lambda controls: residual(x_star, controls),
        (parameters,),
        (parameter_tangent,),
    )[1]
    tangent_action = jax.jit(
        lambda direction: jax.jvp(
            lambda x: residual(x, parameters),
            (x_star,),
            (direction,),
        )[1]
    )

    def matrix_vector(vector: Array) -> Array:
        return tangent_action(vector)

    packed_tangent, iterations, relative_residual, converged = _solve_implicit_system(
        matrix_vector,
        -np.asarray(residual_tangent, dtype=float),
        apply_preconditioner,
        scales,
        split,
        rtol=rtol,
        max_restarts=max_restarts,
    )
    state_tangent = jax.jvp(
        state_at,
        (x_star, parameters),
        (jnp.asarray(packed_tangent), parameter_tangent),
    )[1]
    return MirrorTangentResult(
        tangent=state_tangent,
        iterations=iterations,
        relative_residual=relative_residual,
        converged=converged,
    )


FreeBoundaryQuantity = Callable[[MirrorBoundary, MirrorState, MirrorEnergy, Any], Array]


@dataclass(frozen=True, eq=False)
class FreeBoundaryAdjointConfig:
    """Static exterior quadrature and linear-solver controls."""

    axisymmetric_ntheta: int = 40
    exterior_order: int = 8
    spectral_side_density: bool = False
    gamma: float = 5.0 / 3.0
    rtol: float = 1.0e-9
    max_restarts: int = 30


@dataclass(frozen=True)
class FreeBoundaryAdjointResult:
    """Scalar value, total control gradient, and transpose-solve diagnostics."""

    value: Array
    gradient: FreeBoundaryParameters
    iterations: int
    relative_residual: float
    converged: bool


jax.tree_util.register_dataclass(
    FreeBoundaryAdjointResult,
    data_fields=["value", "gradient"],
    meta_fields=["iterations", "relative_residual", "converged"],
)


def free_boundary_adjoint(
    result: Any,
    parameters: FreeBoundaryParameters,
    discretization: Any,
    quantity: FreeBoundaryQuantity,
    *,
    config: FreeBoundaryAdjointConfig = FreeBoundaryAdjointConfig(),
) -> FreeBoundaryAdjointResult:
    """Differentiate a scalar through a converged axisymmetric exterior solve.

    End-cut radii are fixed. Pressure-profile, flux, current, applied-field,
    and solved lateral-shape responses follow the same primal residual.
    """

    if not bool(result.converged):
        raise ValueError("free-boundary differentiation requires a converged result")
    grid = discretization.grid
    if grid.ntheta != 1:
        raise ValueError("free-boundary adjoint currently supports axisymmetry only")
    if config.rtol <= 0.0 or config.max_restarts < 1:
        raise ValueError("adjoint tolerances and iteration limits must be positive")
    problem = _build_free_equilibrium_problem(
        result.coefficient_boundary,
        result.coefficient_state,
        discretization,
        parameters.external_field,
        axial_flux_derivative=parameters.axial_flux_derivative,
        mass_profile=parameters.mass_profile,
        current_derivative=parameters.current_derivative,
        solve_lambda=False,
        gamma=config.gamma,
        target_central_pressure=result.target_central_pressure,
        initial_mass_scale=float(result.mass_scale),
        exterior_ntheta=config.axisymmetric_ntheta,
        exterior_order=config.exterior_order,
        exterior_spectral_side_density=config.spectral_side_density,
        plasma_scale=result.plasma_scale,
    )
    x_star = jnp.asarray(problem.vectorizer.pack())

    def residual(vector: Array, controls: FreeBoundaryParameters) -> Array:
        return problem.parameterized_residual_function(vector, controls)

    def evaluate_quantity(vector: Array, controls: FreeBoundaryParameters) -> Array:
        _, _, _, boundary, state, plasma, _, _, vacuum = problem.parameterized_components_function(vector, controls)
        value = jnp.asarray(quantity(boundary, state, plasma, vacuum))
        if value.ndim != 0:
            raise ValueError("free-boundary adjoint quantity must return a scalar")
        return value

    equilibrium_residual = np.asarray(residual(x_star, parameters), dtype=float)
    if np.max(np.abs(equilibrium_residual)) > 10.0 * max(float(result.variational_max), 1.0e-12):
        raise ValueError("reconstructed free-boundary residual does not match result")
    boundary_size = problem.vectorizer.boundary_size
    state_size = problem.vectorizer.state_size
    scales = np.ones(3 if problem.vectorizer.calibrate_pressure else 2)

    block_slices = [slice(0, boundary_size), slice(boundary_size, boundary_size + state_size)]
    if problem.vectorizer.calibrate_pressure:
        block_slices.append(slice(boundary_size + state_size, None))
    adjoint = _implicit_adjoint(
        x_star,
        parameters,
        residual,
        evaluate_quantity,
        problem.preconditioner(),
        scales,
        tuple(block_slices),
        rtol=config.rtol,
        max_restarts=config.max_restarts,
    )
    return FreeBoundaryAdjointResult(
        value=adjoint.value,
        gradient=adjoint.gradient,
        iterations=adjoint.iterations,
        relative_residual=adjoint.relative_residual,
        converged=adjoint.converged,
    )


__all__ = [
    "FreeBoundaryAdjointConfig",
    "FreeBoundaryAdjointResult",
    "FreeBoundaryParameters",
    "MirrorAdjointResult",
    "MirrorTangentResult",
    "SplineFixedBoundaryParameters",
    "free_boundary_adjoint",
    "spline_fixed_boundary_adjoint",
    "spline_fixed_boundary_tangent",
    "spline_fixed_boundary_parameters",
]
