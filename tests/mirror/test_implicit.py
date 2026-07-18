"""Implicit differentiation of converged mirror equilibria."""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmex.mirror import (  # noqa: E402
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
    free_boundary_adjoint,
    solve_free_boundary,
    spline_fixed_boundary_adjoint,
    spline_fixed_boundary_tangent,
)
from vmex.mirror.implicit import (  # noqa: E402
    FreeBoundaryAdjointConfig,
    _spline_implicit_problem,
    spline_fixed_boundary_parameters,
)
from vmex.mirror.free_boundary import FreeBoundaryParameters  # noqa: E402
from vmex.mirror.forces import MU0, mass_profile_from_pressure, mirror_energy  # noqa: E402
from vmex.mirror.splines import (  # noqa: E402
    SplineMirrorBoundary,
    SplineMirrorDiscretization,
    SplineMirrorState,
    _initialize_closed_vacuum_stream_function,
    solve_fixed_boundary as solve_spline_fixed_boundary,
)
from vmex.mirror.geometry import evaluate_closed_spline_axis  # noqa: E402


@pytest.fixture(autouse=True)
def _enable_solver_jit():
    previous = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    yield
    jax.config.update("jax_disable_jit", previous)


def test_spline_adjoint_matches_reconverged_central_difference() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, nxi=9),
        ftol=1.0e-12,
        max_iterations=1000,
    )
    source_grid = config.build_grid()
    s, xi = jnp.asarray(source_grid.s), jnp.asarray(source_grid.xi)
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.1 * (1.0 - xi**2)), source_grid)
    discretization = SplineMirrorDiscretization.build(config, elements=3)
    spline_boundary = discretization.fit_boundary(boundary, source_grid)
    mass = 2.0e3 * (1.0 - s)
    current = 1.0e-2 * s
    result = solve_spline_fixed_boundary(
        discretization.fit_state(MirrorState.from_boundary(boundary, source_grid), source_grid),
        spline_boundary,
        discretization,
        config,
        axial_flux_derivative=0.1,
        mass_profile=mass,
        current_derivative=current,
        gradient_tolerance=1.0e-12,
        require_convergence=True,
    )
    parameters = spline_fixed_boundary_parameters(
        spline_boundary,
        axial_flux_derivative=0.1,
        mass_profile=mass,
        current_derivative=current,
    )

    def quantity(state, _energy):
        return state.radius_scale[2, 0, discretization.grid.nxi // 2]

    adjoint = spline_fixed_boundary_adjoint(
        result,
        parameters,
        discretization,
        quantity,
        rtol=1.0e-10,
    )
    nodes = jnp.asarray(discretization.spline.collocation_nodes)
    boundary_direction = 0.01 * (1.0 - nodes**2)[None, :]
    flux_direction = jnp.asarray(0.003)
    mass_direction = 100.0 * (1.0 - s)
    current_direction = 2.0e-3 * s
    predicted = float(
        jnp.vdot(adjoint.gradient.boundary_coefficients, boundary_direction)
        + adjoint.gradient.axial_flux_derivative * flux_direction
        + jnp.vdot(adjoint.gradient.mass_profile, mass_direction)
        + jnp.vdot(adjoint.gradient.current_derivative, current_direction)
    )

    values = []
    epsilon = 1.0e-4
    for sign in (-1.0, 1.0):
        varied_boundary = SplineMirrorBoundary(
            spline_boundary.radius_coefficients + sign * epsilon * boundary_direction
        )
        initial = discretization.transfer_boundary(result.coefficient_state, spline_boundary, varied_boundary)
        varied = solve_spline_fixed_boundary(
            initial,
            varied_boundary,
            discretization,
            config,
            axial_flux_derivative=0.1 + sign * epsilon * flux_direction,
            mass_profile=mass + sign * epsilon * mass_direction,
            current_derivative=current + sign * epsilon * current_direction,
            gradient_tolerance=1.0e-12,
            require_convergence=True,
        )
        values.append(float(quantity(varied.evaluated.state, varied.evaluated.energy)))
    finite_difference = (values[1] - values[0]) / (2.0 * epsilon)

    assert adjoint.converged
    assert adjoint.iterations > 0
    assert adjoint.relative_residual < 1.0e-10
    np.testing.assert_allclose(predicted, finite_difference, rtol=2.0e-6, atol=1.0e-10)


def test_nonaxisymmetric_spline_adjoint_includes_stream_function() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=1, nxi=7),
        ftol=1.0e-12,
        max_iterations=1000,
    )
    source_grid = config.build_grid()
    s = jnp.asarray(source_grid.s)
    theta = jnp.asarray(source_grid.theta)[:, None]
    xi = jnp.asarray(source_grid.xi)[None, :]
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.02 * jnp.cos(theta) * (1.0 - xi**2)), source_grid)
    discretization = SplineMirrorDiscretization.build(config, elements=3)
    spline_boundary = discretization.fit_boundary(boundary, source_grid)
    current = 1.0e-3 * s
    result = solve_spline_fixed_boundary(
        discretization.fit_state(MirrorState.from_boundary(boundary, source_grid), source_grid),
        spline_boundary,
        discretization,
        config,
        axial_flux_derivative=0.1,
        mass_profile=100.0 * (1.0 - s),
        current_derivative=current,
        solve_lambda=True,
        gradient_tolerance=1.0e-12,
        require_convergence=True,
    )
    parameters = spline_fixed_boundary_parameters(
        spline_boundary,
        axial_flux_derivative=0.1,
        mass_profile=100.0 * (1.0 - s),
        current_derivative=current,
    )

    adjoint = spline_fixed_boundary_adjoint(
        result,
        parameters,
        discretization,
        lambda _state, energy: energy.geometry.volume,
        solve_lambda=True,
        rtol=1.0e-9,
    )
    direction = jnp.zeros_like(spline_boundary.radius_coefficients).at[0, 2].set(0.01)
    zero_tangent = jax.tree.map(jnp.zeros_like, parameters)
    tangent = spline_fixed_boundary_tangent(
        result,
        parameters,
        replace(zero_tangent, boundary_coefficients=direction),
        discretization,
        solve_lambda=True,
        rtol=1.0e-9,
    )
    predicted = float(jnp.vdot(adjoint.gradient.boundary_coefficients, direction))
    values = []
    states = []
    epsilon = 2.0e-4
    for sign in (-1.0, 1.0):
        varied_boundary = SplineMirrorBoundary(spline_boundary.radius_coefficients + sign * epsilon * direction)
        varied = solve_spline_fixed_boundary(
            discretization.transfer_boundary(result.coefficient_state, spline_boundary, varied_boundary),
            varied_boundary,
            discretization,
            config,
            axial_flux_derivative=0.1,
            mass_profile=parameters.mass_profile,
            current_derivative=current,
            solve_lambda=True,
            gradient_tolerance=1.0e-12,
            require_convergence=True,
        )
        values.append(float(varied.evaluated.energy.geometry.volume))
        states.append(varied.evaluated.state)
    finite_difference = (values[1] - values[0]) / (2.0 * epsilon)
    state_difference = jax.tree.map(
        lambda upper, lower: (upper - lower) / (2.0 * epsilon),
        states[1],
        states[0],
    )

    assert adjoint.converged
    assert adjoint.relative_residual < 1.0e-8
    assert tangent.converged
    assert tangent.relative_residual < 1.0e-8
    assert float(jnp.max(jnp.abs(result.evaluated.state.lambda_stream))) > 1.0e-4
    np.testing.assert_allclose(predicted, finite_difference, rtol=2.0e-5, atol=1.0e-9)
    for actual, expected in zip(
        jax.tree.leaves(tangent.tangent),
        jax.tree.leaves(state_difference),
        strict=True,
    ):
        relative_error = jnp.linalg.norm(actual - expected) / jnp.maximum(
            jnp.linalg.norm(expected), jnp.finfo(expected.dtype).tiny
        )
        assert float(relative_error) < 2.0e-4


def test_closed_spline_adjoint_differentiates_axis_and_boundary() -> None:
    resolution = MirrorResolution(ns=5, mpol=1, nxi=4)
    config = MirrorConfig(resolution=resolution, ftol=1.0e-12, max_iterations=1000)
    discretization = SplineMirrorDiscretization.build_closed(
        resolution,
        coefficient_count=8,
        quadrature_order=3,
    )
    basis = discretization.spline
    points = jnp.asarray(basis.collocation_nodes)
    axis_coefficients = basis.fit(
        jnp.stack(
            (2.5 * jnp.cos(points), jnp.zeros_like(points), 2.5 * jnp.sin(points)),
            axis=-1,
        ),
        axis=0,
    )

    def axis_at(coefficients):
        return evaluate_closed_spline_axis(
            coefficients,
            basis,
            discretization.grid.z,
            initial_normal=jnp.asarray([0.0, 1.0, 0.0]),
        )

    boundary = SplineMirrorBoundary(jnp.full((resolution.ntheta, basis.size), 0.25))
    radius = jnp.full((resolution.ns, resolution.ntheta, basis.size), 0.25)
    initial = _initialize_closed_vacuum_stream_function(
        SplineMirrorState(radius, jnp.zeros_like(radius)),
        discretization,
        axis_at(axis_coefficients),
        axial_flux_derivative=0.03,
    )
    solve_options = dict(
        axial_flux_derivative=0.03,
        solve_lambda=True,
        gradient_tolerance=1.0e-12,
        require_convergence=True,
    )
    result = solve_spline_fixed_boundary(
        initial,
        boundary,
        discretization,
        config,
        axis=axis_at(axis_coefficients),
        **solve_options,
    )
    parameters = spline_fixed_boundary_parameters(
        boundary,
        axial_flux_derivative=0.03,
        axis_coefficients=axis_coefficients,
    )
    adjoint = spline_fixed_boundary_adjoint(
        result,
        parameters,
        discretization,
        lambda _state, energy: energy.geometry.volume,
        solve_lambda=True,
        rtol=1.0e-9,
    )
    boundary_direction = jnp.full_like(boundary.radius_coefficients, 0.01)
    axis_direction = 0.01 * axis_coefficients
    parameter_direction = replace(
        jax.tree.map(jnp.zeros_like, parameters),
        boundary_coefficients=boundary_direction,
        axis_coefficients=axis_direction,
    )
    predicted = float(
        jnp.vdot(adjoint.gradient.boundary_coefficients, boundary_direction)
        + jnp.vdot(adjoint.gradient.axis_coefficients, axis_direction)
    )
    x_star, _, _, residual, *_ = _spline_implicit_problem(
        result,
        parameters,
        discretization,
        gamma=5.0 / 3.0,
        solve_lambda=True,
    )
    cotangent = jnp.linspace(-0.5, 0.5, x_star.size)
    parameter_jvp = jax.jvp(
        lambda controls: residual(x_star, controls),
        (parameters,),
        (parameter_direction,),
    )[1]
    parameter_vjp = jax.vjp(lambda controls: residual(x_star, controls), parameters)[1](cotangent)[0]
    transpose_product = sum(
        jnp.vdot(direction, pullback)
        for direction, pullback in zip(
            jax.tree.leaves(parameter_direction),
            jax.tree.leaves(parameter_vjp),
            strict=True,
        )
    )
    np.testing.assert_allclose(
        jnp.vdot(parameter_jvp, cotangent),
        transpose_product,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    finite_differences = []
    for epsilon in (4.0e-4, 2.0e-4, 1.0e-4):
        values = []
        for sign in (-1.0, 1.0):
            varied_boundary = SplineMirrorBoundary(
                boundary.radius_coefficients + sign * epsilon * boundary_direction
            )
            varied_axis_coefficients = axis_coefficients + sign * epsilon * axis_direction
            varied = solve_spline_fixed_boundary(
                discretization.transfer_boundary(
                    result.coefficient_state,
                    boundary,
                    varied_boundary,
                ),
                varied_boundary,
                discretization,
                config,
                axis=axis_at(varied_axis_coefficients),
                **solve_options,
            )
            values.append(float(varied.evaluated.energy.geometry.volume))
        finite_differences.append((values[1] - values[0]) / (2.0 * epsilon))

    assert adjoint.converged
    assert adjoint.relative_residual < 1.0e-8
    assert float(jnp.linalg.norm(adjoint.gradient.axis_coefficients)) > 0.0
    np.testing.assert_allclose(finite_differences, predicted, rtol=5.0e-5)
    np.testing.assert_allclose(finite_differences[-1], predicted, rtol=2.0e-5)


@dataclass(frozen=True)
class ParaxialMirrorField:
    """Differentiable divergence-free field with mirror-like axial strength."""

    center_field: jax.Array
    curvature: jax.Array

    def __call__(self, points):
        points = jnp.asarray(points)
        x, y, z = jnp.moveaxis(points, -1, 0)
        return jnp.stack(
            (
                -self.curvature * x * z,
                -self.curvature * y * z,
                self.center_field + self.curvature * z**2,
            ),
            axis=-1,
        )


jax.tree_util.register_dataclass(
    ParaxialMirrorField,
    data_fields=["center_field", "curvature"],
    meta_fields=[],
)


def test_free_boundary_adjoint_rejects_unconverged_and_3d_results() -> None:
    axisymmetric_config = MirrorConfig(resolution=MirrorResolution(ns=3, nxi=5))
    axisymmetric = SplineMirrorDiscretization.build_cgl(axisymmetric_config, elements=2)
    parameters = FreeBoundaryParameters(object(), jnp.asarray(0.1), jnp.asarray(0.0), jnp.asarray(0.0))
    with pytest.raises(ValueError, match="converged"):
        free_boundary_adjoint(
            type("Result", (), {"converged": False})(),
            parameters,
            axisymmetric,
            lambda *_: 0.0,
        )

    nonaxisymmetric_config = MirrorConfig(resolution=MirrorResolution(ns=3, mpol=1, nxi=5))
    nonaxisymmetric = SplineMirrorDiscretization.build_cgl(nonaxisymmetric_config, elements=2)
    with pytest.raises(ValueError, match="axisymmetry"):
        free_boundary_adjoint(
            type("Result", (), {"converged": True})(),
            parameters,
            nonaxisymmetric,
            lambda *_: 0.0,
        )


@pytest.mark.full
def test_free_boundary_field_adjoint_matches_central_difference(_module_jit_enabled) -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, nxi=7),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=300,
    )
    source_grid = config.build_grid()
    discretization = SplineMirrorDiscretization.build_cgl(config, elements=4)
    grid = discretization.grid
    field = ParaxialMirrorField(jnp.asarray(0.08), jnp.asarray(0.02))
    on_axis = field.center_field + field.curvature * jnp.asarray(grid.z) ** 2
    center = grid.nxi // 2
    flux = 0.5 * on_axis[center] * 0.25**2
    nodal_boundary = MirrorBoundary.from_axis_field(flux, on_axis, grid)
    initial_boundary = discretization.fit_boundary(nodal_boundary, source_grid)
    reference_energy = mirror_energy(
        MirrorState.from_boundary(nodal_boundary, grid),
        grid,
        axial_flux_derivative=flux,
    )
    pressure = 0.03 * field.center_field**2 / (2.0 * MU0) * (1.0 - jnp.asarray(grid.s))
    mass_profile = mass_profile_from_pressure(pressure, reference_energy.volume_derivative)
    solve_options = dict(
        axial_flux_derivative=flux,
        mass_profile=mass_profile,
        exterior_ntheta=8,
        exterior_order=6,
        exterior_spectral_side_density=True,
        require_convergence=True,
    )
    result = solve_free_boundary(initial_boundary, discretization, config, field, **solve_options)
    parameters = FreeBoundaryParameters(field, flux, mass_profile, jnp.asarray(0.0))

    def quantity(boundary, _state, _energy, _vacuum):
        return boundary.radius_scale[0, center]

    adjoint = free_boundary_adjoint(
        result,
        parameters,
        discretization,
        quantity,
        config=FreeBoundaryAdjointConfig(
            axisymmetric_ntheta=8,
            exterior_order=6,
            spectral_side_density=True,
            rtol=1.0e-8,
        ),
    )
    direction = ParaxialMirrorField(jnp.asarray(0.01), jnp.asarray(-0.005))
    predicted = float(
        adjoint.gradient.external_field.center_field * direction.center_field
        + adjoint.gradient.external_field.curvature * direction.curvature
    )
    epsilon = 1.0e-4

    def reconverged_value(sign, *, field_direction=None, mass_direction=0.0):
        varied_field = field
        if field_direction is not None:
            varied_field = jax.tree.map(lambda value, delta: value + sign * epsilon * delta, field, field_direction)
        varied_options = solve_options | {"mass_profile": mass_profile + sign * epsilon * mass_direction}
        varied = solve_free_boundary(
            result.coefficient_boundary,
            discretization,
            config,
            varied_field,
            initial_state=result.coefficient_state,
            **varied_options,
        )
        return float(quantity(varied.boundary, None, None, None))

    field_values = [reconverged_value(sign, field_direction=direction) for sign in (-1.0, 1.0)]
    field_finite_difference = (field_values[1] - field_values[0]) / (2.0 * epsilon)

    mass_direction = 0.1 * mass_profile
    mass_predicted = float(jnp.vdot(adjoint.gradient.mass_profile, mass_direction))
    mass_values = [reconverged_value(sign, mass_direction=mass_direction) for sign in (-1.0, 1.0)]
    mass_finite_difference = (mass_values[1] - mass_values[0]) / (2.0 * epsilon)
    assert result.converged
    assert float(result.variational_max) <= config.ftol
    assert adjoint.converged
    assert adjoint.relative_residual < 1.0e-8
    np.testing.assert_allclose(predicted, field_finite_difference, rtol=3.0e-4)
    np.testing.assert_allclose(mass_predicted, mass_finite_difference, rtol=3.0e-4)
