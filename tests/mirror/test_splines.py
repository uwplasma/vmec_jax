"""Cubic B-spline identities, refinement, and differentiation tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import BSpline

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmex.mirror.splines import CubicBSplineBasis  # noqa: E402
from vmex.mirror import (  # noqa: E402
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
)
from vmex.mirror.forces import (  # noqa: E402
    force_gate_zones,
    isotropic_force_residual,
    mirror_energy,
    passes_promotion_gate,
    refinement_convergence,
)
from vmex.mirror.free_boundary import (  # noqa: E402
    _SplineFreeBoundaryVectorizer,
    _spline_boundary_work,
    _spline_boundary_work_residual,
)
from vmex.mirror.geometry import (  # noqa: E402
    contravariant_field,
    divergence_b,
    evaluate_closed_geometry,
    evaluate_closed_spline_axis,
    evaluate_geometry,
    magnetic_field_xyz,
    stellarator_mirror_axis_coefficients,
    stellarator_mirror_section_coefficients,
)
from vmex.mirror.solver import MirrorConvergenceError  # noqa: E402
from vmex.mirror.splines import (  # noqa: E402
    SplineMirrorBoundary,
    SplineMirrorDiscretization,
    SplineMirrorState,
    _SplineStateVectorizer,
    _packed_spline_preconditioner,
    build_stellarator_mirror_hybrid,
    _initialize_closed_vacuum_stream_function,
    initialize_from_cartesian_field,
    solve_fixed_boundary as solve_spline_fixed_boundary,
    solve_fixed_boundary_from_radius,
    trace_closed_field_line,
)


def test_clamped_basis_matches_scipy_and_partitions_unity() -> None:
    basis = CubicBSplineBasis.clamped(np.linspace(-1.0, 1.0, 7))
    points = np.linspace(-1.0, 1.0, 101)
    actual = np.asarray(basis.basis_matrix(points))
    expected = np.column_stack(
        [BSpline(basis.knots, np.eye(basis.size)[index], 3)(points) for index in range(basis.size)]
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(np.sum(actual, axis=1), 1.0, atol=2.0e-15)
    np.testing.assert_allclose(actual[0], np.eye(basis.size)[0], atol=0.0)
    np.testing.assert_allclose(actual[-1], np.eye(basis.size)[-1], atol=0.0)


def test_clamped_derivatives_and_cubic_reproduction() -> None:
    basis = CubicBSplineBasis.clamped(np.linspace(-1.0, 1.0, 8))

    def polynomial(x):
        return 1.2 - 0.4 * x + 0.7 * x**2 - 0.3 * x**3

    coefficients = basis.fit(polynomial(jnp.asarray(basis.collocation_nodes)))
    points = jnp.linspace(-1.0, 1.0, 83)
    np.testing.assert_allclose(basis.evaluate(coefficients, points), polynomial(points), atol=2.0e-14)
    np.testing.assert_allclose(
        basis.evaluate(coefficients, points, derivative=1), -0.4 + 1.4 * points - 0.9 * points**2, atol=8.0e-14
    )
    np.testing.assert_allclose(basis.evaluate(coefficients, points, derivative=2), 1.4 - 1.8 * points, atol=4.0e-13)
    np.testing.assert_allclose(basis.integrate(coefficients), 2.0 * 1.2 + 2.0 * 0.7 / 3.0, atol=2.0e-14)


def test_open_knot_insertion_preserves_curve_and_jax_derivatives() -> None:
    basis = CubicBSplineBasis.clamped(np.linspace(-1.0, 1.0, 6))
    coefficients = jnp.sin(1.7 * jnp.asarray(basis.collocation_nodes)) + 0.2 * basis.collocation_nodes**2
    refined, refined_coefficients = basis.insert_knot(coefficients, 0.13)
    points = jnp.linspace(-1.0, 1.0, 117)
    np.testing.assert_allclose(
        refined.evaluate(refined_coefficients, points), basis.evaluate(coefficients, points), rtol=3.0e-14, atol=3.0e-14
    )

    direction = jnp.linspace(-0.3, 0.4, basis.size)
    primal, tangent = jax.jvp(lambda values: basis.evaluate(values, points), (coefficients,), (direction,))
    np.testing.assert_allclose(primal, basis.evaluate(coefficients, points), atol=0.0)
    np.testing.assert_allclose(tangent, basis.evaluate(direction, points), atol=2.0e-15)
    cotangent = jnp.cos(points)
    reverse = jax.grad(lambda values: jnp.vdot(basis.evaluate(values, points), cotangent))(coefficients)
    np.testing.assert_allclose(reverse, basis.basis_matrix(points).T @ cotangent, atol=2.0e-14)


def test_periodic_spline_closes_through_two_derivatives_and_is_differentiable() -> None:
    basis = CubicBSplineBasis.periodic_uniform(16)
    points = jnp.linspace(0.0, 2.0 * jnp.pi, 129)
    np.testing.assert_allclose(jnp.sum(basis.basis_matrix(points), axis=1), 1.0, atol=2.0e-15)
    for derivative in range(3):
        np.testing.assert_allclose(
            basis.basis_matrix(points[:1], derivative=derivative),
            basis.basis_matrix(points[-1:], derivative=derivative),
            atol=3.0e-14,
        )

    coefficients = jnp.sin(basis.collocation_nodes) + 0.2 * jnp.cos(2.0 * basis.collocation_nodes)
    direction = jnp.linspace(-0.3, 0.4, basis.size)
    tangent = jax.jvp(lambda value: basis.evaluate(value, points), (coefficients,), (direction,))[1]
    np.testing.assert_allclose(tangent, basis.evaluate(direction, points), atol=2.0e-15)
    cotangent = jnp.cos(points)
    reverse = jax.grad(lambda value: jnp.vdot(basis.evaluate(value, points), cotangent))(coefficients)
    np.testing.assert_allclose(reverse, basis.basis_matrix(points).T @ cotangent, atol=3.0e-14)


def test_periodic_refinement_preserves_values_and_two_derivatives() -> None:
    basis = CubicBSplineBasis.periodic_uniform(8)
    coefficients = jnp.stack(
        (
            jnp.cos(basis.collocation_nodes),
            0.2 * jnp.sin(2.0 * basis.collocation_nodes),
        ),
        axis=-1,
    )
    refined, values = basis.refine_periodic_uniform(coefficients, 32, axis=0)
    points = jnp.linspace(0.0, 2.0 * jnp.pi, 193, endpoint=False)
    for derivative in range(3):
        np.testing.assert_allclose(
            refined.evaluate(values, points, derivative=derivative, axis=0),
            basis.evaluate(coefficients, points, derivative=derivative, axis=0),
            rtol=2.0e-13,
            atol=2.0e-13,
        )


def test_stellarator_mirror_axis_has_exact_straights_and_periodic_frame() -> None:
    basis = CubicBSplineBasis.periodic_uniform(32)
    coefficients = stellarator_mirror_axis_coefficients(
        basis,
        straight_length=6.0,
        return_radius=1.0,
    )
    points = np.linspace(*basis.domain, 257, endpoint=False)
    axis = evaluate_closed_spline_axis(
        coefficients,
        basis,
        points,
        initial_normal=jnp.asarray([0.0, 1.0, 0.0]),
    )
    straight = np.asarray(axis.curvature) < 1.0e-10
    assert np.count_nonzero(straight) > 0.35 * points.size
    assert float(axis.closure_error) < 2.0e-14
    assert float(axis.tangent_closure_error) < 2.0e-14
    assert float(axis.frame_closure_error) < 2.0e-14
    np.testing.assert_allclose(jnp.linalg.norm(axis.tangent, axis=-1), 1.0, atol=2.0e-14)
    np.testing.assert_allclose(jnp.sum(axis.tangent * axis.normal, axis=-1), 0.0, atol=2.0e-14)

    gradient = jax.grad(
        lambda value: evaluate_closed_spline_axis(
            value,
            basis,
            points,
            initial_normal=jnp.asarray([0.0, 1.0, 0.0]),
        ).arc_length
    )(coefficients)
    assert np.all(np.isfinite(gradient))
    assert float(jnp.linalg.norm(gradient)) > 1.0


def test_stellarator_mirror_ellipse_rotates_ninety_degrees_between_legs() -> None:
    basis = CubicBSplineBasis.periodic_uniform(32)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 65, endpoint=False)
    coefficients = stellarator_mirror_section_coefficients(
        basis,
        theta,
        semi_major=0.45,
        semi_minor=0.30,
    )
    leg_radii = basis.evaluate(coefficients, jnp.asarray([0.0, jnp.pi]), axis=-1)
    major_index = int(jnp.argmin(jnp.abs(theta)))
    minor_index = int(jnp.argmin(jnp.abs(theta - 0.5 * jnp.pi)))
    np.testing.assert_allclose(leg_radii[major_index], [0.45, 0.30], rtol=3.0e-13)
    np.testing.assert_allclose(leg_radii[minor_index], [0.30, 0.45], rtol=3.0e-3)


def test_stellarator_mirror_toroidal_rotation_winds_ellipse_by_full_turns() -> None:
    """``section_turns`` rotates the ellipse by that many full circuit turns.

    The default reproduces the return-only rotation exactly. A nonzero
    ``section_turns`` superposes a genuine rotating ellipse whose major axis
    winds by that many ``2*pi`` turns around the circuit; the phase of the m=2
    poloidal harmonic (which equals ``-2*alpha``) therefore advances by exactly
    ``-4*pi*section_turns``, and the section stays positive and periodic.
    """

    basis = CubicBSplineBasis.periodic_uniform(64)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 64, endpoint=False)
    default = stellarator_mirror_section_coefficients(basis, theta, semi_major=0.45, semi_minor=0.30)
    explicit_zero = stellarator_mirror_section_coefficients(
        basis, theta, semi_major=0.45, semi_minor=0.30, section_turns=0
    )
    np.testing.assert_array_equal(np.asarray(default), np.asarray(explicit_zero))

    points = jnp.asarray(np.linspace(0.0, 2.0 * np.pi, 257))
    for turns in (1, 2, 3):
        coefficients = stellarator_mirror_section_coefficients(
            basis, theta, semi_major=0.45, semi_minor=0.25, section_turns=turns
        )
        assert float(jnp.min(coefficients)) > 0.0
        curve = np.asarray(basis.evaluate(coefficients, points, axis=-1))
        second_harmonic = np.fft.fft(curve, axis=0)[2]
        winding = np.unwrap(np.angle(second_harmonic))
        np.testing.assert_allclose(winding[-1] - winding[0], -4.0 * np.pi * turns, atol=1.0e-6)


def test_stellarator_mirror_builder_has_positive_nested_spline_geometry() -> None:
    resolution = MirrorResolution(ns=7, mpol=8, nxi=4)
    setup = build_stellarator_mirror_hybrid(resolution)
    state = setup.discretization.evaluate_state(setup.initial_state)
    geometry = evaluate_closed_geometry(state, setup.discretization.grid, setup.axis)
    expected_volume = np.pi * 0.45 * 0.30 * float(setup.axis.arc_length)
    np.testing.assert_allclose(geometry.volume, expected_volume, rtol=3.0e-3)
    assert not bool(geometry.jacobian_sign_changed)
    assert float(setup.axis.closure_error) < 2.0e-14
    assert float(setup.axis.frame_closure_error) < 2.0e-14

    field = contravariant_field(
        state,
        geometry,
        setup.discretization.grid,
        axial_flux_derivative=0.02,
        current_derivative=0.002,
    )
    np.testing.assert_allclose(
        divergence_b(field, geometry, setup.discretization.grid)[1:],
        0.0,
        atol=7.0e-13,
    )


@pytest.mark.full
@pytest.mark.usefixtures("_module_jit_enabled")
def test_closed_circular_limit_reaches_ftol_with_independent_strong_force() -> None:
    resolution = MirrorResolution(ns=5, mpol=1, nxi=4)
    config = MirrorConfig(resolution=resolution, ftol=1.0e-12, max_iterations=1000)
    discretization = SplineMirrorDiscretization.build_closed(
        resolution,
        coefficient_count=8,
        quadrature_order=3,
    )
    basis = discretization.spline
    points = jnp.asarray(basis.collocation_nodes)
    axis = evaluate_closed_spline_axis(
        basis.fit(
            jnp.stack(
                (2.5 * jnp.cos(points), jnp.zeros_like(points), 2.5 * jnp.sin(points)),
                axis=-1,
            ),
            axis=0,
        ),
        basis,
        discretization.grid.z,
        initial_normal=jnp.asarray([0.0, 1.0, 0.0]),
    )
    boundary = SplineMirrorBoundary(jnp.full((resolution.ntheta, basis.size), 0.25))
    radius = jnp.full((resolution.ns, resolution.ntheta, basis.size), 0.25)
    initial = _initialize_closed_vacuum_stream_function(
        SplineMirrorState(radius, jnp.zeros_like(radius)),
        discretization,
        axis,
        axial_flux_derivative=0.03,
    )
    result = solve_spline_fixed_boundary(
        initial,
        boundary,
        discretization,
        config,
        axial_flux_derivative=0.03,
        solve_lambda=True,
        axis=axis,
        require_convergence=True,
    ).evaluated
    assert result.converged
    assert float(result.variational.maximum) <= config.ftol
    assert float(result.staggered_weak_force.maximum) <= config.ftol
    # Minor-radius norm: the former device-length gate of 1.0e-2 scales by
    # exactly a/L = 0.25/(2*pi*2.5) on this closed circular limit.
    assert float(result.force.normalized_rms) < 1.6e-4
    assert float(result.force.device_normalized_rms) < 1.0e-2
    assert float(result.normalized_divergence_rms) < 1.0e-12


@pytest.mark.full
@pytest.mark.usefixtures("_module_jit_enabled")
def test_stellarator_mirror_fixed_boundary_reaches_ftol() -> None:
    resolution = MirrorResolution(ns=5, mpol=2, nxi=4)
    config = MirrorConfig(resolution=resolution, ftol=1.0e-12, max_iterations=1000)
    setup = build_stellarator_mirror_hybrid(
        resolution,
        coefficient_count=16,
        straight_length=8.0,
        return_radius=2.5,
        semi_major=0.45,
        semi_minor=0.30,
        quadrature_order=3,
    )
    result = solve_spline_fixed_boundary(
        setup.initial_state,
        setup.boundary,
        setup.discretization,
        config,
        axial_flux_derivative=0.02,
        current_derivative=0.002,
        solve_lambda=True,
        axis=setup.axis,
        require_convergence=True,
    ).evaluated
    assert result.converged
    assert float(result.variational.maximum) <= config.ftol
    assert float(result.staggered_weak_force.maximum) <= config.ftol
    assert float(result.normalized_divergence_rms) < 1.0e-12
    assert not bool(result.energy.geometry.jacobian_sign_changed)
    line = trace_closed_field_line(
        result.energy.field,
        setup.discretization,
        radial_index=resolution.ns - 1,
        turns=2,
    )
    assert line.theta.shape == (513,)
    assert np.isfinite(float(line.iota))
    assert abs(float(line.iota)) > 1.0e-3


@pytest.mark.full
@pytest.mark.usefixtures("_module_jit_enabled")
def test_stellarator_mirror_toroidal_rotation_raises_transform() -> None:
    """A toroidally rotating ellipse solves and lifts iota above return-only.

    Two full section turns amplify the current-driven transform from the
    return-only ``iota=0.085`` to ``iota=0.141`` at ``s=0.75`` while the
    equilibrium still reaches ftol with a divergence-free field.
    """

    resolution = MirrorResolution(ns=5, mpol=4, nxi=4)
    config = MirrorConfig(resolution=resolution, ftol=1.0e-12, max_iterations=1000)
    setup = build_stellarator_mirror_hybrid(
        resolution,
        coefficient_count=32,
        straight_length=8.0,
        return_radius=2.5,
        semi_major=0.45,
        semi_minor=0.25,
        section_turns=2,
        quadrature_order=3,
    )
    result = solve_spline_fixed_boundary(
        setup.initial_state,
        setup.boundary,
        setup.discretization,
        config,
        axial_flux_derivative=0.02,
        current_derivative=0.002,
        solve_lambda=True,
        axis=setup.axis,
        require_convergence=True,
    ).evaluated
    assert result.converged
    assert float(result.variational.maximum) <= config.ftol
    assert float(result.normalized_divergence_rms) < 1.0e-12
    assert not bool(result.energy.geometry.jacobian_sign_changed)
    line = trace_closed_field_line(
        result.energy.field,
        setup.discretization,
        radial_index=resolution.ns - 2,
        turns=2,
    )
    assert abs(float(line.iota)) > 0.12


_CIRCULAR_HYBRID = dict(
    straight_length=8.0,
    return_radius=2.5,
    semi_major=0.45,
    semi_minor=0.45,
    quadrature_order=3,
)


def test_stellarator_mirror_junction_freeze_fixes_axis_geometry() -> None:
    """axis_coefficient_count freezes the leg-return junction geometry.

    Building the racetrack at a fixed base control count and refining the solve
    basis by exact dyadic subdivision leaves the axis curve unchanged, so the
    junction-transition width stops sharpening. Building the axis directly in a
    finer basis is a genuinely different, sharper-junction curve.
    """

    resolution = MirrorResolution(ns=5, mpol=2, nxi=4)
    points = np.linspace(0.0, 2.0 * np.pi, 401, endpoint=False)

    def axis_curve(coefficient_count: int, axis_coefficient_count: int | None) -> np.ndarray:
        setup = build_stellarator_mirror_hybrid(
            resolution,
            coefficient_count=coefficient_count,
            axis_coefficient_count=axis_coefficient_count,
            **_CIRCULAR_HYBRID,
        )
        return np.asarray(
            setup.discretization.spline.evaluate(setup.axis_coefficients, points, axis=0)
        )

    base = axis_curve(16, 16)
    frozen_32 = axis_curve(32, 16)
    frozen_64 = axis_curve(64, 16)
    # Exact spline refinement keeps the frozen junction curve identical.
    np.testing.assert_allclose(frozen_32, base, atol=1.0e-12)
    np.testing.assert_allclose(frozen_64, base, atol=1.0e-12)
    # Rebuilding the axis directly in the finer basis is a different curve.
    rebuilt_32 = axis_curve(32, None)
    assert np.max(np.abs(rebuilt_32 - base)) > 1.0e-2
    # The solve basis must be a dyadic multiple of the frozen axis basis.
    with pytest.raises(ValueError):
        build_stellarator_mirror_hybrid(
            resolution,
            coefficient_count=48,
            axis_coefficient_count=16,
            **_CIRCULAR_HYBRID,
        )


@pytest.mark.full
@pytest.mark.usefixtures("_module_jit_enabled")
def test_stellarator_mirror_frozen_junction_force_ladder_converges() -> None:
    """The frozen-junction circular-section hybrid passes the promotion gate.

    With the junction geometry frozen at 16 controls, exact refinement of the
    solve basis to 32 and 64 controls drives a monotone decrease of the
    minor-radius bulk force below the 0.05 gate, reproducing the audit's
    device-normalized 0.204 -> 0.176 -> 0.118 all-volume ladder.
    """

    resolution = MirrorResolution(ns=5, mpol=2, nxi=4)
    config = MirrorConfig(resolution=resolution, ftol=1.0e-12, max_iterations=1500)
    flux = 0.02
    device_ladder = []
    bulk_ladder = []
    for coefficient_count in (16, 32, 64):
        setup = build_stellarator_mirror_hybrid(
            resolution,
            coefficient_count=coefficient_count,
            axis_coefficient_count=16,
            axial_flux_derivative=flux,
            **_CIRCULAR_HYBRID,
        )
        result = solve_spline_fixed_boundary(
            setup.initial_state,
            setup.boundary,
            setup.discretization,
            config,
            axial_flux_derivative=flux,
            current_derivative=0.0,
            solve_lambda=True,
            axis=setup.axis,
            require_convergence=False,
        ).evaluated
        assert bool(result.converged)
        assert float(result.variational.maximum) <= config.ftol
        assert float(result.normalized_divergence_rms) < 1.0e-12
        zones = force_gate_zones(result.force)
        device_ladder.append(zones.device_all_volume)
        bulk_ladder.append(zones.bulk)

    # Device-normalized all-volume ladder reproduces the audit and is monotone.
    assert refinement_convergence(device_ladder).monotone
    assert device_ladder[0] > 0.2 > device_ladder[-1]
    # Minor-radius bulk force converges monotonically below the promotion gate.
    bulk = refinement_convergence(bulk_ladder)
    assert bulk.monotone
    assert passes_promotion_gate(bulk_ladder, absolute_gate=5.0e-2)
    assert bulk_ladder[-1] < 2.0e-3


def _spline_polynomial_state():
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=0, nxi=41),
        z_min=-1.4,
        z_max=1.4,
    )
    chebyshev_grid = config.build_grid()
    radius = 0.27 * (1.0 + 0.16 * jnp.asarray(chebyshev_grid.xi) ** 2)
    boundary = MirrorBoundary.from_radius(radius, chebyshev_grid)
    state = MirrorState.from_boundary(boundary, chebyshev_grid)
    discretization = SplineMirrorDiscretization.build(config, elements=6)
    spline_boundary = discretization.fit_boundary(boundary, chebyshev_grid)
    spline_state = discretization.fit_state(state, chebyshev_grid)
    return config, chebyshev_grid, boundary, state, discretization, spline_boundary, spline_state


def test_coefficient_native_state_matches_chebyshev_polynomial_geometry_and_energy() -> None:
    _, chebyshev_grid, _, state, discretization, spline_boundary, spline_state = _spline_polynomial_state()
    projected = discretization.project_fixed_boundary(spline_state, spline_boundary)
    evaluated = discretization.evaluate_state(projected)
    spline_geometry = evaluate_geometry(evaluated, discretization.grid)
    chebyshev_geometry = evaluate_geometry(state, chebyshev_grid)
    np.testing.assert_allclose(spline_geometry.volume, chebyshev_geometry.volume, rtol=3.0e-14)
    actual_ends = evaluated.radius_scale[:, :, [0, -1]]
    boundary_ends = discretization.evaluate_boundary(spline_boundary).radius_scale[:, [0, -1]]
    np.testing.assert_allclose(actual_ends, jnp.broadcast_to(boundary_ends, actual_ends.shape), atol=3.0e-15)

    spline_energy = mirror_energy(evaluated, discretization.grid, axial_flux_derivative=0.1)
    chebyshev_energy = mirror_energy(state, chebyshev_grid, axial_flux_derivative=0.1)
    np.testing.assert_allclose(spline_energy.total, chebyshev_energy.total, rtol=2.0e-12)
    assert projected.radius_coefficients.shape[-1] == discretization.coefficient_count
    assert evaluated.radius_scale.shape[-1] > projected.radius_coefficients.shape[-1]


def test_clamped_projection_preserves_prescribed_nested_cut_profiles() -> None:
    _, _, _, _, discretization, boundary, state = _spline_polynomial_state()
    radial_profile = jnp.linspace(0.82, 1.0, discretization.grid.ns)
    radius = state.radius_coefficients.at[:, :, 0].multiply(radial_profile[:, None])
    radius = radius.at[:, :, -1].multiply((2.0 - radial_profile)[:, None])
    projected = discretization.project_fixed_boundary(
        SplineMirrorState(radius, state.lambda_coefficients),
        boundary,
    )

    np.testing.assert_allclose(projected.radius_coefficients[-1], boundary.radius_coefficients)
    np.testing.assert_allclose(projected.radius_coefficients[1:-1, :, 0], radius[1:-1, :, 0])
    np.testing.assert_allclose(projected.radius_coefficients[1:-1, :, -1], radius[1:-1, :, -1])
    np.testing.assert_allclose(projected.radius_coefficients[0], projected.radius_coefficients[1])


def test_self_similar_cut_condition_uses_lcfs_end_sections() -> None:
    _, _, _, _, discretization, boundary, state = _spline_polynomial_state()
    varied = state.radius_coefficients.at[1:-1, :, 0].multiply(0.91)
    varied = varied.at[1:-1, :, -1].multiply(1.07)
    constrained = discretization.impose_self_similar_cuts(
        SplineMirrorState(varied, state.lambda_coefficients),
        boundary,
    )

    expected_lower = np.broadcast_to(
        np.asarray(boundary.radius_coefficients[:, 0]),
        np.asarray(constrained.radius_coefficients[:, :, 0]).shape,
    )
    expected_upper = np.broadcast_to(
        np.asarray(boundary.radius_coefficients[:, -1]),
        np.asarray(constrained.radius_coefficients[:, :, -1]).shape,
    )
    np.testing.assert_allclose(constrained.radius_coefficients[:, :, 0], expected_lower)
    np.testing.assert_allclose(constrained.radius_coefficients[:, :, -1], expected_upper)


def test_boundary_transfer_preserves_nested_self_similarity() -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=5, mpol=2, nxi=9))
    source_grid = config.build_grid()
    theta = jnp.asarray(source_grid.theta)[:, None]
    xi = jnp.asarray(source_grid.xi)[None, :]
    source_boundary = MirrorBoundary.from_radius(0.3, source_grid)
    target_boundary = MirrorBoundary.from_radius(
        0.24 * (1.0 + 0.08 * jnp.cos(2.0 * theta) * (1.0 - xi**2)),
        source_grid,
    )
    discretization = SplineMirrorDiscretization.build(config, elements=4)
    source = discretization.fit_boundary(source_boundary, source_grid)
    target = discretization.fit_boundary(target_boundary, source_grid)
    state = discretization.fit_state(MirrorState.from_boundary(source_boundary, source_grid), source_grid)
    transferred = discretization.transfer_boundary(state, source, target)
    evaluated = discretization.evaluate_state(transferred)
    evaluated_target = discretization.evaluate_boundary(target).radius_scale

    np.testing.assert_allclose(
        evaluated.radius_scale,
        jnp.broadcast_to(evaluated_target, evaluated.radius_scale.shape),
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    assert not bool(evaluate_geometry(evaluated, discretization.grid).jacobian_sign_changed)


def test_coefficient_native_energy_gradient_matches_central_difference() -> None:
    _, _, _, _, discretization, spline_boundary, spline_state = _spline_polynomial_state()
    projected = discretization.project_fixed_boundary(spline_state, spline_boundary)
    direction = (
        jnp.zeros_like(projected.radius_coefficients)
        .at[2, 0, 2:-2]
        .set(jnp.linspace(-0.2, 0.3, discretization.coefficient_count - 4))
    )

    def objective(radius_coefficients):
        candidate = SplineMirrorState(radius_coefficients, projected.lambda_coefficients)
        evaluated = discretization.evaluate_state(candidate)
        return mirror_energy(evaluated, discretization.grid, axial_flux_derivative=0.1).total

    derivative = jnp.vdot(jax.grad(objective)(projected.radius_coefficients), direction)
    step = 3.0e-6
    finite_difference = (
        objective(projected.radius_coefficients + step * direction)
        - objective(projected.radius_coefficients - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2.0e-7, atol=2.0e-7)


def _free_boundary_spline_fixture():
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=2, nxi=9),
        z_min=-1.4,
        z_max=1.4,
    )
    discretization = SplineMirrorDiscretization.build(config, elements=3, quadrature_order=5)
    theta = jnp.asarray(discretization.grid.theta)[:, None]
    nodes = jnp.asarray(discretization.spline.collocation_nodes)[None, :]
    coefficients = 0.28 * (1.0 + 0.08 * jnp.cos(2.0 * theta) * (1.0 - nodes**2))
    boundary = SplineMirrorBoundary(coefficients)
    radius = jnp.broadcast_to(
        coefficients,
        (discretization.grid.ns,) + coefficients.shape,
    )
    state = SplineMirrorState(radius, jnp.zeros_like(radius))
    return discretization, boundary, state


def test_spline_boundary_work_is_the_discrete_shape_derivative() -> None:
    discretization, boundary, _ = _free_boundary_spline_fixture()
    theta = jnp.asarray(discretization.grid.theta)[:, None]
    xi = jnp.asarray(discretization.grid.xi)[None, :]
    jump = 1.7 + 0.2 * jnp.cos(theta) * (1.0 - xi**2)
    direction = jnp.sin(jnp.arange(boundary.radius_coefficients.size, dtype=float)).reshape(
        boundary.radius_coefficients.shape
    )
    direction = direction.at[:, [0, -1]].set(0.0)
    weights = (
        jnp.asarray(discretization.grid.theta_basis.weights)[:, None]
        * jnp.asarray(discretization.grid.axial_basis.weights)[None, :]
    )

    def pressure_potential(coefficients):
        radius = discretization.evaluate_boundary(SplineMirrorBoundary(coefficients)).radius_scale
        return 0.5 * abs(float(discretization.grid.dz_dxi)) * jnp.sum(jump * radius**2 * weights)

    work = _spline_boundary_work(boundary, jump, discretization)
    derivative = jnp.vdot(work, direction)
    automatic = jnp.vdot(jax.grad(pressure_potential)(boundary.radius_coefficients), direction)
    step = 2.0e-6
    finite_difference = (
        pressure_potential(boundary.radius_coefficients + step * direction)
        - pressure_potential(boundary.radius_coefficients - step * direction)
    ) / (2.0 * step)

    np.testing.assert_allclose(derivative, automatic, rtol=3.0e-14, atol=3.0e-14)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2.0e-10, atol=2.0e-10)


def test_spline_boundary_work_normalization_preserves_constant_stress_ratio() -> None:
    discretization, boundary, _ = _free_boundary_spline_fixture()
    shape = (discretization.grid.ntheta, discretization.grid.nxi)
    residual = _spline_boundary_work_residual(
        boundary,
        2.5 * jnp.ones(shape),
        5.0 * jnp.ones(shape),
        discretization,
    )

    np.testing.assert_allclose(residual[:, 1:-1], 0.5, rtol=2.0e-15, atol=2.0e-15)


def test_spline_free_boundary_map_is_square_and_preserves_constraints() -> None:
    discretization, boundary, state = _free_boundary_spline_fixture()
    vectorizer = _SplineFreeBoundaryVectorizer.build(
        boundary,
        state,
        discretization,
        axial_flux_derivative=0.1,
        solve_lambda=True,
        calibrate_pressure=True,
        initial_mass_scale=1.2,
    )
    varied = vectorizer.pack()
    varied[: vectorizer.boundary_size] *= 1.03
    varied_boundary, varied_state, mass_scale = vectorizer.unpack(varied)
    evaluated_boundary = discretization.evaluate_boundary(varied_boundary).radius_scale
    evaluated_state = discretization.evaluate_state(varied_state)

    assert vectorizer.size == vectorizer.boundary_size + vectorizer.state_size + 1
    assert vectorizer.boundary_size == discretization.grid.ntheta * (discretization.coefficient_count - 2)
    np.testing.assert_allclose(
        varied_boundary.radius_coefficients[:, [0, -1]],
        boundary.radius_coefficients[:, [0, -1]],
    )
    np.testing.assert_allclose(evaluated_state.radius_scale[-1], evaluated_boundary, atol=3.0e-15)
    np.testing.assert_allclose(evaluated_state.radius_scale[0], evaluated_state.radius_scale[1], atol=3.0e-15)
    np.testing.assert_allclose(
        jnp.einsum(
            "j,k,ijk->i",
            discretization.grid.theta_basis.weights,
            discretization.grid.axial_basis.weights,
            evaluated_state.lambda_stream,
        ),
        0.0,
        atol=3.0e-15,
    )
    np.testing.assert_allclose(mass_scale, 1.2)


def _perturbed_cylinder(config, elements, amplitude):
    """Return coefficient inputs for fixed-boundary solver tests."""

    grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.3, grid)
    base = MirrorState.from_boundary(boundary, grid)
    s = jnp.asarray(grid.s)[:, None, None]
    xi = jnp.asarray(grid.xi)[None, None, :]
    initial = MirrorState(
        base.radius_scale + amplitude * s * (1.0 - s) * (1.0 - xi**2),
        base.lambda_stream,
    )
    discretization = SplineMirrorDiscretization.build(config, elements=elements)
    return (
        grid,
        discretization,
        discretization.fit_boundary(boundary, grid),
        discretization.fit_state(initial, grid),
    )


def test_spline_fixed_boundary_solver_recovers_cylindrical_equilibrium() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=0, nxi=9),
        z_min=-1.2,
        z_max=1.2,
        ftol=1.0e-12,
        max_iterations=300,
    )
    _, discretization, spline_boundary, spline_initial = _perturbed_cylinder(config, 4, 0.03)

    result = solve_spline_fixed_boundary(
        spline_initial,
        spline_boundary,
        discretization,
        config,
        axial_flux_derivative=0.1,
        gradient_tolerance=1.0e-12,
        require_convergence=True,
    )

    assert result.evaluated.converged
    assert result.evaluated.iterations > 0
    assert float(result.evaluated.variational.maximum) <= config.ftol
    assert float(result.evaluated.staggered_weak_force.maximum) <= 1.2 * config.ftol
    np.testing.assert_allclose(result.evaluated.state.radius_scale, 0.3, atol=3.0e-13)
    assert result.coefficient_state.radius_coefficients.shape[-1] == 7


def test_solve_fixed_boundary_from_radius_convenience_converges() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=4, nxi=9),
        z_min=-1.2,
        z_max=1.2,
        ftol=1.0e-12,
        max_iterations=1000,
    )

    result = solve_fixed_boundary_from_radius(
        0.3,
        config,
        elements=4,
        axial_flux_derivative=0.1,
    )

    assert result.evaluated.converged
    assert float(result.evaluated.variational.maximum) <= config.ftol
    np.testing.assert_allclose(result.evaluated.state.radius_scale, 0.3, atol=1.0e-11)


def test_spline_solver_raises_when_convergence_is_required() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=0, nxi=9),
        ftol=1.0e-14,
        max_iterations=1,
    )
    _, discretization, spline_boundary, spline_initial = _perturbed_cylinder(config, 4, 0.04)

    with pytest.raises(MirrorConvergenceError) as caught:
        solve_spline_fixed_boundary(
            spline_initial,
            spline_boundary,
            discretization,
            config,
            axial_flux_derivative=0.1,
            require_convergence=True,
        )

    assert not caught.value.result.converged
    assert float(caught.value.result.variational.maximum) > config.ftol


def test_spline_solver_converges_nonaxisymmetric_finite_current_state() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=1, nxi=7),
        z_min=-1.0,
        z_max=1.0,
        ftol=1.0e-12,
        max_iterations=1000,
    )
    source_grid = config.build_grid()
    theta = jnp.asarray(source_grid.theta)[:, None]
    xi = jnp.asarray(source_grid.xi)[None, :]
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.02 * jnp.cos(theta) * (1.0 - xi**2)), source_grid)
    initial = MirrorState.from_boundary(boundary, source_grid)
    discretization = SplineMirrorDiscretization.build(config, elements=3)
    result = solve_spline_fixed_boundary(
        discretization.fit_state(initial, source_grid),
        discretization.fit_boundary(boundary, source_grid),
        discretization,
        config,
        axial_flux_derivative=0.1,
        current_derivative=1.0e-3 * jnp.asarray(source_grid.s),
        solve_lambda=True,
        gradient_tolerance=1.0e-12,
        require_convergence=True,
    )
    evaluated = result.evaluated
    surface_integrals = np.einsum(
        "j,k,ijk->i",
        discretization.grid.theta_basis.weights,
        discretization.grid.axial_basis.weights,
        np.asarray(evaluated.state.lambda_stream),
    )

    assert evaluated.converged
    assert float(evaluated.variational.maximum) <= config.ftol
    assert float(evaluated.staggered_weak_force.maximum) <= 1.1 * config.ftol
    assert float(jnp.max(jnp.abs(evaluated.state.lambda_stream))) > 1.0e-3
    np.testing.assert_allclose(surface_integrals, 0.0, atol=3.0e-15)
    axis_field = np.sqrt(np.asarray(evaluated.energy.b_squared)[0])
    np.testing.assert_allclose(
        axis_field,
        np.broadcast_to(np.mean(axis_field, axis=0), axis_field.shape),
        rtol=2.0e-13,
    )


def test_supplied_field_initializer_recovers_straight_field_line_mirror() -> None:
    from vmex.mirror.analytic import StraightFieldLineMirror

    config = MirrorConfig(
        resolution=MirrorResolution(ns=9, mpol=8, nxi=17),
        z_min=-1.0,
        z_max=1.0,
    )
    source_grid = config.build_grid()
    discretization = SplineMirrorDiscretization.build(config, elements=8)
    fixture = StraightFieldLineMirror(center_field=1.0, axial_scale=2.5)
    theta = jnp.asarray(source_grid.theta)[:, None]
    z = jnp.asarray(source_grid.z)[None, :]
    oversampled_count = 8 * source_grid.ntheta
    oversampled_theta = jnp.linspace(0.0, 2.0 * jnp.pi, oversampled_count, endpoint=False)[:, None]
    oversampled_radius = fixture.boundary_radius(0.03, oversampled_theta, z)
    oversampled_modes = np.rint(np.fft.fftfreq(oversampled_count, d=1.0 / oversampled_count)).astype(int)
    retained = np.abs(oversampled_modes) <= config.resolution.mpol
    coefficients = jnp.fft.fft(oversampled_radius, axis=0)[retained] / oversampled_count
    phase = jnp.exp(1j * theta * jnp.asarray(oversampled_modes[retained])[None])
    boundary = MirrorBoundary.from_radius(
        (phase @ coefficients).real,
        source_grid,
    )
    spline_boundary = discretization.fit_boundary(boundary, source_grid)
    initial = discretization.fit_state(
        MirrorState.from_boundary(boundary, source_grid),
        source_grid,
    )
    initialized = initialize_from_cartesian_field(
        initial,
        spline_boundary,
        discretization,
        fixture.field,
    )
    state = discretization.evaluate_state(initialized.state)
    energy = mirror_energy(
        state,
        discretization.grid,
        axial_flux_derivative=initialized.axial_flux_derivative,
    )
    points = energy.geometry.xyz.reshape((-1, 3))
    supplied = jax.vmap(fixture.field)(points).reshape(energy.geometry.xyz.shape)
    reconstructed = magnetic_field_xyz(energy.field, energy.geometry)
    field_error = jnp.linalg.norm((reconstructed - supplied)[1:])
    field_error /= jnp.linalg.norm(supplied[1:])
    normal = jnp.cross(
        energy.geometry.e_theta_xyz,
        energy.geometry.e_xi_xyz,
    )
    normal /= jnp.linalg.norm(normal, axis=-1, keepdims=True)
    tangency = jnp.sum(supplied * normal, axis=-1)
    tangency /= jnp.linalg.norm(supplied, axis=-1)
    tangency_rms = jnp.sqrt(jnp.mean(tangency[1:] ** 2))
    force = isotropic_force_residual(
        energy,
        discretization.grid,
        state=state,
        axial_flux_derivative=initialized.axial_flux_derivative,
    )
    sampled = initialize_from_cartesian_field(
        initial,
        spline_boundary,
        discretization,
        supplied,
    )
    _, flux_tangent = jax.jvp(
        lambda scale: (
            initialize_from_cartesian_field(
                initial,
                spline_boundary,
                discretization,
                scale * supplied,
            ).axial_flux_derivative
        ),
        (jnp.asarray(1.0),),
        (jnp.asarray(1.0),),
    )

    assert not bool(energy.geometry.jacobian_sign_changed)
    assert float(tangency_rms) < 2.0e-4
    assert float(field_error) < 5.0e-4
    # Minor-radius norm: the former device-length bound of 6.0e-3 scales by
    # a/L = 0.03/2 for this thin analytic tube.
    assert float(force.normalized_rms) < 1.0e-4
    assert float(force.device_normalized_rms) < 6.0e-3
    assert float(jnp.max(jnp.abs(initialized.state.lambda_coefficients))) > 1.0e-6
    assert np.all(np.asarray(initialized.axial_flux_derivative) > 0.0)
    np.testing.assert_allclose(
        initialized.axial_flux_derivative,
        0.5 * fixture.center_field * 0.03**2,
        rtol=2.0e-4,
    )
    np.testing.assert_allclose(
        flux_tangent,
        initialized.axial_flux_derivative,
        rtol=2.0e-12,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        sampled.state.lambda_coefficients,
        initialized.state.lambda_coefficients,
        rtol=2.0e-12,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        sampled.axial_flux_derivative,
        initialized.axial_flux_derivative,
        rtol=2.0e-12,
        atol=2.0e-15,
    )


@pytest.mark.full
def test_equal_end_axisymmetric_mirror_is_independent_of_cut_location(_module_jit_enabled) -> None:
    from vmex.mirror.analytic import AxisymmetricPolynomialMirror

    fixture = AxisymmetricPolynomialMirror(mirror_strength=0.5)
    center_radius = []
    center_axis_field = []
    for half_length, elements in ((0.6, 6), (0.8, 8), (1.0, 10)):
        config = MirrorConfig(
            resolution=MirrorResolution(ns=9, mpol=0, nxi=17),
            z_min=-half_length,
            z_max=half_length,
            ftol=1.0e-12,
            max_iterations=1000,
        )
        discretization = SplineMirrorDiscretization.build(config, elements=elements)
        boundary_samples = fixture.boundary_radius(
            0.12,
            half_length * jnp.asarray(discretization.spline.collocation_nodes),
        )[None]
        boundary = SplineMirrorBoundary(discretization.spline.fit(boundary_samples, axis=-1))
        radius = jnp.broadcast_to(
            boundary.radius_coefficients[None],
            (config.resolution.ns,) + boundary.radius_coefficients.shape,
        )
        initialized = initialize_from_cartesian_field(
            SplineMirrorState(radius, jnp.zeros_like(radius)),
            boundary,
            discretization,
            fixture.field,
        )
        result = solve_spline_fixed_boundary(
            initialized.state,
            boundary,
            discretization,
            config,
            axial_flux_derivative=initialized.axial_flux_derivative,
            gradient_tolerance=config.ftol,
            require_convergence=True,
        ).evaluated
        center = int(np.argmin(np.abs(discretization.grid.z)))
        center_radius.append(float(result.state.radius_scale[-1, 0, center]))
        center_axis_field.append(float(jnp.sqrt(result.energy.b_squared[0, 0, center])))
        assert result.final_linear_residual < 2.0e-9
        assert float(result.variational.maximum) <= config.ftol
        assert float(result.staggered_weak_force.maximum) <= config.ftol
        # Minor-radius norm: the former device-length bounds of 6.0e-3 and
        # 2.0e-3 scale by a/L <= 0.12/1.2 across the three cut locations.
        assert float(result.force.normalized_rms) < 6.0e-4
        assert float(result.force.bulk_normalized_rms) < 2.0e-4
        assert float(result.force.device_normalized_rms) < 6.0e-3

    np.testing.assert_allclose(center_radius, center_radius[0], rtol=3.0e-8)
    np.testing.assert_allclose(center_axis_field, center_axis_field[0], rtol=2.0e-5)


def test_spline_preconditioners_are_traceable_and_locally_exact() -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=5, mpol=2, nxi=9))
    grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.3, grid)
    discretization = SplineMirrorDiscretization.build(config, elements=4)
    spline_boundary = discretization.fit_boundary(boundary, grid)
    state = discretization.fit_state(MirrorState.from_boundary(boundary, grid), grid)
    vectorizer = _SplineStateVectorizer.build(
        state,
        spline_boundary,
        discretization,
        axial_flux_derivative=0.1,
        solve_lambda=True,
    )
    apply, scales, build_local = _packed_spline_preconditioner(discretization, vectorizer)
    size = vectorizer.pack().size
    vector = jnp.asarray(np.random.default_rng(7).normal(size=size))
    assert scales.shape == (2,)
    np.testing.assert_allclose(jax.jit(apply)(vector), apply(vector), rtol=2.0e-13, atol=2.0e-13)

    diagonal = np.linspace(1.0, 3.0, size)
    matrix = np.diag(diagonal)
    stream_row = slice(
        vectorizer.radius_size,
        vectorizer.radius_size + vectorizer.lambda_free_indices.size,
    )
    coupling = np.linspace(0.01, 0.03, vectorizer.lambda_free_indices.size)
    matrix[stream_row, stream_row] += np.outer(coupling, coupling)
    batch_sizes = []

    def matrix_columns(directions):
        batch_sizes.append(directions.shape[0])
        return directions @ matrix.T

    assert build_local is not None
    local = build_local(matrix_columns)
    exact = np.random.default_rng(8).normal(size=size)
    np.testing.assert_allclose(local(matrix @ exact), exact, rtol=3.0e-14, atol=3.0e-14)
    assert max(batch_sizes) <= 32


@pytest.mark.full
def test_large_spline_solve_uses_matrix_free_coefficient_preconditioner(_module_jit_enabled) -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=29, mpol=1, nxi=25),
        z_min=-1.2,
        z_max=1.2,
        ftol=1.0e-12,
        max_iterations=300,
    )
    source_grid, discretization, spline_boundary, spline_initial = _perturbed_cylinder(config, 12, 0.03)
    assert (source_grid.ns - 2) * source_grid.ntheta * (discretization.coefficient_count - 2) > 1024

    result = solve_spline_fixed_boundary(
        spline_initial,
        spline_boundary,
        discretization,
        config,
        axial_flux_derivative=0.1,
        gradient_tolerance=1.0e-12,
        require_convergence=True,
    ).evaluated

    assert result.converged
    assert result.linear_iterations > 0
    assert result.final_linear_residual < 1.0e-5
    assert float(result.variational.maximum) <= config.ftol
    assert float(result.staggered_weak_force.maximum) <= 1.1 * config.ftol
    np.testing.assert_allclose(result.state.radius_scale, 0.3, atol=7.0e-15)


@pytest.mark.full
def test_knot_refined_rotating_ellipse_uses_matrix_free_rescue(_module_jit_enabled) -> None:
    from vmex.mirror.analytic import RotatingEllipseParaxial

    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=8, nxi=17),
        ftol=1.0e-12,
        max_iterations=2000,
    )
    source_grid = config.build_grid()
    theta = jnp.asarray(source_grid.theta)[:, None]
    z = jnp.asarray(source_grid.z)[None, :]
    discretization = SplineMirrorDiscretization.build(config, elements=6)

    def boundary(stage):
        fixture = RotatingEllipseParaxial(
            mirror_strength=0.2 * stage,
            elongation=1.0 + 0.5 * stage,
            rotation=0.5 * jnp.pi * stage,
        )
        return MirrorBoundary.from_radius(fixture.boundary_radius(0.05, theta, z), source_grid)

    source = discretization.fit_boundary(boundary(0.0), source_grid)
    state = discretization.fit_state(MirrorState.from_boundary(boundary(0.0), source_grid), source_grid)
    result = None
    for stage in (0.0, 0.25, 0.5):
        target = discretization.fit_boundary(boundary(stage), source_grid)
        state = discretization.transfer_boundary(state, source, target)
        result = solve_spline_fixed_boundary(
            state,
            target,
            discretization,
            config,
            axial_flux_derivative=0.01,
            solve_lambda=True,
            gradient_tolerance=1.0e-12,
            require_convergence=True,
        )
        state, source = result.coefficient_state, target

    assert result is not None
    assert result.evaluated.linear_iterations > 0
    assert result.evaluated.final_linear_residual < 1.0e-8
    assert float(result.evaluated.variational.maximum) <= config.ftol
    assert float(result.evaluated.staggered_weak_force.maximum) <= config.ftol
