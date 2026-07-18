"""M2 isotropic energy, constraints, and physical-force diagnostics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from solvax import gmres

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmex.mirror import (  # noqa: E402
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
)
from vmex.mirror.forces import (  # noqa: E402
    MU0,
    _interpolate_radius_scale,
    isotropic_force_residual,
    isotropic_staggered_energy_gradient,
    isotropic_staggered_fixed_boundary_gradient,
    isotropic_staggered_weak_residual,
    mass_profile_from_pressure,
    mirror_energy,
    staggered_field_strength,
)
from vmex.mirror.solver import (  # noqa: E402
    SeparableMirrorPreconditioner,
    _bounded_newton_krylov,
    _valid_energy_objective,
)
from vmex.mirror.model import project_fixed_boundary_state  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _enable_solver_jit():
    """Exercise nonlinear solver tests in their production execution mode."""

    previous = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    yield
    jax.config.update("jax_disable_jit", previous)


def _cylinder(*, ns: int = 11, nxi: int = 21, radius: float = 0.3, half_length: float = 1.2):
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=ns, mpol=0, nxi=nxi),
        z_min=-half_length,
        z_max=half_length,
    ).build_grid()
    boundary = MirrorBoundary.from_radius(radius, grid)
    return grid, boundary, MirrorState.from_boundary(boundary, grid)


def test_fixed_boundary_projection_enforces_geometry_and_lambda_gauge() -> None:
    grid = MirrorConfig(resolution=MirrorResolution(ns=7, mpol=3, nxi=13)).build_grid()
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.05 * jnp.cos(2.0 * theta) * xi**2), grid)
    rng = np.random.default_rng(17)
    state = MirrorState(
        radius_scale=jnp.asarray(0.2 + 0.1 * rng.random(grid.shape)),
        lambda_stream=jnp.asarray(1.0 + rng.normal(size=grid.shape)),
    )
    projected = project_fixed_boundary_state(state, boundary, grid)
    np.testing.assert_allclose(projected.radius_scale[-1], boundary.radius_scale)
    expected_cut = np.asarray(state.radius_scale[:, :, 0]).copy()
    expected_cut[-1] = np.asarray(boundary.radius_scale[:, 0])
    first_ring_modes = np.fft.fft(expected_cut[1])
    modes = np.rint(np.fft.fftfreq(grid.ntheta, d=1.0 / grid.ntheta)).astype(int)
    first_ring_modes[np.abs(modes) % 2 == 1] = 0.0
    expected_cut[0] = np.fft.ifft(first_ring_modes).real
    np.testing.assert_allclose(projected.radius_scale[:, :, 0], expected_cut)
    axis_modes = np.fft.fft(np.asarray(projected.radius_scale[0]), axis=0)
    np.testing.assert_allclose(axis_modes[np.abs(modes) % 2 == 1], 0.0, atol=2.0e-16)
    np.testing.assert_allclose(projected.lambda_stream[0], projected.lambda_stream[1])
    surface_mean = np.einsum(
        "j,k,ijk->i",
        grid.theta_basis.weights,
        grid.axial_basis.weights,
        np.asarray(projected.lambda_stream),
    ) / (4.0 * np.pi)
    np.testing.assert_allclose(surface_mean, 0.0, atol=1.0e-15)


def test_radius_interpolation_respects_odd_mode_axis_regularity() -> None:
    grid = MirrorConfig(resolution=MirrorResolution(ns=7, mpol=3, nxi=5)).build_grid()
    s = jnp.asarray(grid.s)[:, None, None]
    theta = jnp.asarray(grid.theta)[None, :, None]
    radius_scale = 0.25 + 0.03 * jnp.sqrt(s) * jnp.cos(theta) + 0.02 * s * jnp.cos(2.0 * theta)
    radius_scale = jnp.broadcast_to(radius_scale, grid.shape)
    fraction = jnp.asarray([0.2, 0.7])[:, None, None, None]

    values, derivatives = _interpolate_radius_scale(radius_scale, grid, fraction)
    s_quadrature = jnp.asarray(grid.s[:-1])[None, :, None, None] + fraction * (grid.s[1] - grid.s[0])
    theta_quadrature = jnp.asarray(grid.theta)[None, None, :, None]
    expected_values = 0.25 + 0.03 * jnp.sqrt(s_quadrature) * jnp.cos(theta_quadrature)
    expected_values += 0.02 * s_quadrature * jnp.cos(2.0 * theta_quadrature)
    expected_derivatives = 0.015 / jnp.sqrt(s_quadrature) * jnp.cos(theta_quadrature)
    expected_derivatives += 0.02 * jnp.cos(2.0 * theta_quadrature)
    expected_values = jnp.broadcast_to(expected_values, values.shape)
    expected_derivatives = jnp.broadcast_to(expected_derivatives, derivatives.shape)

    np.testing.assert_allclose(values, expected_values, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(derivatives, expected_derivatives, rtol=3.0e-13, atol=3.0e-13)
    tangent = jax.jvp(
        lambda scale: _interpolate_radius_scale(scale * radius_scale, grid, fraction)[0],
        (1.0,),
        (0.4,),
    )[1]
    np.testing.assert_allclose(tangent, 0.4 * values, rtol=2.0e-14, atol=2.0e-14)


def test_vacuum_cylinder_has_exact_energy_and_negligible_physical_force() -> None:
    radius, half_length, psi_prime = 0.3, 1.2, 0.1
    grid, boundary, state = _cylinder(radius=radius, half_length=half_length)
    energy = mirror_energy(state, grid, axial_flux_derivative=psi_prime)
    expected_b = 2.0 * psi_prime / radius**2
    expected_volume = 2.0 * np.pi * half_length * radius**2
    expected_energy = expected_b**2 * expected_volume / (2.0 * MU0)
    np.testing.assert_allclose(energy.magnetic, expected_energy, rtol=3.0e-14, atol=3.0e-9)
    np.testing.assert_allclose(energy.pressure_energy, 0.0)

    residual = isotropic_force_residual(
        energy,
        grid,
        state=state,
        axial_flux_derivative=psi_prime,
    )
    assert float(residual.normalized_rms) < 2.0e-13
    assert float(residual.bulk_normalized_rms) < 2.0e-13
    assert float(residual.axis_normalized_rms) < 2.0e-13
    assert float(residual.axis_field_nonuniformity) < 2.0e-15
    np.testing.assert_allclose(residual.component_rms[1:], 0.0, atol=2.0e-14)

    gradient = jax.grad(
        lambda trial: mirror_energy(
            project_fixed_boundary_state(trial, boundary, grid),
            grid,
            axial_flux_derivative=psi_prime,
        ).total
    )(state)
    assert float(jnp.max(jnp.abs(gradient.radius_scale))) / expected_energy < 5.0e-14
    np.testing.assert_allclose(gradient.lambda_stream, 0.0, atol=2.0e-12)


def test_mass_profile_recovers_reference_isotropic_pressure() -> None:
    pressure = 2.5e4
    grid, _, state = _cylinder()
    vacuum = mirror_energy(state, grid, axial_flux_derivative=0.1)
    mass = mass_profile_from_pressure(pressure, vacuum.volume_derivative)
    finite_beta = mirror_energy(
        state,
        grid,
        axial_flux_derivative=0.1,
        mass_profile=mass,
    )
    np.testing.assert_allclose(finite_beta.pressure, pressure, rtol=2.0e-14, atol=2.0e-10)
    expected = pressure * float(vacuum.volume_derivative[0]) / (5.0 / 3.0 - 1.0)
    np.testing.assert_allclose(finite_beta.pressure_energy, expected, rtol=3.0e-14, atol=3.0e-9)
    assert (
        float(
            isotropic_force_residual(
                finite_beta,
                grid,
                state=state,
                axial_flux_derivative=0.1,
                mass_profile=mass,
            ).normalized_rms
        )
        < 2.0e-13
    )


def test_manufactured_radial_pressure_balance_converges_second_order() -> None:
    """Resolve a cylindrical equilibrium with analytic radial ``B_z`` and pressure."""

    residuals = []
    for ns in (9, 17, 33):
        grid, _, _ = _cylinder(ns=ns, nxi=9)
        s = jnp.asarray(grid.s)
        radius, shaping, flux = 0.3, 0.4, 0.1
        radius_scale = radius * jnp.sqrt(1.0 + shaping * s)
        state = MirrorState(
            jnp.broadcast_to(radius_scale[:, None, None], grid.shape),
            jnp.zeros(grid.shape),
        )
        vacuum = mirror_energy(state, grid, axial_flux_derivative=flux)
        radial_jacobian = 0.5 * radius**2 * (1.0 + 2.0 * shaping * s)
        field = flux / radial_jacobian
        pressure = 2.0e5 + (field[-1] ** 2 - field**2) / (2.0 * MU0)
        mass = mass_profile_from_pressure(
            pressure,
            vacuum.volume_derivative,
        )
        energy = mirror_energy(
            state,
            grid,
            axial_flux_derivative=flux,
            mass_profile=mass,
        )
        residuals.append(
            float(
                isotropic_force_residual(
                    energy,
                    grid,
                    state=state,
                    axial_flux_derivative=flux,
                    mass_profile=mass,
                ).normalized_rms
            )
        )

    assert residuals[0] > residuals[1] > residuals[2]
    np.testing.assert_allclose(
        np.asarray(residuals[:-1]) / np.asarray(residuals[1:]),
        4.1,
        rtol=0.08,
    )


def test_staggered_polynomial_force_converges_with_lambda_current_and_pressure() -> None:
    """Check every force component against a closed-form cylindrical state."""

    errors = []
    for ns in (9, 17, 33):
        radius, half_length, stream_amplitude = 0.31, 1.2, 0.008
        grid = MirrorConfig(
            resolution=MirrorResolution(ns=ns, mpol=4, nxi=9),
            z_min=-half_length,
            z_max=half_length,
        ).build_grid()
        s = jnp.asarray(grid.s)[:, None, None]
        theta = jnp.asarray(grid.theta)[None, :, None]
        xi = jnp.asarray(grid.xi)[None, None, :]
        stream_factor = stream_amplitude * s * (1.0 - s)
        lam = stream_factor * jnp.cos(2.0 * theta) * (1.0 - xi**2)
        state = MirrorState(jnp.full(grid.shape, radius), lam)
        flux_profile = 0.09 + 0.015 * jnp.asarray(grid.s)
        current_profile = 0.012 + 0.01 * jnp.asarray(grid.s)
        pressure = 2.0e4 - 3.0e3 * jnp.asarray(grid.s)
        vacuum = mirror_energy(
            state,
            grid,
            axial_flux_derivative=flux_profile,
            current_derivative=current_profile,
        )
        mass = mass_profile_from_pressure(pressure, vacuum.volume_derivative)
        energy = mirror_energy(
            state,
            grid,
            axial_flux_derivative=flux_profile,
            current_derivative=current_profile,
            mass_profile=mass,
        )
        residual = isotropic_force_residual(
            energy,
            grid,
            state=state,
            axial_flux_derivative=flux_profile,
            current_derivative=current_profile,
            mass_profile=mass,
        )

        jacobian = 0.5 * radius**2 * half_length
        lambda_xi = -2.0 * stream_factor * jnp.cos(2.0 * theta) * xi
        lambda_theta = -2.0 * stream_factor * jnp.sin(2.0 * theta) * (1.0 - xi**2)
        field_theta_numerator = current_profile[:, None, None] - lambda_xi
        field_xi_numerator = flux_profile[:, None, None] + lambda_theta
        b_sup_theta = field_theta_numerator / jacobian
        b_sup_xi = field_xi_numerator / jacobian
        current_s_numerator = (
            -4.0 * half_length**2 * stream_factor * jnp.cos(2.0 * theta) * (1.0 - xi**2) / jacobian
            - 2.0 * radius**2 * s * stream_factor * jnp.cos(2.0 * theta) / jacobian
        )
        lambda_theta_s = -2.0 * stream_amplitude * (1.0 - 2.0 * s) * jnp.sin(2.0 * theta) * (1.0 - xi**2)
        lambda_xi_s = -2.0 * stream_amplitude * (1.0 - 2.0 * s) * jnp.cos(2.0 * theta) * xi
        current_theta_numerator = -half_length**2 * (0.015 + lambda_theta_s) / jacobian
        current_xi_numerator = radius**2 * (
            field_theta_numerator + s * (0.01 - lambda_xi_s)
        ) / jacobian
        expected = (
            (current_theta_numerator * b_sup_xi - current_xi_numerator * b_sup_theta) / MU0 + 3.0e3,
            -current_s_numerator * b_sup_xi / MU0,
            current_s_numerator * b_sup_theta / MU0,
        )
        numerical = (residual.covariant_s, residual.covariant_theta, residual.covariant_xi)
        errors.append(
            [
                np.sqrt(np.mean((np.asarray(got)[1:-1] - np.asarray(want)[1:-1]) ** 2))
                / np.sqrt(np.mean(np.asarray(want)[1:-1] ** 2))
                for got, want in zip(numerical, expected, strict=True)
            ]
        )

    errors = np.asarray(errors)
    np.testing.assert_allclose(errors[:-1] / errors[1:], 4.0, rtol=0.08)
    assert np.max(errors[-1]) < 3.0e-3


def test_nonaxisymmetric_coordinates_recover_uniform_cartesian_field() -> None:
    """A shaped self-similar tube must not create a spurious Lorentz force."""

    config = MirrorConfig(
        resolution=MirrorResolution(ns=9, mpol=4, nxi=9),
    )
    grid = config.build_grid()
    theta = jnp.asarray(grid.theta)
    radius_scale = 0.3 * (1.0 + 0.12 * jnp.cos(2.0 * theta))
    radial_jacobian = 0.5 * radius_scale**2
    axial_flux = jnp.sum(jnp.asarray(grid.theta_basis.weights) * radial_jacobian) / (2.0 * jnp.pi)
    stream_derivative = radial_jacobian - axial_flux
    modes = jnp.fft.fft(stream_derivative)
    mode_numbers = jnp.fft.fftfreq(grid.ntheta, 1.0 / grid.ntheta)
    inverse_derivative = jnp.where(mode_numbers == 0.0, 0.0, 1.0 / (1j * mode_numbers))
    stream = jnp.fft.ifft(modes * inverse_derivative).real
    state = MirrorState(
        jnp.broadcast_to(radius_scale[None, :, None], grid.shape),
        jnp.broadcast_to(stream[None, :, None], grid.shape),
    )
    energy = mirror_energy(state, grid, axial_flux_derivative=axial_flux)
    residual = isotropic_force_residual(
        energy,
        grid,
        state=state,
        axial_flux_derivative=axial_flux,
    )

    np.testing.assert_allclose(energy.b_squared, 1.0, rtol=4.0e-15, atol=4.0e-15)
    assert float(residual.normalized_rms) < 1.0e-12
    assert float(residual.axis_field_nonuniformity) < 1.0e-14


def test_optimizer_merit_rejects_crossed_flux_surfaces() -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=5, nxi=7))
    grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.3, grid)
    valid = MirrorState.from_boundary(boundary, grid)
    valid_energy = mirror_energy(valid, grid, axial_flux_derivative=0.1)
    invalid = replace(valid, radius_scale=valid.radius_scale.at[1].set(1.2))
    invalid_energy = mirror_energy(invalid, grid, axial_flux_derivative=0.1)

    assert not bool(valid_energy.geometry.jacobian_sign_changed)
    assert bool(invalid_energy.geometry.jacobian_sign_changed)
    np.testing.assert_allclose(_valid_energy_objective(valid_energy, float(valid_energy.total)), 1.0)
    assert np.isinf(float(_valid_energy_objective(invalid_energy, 1.0)))


def test_staggered_field_strength_is_exact_for_uniform_cylinder() -> None:
    grid, _, state = _cylinder(ns=7, nxi=13)
    mod_b = staggered_field_strength(state, grid, axial_flux_derivative=0.1)

    np.testing.assert_allclose(mod_b, mod_b[0, 0, 0], rtol=3.0e-15, atol=3.0e-15)


def test_staggered_first_variation_matches_autodiff_for_3d_finite_beta() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=3, nxi=11),
        z_min=-1.3,
        z_max=1.1,
    )
    grid = config.build_grid()
    theta = jnp.asarray(grid.theta)[None, :, None]
    xi = jnp.asarray(grid.xi)[None, None, :]
    s = jnp.asarray(grid.s)[:, None, None]
    radius = 0.31 * (1.0 + 0.04 * s * jnp.cos(2.0 * theta) * (1.0 - xi**2) + 0.03 * s * xi)
    lam = 0.006 * s * jnp.sin(theta) * (1.0 - xi**2)
    state = MirrorState(radius, lam)
    kwargs = {
        "axial_flux_derivative": jnp.linspace(0.09, 0.12, grid.ns),
        "current_derivative": jnp.linspace(0.01, 0.025, grid.ns),
        "mass_profile": 1.2e3 * (1.0 - jnp.asarray(grid.s)) ** 2,
    }
    automatic = jax.grad(lambda trial: mirror_energy(trial, grid, **kwargs).total)(state)
    staggered = isotropic_staggered_energy_gradient(state, grid, **kwargs)
    np.testing.assert_allclose(
        staggered.radius_scale,
        automatic.radius_scale,
        rtol=2.0e-12,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        staggered.lambda_stream,
        automatic.lambda_stream,
        rtol=2.0e-12,
        atol=2.0e-8,
    )


def test_staggered_weak_force_matches_fixed_boundary_projection() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=3, nxi=11),
        z_min=-1.3,
        z_max=1.1,
    )
    grid = config.build_grid()
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    boundary = MirrorBoundary.from_radius(
        0.31 * (1.0 + 0.03 * jnp.cos(2.0 * theta) * (1.0 - xi**2)),
        grid,
    )
    state = MirrorState.from_boundary(boundary, grid)
    s = jnp.asarray(grid.s)[:, None, None]
    state = replace(
        state,
        radius_scale=state.radius_scale + 0.01 * s * (1.0 - s) * (1.0 - xi[None] ** 2),
        lambda_stream=0.004 * s * jnp.sin(jnp.asarray(grid.theta))[None, :, None] * (1.0 - xi[None] ** 2),
    )
    kwargs = {
        "axial_flux_derivative": jnp.linspace(0.09, 0.12, grid.ns),
        "current_derivative": jnp.linspace(0.01, 0.025, grid.ns),
        "mass_profile": 1.2e3 * (1.0 - jnp.asarray(grid.s)) ** 2,
    }
    automatic = jax.grad(
        lambda trial: mirror_energy(
            project_fixed_boundary_state(trial, boundary, grid),
            grid,
            **kwargs,
        ).total
    )(state)
    staggered = isotropic_staggered_fixed_boundary_gradient(
        state,
        boundary,
        grid,
        **kwargs,
    )
    np.testing.assert_allclose(
        staggered.radius_scale,
        automatic.radius_scale,
        rtol=3.0e-12,
        atol=3.0e-8,
    )
    np.testing.assert_allclose(
        staggered.lambda_stream,
        automatic.lambda_stream,
        rtol=3.0e-12,
        atol=3.0e-8,
    )
    weak = isotropic_staggered_weak_residual(state, boundary, grid, **kwargs)
    assert float(weak.maximum) > 0.0


def test_radial_gauss_quadrature_controls_lambda_checkerboard_mode() -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=15, mpol=1, nxi=15))
    grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.3, grid)
    base = MirrorState.from_boundary(boundary, grid)
    s = jnp.asarray(grid.s)[:, None, None]
    theta = jnp.asarray(grid.theta)[None, :, None]
    xi = jnp.asarray(grid.xi)[None, None, :]
    mode = 0.01 * jnp.cos(theta) * (1.0 - xi**2)
    alternating = (-1.0) ** jnp.arange(grid.ns)[:, None, None] * mode
    smooth = jnp.sin(jnp.pi * s) * mode
    baseline = mirror_energy(base, grid, axial_flux_derivative=0.1).total

    def excess_energy(lam):
        state = project_fixed_boundary_state(MirrorState(base.radius_scale, lam), boundary, grid)
        return mirror_energy(state, grid, axial_flux_derivative=0.1).total - baseline

    alternating_energy = excess_energy(alternating)
    smooth_energy = excess_energy(smooth)
    assert float(alternating_energy) > 0.2 * float(smooth_energy)
    assert float(alternating_energy) < 2.0 * float(smooth_energy)


def test_flared_tube_manufactured_lorentz_force_converges_spectrally() -> None:
    """Compare the continuum force to a closed-form divergence-free field."""

    relative_errors = []
    for nxi in (9, 13, 17):
        half_length = 1.4
        grid, _, _ = _cylinder(ns=9, nxi=nxi, half_length=half_length)
        xi = jnp.asarray(grid.xi)
        radius = 0.3 * (1.0 + 0.12 * xi**2)
        boundary = MirrorBoundary.from_radius(radius, grid)
        state = MirrorState.from_boundary(boundary, grid)
        flux = 0.1
        residual = isotropic_force_residual(
            mirror_energy(state, grid, axial_flux_derivative=flux),
            grid,
            state=state,
            axial_flux_derivative=flux,
        )

        radius_z = 0.3 * 0.24 * xi / half_length
        radius_zz = jnp.full_like(xi, 0.3 * 0.24 / half_length**2)
        logarithmic_slope = radius_z / radius
        curvature = radius_zz / radius - 3.0 * logarithmic_slope**2
        axial_field = 2.0 * flux / radius**2
        exact_covariant_s = 0.5 * radius**2 * axial_field**2 * curvature / MU0
        numerical = np.asarray(residual.covariant_s)[1:, 0, :]
        error = np.max(np.abs(numerical - np.asarray(exact_covariant_s)))
        relative_errors.append(error / np.max(np.abs(np.asarray(exact_covariant_s))))
        np.testing.assert_allclose(residual.covariant_xi[1:], 0.0, atol=3.0e-11)

    assert relative_errors[1] < 2.0e-3 * relative_errors[0]
    assert relative_errors[2] < 2.0e-3 * relative_errors[1]
    assert relative_errors[-1] < 3.0e-11


def test_separable_preconditioner_is_exact_for_its_model_and_reduces_gmres_work() -> None:
    grid, _, _ = _cylinder(ns=17, nxi=33)
    preconditioner = SeparableMirrorPreconditioner.build(grid)
    rng = np.random.default_rng(42)
    exact = rng.normal(size=preconditioner.size)
    right_hand_side = preconditioner.operator(exact)
    np.testing.assert_allclose(preconditioner.apply(right_hand_side), exact, rtol=2.0e-12, atol=2.0e-12)

    plain = gmres(
        preconditioner.operator,
        right_hand_side,
        rtol=1.0e-11,
        atol=0.0,
        restart=20,
        max_restarts=50,
    )
    accelerated = gmres(
        preconditioner.operator,
        right_hand_side,
        precond=preconditioner.apply,
        rtol=1.0e-11,
        atol=0.0,
        restart=20,
        max_restarts=50,
    )
    assert bool(plain.converged) and bool(accelerated.converged)
    assert int(plain.iterations) >= 10
    assert int(accelerated.iterations) <= 2
    np.testing.assert_allclose(plain.x, exact, rtol=2.0e-9, atol=2.0e-9)
    np.testing.assert_allclose(accelerated.x, exact, rtol=2.0e-12, atol=2.0e-12)


@pytest.mark.parametrize("use_objective", [False, True])
def test_shared_bounded_newton_driver_reports_true_linear_residual(use_objective) -> None:
    matrix = np.diag([2.0, 3.0, 5.0])
    right_hand_side = np.array([0.5, -0.75, 1.0])
    inverse = np.diag(1.0 / np.diag(matrix))
    matrix_jax, inverse_jax = jnp.asarray(matrix), jnp.asarray(inverse)

    def residual(x):
        return matrix @ x - right_hand_side

    objective = (
        (lambda x: 0.5 * x @ matrix @ x - right_hand_side @ x)
        if use_objective
        else None
    )
    records = []
    result = _bounded_newton_krylov(
        np.zeros(3),
        residual,
        lambda _x, _residual: (
            lambda vector: matrix_jax @ vector,
            lambda vector: inverse_jax @ vector,
        ),
        (-np.ones(3), np.ones(3)),
        ftol=1.0e-12,
        max_steps=3,
        record_step=lambda x: records.append(np.array(x)),
        restart=3,
        max_restarts=2,
        linear_rtol=lambda _residual_max: 1.0e-12,
        objective_function=objective,
    )
    solution, steps, iterations, linear_residual, converged, _ = result
    np.testing.assert_allclose(solution, np.linalg.solve(matrix, right_hand_side), atol=1.0e-14)
    assert converged and steps == 1 and iterations > 0 and records
    assert linear_residual < 1.0e-12
