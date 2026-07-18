"""Analytic M1 geometry and divergence-free field tests."""

from __future__ import annotations

from dataclasses import replace

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
)
from vmex.mirror.forces import isotropic_force_residual, mirror_energy  # noqa: E402
from vmex.mirror.geometry import (  # noqa: E402
    contravariant_field,
    divergence_b,
    evaluate_geometry,
    magnetic_field_squared,
    normalized_divergence_rms,
)


def _two_loop_axis_field(z, *, radius: float, separation: float, current: float):
    """Analytic on-axis field of two equal circular loops."""

    z = jnp.asarray(z)
    return sum(
        4.0e-7 * jnp.pi * current * radius**2 / (2.0 * (radius**2 + (z - position) ** 2) ** 1.5)
        for position in (-0.5 * separation, 0.5 * separation)
    )


def _axisymmetric_grid(*, ns: int = 17, nxi: int = 33, half_length: float = 1.8):
    return MirrorConfig(
        resolution=MirrorResolution(ns=ns, mpol=0, nxi=nxi),
        z_min=-half_length,
        z_max=half_length,
    ).build_grid()


def test_cylinder_geometry_matches_exact_metrics_volume_and_field() -> None:
    radius = 0.37
    half_length = 1.8
    grid = _axisymmetric_grid(half_length=half_length)
    state = MirrorState.from_boundary(MirrorBoundary.from_radius(radius, grid), grid)
    geometry = evaluate_geometry(state, grid)

    expected_jacobian = 0.5 * radius**2 * half_length
    np.testing.assert_allclose(geometry.sqrt_g, expected_jacobian, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(geometry.g_sxi, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(geometry.g_thetaxi, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(geometry.g_xixi, half_length**2, rtol=2.0e-14, atol=2.0e-14)
    expected_volume = 2.0 * np.pi * half_length * radius**2
    np.testing.assert_allclose(geometry.volume, expected_volume, rtol=2.0e-14, atol=2.0e-14)
    assert not bool(geometry.jacobian_sign_changed)

    psi_prime = 0.12
    field = contravariant_field(
        state,
        geometry,
        grid,
        axial_flux_derivative=psi_prime,
    )
    expected_b_squared = (2.0 * psi_prime / radius**2) ** 2
    np.testing.assert_allclose(
        magnetic_field_squared(field, geometry),
        expected_b_squared,
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(divergence_b(field, geometry, grid), 0.0, atol=3.0e-14)
    assert float(normalized_divergence_rms(field, geometry, grid)) < 3.0e-14


def test_polynomial_flared_tube_matches_exact_jacobian_and_volume() -> None:
    radius = 0.25
    flare = 0.18
    half_length = 1.4
    grid = _axisymmetric_grid(ns=19, nxi=41, half_length=half_length)
    xi = jnp.asarray(grid.xi)
    boundary_radius = radius * (1.0 + flare * xi**2)
    state = MirrorState.from_boundary(MirrorBoundary.from_radius(boundary_radius, grid), grid)
    geometry = evaluate_geometry(state, grid)

    expected_jacobian = np.broadcast_to(
        0.5 * np.asarray(boundary_radius)[None, None, :] ** 2 * half_length,
        geometry.sqrt_g.shape,
    )
    np.testing.assert_allclose(geometry.sqrt_g, expected_jacobian, rtol=3.0e-13, atol=3.0e-13)
    expected_volume = np.pi * half_length * np.dot(grid.axial_basis.weights, np.asarray(boundary_radius) ** 2)
    np.testing.assert_allclose(geometry.volume, expected_volume, rtol=3.0e-13, atol=3.0e-13)


def test_theta_shaped_geometry_is_positive_and_has_correct_mean_volume() -> None:
    radius = 0.31
    epsilon = 0.12
    half_length = 1.25
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=13, mpol=4, nxi=25),
        z_min=-half_length,
        z_max=half_length,
    ).build_grid()
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    boundary_radius = radius * (1.0 + epsilon * jnp.cos(2.0 * theta) * (1.0 - xi**2))
    state = MirrorState.from_boundary(MirrorBoundary.from_radius(boundary_radius, grid), grid)
    geometry = evaluate_geometry(state, grid)

    assert np.all(np.asarray(geometry.sqrt_g) > 0.0)
    assert not bool(geometry.jacobian_sign_changed)
    expected_volume = (
        0.5
        * half_length
        * np.einsum(
            "j,k,jk->",
            grid.theta_basis.weights,
            grid.axial_basis.weights,
            np.asarray(boundary_radius) ** 2,
        )
    )
    np.testing.assert_allclose(geometry.volume, expected_volume, rtol=4.0e-13, atol=4.0e-13)
    assert np.max(np.abs(np.asarray(geometry.g_stheta))) > 0.0


def test_stream_function_field_is_discretely_divergence_free_and_conserves_flux() -> None:
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=11, mpol=4, nxi=21),
        z_min=-1.7,
        z_max=1.7,
    ).build_grid()
    theta = jnp.asarray(grid.theta)[None, :, None]
    xi = jnp.asarray(grid.xi)[None, None, :]
    s = jnp.asarray(grid.s)[:, None, None]
    boundary = MirrorBoundary.from_radius(0.32 * (1.0 + 0.08 * xi[0, 0] ** 2), grid)
    base = MirrorState.from_boundary(boundary, grid)
    lam = s * (1.0 - s) * (1.0 + 0.2 * xi**3) * jnp.cos(2.0 * theta)
    state = replace(base, lambda_stream=lam)
    geometry = evaluate_geometry(state, grid)
    psi_prime = 0.15 * (1.0 - 0.3 * jnp.asarray(grid.s))
    current_prime = 0.04 * jnp.asarray(grid.s)
    field = contravariant_field(
        state,
        geometry,
        grid,
        axial_flux_derivative=psi_prime,
        current_derivative=current_prime,
    )

    np.testing.assert_allclose(divergence_b(field, geometry, grid), 0.0, rtol=0.0, atol=3.0e-12)
    axial_flux_density = grid.theta_basis.integrate(field.jac_b_xi, axis=1)
    expected_flux = np.broadcast_to(2.0 * np.pi * np.asarray(psi_prime)[:, None], axial_flux_density.shape)
    np.testing.assert_allclose(axial_flux_density, expected_flux, rtol=3.0e-13, atol=3.0e-13)


def test_geometry_volume_is_jax_differentiable_with_analytic_cylinder_gradient() -> None:
    grid = _axisymmetric_grid(ns=9, nxi=17, half_length=1.3)

    def volume(radius):
        state = MirrorState.from_boundary(MirrorBoundary.from_radius(radius, grid), grid)
        return evaluate_geometry(state, grid).volume

    radius = 0.28
    derivative = jax.grad(volume)(radius)
    expected = 4.0 * np.pi * 1.3 * radius
    np.testing.assert_allclose(derivative, expected, rtol=3.0e-13, atol=3.0e-13)
    jax.make_jaxpr(volume)(radius)




def test_two_coil_flux_tube_has_high_field_throats_and_correct_on_axis_field() -> None:
    grid = _axisymmetric_grid(ns=13, nxi=41, half_length=1.0)
    coil_radius, separation, current = 0.8, 2.0, 2.0e5
    bz_reference = _two_loop_axis_field(
        jnp.asarray(grid.z),
        radius=coil_radius,
        separation=separation,
        current=current,
    )
    center = grid.nxi // 2
    center_radius = 0.05
    flux = 0.5 * bz_reference[center] * center_radius**2
    boundary = MirrorBoundary.from_axis_field(flux, bz_reference, grid)
    radius = np.asarray(boundary.radius_scale[0])
    assert radius[center] == pytest.approx(center_radius)
    assert radius[0] < radius[center] and radius[-1] < radius[center]
    np.testing.assert_allclose(radius, radius[::-1], rtol=2.0e-14, atol=2.0e-14)

    state = MirrorState.from_boundary(boundary, grid)
    geometry = evaluate_geometry(state, grid)
    field = contravariant_field(
        state,
        geometry,
        grid,
        axial_flux_derivative=flux,
    )
    bmag_axis = np.sqrt(np.asarray(magnetic_field_squared(field, geometry))[0, 0])
    np.testing.assert_allclose(bmag_axis, bz_reference, rtol=3.0e-13, atol=3.0e-13)


def test_two_coil_paraxial_tensor_residual_decreases_with_tube_radius() -> None:
    grid = _axisymmetric_grid(ns=13, nxi=33, half_length=1.0)
    bz = _two_loop_axis_field(
        jnp.asarray(grid.z),
        radius=0.8,
        separation=2.0,
        current=2.0e5,
    )
    center = grid.nxi // 2
    residuals = []
    for center_radius in (0.1, 0.05, 0.025):
        flux = 0.5 * bz[center] * center_radius**2
        boundary = MirrorBoundary.from_axis_field(flux, bz, grid)
        state = MirrorState.from_boundary(boundary, grid)
        energy = mirror_energy(state, grid, axial_flux_derivative=flux)
        residuals.append(
            float(
                isotropic_force_residual(
                    energy,
                    grid,
                    state=state,
                    axial_flux_derivative=flux,
                ).normalized_rms
            )
        )
    assert residuals[0] > residuals[1] > residuals[2]
    np.testing.assert_allclose(
        np.asarray(residuals[:-1]) / np.asarray(residuals[1:]),
        2.0,
        rtol=3.0e-3,
    )
