"""Independent analytic gates for nonaxisymmetric paraxial mirrors."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmex.mirror.analytic import (  # noqa: E402
    AxisymmetricPolynomialMirror,
    RotatingEllipseParaxial,
    StraightFieldLineMirror,
)


def test_axisymmetric_polynomial_mirror_is_exact_and_flux_preserving() -> None:
    fixture = AxisymmetricPolynomialMirror(
        center_field=0.8,
        half_length=1.2,
        mirror_strength=0.6,
    )
    point = jnp.asarray([0.13, -0.07, 0.4])
    jacobian = jax.jacfwd(fixture.field)(point)
    curl = jnp.asarray(
        (
            jacobian[2, 1] - jacobian[1, 2],
            jacobian[0, 2] - jacobian[2, 0],
            jacobian[1, 0] - jacobian[0, 1],
        )
    )
    np.testing.assert_allclose(curl, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(jnp.trace(jacobian), 0.0, atol=2.0e-15)

    z = jnp.linspace(-1.0, 1.0, 21)
    radius = fixture.boundary_radius(0.12, z)
    np.testing.assert_allclose(
        fixture.poloidal_flux(radius, z),
        fixture.poloidal_flux(0.12, 0.0),
        rtol=3.0e-15,
        atol=3.0e-15,
    )
    np.testing.assert_allclose(radius[0], radius[-1], atol=2.0e-15)
    np.testing.assert_allclose(
        fixture.field(jnp.asarray([0.0, 0.0, 0.4]))[2],
        fixture.axis_field(0.4),
    )
    uniform = AxisymmetricPolynomialMirror(mirror_strength=0.0)
    np.testing.assert_allclose(uniform.boundary_radius(0.12, z), 0.12)


def test_rotating_ellipse_flux_vacuum_identity_and_angle() -> None:
    fixture = RotatingEllipseParaxial(elongation=2.4, mirror_strength=0.7)
    z = jnp.linspace(-0.9, 0.9, 13)
    determinants = jax.vmap(fixture.flux_determinant)(z)
    expected = fixture.reference_field / fixture.axis_field(z)
    np.testing.assert_allclose(determinants, expected, rtol=3.0e-15, atol=3.0e-15)
    np.testing.assert_allclose(jax.vmap(fixture.consistency_residual)(z), 0.0, atol=8.0e-16)
    np.testing.assert_allclose(jax.vmap(fixture.riccati_residual)(z), 0.0, atol=2.0e-15)

    def measured_angle(zz):
        covariance = fixture.section_matrix(zz) @ fixture.section_matrix(zz).T
        eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
        major = eigenvectors[:, jnp.argmax(eigenvalues)]
        return jnp.arctan2(major[1], major[0])

    measured = jax.vmap(measured_angle)(z)
    expected_angle = fixture.orientation(z) - fixture.orientation(0.0)
    angle_error = 0.5 * jnp.arctan2(
        jnp.sin(2.0 * (measured - expected_angle)),
        jnp.cos(2.0 * (measured - expected_angle)),
    )
    np.testing.assert_allclose(angle_error, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(expected_angle[-1] - expected_angle[0], 0.45 * jnp.pi, atol=2.0e-15)
    endpoint_turn = fixture.orientation(1.0) - fixture.orientation(-1.0)
    np.testing.assert_allclose(endpoint_turn, 0.5 * jnp.pi, atol=2.0e-15)


def test_rotating_ellipse_field_is_solenoidal_and_tangent() -> None:
    fixture = RotatingEllipseParaxial(elongation=1.7, mirror_strength=0.3)
    radius, alpha, z = 0.04, 0.7, 0.3
    point = fixture.section(radius, alpha, z)
    field = fixture.field(point)
    tangent = jax.jacfwd(lambda zz: fixture.section(radius, alpha, zz))(z)
    np.testing.assert_allclose(
        jnp.cross(field, tangent),
        0.0,
        atol=3.0e-15,
    )
    jacobian = jax.jacfwd(fixture.field)(point)
    np.testing.assert_allclose(jnp.trace(jacobian), 0.0, atol=2.0e-14)
    np.testing.assert_allclose(
        fixture.field(jnp.asarray([0.0, 0.0, z]))[2],
        fixture.axis_field(z),
    )


def test_rotating_ellipse_general_and_center_quadrupole_agree() -> None:
    fixture = RotatingEllipseParaxial(elongation=1.8, mirror_strength=0.4)
    general = np.asarray(fixture.quadrupole(jnp.asarray(0.0)))
    closed = np.asarray(fixture.center_quadrupole())
    np.testing.assert_allclose(general, closed, rtol=3.0e-12, atol=3.0e-12)

    alpha = jnp.linspace(0.0, 2.0 * jnp.pi, 17, endpoint=False)
    radii = jnp.asarray([2.0e-3, 1.0e-3, 5.0e-4])
    recovered = jax.vmap(
        lambda radius: (
            (jax.vmap(lambda angle: fixture.field_strength(radius, angle, 0.0))(alpha) - fixture.axis_field(0.0))
            / radius**2
        )
    )(radii)
    design = closed[0] + closed[1] * np.cos(2.0 * np.asarray(alpha))
    design += closed[2] * np.sin(2.0 * np.asarray(alpha))
    expected = np.broadcast_to(design, recovered.shape)
    np.testing.assert_allclose(recovered, expected, rtol=2.0e-7, atol=2.0e-9)

    first_order = (
        jax.vmap(lambda angle: fixture.field_strength(1.0e-4, angle, 0.0))(alpha)
        - jax.vmap(lambda angle: fixture.field_strength(-1.0e-4, angle, 0.0))(alpha)
    ) / 2.0e-4
    np.testing.assert_allclose(first_order, 0.0, atol=2.0e-12)
    section = fixture.section(0.1, alpha, 0.2)
    assert section.shape == (alpha.size, 3)
    polar_angle = jnp.arctan2(section[:, 1], section[:, 0])
    polar_radius = jnp.linalg.norm(section[:, :2], axis=1)
    np.testing.assert_allclose(fixture.boundary_radius(0.1, polar_angle, 0.2), polar_radius, atol=3.0e-16)
    assert np.isfinite(float(jax.grad(lambda zz: fixture.field_strength(0.01, 0.3, zz))(0.1)))


def test_sflm_field_is_curl_free_and_paraxially_solenoidal() -> None:
    fixture = StraightFieldLineMirror(center_field=0.8, axial_scale=2.0)
    point = jnp.asarray([0.03, -0.02, 0.5])
    jacobian = jax.jacfwd(fixture.field)(point)
    curl = jnp.asarray(
        (
            jacobian[2, 1] - jacobian[1, 2],
            jacobian[0, 2] - jacobian[2, 0],
            jacobian[1, 0] - jacobian[0, 1],
        )
    )
    np.testing.assert_allclose(curl, 0.0, atol=2.0e-14)

    def divergence(radius):
        location = jnp.asarray([radius, -0.7 * radius, 0.5])
        return jnp.trace(jax.jacfwd(fixture.field)(location))

    coarse = abs(float(divergence(0.04)))
    fine = abs(float(divergence(0.02)))
    assert fine < 0.27 * coarse
    np.testing.assert_allclose(fixture.field(jnp.asarray([0.0, 0.0, 0.5]))[2], fixture.axis_field(0.5))


def test_sflm_labels_sections_and_field_line_error_have_expected_order() -> None:
    fixture = StraightFieldLineMirror(center_field=1.2, axial_scale=2.5)
    z = jnp.linspace(-1.0, 1.0, 11)
    line = fixture.field_line(0.04, -0.03, z)
    labels = jax.vmap(fixture.clebsch_labels)(line)
    expected_labels = jnp.broadcast_to(jnp.asarray([0.04, -0.03]), labels.shape)
    np.testing.assert_allclose(labels, expected_labels, atol=2.0e-15)
    np.testing.assert_allclose(jnp.diff(line, n=2, axis=0), 0.0, atol=2.0e-15)

    alpha = jnp.linspace(0.0, 2.0 * jnp.pi, 65, endpoint=False)
    section = fixture.section(0.05, alpha, 0.8)
    semi_x = np.max(np.abs(np.asarray(section[:, 0])))
    semi_y = np.max(np.abs(np.asarray(section[:, 1])))
    np.testing.assert_allclose(semi_x / semi_y, fixture.ellipticity(0.8), rtol=5.0e-4)
    area_factor = (semi_x * semi_y) / 0.05**2
    np.testing.assert_allclose(area_factor * fixture.axis_field(0.8), fixture.center_field, rtol=5.0e-4)
    polar_angle = jnp.arctan2(section[:, 1], section[:, 0])
    polar_radius = jnp.linalg.norm(section[:, :2], axis=1)
    np.testing.assert_allclose(
        fixture.boundary_radius(0.05, polar_angle, 0.8),
        polar_radius,
        atol=3.0e-16,
    )

    def tangent_error(radius):
        point = fixture.field_line(radius, 0.6 * radius, jnp.asarray([0.4]))[0]
        tangent = jnp.asarray((radius / fixture.axial_scale, -0.6 * radius / fixture.axial_scale, 1.0))
        field = fixture.field(point)
        return jnp.linalg.norm(jnp.cross(field, tangent)) / jnp.linalg.norm(field)

    coarse = float(tangent_error(0.05))
    fine = float(tangent_error(0.025))
    assert fine < 0.27 * coarse
