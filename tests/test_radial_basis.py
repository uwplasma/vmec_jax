"""High-order radial B-spline and magnetic-axis regularity tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import BSpline

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmex.core.radial_basis import (  # noqa: E402
    BSplineBasis,
    CubicBSplineBasis,
    evaluate_regularized_mode,
)


@pytest.mark.parametrize("degree", [3, 5, 7])
@pytest.mark.parametrize("derivative", [0, 1, 2])
def test_clamped_basis_matches_scipy(degree: int, derivative: int) -> None:
    breakpoints = np.asarray([0.0, 0.04, 0.15, 0.37, 0.70, 1.0])
    basis = BSplineBasis.clamped(breakpoints, degree=degree)
    points = np.linspace(0.0, 1.0, 151)
    actual = np.asarray(basis.basis_matrix(points, derivative=derivative))
    expected = np.column_stack(
        [BSpline(basis.knots, np.eye(basis.size)[index], degree)(points, nu=derivative) for index in range(basis.size)]
    )
    tolerance = {0: 5.0e-14, 1: 2.0e-11, 2: 2.0e-8}[derivative]
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    if derivative == 0:
        np.testing.assert_allclose(np.sum(actual, axis=1), 1.0, atol=5.0e-14)
        assert np.max(np.count_nonzero(np.abs(actual) > 1.0e-14, axis=1)) <= degree + 1


@pytest.mark.parametrize("degree", [3, 5, 7])
def test_polynomial_reproduction_derivatives_and_quadrature(degree: int) -> None:
    basis = BSplineBasis.clamped(np.asarray([0.0, 0.08, 0.24, 0.51, 0.76, 1.0]), degree=degree)
    powers = jnp.arange(degree + 1)
    weights = (-1.0) ** powers / (powers + 1.0)

    def polynomial(x):
        return jnp.sum(weights * x[..., None] ** powers, axis=-1)

    coefficients = basis.fit(polynomial(jnp.asarray(basis.collocation_nodes)))
    points = jnp.linspace(0.0, 1.0, 101)
    expected_first = jnp.sum(weights[1:] * powers[1:] * points[..., None] ** (powers[1:] - 1), axis=-1)
    expected_second = jnp.sum(
        weights[2:] * powers[2:] * (powers[2:] - 1) * points[..., None] ** (powers[2:] - 2),
        axis=-1,
    )
    tolerance = {3: 2.0e-12, 5: 2.0e-11, 7: 3.0e-10}[degree]
    np.testing.assert_allclose(basis.evaluate(coefficients, points), polynomial(points), atol=tolerance)
    np.testing.assert_allclose(
        basis.evaluate(coefficients, points, derivative=1),
        expected_first,
        atol=20.0 * tolerance,
    )
    np.testing.assert_allclose(
        basis.evaluate(coefficients, points, derivative=2),
        expected_second,
        atol=300.0 * tolerance,
    )
    np.testing.assert_allclose(
        basis.integrate(coefficients),
        jnp.sum(weights / (powers + 1.0)),
        atol=10.0 * tolerance,
    )


@pytest.mark.parametrize("degree", [3, 5, 7])
def test_open_knot_insertion_is_exact_and_differentiable(degree: int) -> None:
    basis = BSplineBasis.clamped(np.linspace(0.0, 1.0, 7), degree=degree)
    coefficients = jnp.sin(1.3 * jnp.asarray(basis.collocation_nodes))
    refined, refined_coefficients = basis.insert_knot(coefficients, 0.43)
    points = jnp.linspace(0.0, 1.0, 137)
    np.testing.assert_allclose(
        refined.evaluate(refined_coefficients, points),
        basis.evaluate(coefficients, points),
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    direction = jnp.linspace(-0.2, 0.3, basis.size)
    tangent = jax.jvp(
        lambda value: basis.insert_knot(value, 0.43)[1],
        (coefficients,),
        (direction,),
    )[1]
    np.testing.assert_allclose(
        refined.evaluate(tangent, points),
        basis.evaluate(direction, points),
        rtol=3.0e-12,
        atol=3.0e-12,
    )


@pytest.mark.parametrize("degree", [3, 5, 7])
def test_periodic_refinement_closure_and_transfer_transpose(degree: int) -> None:
    basis = BSplineBasis.periodic_uniform(16, degree=degree)
    coefficients = jnp.sin(basis.collocation_nodes) + 0.2 * jnp.cos(2.0 * basis.collocation_nodes)
    refined, refined_coefficients = basis.refine_periodic_uniform(coefficients, 32)
    points = jnp.linspace(0.0, 2.0 * jnp.pi, 193)
    tolerance = {3: 3.0e-11, 5: 3.0e-9, 7: 3.0e-7}[degree]
    np.testing.assert_allclose(jnp.sum(basis.basis_matrix(points), axis=1), 1.0, atol=3.0e-13)
    for derivative in range(3):
        np.testing.assert_allclose(
            basis.basis_matrix(points[:1], derivative=derivative),
            basis.basis_matrix(points[-1:], derivative=derivative),
            atol=tolerance,
        )
        np.testing.assert_allclose(
            refined.evaluate(refined_coefficients, points, derivative=derivative),
            basis.evaluate(coefficients, points, derivative=derivative),
            rtol=tolerance,
            atol=tolerance,
        )

    direction = jnp.linspace(-0.4, 0.5, basis.size)
    cotangent = jnp.cos(jnp.arange(refined.size))
    prolongated = basis.transfer(direction, refined)
    restricted = basis.transfer_transpose(cotangent, refined)
    np.testing.assert_allclose(
        jnp.vdot(prolongated, cotangent),
        jnp.vdot(direction, restricted),
        rtol=2.0e-13,
        atol=2.0e-13,
    )


@pytest.mark.parametrize("degree", [3, 5, 7])
def test_clamped_nested_transfer_is_exact_and_has_exact_transpose(degree: int) -> None:
    source = BSplineBasis.clamped([0.0, 0.2, 0.5, 0.8, 1.0], degree=degree)
    target = BSplineBasis.clamped([0.0, 0.2, 0.37, 0.5, 0.8, 1.0], degree=degree)
    coefficients = jnp.cos(jnp.asarray(source.collocation_nodes))
    transferred = source.transfer(coefficients, target)
    points = jnp.linspace(0.0, 1.0, 151)
    np.testing.assert_allclose(
        target.evaluate(transferred, points),
        source.evaluate(coefficients, points),
        rtol=4.0e-11,
        atol=4.0e-11,
    )
    cotangent = jnp.sin(jnp.arange(target.size))
    np.testing.assert_allclose(
        jnp.vdot(transferred, cotangent),
        jnp.vdot(coefficients, source.transfer_transpose(cotangent, target)),
        rtol=2.0e-13,
        atol=2.0e-13,
    )


@pytest.mark.parametrize("mode_m", [0, 1, 2, 3, 4])
def test_regularized_modes_have_analytic_axis_limits_under_jit(mode_m: int) -> None:
    basis = BSplineBasis.clamped(np.linspace(0.0, 1.0, 7), degree=5)
    nodes = jnp.asarray(basis.collocation_nodes)
    coefficients = basis.fit(1.0 + 0.3 * nodes - 0.2 * nodes**2)
    rho = jnp.asarray([0.0, 1.0e-8, 0.02, 0.3, 0.8, 1.0])
    s = rho**2
    q = 1.0 + 0.3 * s - 0.2 * s**2
    q_s = 0.3 - 0.4 * s
    q_ss = -0.4 * jnp.ones_like(s)
    expected = [rho**mode_m * q]
    expected.append(
        mode_m * jnp.where(mode_m > 0, rho ** max(mode_m - 1, 0), 0.0) * q + 2.0 * rho ** (mode_m + 1) * q_s
    )
    expected.append(
        mode_m * (mode_m - 1) * jnp.where(mode_m > 1, rho ** max(mode_m - 2, 0), 0.0) * q
        + 2.0 * (2 * mode_m + 1) * rho**mode_m * q_s
        + 4.0 * rho ** (mode_m + 2) * q_ss
    )
    for derivative in range(3):
        actual = jax.jit(
            lambda radial_coordinate: evaluate_regularized_mode(
                basis,
                coefficients,
                radial_coordinate,
                mode_m,
                derivative=derivative,
            )
        )(s)
        np.testing.assert_allclose(actual, expected[derivative], atol=2.0e-9)
        assert np.all(np.isfinite(actual))


def test_basis_validation_and_cubic_compatibility_alias() -> None:
    assert CubicBSplineBasis.clamped([0.0, 1.0]).degree == 3
    with pytest.raises(ValueError, match="one-dimensional"):
        BSplineBasis.clamped([0.0])
    with pytest.raises(ValueError, match="degree"):
        BSplineBasis.clamped([0.0, 1.0], degree=4)
    with pytest.raises(ValueError, match="strictly increasing"):
        BSplineBasis.clamped([0.0, 0.5, 0.5, 1.0])
    with pytest.raises(ValueError, match="size"):
        BSplineBasis.periodic_uniform(5, degree=7)
    with pytest.raises(ValueError, match="stop > start"):
        BSplineBasis.periodic_uniform(8, domain=(1.0, 0.0))
    with pytest.raises(ValueError, match="positive"):
        BSplineBasis.clamped([0.0, 1.0], quadrature_order=0)
    with pytest.raises(ValueError, match="derivatives"):
        BSplineBasis.clamped([0.0, 1.0]).basis_matrix([0.5], derivative=3)
    with pytest.raises(ValueError, match="strictly inside"):
        BSplineBasis.clamped([0.0, 1.0]).insert_knot(jnp.ones(4), 0.0)
    with pytest.raises(ValueError, match="derivatives"):
        evaluate_regularized_mode(BSplineBasis.clamped([0.0, 1.0]), jnp.ones(4), jnp.ones(1), 0, derivative=3)


def test_basis_operation_shape_topology_and_refinement_errors() -> None:
    open_basis = BSplineBasis.clamped([0.0, 0.5, 1.0])
    periodic = BSplineBasis.periodic_uniform(8)
    refined_periodic = BSplineBasis.periodic_uniform(16)
    with pytest.raises(ValueError, match="coefficient axis"):
        open_basis.evaluate(jnp.ones(open_basis.size - 1), [0.5])
    with pytest.raises(ValueError, match="exactly one"):
        open_basis.fit(jnp.ones(2), nodes=[0.2, 0.8])
    with pytest.raises(ValueError, match="topology and domain"):
        open_basis.transfer_matrix_to(periodic)
    with pytest.raises(ValueError, match="source coefficient"):
        periodic.transfer(jnp.ones(7), refined_periodic)
    with pytest.raises(ValueError, match="target cotangent"):
        periodic.transfer_transpose(jnp.ones(15), refined_periodic)
    with pytest.raises(ValueError, match="refine_periodic_uniform"):
        periodic.insert_knot(jnp.ones(periodic.size), 0.4)
    with pytest.raises(ValueError, match="must be new"):
        open_basis.insert_knot(jnp.ones(open_basis.size), 0.5)
    with pytest.raises(ValueError, match="coefficient axis"):
        open_basis.insert_knot(jnp.ones(open_basis.size - 1), 0.4)
    with pytest.raises(ValueError, match="periodic basis"):
        open_basis.refine_periodic_uniform(jnp.ones(open_basis.size), 2 * open_basis.size)
    with pytest.raises(ValueError, match="dyadic multiple"):
        periodic.refine_periodic_uniform(jnp.ones(periodic.size), 7)
    with pytest.raises(ValueError, match="dyadic multiple"):
        periodic.refine_periodic_uniform(jnp.ones(periodic.size), 24)
    with pytest.raises(ValueError, match="coefficient axis"):
        periodic.refine_periodic_uniform(jnp.ones(periodic.size - 1), 16)


def test_equality_and_hash_follow_content_not_identity() -> None:
    """A basis rides in jit pytree metadata: two equal-content builds must be
    one compilation-cache key, and different content must not collide."""
    a = BSplineBasis.clamped(np.linspace(0.0, 1.0, 5))
    b = BSplineBasis.clamped(np.linspace(0.0, 1.0, 5))
    c = BSplineBasis.clamped(np.linspace(0.0, 1.0, 6))
    assert a is not b
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != object()
