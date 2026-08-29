"""Independent tests for the high-order toroidal strong-force oracle."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core.fourier import Resolution
from vmex.core.input import VmecInput
from vmex.core.profiles import MU0
from vmex.core.radial_basis import BSplineBasis
from vmex.core.solver import _initial_state, prepare_runtime, resolution_from_input
from vmex.core.strong_force import (
    HighOrderEquilibriumState,
    certify_strong_force,
    evaluate_strong_force,
    high_order_state_from_wout,
    lift_high_order_state,
    plot_strong_force_report,
)
from vmex.core.wout import wout_from_state

jax.config.update("jax_enable_x64", True)


def _constant_toroidal_field_state(*, degree: int = 5, desc_orientation: bool = False) -> HighOrderEquilibriumState:
    """Concentric circular surfaces carrying constant toroidal field B=-e_phi."""

    basis = BSplineBasis.clamped(np.linspace(0.0, 1.0, 5), degree=degree)
    m = np.asarray([0, 1])
    n = np.asarray([0, 0])
    zeros = np.zeros((2, basis.size))
    r_cos = zeros.copy()
    z_sin = zeros.copy()
    r_cos[0] = 10.0
    r_cos[1] = 1.0
    z_sign = -1.0 if desc_orientation else 1.0
    z_sin[1] = z_sign
    # sqrt(g) is negative for this (rho, theta, zeta) parameterization.  A
    # constant phipf=1/2 gives B=-e_phi when a=1.
    phipf = np.full((basis.size,), 0.5)
    profile_zero = np.zeros((basis.size,))
    return HighOrderEquilibriumState(
        radial_basis=basis,
        m=m,
        n=n,
        nfp=1,
        R_cos=jnp.asarray(r_cos),
        R_sin=jnp.asarray(zeros),
        Z_cos=jnp.asarray(zeros),
        Z_sin=jnp.asarray(z_sin),
        L_cos=jnp.asarray(zeros),
        L_sin=jnp.asarray(zeros),
        phipf=jnp.asarray(phipf),
        chipf=jnp.asarray(profile_zero),
        pressure=jnp.asarray(profile_zero),
        jacobian_sign=1 if desc_orientation else -1,
        boundary_R_cos=jnp.asarray([10.0, 1.0]),
        boundary_R_sin=jnp.asarray([0.0, 0.0]),
        boundary_Z_cos=jnp.asarray([0.0, 0.0]),
        boundary_Z_sin=jnp.asarray([0.0, z_sign]),
    )


def test_constant_toroidal_field_matches_analytic_current_and_force():
    state = _constant_toroidal_field_state()
    rho = jnp.asarray([0.03, 0.2, 0.7, 0.95])
    theta = jnp.asarray([0.1, 1.2, 2.4, 5.2])
    zeta = jnp.asarray([0.4, 1.1, 3.0, 5.7])
    result = evaluate_strong_force(state, rho, theta, zeta)

    radius = 10.0 + rho * jnp.cos(theta)
    e_phi = jnp.stack((-jnp.sin(zeta), jnp.cos(zeta), jnp.zeros_like(zeta)), axis=-1)
    e_R = jnp.stack((jnp.cos(zeta), jnp.sin(zeta), jnp.zeros_like(zeta)), axis=-1)
    expected_B = -e_phi
    expected_J = -jnp.asarray([0.0, 0.0, 1.0]) / (MU0 * radius[:, None])
    expected_force = -e_R / (MU0 * radius[:, None])

    np.testing.assert_allclose(result.B, expected_B, rtol=2e-12, atol=2e-12)
    # Nested second derivatives leave sub-pico-T/m Cartesian cancellation
    # noise in components whose analytic value is zero.
    np.testing.assert_allclose(result.J, expected_J, rtol=2e-11, atol=2e-7)
    np.testing.assert_allclose(result.force, expected_force, rtol=2e-11, atol=2e-7)
    assert np.all(np.isfinite(result.force_rho))
    assert np.all(np.isfinite(result.force_helical))


def test_independent_oracle_agrees_with_desc_pointwise_current_and_force():
    oracle_path = Path(__file__).parent / "data" / "desc_strong_force_circular.json"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    nodes = np.asarray(oracle["nodes"])
    result = evaluate_strong_force(
        _constant_toroidal_field_state(desc_orientation=True),
        jnp.asarray(nodes[:, 0]),
        jnp.asarray(nodes[:, 1]),
        jnp.asarray(nodes[:, 2]),
    )
    e_R = np.stack(
        (np.cos(nodes[:, 2]), np.sin(nodes[:, 2]), np.zeros(nodes.shape[0])),
        axis=-1,
    )
    e_phi = np.stack(
        (-np.sin(nodes[:, 2]), np.cos(nodes[:, 2]), np.zeros(nodes.shape[0])),
        axis=-1,
    )

    def cylindrical(vectors):
        vectors = np.asarray(vectors)
        return np.stack(
            (
                np.sum(vectors * e_R, axis=-1),
                np.sum(vectors * e_phi, axis=-1),
                vectors[:, 2],
            ),
            axis=-1,
        )

    np.testing.assert_allclose(result.sqrt_g, oracle["sqrt_g"], rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(cylindrical(result.B), oracle["B"], rtol=3e-13, atol=3e-13)
    # Radial/azimuthal current and the orthogonal force components vanish
    # analytically in this case and are cancellations of nested coordinate
    # derivatives. Bound their cross-device roundoff against the corresponding
    # nonzero physical scale rather than with CPU-specific absolute thresholds
    # (Linux x64 reaches about 2.2e-8 here).
    oracle_J = np.asarray(oracle["J"])
    current_roundoff = float(5.0e-13 * np.max(np.abs(oracle_J)))
    oracle_force = np.asarray(oracle["F"])
    force_roundoff = float(5.0e-13 * np.max(np.abs(oracle_force)))
    np.testing.assert_allclose(
        cylindrical(result.J), oracle_J, rtol=3e-10, atol=current_roundoff
    )
    np.testing.assert_allclose(
        cylindrical(result.force), oracle_force, rtol=3e-10, atol=force_roundoff
    )
    np.testing.assert_allclose(
        result.force_rho, oracle["F_rho"], rtol=3e-10, atol=force_roundoff
    )
    np.testing.assert_allclose(
        result.force_helical, oracle["F_helical"], rtol=3e-10, atol=force_roundoff
    )


def test_point_evaluation_is_jittable_and_finite_near_axis():
    state = _constant_toroidal_field_state(degree=7)
    evaluate = jax.jit(lambda rho, theta, zeta: evaluate_strong_force(state, rho, theta, zeta).force)
    result = evaluate(
        jnp.asarray([1.0e-4, 1.0e-3, 1.0e-2]),
        jnp.asarray([0.2, 1.3, 4.1]),
        jnp.asarray([0.5, 2.0, 5.0]),
    )
    assert np.all(np.isfinite(result))
    assert float(jnp.max(jnp.linalg.norm(result, axis=-1))) < 9.0e4


def test_point_force_jvp_with_respect_to_high_order_coefficients():
    state = _constant_toroidal_field_state()
    tangent = jax.tree.map(jnp.zeros_like, state)
    tangent = replace(tangent, R_cos=tangent.R_cos.at[1, 0].set(1.0))

    def evaluate(candidate):
        return evaluate_strong_force(candidate, jnp.asarray(0.4), jnp.asarray(0.7), jnp.asarray(1.2)).force

    primal, derivative = jax.jit(lambda x, dx: jax.jvp(evaluate, (x,), (dx,)))(state, tangent)
    assert np.all(np.isfinite(primal))
    assert np.all(np.isfinite(derivative))
    assert float(jnp.linalg.norm(derivative)) > 0.0


def test_overintegrated_certificate_reports_dimensional_and_normalized_force():
    report = certify_strong_force(
        _constant_toroidal_field_state(),
        angular_multiplier=1,
        radial_order_increment=0,
    )
    assert float(report.absolute_l2) == pytest.approx(7.97e4, rel=0.02)
    assert float(report.absolute_linf) > float(report.absolute_l2)
    assert float(report.normalized_linf) == pytest.approx(2.0, rel=1e-10)
    assert float(report.minimum_signed_jacobian) > 9.0
    assert float(report.gauge_residual) == 0.0
    assert "JxB-grad(p)" in report.normalization
    assert "field-period" in report.coordinate_convention

    figure, axes = plot_strong_force_report({"analytic": report})
    assert len(axes) == 2
    assert "strong-force" in figure._suptitle.get_text()
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_legacy_lift_preserves_axis_boundary_and_lambda_gauge():
    path = "examples/data/input.circular_tokamak"
    inp = VmecInput.from_file(path)
    base = resolution_from_input(inp)
    resolution = Resolution(
        mpol=base.mpol,
        ntor=base.ntor,
        ntheta=base.ntheta,
        nzeta=base.nzeta,
        nfp=base.nfp,
        lasym=base.lasym,
        ns=9,
    )
    runtime = prepare_runtime(inp, resolution)
    high_order = lift_high_order_state(_initial_state(runtime.setup), runtime, degree=5, max_spans=3)

    for coefficients, target in (
        (high_order.R_cos, high_order.boundary_R_cos),
        (high_order.R_sin, high_order.boundary_R_sin),
        (high_order.Z_cos, high_order.boundary_Z_cos),
        (high_order.Z_sin, high_order.boundary_Z_sin),
    ):
        edge = high_order.radial_basis.evaluate(coefficients, 1.0, axis=-1)
        np.testing.assert_allclose(edge, target, rtol=0.0, atol=2e-14)
    gauge = (high_order.m == 0) & (high_order.n == 0)
    assert np.max(np.abs(np.asarray(high_order.L_cos)[gauge])) == 0.0
    assert np.max(np.abs(np.asarray(high_order.L_sin)[gauge])) == 0.0
    assert np.all(np.isfinite(np.asarray(high_order.pressure)))

    legacy_state = _initial_state(runtime.setup)
    wout = wout_from_state(
        inp=inp,
        state=legacy_state,
        fsqr=1.0,
        fsqz=1.0,
        fsql=1.0,
        converged=False,
    )
    imported = high_order_state_from_wout(
        wout,
        inp=inp,
        radial_basis=high_order.radial_basis,
    )
    np.testing.assert_allclose(imported.boundary_R_cos, high_order.boundary_R_cos, rtol=0.0, atol=2e-13)


@pytest.mark.parametrize("name", ["R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin"])
def test_state_rejects_malformed_fourier_tables(name):
    state = _constant_toroidal_field_state()
    values = state.__dict__.copy()
    values[name] = jnp.zeros((1, state.radial_basis.size))
    with pytest.raises(ValueError, match=name):
        HighOrderEquilibriumState(**values)


def test_state_rejects_periodic_radial_basis():
    state = _constant_toroidal_field_state()
    values = state.__dict__.copy()
    values["radial_basis"] = BSplineBasis.periodic_uniform(8, degree=5)
    with pytest.raises(ValueError, match="clamped radial basis"):
        HighOrderEquilibriumState(**values)


def test_state_validation_and_empty_plot_errors():
    state = _constant_toroidal_field_state()
    with pytest.raises(ValueError, match="nfp must be positive"):
        replace(state, nfp=0)
    with pytest.raises(ValueError, match="m and n"):
        replace(state, n=np.asarray([0]))
    with pytest.raises(ValueError, match="pressure"):
        replace(state, pressure=jnp.zeros((state.radial_basis.size - 1,)))
    with pytest.raises(ValueError, match="at least one"):
        plot_strong_force_report({})
