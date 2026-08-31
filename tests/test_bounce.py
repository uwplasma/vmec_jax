"""Analytic and differentiation checks for the bounce-action kernel."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

import jax
import jax.numpy as jnp

from vmex.core.bounce import (
    bounce_action,
    bounce_action_from_boozer,
    trace_boozer_field_lines,
)


def _sinusoidal_field(amplitude=0.2, n=1024):
    phi = jnp.arange(n, dtype=jnp.float64) * (2.0 * jnp.pi / n)
    return 1.0 + amplitude * jnp.cos(phi)


def test_input_validation():
    with pytest.raises(ValueError, match="samples"):
        bounce_action(jnp.ones(2), 1.0)
    with pytest.raises(ValueError, match="pitch"):
        bounce_action(jnp.ones(8), jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="quadrature_order"):
        bounce_action(jnp.ones(8), 1.0, quadrature_order=1)
    with pytest.raises(ValueError, match="max_wells"):
        bounce_action(jnp.ones(8), 1.0, max_wells=0)
    common = dict(
        xm_b=[0.0], xn_b=[0.0], iota_b=[0.4], G_b=[1.0], I_b=[0.0],
        nfp=1, alpha=[0.0])
    with pytest.raises(ValueError, match="bmnc_b"):
        trace_boozer_field_lines(bmnc_b=[1.0], **common)
    with pytest.raises(ValueError, match="points_per_period"):
        trace_boozer_field_lines(bmnc_b=[[1.0]], points_per_period=4, **common)
    with pytest.raises(ValueError, match="bmns_b"):
        trace_boozer_field_lines(bmnc_b=[[1.0]], bmns_b=[[0.0, 0.1]], **common)


def test_sinusoidal_well_matches_adaptive_quadrature():
    amplitude = 0.2
    result = bounce_action(
        _sinusoidal_field(amplitude), 1.0, quadrature_order=48)
    reference = 2.0 * quad(
        lambda phi: np.sqrt(max(-amplitude * np.cos(phi), 0.0)),
        0.5 * np.pi, 1.5 * np.pi, epsabs=1e-13)[0]
    assert result["well_mask"][0, 0]
    np.testing.assert_allclose(result["action"][0, 0], reference, rtol=3e-6)


def test_multiple_wells_and_absent_well_status():
    phi = jnp.arange(1024, dtype=jnp.float64) * (2.0 * jnp.pi / 1024)
    double = bounce_action(1.0 + 0.2 * jnp.cos(2.0 * phi), 1.0)
    assert int(jnp.sum(double["well_mask"])) == 2
    np.testing.assert_allclose(
        np.asarray(double["action"][double["well_mask"]]),
        np.repeat(np.asarray(double["action"][0, 0]), 2),
        rtol=2e-12,
    )

    absent = bounce_action(jnp.ones(64), jnp.array([0.9, 1.1]))
    assert np.all(np.asarray(absent["absent"]))
    assert np.all(np.isnan(np.asarray(absent["action"])))


def test_topology_change_is_reported():
    field = _sinusoidal_field()
    result = bounce_action(
        field, jnp.array([1.0 / 0.8, 1.0 / 1.2]),
        topology_tolerance=1e-5)
    assert result["marginal"][0]
    assert result["merged"][1]
    assert not np.any(np.asarray(result["usable_mask"]))


def test_field_resolution_and_well_capacity_contract():
    pitch = 1.0 / 1.05
    root = np.arccos(0.25)
    reference = 2.0 * quad(
        lambda phi: np.sqrt(max(1.0 - pitch * (1.0 + 0.2 * np.cos(phi)), 0.0)),
        root, 2.0 * np.pi - root, epsabs=1e-13)[0]
    errors = []
    for n in (64, 128, 256):
        value = bounce_action(
            _sinusoidal_field(n=n), pitch, quadrature_order=24)["action"][0, 0]
        errors.append(abs(float(value) - reference))
    assert errors[2] < errors[1] < errors[0]

    phi = jnp.arange(256, dtype=jnp.float64) * (2.0 * jnp.pi / 256)
    limited = bounce_action(
        1.0 + 0.2 * jnp.cos(2.0 * phi), pitch, max_wells=1)
    assert limited["overflow"][0]
    assert not np.any(np.asarray(limited["usable_mask"]))

    cut = bounce_action(
        _sinusoidal_field(n=65)[16:49], 1.0, length=np.pi,
        periodic=False)
    assert cut["truncated"][0]

    phi = jnp.linspace(jnp.pi, 6.0 * jnp.pi, 641)
    interior = bounce_action(
        1.0 + 0.2 * jnp.cos(phi), 1.0, length=5.0 * np.pi,
        periodic=False, max_wells=3)
    assert interior["truncated"][0]
    assert int(jnp.sum(interior["usable_mask"])) == 2


def test_quadrature_resolution_converges_for_shaped_well():
    phi = jnp.arange(128, dtype=jnp.float64) * (2.0 * jnp.pi / 128)
    field = (
        1.2 + 0.11 * jnp.cos(phi + 4.5)
        + 0.135 * jnp.cos(3.0 * phi + 1.45)
        + 0.023 * jnp.cos(5.0 * phi + 5.33)
    )
    dl_dphi = 1.0 + 0.12 * jnp.cos(2.0 * phi + 5.36)

    def evaluate(order):
        return jnp.nansum(
            bounce_action(
                field, 0.767, dl_dphi=dl_dphi,
                quadrature_order=order)["action"])

    coarse, default, reference = map(evaluate, (32, 64, 128))
    assert abs(default - reference) < 0.1 * abs(coarse - reference)


def test_jit_jvp_vjp_and_finite_difference_agree():
    def value(amplitude):
        out = bounce_action(
            _sinusoidal_field(amplitude, n=512), 1.0,
            quadrature_order=32)
        return out["action"][0, 0]

    amplitude = jnp.asarray(0.2, dtype=jnp.float64)
    compiled = jax.jit(value)(amplitude)
    primal, tangent = jax.jvp(value, (amplitude,), (jnp.ones_like(amplitude),))
    _, pullback = jax.vjp(value, amplitude)
    reverse = pullback(jnp.ones_like(primal))[0]
    step = 1.0e-5
    finite_difference = (value(amplitude + step) - value(amplitude - step)) / (2 * step)
    np.testing.assert_allclose(compiled, primal, rtol=2e-13)
    np.testing.assert_allclose(tangent, reverse, rtol=2e-12)
    np.testing.assert_allclose(tangent, finite_difference, rtol=3e-5)


def test_sampled_field_jvp_vjp_transpose_identity():
    field = _sinusoidal_field(n=256)
    direction = 0.03 * jnp.sin(
        jnp.arange(256, dtype=field.dtype) * (4.0 * jnp.pi / 256))

    def actions(values):
        result = bounce_action(values, jnp.array([0.9, 1.0]))
        return jnp.where(result["well_mask"], result["action"], 0.0)

    _, tangent = jax.jvp(actions, (field,), (direction,))
    _, pullback = jax.vjp(actions, field)
    cotangent = jnp.arange(tangent.size, dtype=field.dtype).reshape(tangent.shape)
    transpose = pullback(cotangent)[0]
    np.testing.assert_allclose(
        jnp.vdot(cotangent, tangent), jnp.vdot(transpose, direction),
        rtol=2e-12, atol=2e-12)


def test_boozer_trace_uses_physical_pitch_and_line_element():
    def evaluate(amplitude):
        return bounce_action_from_boozer(
            bmnc_b=jnp.array([[1.0, amplitude]]),
            xm_b=jnp.array([0.0, 0.0]),
            xn_b=jnp.array([0.0, 2.0]),
            iota_b=jnp.array([0.4]),
            G_b=jnp.array([2.0]),
            I_b=jnp.array([0.5]),
            nfp=2,
            alpha=jnp.array([0.0, 0.7]),
            points_per_period=128,
            num_periods=2,
            bmns_b=jnp.array([[0.0, 0.03]]),
            pitch=jnp.array([1.0]),
            max_wells=3,
        )

    out = evaluate(0.2)
    assert out["bmag"].shape == (1, 2, 257)
    assert np.all(np.asarray(out["well_mask"][..., :2]))
    np.testing.assert_allclose(
        np.asarray(out["action"][0, 0, :2]),
        np.asarray(out["action"][0, 1, :2]),
        rtol=2e-12,
    )
    def value(amplitude):
        return jnp.nansum(evaluate(amplitude)["action"])

    derivative = jax.grad(value)(0.2)
    step = 1.0e-5
    finite_difference = (value(0.2 + step) - value(0.2 - step)) / (2 * step)
    np.testing.assert_allclose(derivative, finite_difference, rtol=3e-5)


def test_matches_desc_bounce1d_when_available():
    """Independent spline/root/quadrature oracle on a bounded two-well trace."""
    pytest.importorskip("desc")
    from desc.grid import Grid
    from desc.integrals import Bounce1D

    zeta = np.linspace(-2.0 * np.pi, 2.0 * np.pi, 2049)
    bmag = 1.0 + 0.2 * np.cos(zeta)
    derivative = -0.2 * np.sin(zeta)
    grid = Grid.create_meshgrid([1.0, 0.0, zeta], coordinates="raz")
    desc = Bounce1D(grid, {
        "B^zeta": bmag,
        "B^zeta_z|r,a": derivative,
        "|B|": bmag,
        "|B|_z|r,a": derivative,
    })
    pitch_inv = np.array([[1.05]])
    points = desc.points(pitch_inv, num_well=2)
    oracle = 2.0 * desc.integrate(
        lambda data, B, pitch: jnp.sqrt(jnp.maximum(1.0 - pitch * B, 0.0)),
        pitch_inv,
        points=points,
    ).sum()

    ours = bounce_action(
        jnp.asarray(bmag), 1.0 / pitch_inv[0, 0], length=4.0 * np.pi,
        periodic=False, max_wells=2)
    np.testing.assert_allclose(jnp.nansum(ours["action"]), oracle, rtol=1e-6)


def test_nemov_drift_kernels_match_adaptive_quadrature():
    """The optional per-well kernels reproduce mapped adaptive quadrature.

    Sinusoidal well ``B = 1 + 0.2 cos(phi)`` at ``pitch = 1``, integrand
    ``f = 0.3 + sin(phi)^2``, line element ``1 + 0.1 cos(phi)``. References
    use the same singularity-removing sine map with scipy adaptive
    quadrature; measured kernel agreement at ``n = 1024``, order 48 is
    1.9e-6 (bounce time), 1.1e-6 (drift), 1.6e-6 (parallel), asserted at
    1e-5. The softmin ``argmin_f`` averages ``f`` around the field minimum
    (measured offset 0.010 from ``f(pi)`` for this rapidly varying ``f``,
    asserted at 0.05).
    """
    amplitude, pitch, n = 0.2, 1.0, 1024
    phi = jnp.arange(n, dtype=jnp.float64) * (2.0 * jnp.pi / n)
    field = 1.0 + amplitude * jnp.cos(phi)
    integrand = 0.3 + jnp.sin(phi) ** 2
    dl_dphi = 1.0 + 0.1 * jnp.cos(phi)
    out = bounce_action(
        field, pitch, dl_dphi=dl_dphi, quadrature_order=48,
        drift_integrands={"f": integrand},
        parallel_integrands={"f": integrand},
        argmin_integrands={"f": integrand})

    mid, half = np.pi, 0.5 * np.pi

    def mapped(g):
        return quad(
            lambda t: g(mid + half * np.sin(t)) * half * np.cos(t),
            -np.pi / 2, np.pi / 2, epsabs=1e-13, limit=200)[0]

    def u(p):
        return max(1.0 - pitch * (1.0 + amplitude * np.cos(p)), 1e-300)

    def line(p):
        return 1.0 + 0.1 * np.cos(p)

    def f(p):
        return 0.3 + np.sin(p) ** 2

    def weight(p):
        return 1.0 - 0.5 * pitch * (1.0 + amplitude * np.cos(p))
    np.testing.assert_allclose(
        out["bounce_time"][0, 0], mapped(lambda p: 2.0 * line(p) / np.sqrt(u(p))),
        rtol=1e-5)
    np.testing.assert_allclose(
        out["drift_f"][0, 0],
        mapped(lambda p: f(p) * weight(p) / np.sqrt(u(p)) * line(p)), rtol=1e-5)
    np.testing.assert_allclose(
        out["parallel_f"][0, 0],
        mapped(lambda p: f(p) * np.sqrt(u(p)) * line(p)), rtol=1e-5)
    assert abs(float(out["argmin_f"][0, 0]) - f(np.pi)) < 0.05


def test_kernel_floor_caps_inverse_speed_kernels():
    """``kernel_floor`` regularizes exactly the inverse-speed kernels.

    Sinusoidal well at ``pitch = 1``: the exact bounce time is the mapped
    adaptive integral of ``2/sqrt(u)``; with a floor ``delta`` it must equal
    the adaptive integral of ``2/sqrt(u + delta)`` (measured 3e-6, asserted
    1e-4), be strictly smaller, and converge back to the exact kernel as
    ``delta -> 0``.  The ``parallel_*`` kernels carry no inverse speed and
    must be untouched by the floor.
    """
    amplitude, pitch, delta = 0.2, 1.0, 0.05
    field = _sinusoidal_field(amplitude)
    kwargs = dict(
        quadrature_order=48, drift_integrands={"one": jnp.ones(1024)},
        parallel_integrands={"one": jnp.ones(1024)})
    exact = bounce_action(field, pitch, **kwargs)
    floored = bounce_action(field, pitch, kernel_floor=delta, **kwargs)

    mid, half = np.pi, 0.5 * np.pi

    def mapped(g):
        return quad(
            lambda t: g(mid + half * np.sin(t)) * half * np.cos(t),
            -np.pi / 2, np.pi / 2, epsabs=1e-13, limit=200)[0]

    def u(p):
        return max(1.0 - pitch * (1.0 + amplitude * np.cos(p)), 0.0)

    reference = mapped(lambda p: 2.0 / np.sqrt(u(p) + delta))
    np.testing.assert_allclose(floored["bounce_time"][0, 0], reference, rtol=1e-4)
    assert float(floored["bounce_time"][0, 0]) < float(exact["bounce_time"][0, 0])
    np.testing.assert_allclose(
        floored["parallel_one"][0, 0], exact["parallel_one"][0, 0], rtol=1e-12)

    tiny = bounce_action(field, pitch, kernel_floor=1.0e-12, **kwargs)
    np.testing.assert_allclose(
        tiny["bounce_time"][0, 0], exact["bounce_time"][0, 0], rtol=1e-5)


def test_drift_kernels_are_nan_outside_wells_and_differentiable():
    """Kernel extensions keep the NaN-for-invalid contract and stay traceable."""
    field = _sinusoidal_field(n=512)
    out = bounce_action(
        field, jnp.array([0.5, 1.0]), drift_integrands={"one": jnp.ones(512)})
    assert np.all(np.isnan(np.asarray(out["bounce_time"][0])))  # untrapped pitch
    assert np.isfinite(float(out["drift_one"][1, 0]))

    def total(amplitude):
        result = bounce_action(
            _sinusoidal_field(amplitude, n=512), 1.0, quadrature_order=32,
            drift_integrands={"f": jnp.ones(512)},
            parallel_integrands={"f": jnp.ones(512)},
            argmin_integrands={"f": _sinusoidal_field(amplitude, n=512)})
        return (result["bounce_time"][0, 0] + result["drift_f"][0, 0]
                + result["parallel_f"][0, 0] + result["argmin_f"][0, 0])

    amplitude = jnp.asarray(0.2, dtype=jnp.float64)
    primal, tangent = jax.jvp(total, (amplitude,), (jnp.ones_like(amplitude),))
    step = 1.0e-5
    finite_difference = (
        total(amplitude + step) - total(amplitude - step)) / (2.0 * step)
    assert np.isfinite(float(primal)) and np.isfinite(float(tangent))
    np.testing.assert_allclose(tangent, finite_difference, rtol=2e-4)
