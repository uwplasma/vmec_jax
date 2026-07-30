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
