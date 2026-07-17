"""Unit tests for :mod:`vmex.core.step` (evolve.f / restart.f port).

The parity-critical constants and update rules are asserted directly
against the closed-form definitions recorded in the module docstring:
the ndamp damping recurrence, the momentum update, and the three restart
outcomes (STEP_OK / RESTART_JACOBIAN / RESTART_GROWTH) including the
time-step rescaling, velocity zeroing, and best-residual bookkeeping.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core import step


def test_parity_constants():
    assert step.NDAMP == 10
    assert step.DAMPING_CAP == 0.15
    assert step.JACOBIAN_RESET_FACTOR == 0.90
    assert step.GROWTH_BACKOFF_DIVISOR == 1.03
    assert step.GROWTH_LIMIT == 1.0e4
    assert step.GROWTH_MIN_ITERATIONS == 10


def test_initial_control():
    c = step.StepControl.initial(0.9)
    assert float(c.time_step) == pytest.approx(0.9)
    np.testing.assert_allclose(np.asarray(c.inv_tau), np.full(10, 0.15 / 0.9))
    assert float(c.fsq_total_prev) == 0.0
    assert math.isinf(float(c.residual_best_precond))
    assert math.isinf(float(c.residual_best_raw))
    assert int(c.iter_last_reset) == 0
    assert int(c.jacobian_resets) == 0


def test_damping_fresh_start_uses_cap_window():
    c = step.StepControl.initial(0.9)
    b1, fac, c2 = step.damping_coefficients(c, jnp.asarray(0), jnp.asarray(1e-3))
    # fresh window: dtau = dt * mean(0.15/dt) / 2 = 0.075
    assert float(b1) == pytest.approx(1.0 - 0.075)
    assert float(fac) == pytest.approx(1.0 / 1.075)
    assert float(c2.fsq_total_prev) == pytest.approx(1e-3)


def test_damping_decrement_from_residual_ratio_and_cap():
    dt = 0.9
    c = step.StepControl.initial(dt)
    c = step.StepControl(
        time_step=c.time_step, inv_tau=jnp.zeros(10),
        fsq_total_prev=jnp.asarray(1.0),
        residual_best_precond=c.residual_best_precond,
        residual_best_raw=c.residual_best_raw,
        iter_last_reset=jnp.asarray(0), jacobian_resets=c.jacobian_resets,
    )
    # |log(fsq/prev)| = 0.05 < cap: decrement is exactly 0.05
    fsq = math.exp(-0.05)
    b1, fac, c2 = step.damping_coefficients(c, jnp.asarray(5), jnp.asarray(fsq))
    assert float(c2.inv_tau[-1]) == pytest.approx(0.05 / dt)
    np.testing.assert_allclose(np.asarray(c2.inv_tau[:-1]), 0.0)
    assert float(b1) == pytest.approx(1.0 - dt * (0.05 / dt / 10.0) / 2.0)

    # a huge residual jump is capped at DAMPING_CAP
    _, _, c3 = step.damping_coefficients(c, jnp.asarray(5), jnp.asarray(1e9))
    assert float(c3.inv_tau[-1]) == pytest.approx(0.15 / dt)

    # zero previous (or current) residual: decrement is 0, no NaNs
    c_zero = step.StepControl(
        time_step=c.time_step, inv_tau=jnp.zeros(10),
        fsq_total_prev=jnp.asarray(0.0),
        residual_best_precond=c.residual_best_precond,
        residual_best_raw=c.residual_best_raw,
        iter_last_reset=jnp.asarray(0), jacobian_resets=c.jacobian_resets,
    )
    b1z, _, c4 = step.damping_coefficients(c_zero, jnp.asarray(5), jnp.asarray(1e-3))
    assert float(c4.inv_tau[-1]) == 0.0
    assert np.isfinite(float(b1z))


def test_momentum_update_algebra():
    xc = {"a": jnp.asarray([1.0, 2.0])}
    xcdot = {"a": jnp.asarray([0.5, -0.5])}
    force = {"a": jnp.asarray([2.0, 4.0])}
    b1, fac, dt = jnp.asarray(0.925), jnp.asarray(1.0 / 1.075), jnp.asarray(0.9)
    new_xc, new_v = step.momentum_update(xc, xcdot, force, b1, fac, dt)
    v_ref = (1.0 / 1.075) * (0.925 * np.asarray([0.5, -0.5]) + 0.9 * np.asarray([2.0, 4.0]))
    np.testing.assert_allclose(np.asarray(new_v["a"]), v_ref, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(new_xc["a"]),
                               np.asarray([1.0, 2.0]) + 0.9 * v_ref, rtol=1e-14)


def _control(best=1e-6, iter_last_reset=0):
    c = step.StepControl.initial(0.9)
    return step.StepControl(
        time_step=c.time_step, inv_tau=c.inv_tau, fsq_total_prev=c.fsq_total_prev,
        residual_best_precond=jnp.asarray(best), residual_best_raw=jnp.asarray(best),
        iter_last_reset=jnp.asarray(iter_last_reset), jacobian_resets=jnp.asarray(0),
    )


def test_restart_decision_kinds_and_best_tracking():
    c = _control(best=1e-6)
    ok = jnp.asarray(False)

    # normal step: residual below limit, bests updated to the new minimum
    kind, c2 = step.restart_decision(c, jnp.asarray(20), jnp.asarray(1e-7),
                                     jnp.asarray(1e-7), ok)
    assert int(kind) == step.STEP_OK
    assert float(c2.residual_best_precond) == pytest.approx(1e-7)

    # growth > 1e4 x best after > 10 iterations since reset -> irst=3
    kind, _ = step.restart_decision(c, jnp.asarray(20), jnp.asarray(1e-1),
                                    jnp.asarray(1e-1), ok)
    assert int(kind) == step.RESTART_GROWTH

    # same growth too soon after the last reset -> tolerated
    c_recent = _control(best=1e-6, iter_last_reset=15)
    kind, _ = step.restart_decision(c_recent, jnp.asarray(20), jnp.asarray(1e-1),
                                    jnp.asarray(1e-1), ok)
    assert int(kind) == step.STEP_OK

    # a Jacobian sign change always wins
    kind, _ = step.restart_decision(c, jnp.asarray(20), jnp.asarray(1e-1),
                                    jnp.asarray(1e-1), jnp.asarray(True))
    assert int(kind) == step.RESTART_JACOBIAN


@pytest.mark.parametrize("kind,dt_factor", [
    (step.STEP_OK, 1.0),
    (step.RESTART_JACOBIAN, step.JACOBIAN_RESET_FACTOR),
    (step.RESTART_GROWTH, 1.0 / step.GROWTH_BACKOFF_DIVISOR),
])
def test_apply_restart(kind, dt_factor):
    xc = {"a": jnp.asarray([2.0, 3.0])}
    xcdot = {"a": jnp.asarray([0.5, -0.5])}
    saved = {"a": jnp.asarray([1.0, 1.0])}
    c = _control(best=1e-6)
    it = jnp.asarray(42)

    new_xc, new_v, new_saved, c2 = step.apply_restart(xc, xcdot, saved, c,
                                                      jnp.asarray(kind), it)
    assert float(c2.time_step) == pytest.approx(0.9 * dt_factor)
    if kind == step.STEP_OK:
        np.testing.assert_array_equal(np.asarray(new_xc["a"]), np.asarray(xc["a"]))
        np.testing.assert_array_equal(np.asarray(new_v["a"]), np.asarray(xcdot["a"]))
        # current state becomes the new save point
        np.testing.assert_array_equal(np.asarray(new_saved["a"]), np.asarray(xc["a"]))
        assert int(c2.jacobian_resets) == 0
        assert int(c2.iter_last_reset) == 0
        assert float(c2.residual_best_precond) == pytest.approx(1e-6)
    else:
        # state restored, velocity zeroed, bookkeeping reset
        np.testing.assert_array_equal(np.asarray(new_xc["a"]), np.asarray(saved["a"]))
        np.testing.assert_array_equal(np.asarray(new_v["a"]), 0.0)
        np.testing.assert_array_equal(np.asarray(new_saved["a"]), np.asarray(saved["a"]))
        assert int(c2.iter_last_reset) == 42
        assert math.isinf(float(c2.residual_best_precond))
        assert math.isinf(float(c2.residual_best_raw))
        expected_resets = 1 if kind == step.RESTART_JACOBIAN else 0
        assert int(c2.jacobian_resets) == expected_resets
