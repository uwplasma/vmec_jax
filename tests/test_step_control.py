"""Unit tests for :mod:`vmex.core.step` (evolve.f / restart.f port).

The parity-critical constants and update rules are asserted directly
against the closed-form definitions recorded in the module docstring:
the ndamp damping recurrence, the momentum update, and the three restart
outcomes (STEP_OK / RESTART_JACOBIAN / RESTART_GROWTH) including the
time-step rescaling, velocity zeroing, and best-residual bookkeeping.

The final section pins the JAC75 recovery-chain CONTRACT at the
``vmex.core.solver._solve_stage`` level, driving MULTIPLE consecutive
75-reset events structurally (the loop is faked, the recovery policy is
real): each retry must restart the recorded best finite checkpoint (never
a reconstructed cold state), reduce ``DELT`` before the next update, keep
the iteration-1 axis transfer disabled, and never replay the identical
(state, time-step) attempt.
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
        assert math.isinf(float(c2.residual_best_precond))
        assert math.isinf(float(c2.residual_best_raw))
        # ``ijacob`` advances only on a Jacobian reset (restart.f irst == 2
        # guard), but ``iter1`` is rebased on ANY non-OK restart: the caller,
        # evolve.f TimeStepControl, runs ``iter1 = iter2`` unconditionally
        # inside ``IF (irst .NE. 1)``.  The production loop in solver.py does
        # the same, and this helper must match the full call path.
        expected_resets = 1 if kind == step.RESTART_JACOBIAN else 0
        assert int(c2.jacobian_resets) == expected_resets
        assert int(c2.iter_last_reset) == 42


# ---------------------------------------------------------------------------
# Recovery-chain contract: consecutive JAC75 events through _solve_stage
# ---------------------------------------------------------------------------


def _recovery_runtime():
    """Small real runtime (solovev, ns=5) for the structural recovery tests."""
    from pathlib import Path

    from vmex.core import solver
    from vmex.core.input import VmecInput

    deck = Path(__file__).resolve().parents[1] / "examples" / "data" / "input.solovev"
    inp = VmecInput.from_file(str(deck))
    resolution = solver.resolution_from_input(inp, ns=5)
    return solver.prepare_runtime(inp, resolution, max_iterations=8)


def _install_fake_loop(monkeypatch, rt, calls, *, failures: int):
    """Replace ``solver._run_loop`` with a JAC75-emitting fake.

    The first ``failures`` attempts return a JAC75 carry whose ``xstore``
    is a DISTINCT perturbed checkpoint (as the real loop would record);
    later attempts succeed.  The recovery POLICY under test —
    ``_solve_stage``'s checkpoint/DELT/axis handling — runs unmodified.
    """
    import dataclasses

    import jax
    import jax.numpy as jnp

    from vmex.core import solver
    from vmex.core.errors import JAC75_FLAG, SUCCESSFUL_TERM_FLAG

    def fake_run_loop(state0, loop_rt, *, mode, ijacob, verbose, emit,
                      use_fft=False, emit_banner=True, emit_legend=True,
                      initial_xcdot=None, initial_residuals=None):
        n = len(calls)
        calls.append(dict(
            state={
                name: np.asarray(getattr(state0, name)).copy()
                for name in ("R_cos", "Z_sin", "L_sin")
            },
            lmove_axis=bool(loop_rt.lmove_axis),
            delt0=float(loop_rt.time_step0),
            residuals=None if initial_residuals is None else tuple(
                float(v) for v in initial_residuals),
        ))
        carry = solver._initial_carry(state0, loop_rt, ijacob=ijacob)
        if n < failures:
            checkpoint = jax.tree.map(
                lambda x: x + 1e-3 * (n + 1), state0)
            return dataclasses.replace(
                carry,
                done=jnp.asarray(True),
                ier=jnp.asarray(JAC75_FLAG, dtype=carry.ier.dtype),
                ijacob=jnp.asarray(75, dtype=carry.ijacob.dtype),
                xstore=checkpoint,
                fsqr=jnp.asarray(0.5 + n, dtype=carry.fsqr.dtype),
                fsqz=jnp.asarray(0.25 + n, dtype=carry.fsqz.dtype),
                fsql=jnp.asarray(0.125 + n, dtype=carry.fsql.dtype),
            )
        return dataclasses.replace(
            carry,
            done=jnp.asarray(True),
            ier=jnp.asarray(SUCCESSFUL_TERM_FLAG, dtype=carry.ier.dtype),
        )

    monkeypatch.setattr(solver, "_run_loop", fake_run_loop)

    def no_axis_rebuild(*args, **kwargs):  # pragma: no cover - contract trap
        raise AssertionError(
            "reguess_initial_axis ran during a JAC75 retry: the recovery "
            "must continue the checkpoint, never rebuild a cold axis state")

    monkeypatch.setattr(solver, "reguess_initial_axis", no_axis_rebuild)


def test_consecutive_jac75_retries_restore_checkpoints_and_reduce_delt(
    monkeypatch,
) -> None:
    """TWO consecutive JAC75 events: checkpoint restored, DELT halved, no
    cold axis rebuild, no identical replay — then success on attempt 3."""
    from vmex.core import solver
    from vmex.core.errors import SUCCESSFUL_TERM_FLAG

    rt = _recovery_runtime()
    calls: list[dict] = []
    _install_fake_loop(monkeypatch, rt, calls, failures=2)

    lines: list[str] = []
    carry = solver._solve_stage(
        rt, None, mode="cli", verbose=True,
        emit=lambda value="", end="\n": lines.append(str(value)),
        jacobian_retries=2,
    )

    assert int(carry.ier) == SUCCESSFUL_TERM_FLAG
    assert len(calls) == 3
    banners = [ln for ln in lines if "JACOBIAN RECOVERY RETRY" in ln]
    assert len(banners) == 2
    assert "JACOBIAN RECOVERY RETRY 1/2" in banners[0]
    assert "JACOBIAN RECOVERY RETRY 2/2" in banners[1]

    interior = calls[0]["state"]
    # attempt 1: fresh interior guess, axis transfer allowed, input DELT
    assert calls[0]["lmove_axis"] is True
    assert calls[0]["delt0"] == pytest.approx(float(rt.time_step0))

    # each retry restarts the PREVIOUS attempt's recorded best checkpoint
    for attempt in (1, 2):
        expected = {
            name: calls[attempt - 1]["state"][name] + 1e-3 * attempt
            for name in ("R_cos", "Z_sin")
        }
        for name, want in expected.items():
            np.testing.assert_array_equal(
                calls[attempt]["state"][name], want,
                err_msg=f"retry {attempt} did not restart the recorded "
                        f"checkpoint ({name})",
            )
        # ... and NOT a reconstructed cold state
        assert not np.array_equal(
            calls[attempt]["state"]["R_cos"], interior["R_cos"])
        # the iteration-1 axis transfer stays disabled on retries
        assert calls[attempt]["lmove_axis"] is False
        # residual continuation from the failing attempt is carried
        assert calls[attempt]["residuals"] == pytest.approx(
            (0.5 + attempt - 1, 0.25 + attempt - 1, 0.125 + attempt - 1))

    # DELT reduced BEFORE the next update: min(0.5, 0.5 * previous)
    d0 = float(rt.time_step0)
    assert calls[1]["delt0"] == pytest.approx(min(0.5, 0.5 * d0))
    assert calls[2]["delt0"] == pytest.approx(min(0.5, 0.5 * calls[1]["delt0"]))
    assert calls[0]["delt0"] > calls[1]["delt0"] > calls[2]["delt0"]

    # no identical replay: every attempt is a distinct (state, DELT) pair
    for a in range(3):
        for b in range(a + 1, 3):
            assert calls[a]["delt0"] != calls[b]["delt0"]
            assert not np.array_equal(
                calls[a]["state"]["R_cos"], calls[b]["state"]["R_cos"])


def test_jac75_chain_exhausts_retries_with_typed_error(monkeypatch) -> None:
    """When every retry fails, the chain surfaces the JAC75 class typed —
    after restarting a genuine checkpoint at each event, never a cold state."""
    import pytest as _pytest

    from vmex.core import solver
    from vmex.core.errors import JAC75_FLAG, VmecJacobianError

    rt = _recovery_runtime()
    calls: list[dict] = []
    _install_fake_loop(monkeypatch, rt, calls, failures=99)

    carry = solver._solve_stage(
        rt, None, mode="cli", verbose=False, emit=lambda *a, **k: None,
        jacobian_retries=2,
    )
    assert int(carry.ier) == JAC75_FLAG
    assert len(calls) == 3          # initial + exactly jacobian_retries
    assert calls[1]["lmove_axis"] is False
    assert calls[2]["lmove_axis"] is False
    with _pytest.raises(VmecJacobianError):
        solver._finalize(carry, rt)
