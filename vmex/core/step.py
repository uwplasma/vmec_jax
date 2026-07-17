"""Damped second-order Richardson time stepping and restart control.

Advances the spectral state ``xc`` with the momentum ("conjugate gradient
without line searches", after P. Garabedian) scheme of VMEC2000, including
the ``ndamp``-window adaptive damping, the Jacobian-reset back-off, and the
residual-growth back-off.

VMEC2000 counterparts: ``Sources/TimeStep/evolve.f`` (the ``dtau`` damping
recurrence and momentum update, and ``TimeStepControl``), and
``Sources/TimeStep/restart.f`` (``restart_iter`` state restore and time-step
rescaling).  The parity-critical constants are recorded in Appendix D and asserted here:

- damping window ``NDAMP = 10``; per-step decrement capped at ``0.15``
- velocity update ``v' = (1-dtau)/(1+dtau) * v + delt * F;  x' = x + delt*v'``
- Jacobian reset (irst=2): restore state, zero velocity, ``delt *= 0.90``
- residual-growth back-off (irst=3): growth > 1e4 x best after > 10 steps
  since the last reset; restore state, zero velocity, ``delt /= 1.03``

All functions are pure and jit-compatible; conditions are traced values
(no Python branching on array data), so a single compiled step serves both
the differentiable and the CLI execution lanes (§5.3).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

NDAMP = 10
#: Cap on the per-step damping decrement (VMEC2000 ``cp15`` in evolve.f).
DAMPING_CAP = 0.15
#: Time-step factor on a Jacobian reset (restart.f, irst=2).
JACOBIAN_RESET_FACTOR = 0.90
#: Time-step divisor on residual growth (restart.f, irst=3).
GROWTH_BACKOFF_DIVISOR = 1.03
#: Residual-growth ratio that triggers irst=3 (evolve.f ``TimeStepControl``).
GROWTH_LIMIT = 1.0e4
#: Minimum iterations since the last reset before irst=3 can trigger.
GROWTH_MIN_ITERATIONS = 10

# Restart kinds (VMEC2000 ``irst`` values).
STEP_OK = 1
RESTART_JACOBIAN = 2
RESTART_GROWTH = 3


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class StepControl:
    """Traced scalar state of the time-step controller.

    Attributes mirror VMEC2000 module variables: ``time_step`` (delt),
    ``inv_tau`` (otau damping history), ``fsq_total_prev`` (fsq1),
    ``residual_best_precond`` / ``residual_best_raw`` (res0/res1),
    ``iter_last_reset`` (iter1), ``jacobian_resets`` (ijacob).
    """

    time_step: jax.Array
    inv_tau: jax.Array  # (NDAMP,)
    fsq_total_prev: jax.Array
    residual_best_precond: jax.Array
    residual_best_raw: jax.Array
    iter_last_reset: jax.Array
    jacobian_resets: jax.Array

    @staticmethod
    def initial(time_step: float) -> "StepControl":
        dt = jnp.asarray(float(time_step))
        return StepControl(
            time_step=dt,
            inv_tau=jnp.full((NDAMP,), DAMPING_CAP) / dt,
            fsq_total_prev=jnp.asarray(0.0),
            residual_best_precond=jnp.asarray(jnp.inf),
            residual_best_raw=jnp.asarray(jnp.inf),
            iter_last_reset=jnp.asarray(0),
            jacobian_resets=jnp.asarray(0),
        )


def damping_coefficients(
    control: StepControl, iteration: jax.Array, fsq_total: jax.Array
) -> tuple[jax.Array, jax.Array, StepControl]:
    """Advance the ndamp damping window and return (b1, fac) (evolve.f).

    ``fsq_total`` is the current preconditioned residual sum
    ``fsqr1 + fsqz1 + fsql1``.  On the first step after a (re)start
    (``iteration == iter_last_reset``) the window resets to the cap.
    """
    dt = control.time_step
    fresh = iteration == control.iter_last_reset
    prev = control.fsq_total_prev
    ratio_ok = (prev != 0.0) & (fsq_total != 0.0)
    decrement = jnp.where(
        ratio_ok, jnp.minimum(jnp.abs(jnp.log(jnp.where(ratio_ok, fsq_total / prev, 1.0))), DAMPING_CAP), 0.0
    )
    shifted = jnp.concatenate([control.inv_tau[1:], (decrement / dt)[None]])
    inv_tau = jnp.where(fresh, jnp.full((NDAMP,), DAMPING_CAP) / dt, shifted)
    dtau = dt * jnp.mean(inv_tau) / 2.0
    b1 = 1.0 - dtau
    fac = 1.0 / (1.0 + dtau)
    return b1, fac, replace(control, inv_tau=inv_tau, fsq_total_prev=fsq_total)


def momentum_update(xc, xcdot, force, b1: jax.Array, fac: jax.Array, time_step: jax.Array):
    """One momentum step: v' = fac*(b1*v + delt*F); x' = x + delt*v' (evolve.f).

    ``xc``/``xcdot``/``force`` may be any matching pytree of arrays.
    """
    new_xcdot = jax.tree.map(lambda v, f: fac * (b1 * v + time_step * f), xcdot, force)
    new_xc = jax.tree.map(lambda x, v: x + time_step * v, xc, new_xcdot)
    return new_xc, new_xcdot


def restart_decision(
    control: StepControl,
    iteration: jax.Array,
    fsq_raw_total: jax.Array,
    fsq_precond_total: jax.Array,
    jacobian_sign_changed: jax.Array,
) -> tuple[jax.Array, StepControl]:
    """Classify the step: STEP_OK, RESTART_JACOBIAN, or RESTART_GROWTH.

    Ports evolve.f ``TimeStepControl``: track the best raw/preconditioned
    residuals seen since the last reset; flag irst=3 when either grows by
    more than ``GROWTH_LIMIT`` x best after ``GROWTH_MIN_ITERATIONS`` steps.
    A Jacobian sign change always wins (irst=2).
    """
    best_precond = jnp.minimum(control.residual_best_precond, fsq_precond_total)
    best_raw = jnp.minimum(control.residual_best_raw, fsq_raw_total)
    old_enough = (iteration - control.iter_last_reset) > GROWTH_MIN_ITERATIONS
    grew = (fsq_precond_total > GROWTH_LIMIT * best_precond) | (fsq_raw_total > GROWTH_LIMIT * best_raw)
    kind = jnp.where(
        jacobian_sign_changed,
        RESTART_JACOBIAN,
        jnp.where(old_enough & grew, RESTART_GROWTH, STEP_OK),
    )
    return kind, replace(control, residual_best_precond=best_precond, residual_best_raw=best_raw)


def apply_restart(xc, xcdot, xc_saved, control: StepControl, kind: jax.Array, iteration: jax.Array):
    """Apply restart.f: restore saved state, zero velocity, rescale delt.

    For ``STEP_OK`` this is the identity on (xc, xcdot) and records the
    current state as the new save point.  Returns
    ``(xc, xcdot, xc_saved, control)``.
    """
    ok = kind == STEP_OK
    jac = kind == RESTART_JACOBIAN
    growth = kind == RESTART_GROWTH

    new_xc = jax.tree.map(lambda x, s: jnp.where(ok, x, s), xc, xc_saved)
    new_xcdot = jax.tree.map(lambda v: jnp.where(ok, v, jnp.zeros_like(v)), xcdot)
    new_saved = jax.tree.map(lambda s, x: jnp.where(ok, x, s), xc_saved, xc)

    time_step = control.time_step * jnp.where(
        jac, JACOBIAN_RESET_FACTOR, jnp.where(growth, 1.0 / GROWTH_BACKOFF_DIVISOR, 1.0)
    )
    new_control = replace(
        control,
        time_step=time_step,
        jacobian_resets=control.jacobian_resets + jac.astype(control.jacobian_resets.dtype),
        iter_last_reset=jnp.where(ok, control.iter_last_reset, iteration),
        residual_best_precond=jnp.where(ok, control.residual_best_precond, jnp.inf),
        residual_best_raw=jnp.where(ok, control.residual_best_raw, jnp.inf),
    )
    return new_xc, new_xcdot, new_saved, new_control
