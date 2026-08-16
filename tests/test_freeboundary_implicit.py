"""Derivative certificates for the coupled free-boundary root."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_lasym_free_case import lasym_free_field, lasym_free_input
from vmex.core import implicit as im
from vmex.core.freeboundary_implicit import (
    make_free_boundary_config,
    solve_free_boundary_implicit,
    solve_free_boundary_implicit_status,
)
from vmex.core import freeboundary_implicit as fbi
from vmex.core.errors import AdjointSolveError, VmecJacobianError


DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
pytestmark = pytest.mark.usefixtures("_module_jit_enabled")


def test_free_boundary_status_callback_turns_trial_error_into_status(monkeypatch):
    """An invalid optimizer trial returns status 1 instead of crossing JAX."""
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([8]),
        ftol_array=np.array([1.0e-6]), niter_array=np.array([20]),
    )
    field = lasym_free_field()
    cfg = make_free_boundary_config(
        inp, field, ns=8, ftol=1.0e-6, max_iterations=20,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current),
        device="cpu",
    )
    assert cfg.implicit.device.platform == "cpu"

    def fail(*_args, **_kwargs):
        raise VmecJacobianError("invalid trial")

    monkeypatch.setattr(fbi, "_host_solve_and_mask", fail)
    _state, status, fsq, ratio = solve_free_boundary_implicit_status(
        im.params_from_input(inp), field.extcur, cfg)
    assert status == 1
    assert np.isinf(fsq) and np.isinf(ratio)
    assert cfg.resolution.ns == 8

    monkeypatch.setattr(fbi, "_host_solve_and_mask",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            RuntimeError("implementation bug")))
    with pytest.raises(RuntimeError, match="implementation bug"):
        fbi._host_solve_and_mask_status(cfg, im.params_from_input(inp), field.extcur)


def test_free_boundary_config_rejects_fixed_boundary_input():
    inp = dataclasses.replace(lasym_free_input(DATA), lfreeb=False)
    with pytest.raises(ValueError, match="LFREEB"):
        make_free_boundary_config(inp, lasym_free_field())


def test_free_boundary_warm_failure_retries_once_from_cold(monkeypatch):
    """A bad cached state is discarded, but implementation errors are not."""
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([8]),
        ftol_array=np.array([1.0e-6]), niter_array=np.array([20]))
    field = lasym_free_field()
    cfg = make_free_boundary_config(inp, field, ns=8, ftol=1.0e-6,
                                    max_iterations=20)
    runtime = im._template_runtime(cfg.implicit)
    state = im._initial_state(runtime.setup)
    seed = object(); fbi._FREE_HOT_CACHE[cfg] = seed
    fbi._FREE_MASK_CACHE[fbi._mask_key(cfg)] = jax.tree.map(jnp.zeros_like, state)
    calls = []

    def solve(*_args, initial_state=None, **_kwargs):
        calls.append(initial_state)
        if initial_state is seed:
            raise VmecJacobianError("bad warm state")
        result = SimpleNamespace(state=state, fsqr=0.0, fsqz=0.0, fsql=0.0,
                                 converged=True)
        return SimpleNamespace(result=result, continuation_state=state,
                               rcon0=runtime.rcon0, zcon0=runtime.zcon0)

    monkeypatch.setattr(fbi, "_solve_free_boundary_stage", solve)
    solved, *_ = fbi._host_solve_and_mask(cfg, im.params_from_input(inp), field)
    assert calls == [seed, None]
    np.testing.assert_allclose(solved.R_cos, state.R_cos)


def test_free_boundary_host_adjoint_rejects_a_false_solver_success(monkeypatch):
    """The true transpose residual, not SciPy's info flag, certifies a gradient."""
    cfg = SimpleNamespace(adjoint_tol=1.0e-10, adjoint_gcrot_m=3,
                          adjoint_gcrot_k=1, adjoint_maxiter=2)
    monkeypatch.setattr(fbi, "gcrotmk", lambda *_args, **_kwargs: (np.zeros(2), 0))
    def residual(z, *_args):
        return z

    with pytest.raises(AdjointSolveError, match="host GCROT"):
        fbi._host_adjoint(residual, jnp.zeros(2), None, None, None, None, None,
                          jnp.ones(2), cfg)


def test_free_boundary_current_gradient_matches_resolve_finite_difference():
    """The implicit coil-current response agrees with two independent solves."""
    inp = lasym_free_input(DATA)
    inp = dataclasses.replace(
        inp, ns_array=np.array([16]), ftol_array=np.array([1.0e-7]),
        niter_array=np.array([2500]),
    )
    field = lasym_free_field()
    params = im.params_from_input(inp)
    cfg = make_free_boundary_config(
        inp, field, ns=16, ftol=1.0e-7, max_iterations=2500,
        adjoint_tol=1.0e-8, adjoint_maxiter=100,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current),
    )

    def objective(current):
        state, _, _, _ = solve_free_boundary_implicit_status(params, current, cfg)
        return jnp.mean(state.R_cos[-1] ** 2 + state.Z_sin[-1] ** 2)

    current = field.extcur
    strict_state = solve_free_boundary_implicit(params, current, cfg)
    assert bool(jnp.all(jnp.isfinite(strict_state.R_cos)))
    # Exercise the strict public transform as well; the status lane below is
    # the optimizer-facing path and shares this hot forward state.
    _strict_value, _unused_pullback = jax.vjp(
        lambda value: jnp.mean(solve_free_boundary_implicit(
            params, value, cfg).R_cos[-1] ** 2), current)
    derivative = jax.grad(objective)(current)[0]
    step = 2.0e-4
    finite_difference = (
        objective(current + step) - objective(current - step)
    ) / (2.0 * step)
    # The host solve uses adaptive vacuum cadence and stops at finite force
    # tolerance; the independent re-solve FD therefore has percent-level
    # noise on this deliberately coarse nightly case.
    np.testing.assert_allclose(
        derivative, finite_difference, rtol=2.0e-2, atol=2.0e-4
    )
