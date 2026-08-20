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


def test_free_boundary_config_validates_adjoint_solver():
    inp, field = lasym_free_input(DATA), lasym_free_field()
    with pytest.raises(ValueError, match="'boundary_schur' or 'coupled_gcrot'"):
        make_free_boundary_config(inp, field, adjoint_solver="dense")


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
    # These are module-level caches keyed on (resolution, lconm1, ncurr), so
    # monkeypatch.setitem restores both entries at teardown; a bare assignment
    # would hand the all-zero mask to the next free-boundary case sharing that
    # key, which reaches the Schur lane as an edge basis with no columns.
    seed = object()
    monkeypatch.setitem(fbi._FREE_HOT_CACHE, cfg, seed)
    monkeypatch.setitem(fbi._FREE_MASK_CACHE, fbi._mask_key(cfg),
                        jax.tree.map(jnp.zeros_like, state))
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

    monkeypatch.setattr(
        fbi, "gcrotmk", lambda *_args, **_kwargs: (np.full(2, np.nan), 0))
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


def test_boundary_schur_adjoint_reproduces_the_coupled_gcrot_gradient():
    """Both adjoint solvers invert the same converged plasma-vacuum Jacobian.

    ``coupled_gcrot`` is matrix-free on the full coupled transpose;
    ``boundary_schur`` eliminates the block-tridiagonal bulk exactly and
    solves only NESTOR's edge response (``(I + U.T A.T^-1 E.T U) mu = ...``).
    Agreement between them on one converged root certifies the radial
    elimination itself; the nightly ``full`` test below anchors both against
    an independent re-solve finite difference.

    Both lanes are certified only to ``10 x adjoint_tol x ||rhs||``, so they
    are compared at the percent level, which a wrong Schur complement (rather
    than a differently converged Krylov solve) would miss by far more.
    """
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([8]),
        ftol_array=np.array([1.0e-8]), niter_array=np.array([2500]))
    field = lasym_free_field()
    params = im.params_from_input(inp)

    def configure(solver):
        return make_free_boundary_config(
            inp, field, ns=8, ftol=1.0e-8, max_iterations=2500,
            adjoint_tol=1.0e-5, adjoint_maxiter=100, adjoint_solver=solver,
            schur_probe_chunk_size=4,
            field_from_parameters=lambda current: dataclasses.replace(
                field, extcur=current),
            device="cpu")

    def gradient(cfg):
        def objective(current):
            state, _, _, _ = solve_free_boundary_implicit_status(
                params, current, cfg)
            return jnp.mean(state.R_cos[-1] ** 2 + state.Z_sin[-1] ** 2)

        return np.asarray(jax.grad(objective)(field.extcur))

    coupled_cfg, schur_cfg = configure("coupled_gcrot"), configure("boundary_schur")
    coupled = gradient(coupled_cfg)
    # Re-enter the same root warm: the comparison is about the adjoint, and a
    # second cold ladder would only add solver noise to it.
    fbi._FREE_HOT_CACHE[schur_cfg] = fbi._FREE_HOT_CACHE[coupled_cfg]
    schur = gradient(schur_cfg)

    assert np.all(np.isfinite(coupled)) and np.max(np.abs(coupled)) > 0.0
    np.testing.assert_allclose(schur, coupled, rtol=2.0e-2, atol=1.0e-8)


@pytest.mark.full
def test_boundary_schur_current_gradient_matches_resolve_finite_difference():
    """The reduced adjoint retains a nontrivial external-field derivative."""
    base = lasym_free_field()
    zeros = np.zeros_like(base.br)
    field = dataclasses.replace(
        base, br=np.concatenate((zeros, base.br)),
        bp=np.concatenate((base.bp, zeros)),
        bz=np.concatenate((base.bz, zeros)), extcur=np.ones(2))
    inp = dataclasses.replace(
        lasym_free_input(DATA), extcur=np.ones(2), ns_array=np.array([8]),
        ftol_array=np.array([1.0e-8]), niter_array=np.array([6000]))
    params = im.params_from_input(inp)
    cfg = make_free_boundary_config(
        inp, field, ns=8, ftol=1.0e-8, max_iterations=6000,
        adjoint_tol=1.0e-7, adjoint_maxiter=100,
        adjoint_solver="boundary_schur", schur_probe_chunk_size=4,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current), device="cpu")

    def objective(current):
        state, _, _, _ = solve_free_boundary_implicit_status(
            params, current, cfg)
        return jnp.mean(state.R_cos[-1]**2 + state.Z_sin[-1]**2)

    derivative = jax.grad(objective)(field.extcur)[1]
    step = 1.0e-3
    direction = jnp.array([0.0, 1.0])
    values = []
    for sign in (-1.0, 1.0):
        # Independent cold re-solves avoid continuation-history hysteresis in
        # this deliberately coarse nonlinear free-boundary certificate.
        fbi._FREE_HOT_CACHE.pop(cfg, None)
        values.append(objective(field.extcur + sign * step * direction))
    finite_difference = (values[1] - values[0]) / (2.0 * step)
    np.testing.assert_allclose(
        derivative, finite_difference, rtol=5.0e-2, atol=3.0e-4)


def test_presf_ns_scale_is_differentiated_in_the_adjoint_lanes():
    """``presf_ns_scale`` tracks ``am``; the adjoint lanes may not freeze it.

    ``funct3d.f``'s edge force carries ``presf_ns_scale * pressure[-1]``, and
    the ratio ``pmass(1)/pmass(hs*(ns-1.5))`` is a smooth function of ``am``,
    which is a differentiated parameter.  Taking the host float computed from
    the reference input gives the right value at the reference point and no
    derivative at all -- the same failure mode as the frozen lasym ``delta``
    branch, and invisible to any check that compares the traceable lane with
    itself.  This deck is ``power_series`` with a live derivative; a
    ``two_power`` deck has ``p(1) = 0`` and could not show it.
    """
    from vmex.core.freeboundary import (
        _presf_ns_scale, _presf_ns_scale_traceable,
    )

    inp, ns = lasym_free_input(DATA), 9
    assert inp.pmass_type == "power_series"
    params = im.params_from_input(inp)
    host = _presf_ns_scale(inp, ns)
    np.testing.assert_allclose(
        float(_presf_ns_scale_traceable(params, inp, ns)), host,
        rtol=1e-14, atol=0.0)

    am = np.asarray(inp.am, dtype=float)
    active = int(np.max(np.nonzero(am)[0])) + 1
    grad = jax.grad(lambda p: _presf_ns_scale_traceable(p, inp, ns))(params)
    analytic = np.asarray(grad.am, dtype=float)[:active]

    def shifted(k, delta):
        return dataclasses.replace(
            inp, am=np.where(np.arange(am.size) == k, am + delta, am))

    finite = np.empty(active)
    for k in range(active):
        step = 1.0e-6 * max(abs(am[k]), 1.0)
        finite[k] = (_presf_ns_scale(shifted(k, step), ns)
                     - _presf_ns_scale(shifted(k, -step), ns)) / (2.0 * step)

    # The frozen host float reported exactly zero for all of these, so the
    # tolerance only has to separate a live derivative from a missing one.
    # This deck's am coefficients reach 5e7 against a ratio of 0.32, which
    # caps central differences on the host float at a few times 1e-4.
    assert np.max(np.abs(finite)) > 1e-6, f"probe is degenerate: {finite}"
    np.testing.assert_allclose(analytic, finite, rtol=1e-3, atol=0.0)

    # pres_scale cancels in the ratio, so its derivative is genuinely zero.
    assert float(np.asarray(grad.pres_scale)) == 0.0
