"""Zero-crash penalty-path tests for ``optimize.least_squares`` (plan Item
I.2): a mid-campaign trial whose equilibrium solve fails must be penalized
(large finite residual, trust region backs off), never crash.  All four
failure lanes are exercised deterministically by making the host solve fail
on chosen calls: the jac=None ``fun`` body, the exception-free implicit
callback status, the finite differentiated penalty, and the final diagnostic
cold re-solve.  Each campaign must complete with a finite cost and no callback
traceback.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from vmex.core import implicit as im  # noqa: E402
from vmex.core import optimize as opt  # noqa: E402
from vmex.core.errors import VmecJacobianError  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")  # full solves: jitted

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
OBJECTIVE = [(opt.aspect_ratio, 4.0, 1.0)]


def _boom() -> VmecJacobianError:
    return VmecJacobianError(
        "INITIAL JACOBIAN CHANGED SIGN!",
        hint="deterministic stand-in for a self-intersecting trial boundary")


def test_status_callback_builds_safe_mask_before_seed_cache(monkeypatch):
    """Even an unprimed failed trial returns a shape-safe zero mask."""
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    cfg = im.make_config(inp, ftol=1.0e-10, max_iterations=10)
    params = im.params_from_input(inp)
    params_np = jax.tree.map(np.asarray, params)
    saved = dict(im._MASK_CACHE)
    try:
        im._MASK_CACHE.clear()
        monkeypatch.setattr(
            im,
            "_host_solve_and_mask_impl",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(_boom()),
        )
        state, mask, status, fsq, fsq_ratio = im._host_solve_and_mask_status(cfg, params_np)
        assert int(status) == 1
        assert np.isinf(fsq) and np.isinf(fsq_ratio)
        assert all(np.all(value == 0.0) for value in jax.tree.leaves(mask))
        assert jax.tree.structure(state) == jax.tree.structure(mask)
    finally:
        im._MASK_CACHE.clear()
        im._MASK_CACHE.update(saved)


def test_status_callback_exposes_under_converged_fsq(monkeypatch):
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    cfg = im.make_config(inp, ftol=1.0e-14, max_iterations=1, max_fsq_ratio=1.0e-12)
    params = im.params_from_input(inp)
    runtime = im.runtime_from_params(params, cfg)
    state = im._initial_state(runtime.setup)
    mask = jax.tree.map(jax.numpy.zeros_like, state)
    result = SimpleNamespace(converged=False, fsqr=2.0e-10, fsqz=3.0e-10, fsql=0.0)

    def under_converged(config, params_np):
        im._LAST_SOLVE[config] = (im._params_key(params_np), result)
        return jax.tree.map(np.asarray, state), jax.tree.map(np.asarray, mask)

    monkeypatch.setattr(im, "_host_solve_and_mask_impl", under_converged)
    _, _, status, fsq, fsq_ratio = im._host_solve_and_mask_status(
        cfg, jax.tree.map(np.asarray, params)
    )
    assert int(status) == 2
    assert np.isfinite(fsq) and fsq > 0.0
    np.testing.assert_allclose(fsq_ratio, fsq / cfg.ftol)


def test_fd_lane_penalty_path(monkeypatch, capsys):
    """jac=None: a failed trial solve is penalized and the campaign completes."""
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    real = opt.solve_equilibrium
    calls = {"n": 0, "failed": 0}

    def flaky(trial, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # first FD probe after the (required-good) seed eval
            calls["failed"] += 1
            raise _boom()
        return real(trial, **kwargs)

    monkeypatch.setattr(opt, "solve_equilibrium", flaky)
    res = opt.least_squares(OBJECTIVE, inp, max_mode=1, max_nfev=4,
                            diff_step=1e-4, verbose=1)
    out = capsys.readouterr().out
    assert calls["failed"] == 1
    assert "Cost" in out
    assert "VmecJacobianError" not in out
    assert "Traceback" not in out
    assert np.isfinite(res.cost)
    assert res.failed_trials == 1
    assert isinstance(res.input, VmecInput)


def test_implicit_lane_fun_penalty_path(monkeypatch, capsys):
    """jac='implicit': every failed trial solve penalizes; campaign completes.

    The poison hits new-parameter-key host solves (trial boundaries) only, so
    the seed evaluation, the ``fun(x0)``/``jac(x0)`` memo hits and the final
    diagnostic re-solve stay healthy while every trust-region trial fails —
    the campaign must ride the penalty residual to a clean finish at ``x0``.
    """
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    real = im._host_solve
    calls = {"new": 0, "poisoned": 0}

    def flaky(cfg, params):
        hit = im._LAST_SOLVE.get(cfg)
        if hit is None or hit[0] != im._params_key(params):
            calls["new"] += 1
            if calls["new"] >= 2:  # first new key = the x0 seed solve
                calls["poisoned"] += 1
                raise _boom()
        return real(cfg, params)

    monkeypatch.setattr(im, "_host_solve", flaky)
    res = opt.least_squares(OBJECTIVE, inp, max_mode=1, jac="implicit",
                            max_nfev=4, verbose=1)
    out = capsys.readouterr().out
    assert calls["poisoned"] >= 1
    assert "Cost" in out                 # scipy accepted-iteration table
    assert "VmecJacobianError" not in out
    assert "Traceback" not in out        # no exception crossed pure_callback
    assert np.isfinite(res.cost)
    np.testing.assert_allclose(res.x, opt.pack_boundary(inp, 1))  # stayed at x0


def test_minimize_penalty_path(monkeypatch, capsys):
    """A failed scalarized trial gets the smooth consistent penalty pair."""
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    real = im._host_solve
    calls = {"new": 0, "poisoned": 0}

    def flaky(cfg, params):
        hit = im._LAST_SOLVE.get(cfg)
        if hit is None or hit[0] != im._params_key(params):
            calls["new"] += 1
            if calls["new"] >= 2:
                calls["poisoned"] += 1
                raise _boom()
        return real(cfg, params)

    monkeypatch.setattr(im, "_host_solve", flaky)
    res = opt.minimize(
        OBJECTIVE, inp, max_mode=1, verbose=1,
        options={"maxiter": 2, "maxls": 3})
    out = capsys.readouterr().out
    assert calls["poisoned"] >= 1
    assert "VmecJacobianError" not in out
    assert "Traceback" not in out
    assert np.isfinite(res.cost)
    assert res.monitor is not None


def test_implicit_lane_status_penalty_and_diagnostic_resolve(monkeypatch, capsys):
    """Failed differentiated trials use a penalty; diagnostics re-solve cold.

    The scipy driver evaluates ``jac`` at exactly the accepted iterate
    ``fun`` just solved (a memo-hit host solve), so poisoning memo-hit
    solves after the first ``jac(x0)`` fails later differentiated trials,
    which follow the status-safe penalty branch. ``solve_equilibrium`` fails
    whenever hot-seeded, forcing the final diagnostic's cold-solve fallback.
    """
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    real = im._host_solve
    calls = {"repeat": 0, "poisoned": 0}

    def flaky(cfg, params):
        hit = im._LAST_SOLVE.get(cfg)
        if hit is not None and hit[0] == im._params_key(params):
            calls["repeat"] += 1
            # repeats 1-2: fun(x0), jac(x0); later repeats are the
            # differentiated evaluations of accepted steps -> fail those.
            if calls["repeat"] >= 3:
                calls["poisoned"] += 1
                raise _boom()
        return real(cfg, params)

    real_solve_eq = opt.solve_equilibrium
    seeded = {"n": 0}

    def flaky_solve_eq(trial, *, initial_state=None, **kwargs):
        if initial_state is not None:  # the hot-seeded diagnostic re-solve
            seeded["n"] += 1
            raise _boom()
        return real_solve_eq(trial, **kwargs)

    monkeypatch.setattr(im, "_host_solve", flaky)
    monkeypatch.setattr(opt, "solve_equilibrium", flaky_solve_eq)
    res = opt.least_squares(OBJECTIVE, inp, max_mode=1, jac="implicit",
                            max_nfev=4, verbose=1)
    out = capsys.readouterr().out
    assert calls["poisoned"] >= 1
    assert "VmecJacobianError" not in out
    assert "Traceback" not in out
    assert seeded["n"] == 1
    assert np.isfinite(res.cost)
    assert res.equilibrium is not None  # cold-solve fallback delivered it
    assert res.equilibrium.result.converged
