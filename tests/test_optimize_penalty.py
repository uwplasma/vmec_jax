"""Zero-crash penalty-path tests for ``optimize.least_squares`` (plan Item
I.2): a mid-campaign trial whose equilibrium solve fails must be penalized
(large finite residual, trust region backs off), never crash.  All four
except-bodies are exercised deterministically by making the host solve fail
on chosen calls: the jac=None ``fun`` body, the jac="implicit" ``fun`` body
(which also exercises the Item I.1 typed-error relay under jit), the
last-valid-Jacobian fallback, and the final diagnostic cold re-solve.
Each campaign must complete with a finite cost and the penalty prints.
"""

from __future__ import annotations

from pathlib import Path

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
    assert "trial solve failed" in out  # the penalty branch executed
    assert np.isfinite(res.cost)
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
    assert "trial solve failed" in out  # the penalty branch executed
    assert "VmecJacobianError" in out   # Item I.1 relay: short sentinel, typed name
    assert np.isfinite(res.cost)
    np.testing.assert_allclose(res.x, opt.pack_boundary(inp, 1))  # stayed at x0


def test_minimize_penalty_path(monkeypatch, capsys):
    """A failed scalarized trial reuses the last finite reverse gradient."""
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
    assert "trial solve/gradient failed" in out
    assert np.isfinite(res.cost)


def test_implicit_lane_jac_fallback_and_diagnostic_resolve(monkeypatch, capsys):
    """jac='implicit': failed Jacobian reuses the last valid one; final
    diagnostic re-solve falls back to a cold solve when the hot seed fails.

    The scipy driver evaluates ``jac`` at exactly the accepted iterate
    ``fun`` just solved (a memo-hit host solve), so poisoning memo-hit
    solves after the first ``jac(x0)`` fails every later Jacobian while all
    trial solves stay healthy.  ``solve_equilibrium`` additionally fails
    whenever hot-seeded, forcing the final diagnostic's cold-solve fallback.
    """
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    real = im._host_solve
    calls = {"repeat": 0, "poisoned": 0}

    def flaky(cfg, params):
        hit = im._LAST_SOLVE.get(cfg)
        if hit is not None and hit[0] == im._params_key(params):
            calls["repeat"] += 1
            # repeats 1-3: residual pre-size, fun(x0), jac(x0); later repeats
            # are the Jacobians of accepted steps -> fail those.
            if calls["repeat"] >= 4:
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
    assert "trial jacobian failed" in out  # last-valid-Jacobian fallback ran
    assert seeded["n"] == 1
    assert np.isfinite(res.cost)
    assert res.equilibrium is not None  # cold-solve fallback delivered it
    assert res.equilibrium.result.converged


def test_recycled_jacobian_lane_scan_shapes() -> None:
    """The drift-gated recycled Jacobian lane (recycle=True) must run.

    Exercises optimize.jacobian_rows_recycled's lax.scan carry — the path that
    threads and drift-gates the GCROT pair across dof chunks. A shape bug there
    (gating against the full stacked recs leaf instead of a single lane) makes
    lax.scan reject the carry-type mismatch; this pins that the lane completes
    on a real (small) equilibrium and returns a usable result. The recycled
    lane is otherwise opt-in and was previously uncovered in CI.
    """
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    res = opt.least_squares(
        OBJECTIVE, inp, max_mode=1, jac="implicit", recycle=True,
        max_nfev=3, verbose=0,
    )
    assert isinstance(res.input, VmecInput)
    assert np.all(np.isfinite(np.asarray(res.x)))
