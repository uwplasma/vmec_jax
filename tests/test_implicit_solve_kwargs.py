"""Regression tests for implicit-lane forward-solver controls."""

from __future__ import annotations

from pathlib import Path
import dataclasses

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

from vmex.core import optimize as opt
from vmex.core.input import VmecInput


DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.mark.usefixtures("_module_jit_enabled")
def test_implicit_least_squares_honors_multigrid_solve_kwargs():
    """Implicit and FD lanes must see the same requested radial ladder.

    Before the regression fix, ``jac=None`` honored these controls while
    ``jac='implicit'`` silently used the deck's ``NS_ARRAY``/``FTOL_ARRAY``/
    ``NITER_ARRAY``.  The resulting Jacobians were compared at different
    equilibria and had different residual-row shapes.
    """
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    inp = inp.change_resolution(mpol=3, ntor=0, ntheta=12, nzeta=4)
    inp = dataclasses.replace(
        inp,
        ns_array=np.asarray([5]),
        ftol_array=np.asarray([1.0e-10]),
        niter_array=np.asarray([1000]),
    )
    qh = opt.QuasisymmetryRatioResidual([0.5], 1, -1)
    strict = {
        "ns_array": [7],
        "ftol_array": [1.0e-11],
        "niter_array": [1200],
        "device": "cpu",
    }

    def run(solve_kwargs):
        return opt.least_squares(
            [(qh, 0.0, 1.0), (opt.aspect_ratio, 4.0, 1.0)],
            inp,
            max_mode=1,
            jac="implicit",
            hot_restart=False,
            warm_start=None,
            solve_kwargs=solve_kwargs,
            max_nfev=1,
        )

    got = run(strict)
    alternate = run({
        "ns_array": [5],
        "ftol_array": [1.0e-6],
        "niter_array": [5],
        "device": "cpu",
    })
    expected_eq = opt.solve_equilibrium(inp, **strict)
    expected = np.concatenate([
        np.asarray(qh.residuals_state(
            expected_eq.state, expected_eq.runtime), dtype=float).ravel(),
        np.atleast_1d(np.asarray(
            opt.aspect_ratio(expected_eq.state, expected_eq.runtime) - 4.0,
            dtype=float,
        )).ravel(),
    ])
    assert got.fun.shape == expected.shape
    assert got.jac.shape[0] == expected.size
    # The implicit lane anchors its state with the fixed-point refinement,
    # whereas the independent diagnostic solve returns the ordinary host
    # stopping point.  Their values therefore need not be bitwise equal; the
    # row-shape equality is the contract that prevents the old mismatched
    # discretization.  Both are nevertheless finite and use the requested
    # equilibrium ladder.
    assert np.all(np.isfinite(got.fun))
    assert np.asarray(got.input.ns_array).tolist() == [7]
    assert np.asarray(got.input.ftol_array).tolist() == [1.0e-11]
    assert np.asarray(got.input.niter_array).tolist() == [1200]
    # Behavioral check: changing the requested ladder changes the computed
    # residual, rather than merely rewriting an input object. Before the fix,
    # both calls silently used the deck's [5], 1e-10, 1000 ladder and produced
    # the same implicit residual.
    assert alternate.fun.shape != got.fun.shape
    assert not np.isclose(
        np.linalg.norm(np.asarray(alternate.fun, dtype=float)),
        np.linalg.norm(np.asarray(got.fun, dtype=float)),
        rtol=1.0e-7,
        atol=1.0e-10,
    )
