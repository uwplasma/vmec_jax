"""Current-free vacuum convergence at the weak variational m=1,n=0 limit."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmex.core import solver
from vmex.core.errors import MORE_ITER_FLAG, VmecConvergenceError
from vmex.core.input import VmecInput
from vmex.core.multigrid import solve_multigrid
from vmex.core.wout import wout_from_state

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
DECK = DATA / "input.near_degenerate_vacuum_nfp3"
pytestmark = pytest.mark.usefixtures("_module_jit_enabled")


def test_current_free_vacuum_nonconvergence_suggests_lforbal():
    carry = SimpleNamespace(
        ier=MORE_ITER_FLAG,
        fsqr=2e-11,
        fsqz=1e-11,
        fsql=1e-12,
        iteration=3500,
    )
    runtime = SimpleNamespace(
        lforbal=False,
        ftol=1e-11,
        setup=SimpleNamespace(ncurr=1, mass=np.zeros(3), icurv=np.zeros(3)),
    )
    with pytest.raises(VmecConvergenceError) as excinfo:
        solver._finalize(carry, runtime)
    assert "review LFORBAL=T" in excinfo.value.hint

    runtime.setup.mass = np.ones(3)
    with pytest.raises(VmecConvergenceError) as excinfo:
        solver._finalize(carry, runtime)
    assert excinfo.value.hint == "increase NITER or loosen FTOL"


@pytest.mark.full
def test_lforbal_current_free_vacuum_matches_vmec2000():
    inp = VmecInput.from_file(DECK)
    result = solve_multigrid(inp, device="cpu", verbose=False)
    assert result.converged
    assert result.iterations == 941
    np.testing.assert_allclose(
        [result.fsqr, result.fsqz, result.fsql],
        [9.125353472079168e-12, 5.377544421551709e-12, 2.3166943756361613e-12],
        rtol=2e-6,
    )

    wout = wout_from_state(
        inp=inp,
        state=result.state,
        fsqr=float(result.fsqr),
        fsqz=float(result.fsqz),
        fsql=float(result.fsql),
        niter=int(result.iterations),
        converged=True,
    )
    np.testing.assert_allclose(
        [
            wout.volume_p,
            wout.Rmajor_p,
            wout.Aminor_p,
            wout.aspect,
            wout.b0,
            wout.wb,
        ],
        [
            2.154573046149519,
            1.0000000011877124,
            0.3303815150855631,
            3.0268037269843306,
            0.09382366056057204,
            0.0002423456557482906,
        ],
        rtol=5e-10,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        [
            wout.rmnc[0, 0],
            wout.rmnc[12, 0],
            wout.rmnc[12, 11],
            wout.zmns[12, 11],
            wout.bmnc[12, 0],
            wout.bmnc[-1, 1],
            wout.iotaf[-1],
        ],
        [
            0.9848767179889502,
            0.9949157880476165,
            0.28902906452588345,
            0.19115540561752123,
            0.09693177967085653,
            -2.2158421531406352e-05,
            -1.1631776764882045e-08,
        ],
        rtol=2e-8,
        atol=2e-13,
    )
