"""R1 convergence protection: from a circular-torus seed, the QS building
blocks reach real residual reduction via implicit-gradient continuation
(``jac="implicit"`` + ESS, the exact examples/optimization path).  All
``full``-marked (nightly): QA runs two stages to its precise bound
(< 1e-3); QH and QP run a single bounded stage (QP is basin-sensitive).
The full precise campaigns — QA 1.70e-04 (max_mode 2), QH 5.83e-05
(max_mode 5) — are guarded by the example scripts + README; measured
2026-07-11 (office 36-core CPU): QA 2.043e-01 -> 9.82e-03 -> 1.70e-04, QH
6.908e-01 -> 1.401e-01 (stage 1), QP 4.458e-01 -> 9.4e-02 (basin-limited).
Bounds below carry margin over those.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("jax")

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

from vmex.core.input import VmecInput
from vmex.core import optimize as opt

DATA = __import__("pathlib").Path(__file__).resolve().parents[1] / "examples" / "data"


def _nfp2_seed(kick: float = 0.0) -> VmecInput:
    inp = VmecInput.from_file(DATA / "input.minimal_seed_nfp2")
    if kick:
        rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
        rbc[inp.ntor + 1, 1] += kick
        zbs[inp.ntor + 1, 1] += kick
        inp = dataclasses.replace(inp, rbc=rbc, zbs=zbs)
    return inp


def _qh_seed() -> VmecInput:
    ntor = mpol = 6  # one harmonic above max_mode 5
    a = 0.125
    rbc = np.zeros((2 * ntor + 1, mpol))
    zbs = np.zeros((2 * ntor + 1, mpol))
    rbc[ntor, 0] = 1.0
    rbc[ntor, 1] = a
    zbs[ntor, 1] = a
    return VmecInput(nfp=4, mpol=mpol, ntor=ntor, rbc=rbc, zbs=zbs,
                     phiedge=np.pi * a ** 2, lasym=False, lfreeb=False,
                     mgrid_file="NONE", ncurr=1, curtor=0.0, pres_scale=0.0,
                     ns_array=[35], ftol_array=[1e-13], niter_array=[3000], delt=0.9)


@pytest.mark.full
def test_qa_reaches_precise():
    """QA (nfp2, helicity (1,0)) reaches *precise* QS via implicit continuation.

    Measured (office A4000): 2.043e-01 -> 9.82e-03 (max_mode=1) -> 1.70e-04
    (max_mode=2).  A bounded 30-evaluation replay reaches 2.62e-04 with aspect
    6.00008. This two-stage nightly run protects the headline precise-QA claim;
    the bound (< 1e-3) carries margin over both measurements.
    """
    inp = _nfp2_seed(kick=0.01)  # helical kick breaks the axisymmetric saddle
    qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), 1, 0)
    seed = float(qs.total(opt.solve_equilibrium(inp)))
    terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, 6.0, 1.0), (opt.mean_iota, 0.42, 10.0)]
    r = opt.least_squares(terms, inp, max_mode=(1, 2), jac="implicit", use_ess=True,
                          max_nfev=30, ftol=1e-9, xtol=1e-10)
    final = float(qs.total(r.equilibrium))
    assert final < 1e-3, f"QA QS {seed:.3e} -> {final:.3e} (expected precise < 1e-3)"
    assert abs(float(opt.aspect_ratio(r.equilibrium.state, r.equilibrium.runtime)) - 6.0) < 0.05


@pytest.mark.full
def test_qh_implicit_converges():
    """QH (nfp4, helicity (1,-1)) descends from the axisymmetric seed via
    implicit — no kick needed (implicit escapes the saddle where the even QS
    residual makes FD gradients vanish).  The full continuation reaches
    precise QH (6.908e-01 -> 1.401e-01 -> ... -> 5.83e-05 by max_mode 5,
    guarded by ``QH_optimization.py`` + README).  This nightly asserts only
    the single-stage bound < 0.16 (measured 0.140; multi-stage would exceed
    the hosted-worker budget at ~101 s/eval); a 40-evaluation replay reached
    2.12e-03, ample margin."""
    inp = _qh_seed()
    qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), 1, -1)
    seed = float(qs.total(opt.solve_equilibrium(inp)))
    terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, 8.0, 1.0)]
    r = opt.least_squares(terms, inp, max_mode=1, jac="implicit", use_ess=True,
                          max_nfev=40, ftol=1e-9, xtol=1e-10)
    final = float(qs.total(r.equilibrium))
    assert final < 0.16, f"QH QS {seed:.3e} -> {final:.3e} (measured 0.140; bound < 0.16)"


@pytest.mark.full
def test_qp_implicit_descends():
    """QP (nfp2, helicity (0,1)) descends via implicit to its documented basin.

    Basin-limited (not precise): a bounded ten-evaluation max-mode-1 smoke at
    ns=25 reaches QS 4.461e-01 -> 1.402e-01 (CPU replay, 2026-07-22). Near-axis
    theory forbids exact QP, and longer campaigns are rounding-sensitive. This
    test guards substantial descent without turning a physics smoke into a
    resource test.
    """
    inp = dataclasses.replace(_nfp2_seed(), ns_array=[25])
    qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), 0, 1)
    seed = float(qs.total(opt.solve_equilibrium(inp)))

    def iota_shortfall(state, rt):
        import jax.numpy as jnp
        return jnp.maximum(0.15 - jnp.abs(opt.mean_iota(state, rt)), 0.0)

    terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, 6.0, 1.0),
             (iota_shortfall, 0.0, 100.0), (opt.mirror_ratio, 0.20, 10.0)]
    r = opt.least_squares(terms, inp, max_mode=1, jac="implicit", use_ess=True,
                          max_nfev=10, ftol=1e-9, xtol=1e-10)
    final = float(qs.total(r.equilibrium))
    assert final < 0.85 * seed, (
        f"QP QS {seed:.3e} -> {final:.3e} "
        "(expected at least 15% descent)"
    )
