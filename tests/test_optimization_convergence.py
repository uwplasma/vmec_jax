"""R1 convergence protection: from a circular-torus seed, the QS optimization
building blocks reach a real, non-trivial residual reduction (not just the
``VMEX_EXAMPLES_CI`` smoke budget of ``test_examples.py``).

These are ``full``-marked (nightly only, ``RUN_FULL=1``): each runs *real*
implicit-gradient continuation (``jac="implicit"`` + ESS, the exact path the
``examples/optimization`` scripts use) and asserts the achieved
``QuasisymmetryRatioResidual.total`` bound.  QA runs two continuation stages
to its precise bound (< 1e-3); QH and QP run a single stage.  The deep
implicit Jacobian is launch-bound (one preconditioned GMRES per boundary dof,
~101 s/eval at max_mode 2), so the *full* precise campaigns -- QA 1.70e-04
(max_mode 2) and QH 5.83e-05 (max_mode 5, ~100 min just for the max_mode-2
stage) -- are recorded/guarded in the example scripts + README, and the
nightly guards the single-stage bounds that fit the ``full`` job's shared
150-min budget.

Measured on the office 36-core CPU (2026-07-11, implicit Jacobian CPU-pinned):
QA 2.043e-01 -> 9.82e-03 (max_mode 1) -> 1.70e-04 (2, precise); QH 6.908e-01
-> 1.401e-01 (1), continuing to 5.83e-05 by max_mode 5 in the example; QP
4.458e-01 -> 9.4e-02 (basin-limited, same basin to max_mode 5).  Bounds below
carry margin over those.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("jax")

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
    (max_mode=2).  This two-stage nightly run protects the headline precise-QA
    claim; the bound (< 1e-3) carries margin over the measured 1.7e-4.
    """
    inp = _nfp2_seed(kick=0.01)  # helical kick breaks the axisymmetric saddle
    qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), 1, 0)
    seed = float(qs.total(opt.solve_equilibrium(inp)))
    terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, 6.0, 1.0), (opt.mean_iota, 0.42, 10.0)]
    r = opt.least_squares(terms, inp, max_mode=(1, 2), jac="implicit", use_ess=True,
                          max_nfev=80, ftol=1e-9, xtol=1e-10)
    final = float(qs.total(r.equilibrium))
    assert final < 1e-3, f"QA QS {seed:.3e} -> {final:.3e} (expected precise < 1e-3)"
    assert abs(float(opt.aspect_ratio(r.equilibrium.state, r.equilibrium.runtime)) - 6.0) < 0.05


@pytest.mark.full
def test_qh_implicit_converges():
    """QH (nfp4, helicity (1,-1)) descends from the axisymmetric seed via implicit.

    Implicit escapes the exact-axisymmetric saddle where finite differences
    stall (the QS residual is even in the symmetry-breaking harmonic, so the
    FD gradient vanishes) — no seed kick needed.  The full continuation reaches
    *precise* QH — QS 6.908e-01 -> 1.401e-01 (max_mode 1) -> 2.79e-03 (2) ->
    2.41e-04 (3) -> ... -> 5.83e-05 (max_mode 5), aspect 8.000 (office 36-core
    CPU, 2026-07-11; Landreman-Paul literature value ~2e-3) — and that precise
    result is guarded by ``QH_optimization.py`` + the README table.

    This nightly test asserts only the single-stage bound: the deep implicit
    Jacobian is launch-bound (~101 s/eval at max_mode 2, and QH needs ~60 evals
    there — ~100 min), so a multi-stage precise assertion would exceed the
    ``full`` job's shared 150-min budget.  The bound below (< 0.16) tightens the
    prior < 0.30 over the measured 0.140.
    """
    inp = _qh_seed()
    qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), 1, -1)
    seed = float(qs.total(opt.solve_equilibrium(inp)))
    terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, 8.0, 1.0)]
    r = opt.least_squares(terms, inp, max_mode=1, jac="implicit", use_ess=True,
                          max_nfev=60, ftol=1e-9, xtol=1e-10)
    final = float(qs.total(r.equilibrium))
    assert final < 0.16, f"QH QS {seed:.3e} -> {final:.3e} (measured 0.140; bound < 0.16)"


@pytest.mark.full
def test_qp_implicit_descends():
    """QP (nfp2, helicity (0,1)) descends via implicit to its documented basin.

    Basin-limited (not precise): QS 4.458e-01 -> ~9.4e-02 at max_mode 1, and
    stays at that basin through max_mode 5 (measured office CPU 2026-07-11;
    near-axis theory forbids exact QP).  This single stage guards the descent
    into the ~9.4e-2 basin.
    """
    inp = _nfp2_seed()
    qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), 0, 1)
    seed = float(qs.total(opt.solve_equilibrium(inp)))

    def iota_shortfall(state, rt):
        import jax.numpy as jnp
        return jnp.maximum(0.15 - jnp.abs(opt.mean_iota(state, rt)), 0.0)

    terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, 6.0, 1.0),
             (iota_shortfall, 0.0, 100.0), (opt.mirror_ratio, 0.20, 10.0)]
    r = opt.least_squares(terms, inp, max_mode=1, jac="implicit", use_ess=True,
                          max_nfev=60, ftol=1e-9, xtol=1e-10)
    final = float(qs.total(r.equilibrium))
    assert final < 0.20, f"QP QS {seed:.3e} -> {final:.3e} (expected descent to ~9.4e-2 basin)"
