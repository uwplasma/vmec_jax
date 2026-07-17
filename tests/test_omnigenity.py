"""Validation gates for the traceable omnigenity/QI objective (R26h.h2).

- **Transform parity**: the traceable Boozer ``|B|`` spectrum
  (:func:`vmex.core.omnigenity.boozer_bmnc_state`) matches the host
  booz_xform_jax route (:func:`vmex.core.optimize.boozer_modes_from_wout`)
  mode-by-mode on a converged 3D deck.
- **Physics**: the residual is exactly zero on an analytic QI (pure-QP)
  ``|B|`` and large on a QA-like ``|B|``; on solved decks the bundled QI
  configuration (``input.nfp1_QI``) scores far below a circular tokamak.
- **Consistency**: the deck ordering agrees with the wout-engine QI total
  (:func:`vmex.core.optimize.quasi_isodynamic_residual_from_wout`).
- **Differentiability**: ``jax.grad`` w.r.t. the state is finite and nonzero,
  and the residual composes through ``least_squares(..., jac="implicit")``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

from vmex.core import omnigenity as omn
from vmex.core import optimize as opt
from vmex.core.input import VmecInput

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")  # full solves: run jitted

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
SURFACES = (0.25, 0.5, 0.75)
FAST = dict(mboz=10, nboz=10, nphi=61, nalpha=13, n_levels=8)


@pytest.fixture(scope="module")
def qi_eq():
    """Converged bundled QI configuration (nfp=1, mpol=ntor=6)."""
    eq = opt.solve_equilibrium(VmecInput.from_file(DATA_DIR / "input.nfp1_QI"))
    assert eq.result.converged
    return eq


@pytest.fixture(scope="module")
def tok_eq():
    """Circular tokamak: strongly non-omnigenous (M=0, N=1) reference."""
    eq = opt.solve_equilibrium(VmecInput.from_file(DATA_DIR / "input.circular_tokamak"))
    assert eq.result.converged
    return eq


@pytest.fixture(scope="module")
def qa_eq():
    """Landreman-Paul QA (low res): quasi-axisymmetric, i.e. not QI."""
    return opt.solve_equilibrium(
        VmecInput.from_file(DATA_DIR / "input.LandremanPaul2021_QA_lowres"))


# ---------------------------------------------------------------------------
# Traceable Boozer transform parity vs booz_xform_jax
# ---------------------------------------------------------------------------


@pytest.mark.full
def test_boozer_spectrum_matches_booz_xform(qi_eq):
    pytest.importorskip("booz_xform_jax")
    # 0.53 is not equidistant between half surfaces (0.5 is a snapping tie).
    trace = omn.boozer_bmnc_state(qi_eq.state, qi_eq.runtime, surfaces=[0.53],
                                  mboz=10, nboz=10, oversample=2)
    host = opt.boozer_modes_from_wout(qi_eq, surfaces=[0.53], mboz=10, nboz=10)
    assert float(trace["iota_b"][0]) == pytest.approx(float(host["iota_b"][0]), rel=1e-8)
    assert float(trace["s_b"][0]) == pytest.approx(float(host["s_b"][0]), abs=1e-12)
    lookup = {(int(m), int(n)): float(v) for m, n, v in
              zip(host["xm_b"], host["xn_b"], np.asarray(host["bmnc_b"])[0])}
    bt = np.asarray(trace["bmnc_b"])[0]
    b00 = abs(lookup[(0, 0)])
    checked = 0
    for j, (m, n) in enumerate(zip(np.asarray(trace["xm_b"], dtype=int),
                                   np.asarray(trace["xn_b"], dtype=int))):
        bw = lookup.get((m, n))
        if bw is None:
            continue
        # measured floor between the two discretizations: <= 6.3e-5 * b00
        assert abs(bt[j] - bw) < 2e-4 * b00, (m, n)
        if abs(bw) > 1e-2 * b00:  # dominant modes additionally match tightly
            assert bt[j] == pytest.approx(bw, rel=5e-3), (m, n)
        checked += 1
    assert checked >= 50  # the comparison actually covered the spectrum


# ---------------------------------------------------------------------------
# Physics: analytic limits + solved decks
# ---------------------------------------------------------------------------


def _analytic_residual(bmnc, modes, nfp, iota=0.412):
    xm = np.asarray([m for (m, _) in modes], dtype=float)
    xn = np.asarray([n for (_, n) in modes], dtype=float)
    return omn.omnigenity_residual(
        bmnc_b=np.asarray([bmnc], dtype=float), xm_b=xm, xn_b=xn,
        iota_b=[iota], nfp=nfp, nphi=61, nalpha=13, n_levels=8)


def test_residual_zero_on_analytic_qi_field():
    """B = 1 + 0.1 cos(nfp * zeta_B) is exactly omnigenous with M=0, N=1."""
    out = _analytic_residual([1.0, 0.1], [(0, 0), (0, 3)], nfp=3)
    assert float(out["total"]) < 1e-24

    # QA-like field on the same grid: contours close toroidally, not
    # poloidally -- the (M=0, N=1) omnigenity distance must be large.
    out_qa = _analytic_residual([1.0, 0.1], [(0, 0), (1, 0)], nfp=3)
    assert float(out_qa["total"]) > 1e-3


def test_qi_deck_far_below_circular_tokamak(qi_eq, tok_eq):
    qi = omn.QIResidual(SURFACES, **FAST)
    total_qi = float(qi.total(qi_eq))
    total_tok = float(qi.total(tok_eq))
    assert np.isfinite(total_qi) and np.isfinite(total_tok)
    assert total_tok > 1e-2                      # tokamak: badly non-QI
    assert total_qi < total_tok / 20.0           # QI deck: far below


@pytest.mark.full
def test_ordering_matches_wout_engine(qi_eq, tok_eq, qa_eq):
    """Deck ordering agrees with the host booz_xform QI residual."""
    pytest.importorskip("booz_xform_jax")
    qi = omn.QIResidual(SURFACES, **FAST)
    eqs = {"qi": qi_eq, "tok": tok_eq, "qa": qa_eq}
    mine = {k: float(qi.total(eq)) for k, eq in eqs.items()}
    wout = {k: float(opt.quasi_isodynamic_residual_from_wout(
        eq, surfaces=list(SURFACES))["total"]) for k, eq in eqs.items()}
    assert sorted(mine, key=mine.get) == sorted(wout, key=wout.get)
    assert min(mine, key=mine.get) == "qi"


# ---------------------------------------------------------------------------
# Differentiability + implicit-lane composability
# ---------------------------------------------------------------------------


def test_grad_wrt_state_is_finite_and_nonzero(qi_eq):
    """The state gradient the implicit-gradient lane composes with."""
    qi = omn.QIResidual(SURFACES, **FAST)
    grad = jax.grad(lambda st: qi.total_state(st, qi_eq.runtime))(qi_eq.state)
    leaves = jax.tree.leaves(grad)
    assert leaves
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)
    assert any(np.any(np.asarray(leaf) != 0.0) for leaf in leaves)
    r = np.asarray(qi.residuals_state(qi_eq.state, qi_eq.runtime))
    assert float(np.sum(r * r)) == pytest.approx(float(qi.total(qi_eq)), rel=1e-12)


@pytest.mark.full
def test_least_squares_implicit_composes():
    """QIResidual works as a jac='implicit' objective term (2 trial solves)."""
    inp = VmecInput.from_file(DATA_DIR / "input.minimal_seed_nfp2")
    qi = omn.QIResidual((0.3, 0.7), mboz=8, nboz=8, nphi=41, nalpha=9, n_levels=6)
    result = opt.least_squares(
        [(qi, 0.0, 1.0), (opt.aspect_ratio, 6.0, 1.0)], inp, max_mode=1,
        jac="implicit", use_ess=True, max_nfev=2, ftol=1e-12, xtol=1e-14)
    assert np.all(np.isfinite(result.fun))
    assert np.isfinite(float(result.cost))
