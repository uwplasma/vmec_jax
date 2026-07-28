"""Stability-selected NESTOR analytic ``T^{+/-}_l`` recurrence.

``analyt.f`` (VMEC2000, inherited by the original ``vmex`` port) fills the
singular-integral family

    ``T_l = int_{-1}^{1} x^l / sqrt(A x^2 + 2 d x + B) dx``

(``T^+``: ``A = adp, B = adm``; ``T^-`` swapped; ``d = cma``) with a FORWARD
three-term recurrence.  Its homogeneous modes are the roots of
``A r^2 + 2 d r + B = 0`` — complex conjugates of modulus ``sqrt(B/A)``
whenever the metric is positive definite — so rounding in ``T_0`` is
amplified by ``(B/A)^{l/2}``.  Once ``(mf + nf) * ln(B/A)`` is large the
forward pass returns garbage: in free boundary this sets in around
``mpol``/``ntor`` ~ 12.  VMEC2000 and any straight port compute the *same*
wrong integrals, so two-code parity cannot see the defect; these tests pin
the fixed implementation (``vmex.core.vacuum._tl_stable``, which follows the
per-point forward/backward selection of vmecpp's
``free_boundary/singular_integrals/singular_integrals.cc``, commit f5dbf76,
with re-dimensioned switch threshold and Miller tail) against an
*independent* high-precision reference.

Reference choice
----------------
``mpmath`` is not available in the test environment, so the reference
evaluates the integral definition directly with ``scipy.integrate.quad``
(adaptive Gauss-Kronrod, ``epsrel = 1e-13``, ``epsabs = 0``).  The
integrand is smooth (the quadratic has no real roots on the positive-
definite parameter domain), and quad's self-reported error — asserted below
— stays under ``1e-11`` relative on the whole grid, comfortably beyond the
``1e-10`` gates.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from scipy.integrate import IntegrationWarning, quad  # noqa: E402

from vmex.core import vacuum as V  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

#: The forward/backward switch of the implementation under test:
#: backward iff ``lmax * ln(B/A) > _TL_LOG_GROWTH_THRESHOLD``.
THRESHOLD = V._TL_LOG_GROWTH_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers: closed-form T_0, quadrature reference, legacy forward recurrence
# ---------------------------------------------------------------------------


def _t0_closed(A, B, d):
    """Closed-form ``T_0`` exactly as ``analyt.f``/``vacuum.py`` build it.

    With ``4c = A + B + 2d`` and ``4a = A + B - 2d`` (so ``sqrtc = 2 sqrt(c)``
    and ``sqrta = 2 sqrt(a)`` are the boundary values ``sqrt(Q(+/-1))``).
    """
    sqa = np.sqrt(A)
    sqrtc = np.sqrt(A + B + 2.0 * d)
    sqrta = np.sqrt(A + B - 2.0 * d)
    return np.log((sqa * sqrtc + A + d) / (sqa * sqrta - A + d)) / sqa


def _tl_quad_reference(A, B, d, lmax):
    """``T_l`` for ``l = 0..lmax`` by adaptive quadrature; values + error."""
    vals = np.empty(lmax + 1)
    errs = np.empty(lmax + 1)
    with warnings.catch_warnings():
        # Near-endpoint mass at large l trips quad's roundoff heuristic; the
        # returned error estimate (asserted by the caller) stays ~1e-13 rel.
        warnings.simplefilter("ignore", IntegrationWarning)
        for ell in range(lmax + 1):
            vals[ell], errs[ell] = quad(
                lambda x, p=ell: x**p / np.sqrt(A * x * x + 2.0 * d * x + B),
                -1.0, 1.0, epsabs=0.0, epsrel=1e-13, limit=400,
            )
    return vals, errs


def _tl_forward_legacy(A, B, d, sqrtc, sqrta, t0, lmax):
    """The pre-fix ``analyt.f`` forward recurrence, op-for-op (NumPy).

    This is the exact update the shipped code ran before the fix
    (``vacuum.py`` used to advance ``tlp/tlm`` in-loop with this
    arithmetic); reimplemented here so the tests can compare against the
    OLD behavior without keeping dead code in the library.
    """
    t = np.empty((lmax + 1,) + np.shape(t0))
    t[0] = t0
    t_prev = np.zeros_like(np.asarray(t0, dtype=float))
    cur = np.asarray(t0, dtype=float)
    sign1 = 1.0
    fl1 = 0.0
    for _ell in range(lmax):
        fl = fl1
        fl1 = fl1 + 1.0
        fl2 = 2.0 * fl1 - 1.0
        sign1 = -sign1
        nxt = ((sqrtc + sign1 * sqrta) - fl2 * d * cur - fl * B * t_prev) / (A * fl1)
        t_prev = cur
        cur = nxt
        t[int(fl1)] = nxt
    return t


def _param_grid(lmax):
    """(A, B, d) rows covering B/A ~ [0.1, 50] incl. both switch sides.

    ``d = delta * sqrt(A*B)`` with ``|delta| < 1`` keeps the quadratic
    positive definite (``A B > d^2``, the NESTOR geometry guarantee).
    """
    thr = float(np.exp(THRESHOLD / lmax))
    ratios = [
        0.1, 0.5, 0.99, 1.01, 1.05, 1.2, 1.4,
        thr * 0.999, thr * 1.001,  # straddle the forward/backward switch
        1.5, 1.8, 2.2, 3.0, 5.0, 10.0, 20.0, 50.0,
    ]
    deltas = [-0.85, -0.3, 0.3, 0.85]
    rows = [(1.0, r, de * np.sqrt(r)) for r in ratios for de in deltas]
    return np.array(rows)


def _tl_impl(A, B, d, lmax):
    """Call the implementation under test on a batch of (A, B, d) rows."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    d = np.asarray(d, dtype=float)
    sqrtc = np.sqrt(A + B + 2.0 * d)
    sqrta = np.sqrt(A + B - 2.0 * d)
    t0 = _t0_closed(A, B, d)
    out = V._tl_stable(
        jnp.asarray(A), jnp.asarray(B), jnp.asarray(d),
        jnp.asarray(sqrtc), jnp.asarray(sqrta), jnp.asarray(t0), lmax,
    )
    return np.asarray(out), sqrtc, sqrta, t0


# ---------------------------------------------------------------------------
# (a) high-precision reference, stable AND unstable regions
# ---------------------------------------------------------------------------


def test_stability_selected_recurrence_matches_quadrature():
    """New ``T_l`` matches the integral to rtol 1e-10 on the whole grid.

    The grid spans ``B/A`` from 0.1 to 50 with ``l`` up to 45 — deep into
    the region where the legacy forward recurrence has lost *all* digits
    (relative error up to ~1e25 at ``B/A = 50``, see the divergence test).
    """
    lmax = 45
    grid = _param_grid(lmax)
    tl, _, _, _ = _tl_impl(grid[:, 0], grid[:, 1], grid[:, 2], lmax)

    worst = 0.0
    for i, (A, B, d) in enumerate(grid):
        ref, referr = _tl_quad_reference(A, B, d, lmax)
        # Reference quality gate: quad's own error estimate must sit well
        # below the comparison tolerance for the comparison to mean anything.
        assert np.all(np.abs(referr) <= 1e-11 * np.abs(ref)), (
            f"quad reference too loose at B/A={B / A:.4g}, d={d:+.3g}"
        )
        rel = np.max(np.abs(tl[:, i] - ref) / np.abs(ref))
        worst = max(worst, rel)
        assert rel <= 1e-10, (
            f"T_l mismatch: B/A={B / A:.4g} d={d:+.3g} "
            f"(selected={'backward' if lmax * np.log(B / A) > THRESHOLD else 'forward'}) "
            f"max rel err {rel:.3e}"
        )
    # Measured headroom (documented, not asserted): worst ~1.2e-11, on the
    # forward side just below the switch; backward side reaches ~1e-15.
    assert worst < 1e-10


# ---------------------------------------------------------------------------
# (b) bit-compatibility with the legacy forward recurrence below onset
# ---------------------------------------------------------------------------


def test_forward_branch_reproduces_legacy_below_onset():
    """Below the switch the new path IS the old forward recurrence.

    The selection masks pass the untouched operands through ``jnp.where``
    on forward-selected points and the update arithmetic is op-for-op the
    legacy loop, so in eager mode (the default test lane) the agreement is
    exact; the assert allows 1e-14 relative slack per the acceptance spec.
    Every shipped golden/parity number below the onset is therefore pinned
    unchanged.
    """
    lmax = 10  # stable: lmax * ln(B/A) <= 10*ln(2.5) ~ 9.2 < threshold 16.1
    ratios = np.array([0.1, 0.5, 0.9, 1.01, 1.2, 1.5, 2.0, 2.5])
    deltas = np.array([-0.7, -0.2, 0.4, 0.8])
    rows = np.array([(1.0, r, de * np.sqrt(r)) for r in ratios for de in deltas])
    A, B, d = rows[:, 0], rows[:, 1], rows[:, 2]
    assert np.all(lmax * np.log(B / A) < THRESHOLD)

    tl_new, sqrtc, sqrta, t0 = _tl_impl(A, B, d, lmax)
    tl_old = _tl_forward_legacy(A, B, d, sqrtc, sqrta, t0, lmax)

    np.testing.assert_allclose(tl_new, tl_old, rtol=1e-14, atol=0.0)
    # Stronger, measured property on the default (eager) lane: bit-identical.
    assert np.array_equal(tl_new, tl_old)


# ---------------------------------------------------------------------------
# (c) the legacy forward recurrence really is broken above onset
# ---------------------------------------------------------------------------


def test_legacy_forward_recurrence_diverges_above_onset():
    """Documents why the fix exists: forward-only ``T_l`` loses everything.

    At ``B/A = 10`` the spurious mode grows like ``10^{l/2}``; by ``l = 45``
    the legacy result is off by more than a million times the true value
    (measured ~1e7-1e8), while the stability-selected result stays at
    ~1e-15.  Both VMEC2000 and the pre-fix port produced the legacy values,
    which is why two-code parity gates could never catch this.
    """
    lmax = 45
    A, B = 1.0, 10.0
    for delta in (-0.6, 0.3):
        d = delta * np.sqrt(A * B)
        ref, _ = _tl_quad_reference(A, B, d, lmax)
        sqrtc = np.sqrt(A + B + 2.0 * d)
        sqrta = np.sqrt(A + B - 2.0 * d)
        t0 = _t0_closed(A, B, d)
        legacy = _tl_forward_legacy(A, B, d, sqrtc, sqrta, t0, lmax)
        rel_legacy = np.max(np.abs(legacy - ref) / np.abs(ref))
        assert rel_legacy > 1e6, (
            f"expected legacy forward blow-up, got {rel_legacy:.3e}"
        )

        tl_new, _, _, _ = _tl_impl([A], [B], [d], lmax)
        rel_new = np.max(np.abs(tl_new[:, 0] - ref) / np.abs(ref))
        assert rel_new <= 1e-10, f"fixed path off by {rel_new:.3e}"


# ---------------------------------------------------------------------------
# (d) integration smoke: shipped low-mode free-boundary output unchanged
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cth_vacuum_inputs():
    """Boundary + bexni of the CTH-like LASYM parity fixture (initial state)."""
    FB = pytest.importorskip("vmex.core.freeboundary")
    pytest.importorskip("netCDF4")
    from vmex.core.input import VmecInput
    from vmex.core.mgrid import MgridField, read_mgrid
    from vmex.core.solver import _initial_state, prepare_runtime, resolution_from_input

    deck = REPO / "examples" / "data" / "input.cth_like_free_bdy_lasym_small"
    mgrid_path = REPO / "examples" / "data" / "mgrid_cth_like_lasym_small.nc"
    inp = VmecInput.from_file(deck)
    res = resolution_from_input(inp)
    rt = prepare_runtime(inp, res)
    state = _initial_state(rt.setup)
    basis = V.vacuum_basis(
        mf=int(inp.mpol) + 1, nf=int(inp.ntor),
        ntheta3=res.ntheta3, nzeta=res.nzeta, nfp=res.nfp,
        lasym=res.lasym, wint=np.asarray(rt.trig.wint),
    )
    rmnc, zmns, rmns, zmnc = FB._edge_fourier(state, rt)
    boundary = FB.boundary_from_coefficients(
        rmnc=rmnc, zmns=zmns, rmns=rmns, zmnc=zmnc, modes=rt.modes, basis=basis
    )
    ctor, _rbtor, axis_r, axis_z, _bsq3, _pres = FB._vacuum_scalars(state, rt)
    mg = read_mgrid(mgrid_path)
    field = MgridField.from_mgrid_data(
        mg, extcur=np.asarray(inp.extcur, dtype=float)[: mg.nextcur]
    )
    phi = (np.asarray(basis.zeta) * basis.onp).reshape(basis.ntheta3, basis.nzeta)
    br_c, bp_c, bz_c = field.b_cyl(np.asarray(boundary.R), phi, np.asarray(boundary.Z))
    br_a, bp_a, bz_a = FB.axis_current_field(
        R=np.asarray(boundary.R), Z=np.asarray(boundary.Z),
        axis_r=np.asarray(axis_r), axis_z=np.asarray(axis_z),
        nfp=res.nfp, plascur=float(ctor),
    )
    ext = FB.external_field_channels(
        boundary=boundary,
        br=np.asarray(br_c) + br_a, bp=np.asarray(bp_c) + bp_a,
        bz=np.asarray(bz_c) + bz_a,
        basis=basis, signgs=int(rt.setup.signgs),
    )
    return dict(
        basis=basis, boundary=boundary, signgs=int(rt.setup.signgs),
        bexni=jnp.asarray(np.asarray(ext["bexni"], dtype=float)),
    )


def _fixture_recurrence_operands(basis, boundary):
    """(adp, adm, cma, sqrtc, sqrta, tlp0, tlm0) as ``_analytic_terms`` forms."""
    onp = float(basis.onp)
    Rf = np.reshape(np.asarray(boundary.R), (-1,))
    Ruf = np.reshape(np.asarray(boundary.Ru), (-1,))
    Rvf = np.reshape(np.asarray(boundary.Rv), (-1,))
    Zuf = np.reshape(np.asarray(boundary.Zu), (-1,))
    Zvf = np.reshape(np.asarray(boundary.Zv), (-1,))
    guu = Ruf * Ruf + Zuf * Zuf
    guv = (Ruf * Rvf + Zuf * Zvf) * (2.0 * onp)
    gvv = (Rvf * Rvf + Zvf * Zvf + Rf * Rf) * (onp * onp)
    adp = guu + guv + gvv
    adm = guu - guv + gvv
    cma = gvv - guu
    sqrtc = 2.0 * np.sqrt(gvv)
    sqrta = 2.0 * np.sqrt(guu)
    sq1 = np.sqrt(adp)
    sq2 = np.sqrt(adm)
    tlp0 = (1.0 / sq1) * np.log((sq1 * sqrtc + adp + cma) / (sq1 * sqrta - adp + cma))
    tlm0 = (1.0 / sq2) * np.log((sq2 * sqrtc + adm + cma) / (sq2 * sqrta - adm + cma))
    return adp, adm, cma, sqrtc, sqrta, tlp0, tlm0


def test_cth_fixture_below_onset_and_bit_identical(cth_vacuum_inputs, monkeypatch):
    """The shipped low-mode free-boundary vacuum output is unchanged.

    Three stacked guarantees on the real CTH-like LASYM fixture at its
    normal resolution:

    1. every evaluation point of both ``T^+`` and ``T^-`` sits below the
       switch (measured margin ~2x), so the forward branch is selected
       everywhere;
    2. eager lane (the conftest default): the full NESTOR solve with the
       new selection is BIT-IDENTICAL to the same solve forced through the
       forward-only legacy recurrence;
    3. jitted lane (how production runs): outputs agree to max-normalized
       ~1e-11 — two *different* XLA programs are never guaranteed
       bit-equal (fusion/FMA contraction), and the observed ~1e-12-level
       reordering noise is the same class the free-boundary golden
       docstrings already document for any graph change.
    """
    basis = cth_vacuum_inputs["basis"]
    boundary = cth_vacuum_inputs["boundary"]
    bexni = cth_vacuum_inputs["bexni"]
    signgs = cth_vacuum_inputs["signgs"]
    lmax = int(basis.mf) + int(basis.nf)

    # 1. below onset everywhere, for both integral families
    adp, adm, cma, sqrtc, sqrta, tlp0, tlm0 = _fixture_recurrence_operands(
        basis, boundary
    )
    crit_p = lmax * np.log(np.maximum(adm, 1e-300) / adp)
    crit_m = lmax * np.log(np.maximum(adp, 1e-300) / adm)
    assert float(np.max(crit_p)) < THRESHOLD
    assert float(np.max(crit_m)) < THRESHOLD

    def forward_only(A, B, c, sc, sa, t0, lm):
        return V._tl_forward(A, B, c, sc, sa, t0, lm)

    # 2. eager: bit-identical end to end (assumes the conftest default lane)
    # Enforce the eager lane rather than assuming it: under xdist a prior
    # module's jit-enabling module fixture can still be live on this worker
    # (lazy teardown), which silently turned this comparison into the
    # jitted one and broke bit-identity at the documented ~1e-12 level.
    prev_disable_jit = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", True)
    try:
        new_solver = V.make_vacuum_solver(basis, signgs=signgs)
        new_out = new_solver.full(boundary, bexni)
        with monkeypatch.context() as mp:
            mp.setattr(V, "_tl_stable", forward_only)
            old_out = V.make_vacuum_solver(basis, signgs=signgs).full(
                boundary, bexni)
        names = ("potvac", "mode_matrix", "bvec_nonsing", "rhs", "gsource",
                 "grpmn")
        for name, a, b in zip(names, new_out, old_out, strict=True):
            assert np.array_equal(np.asarray(a), np.asarray(b)), (
                f"eager {name} not bit-identical"
            )

        # ... and the T arrays themselves, for both families
        for A, B, t0 in ((adp, adm, tlp0), (adm, adp, tlm0)):
            args = tuple(
                jnp.asarray(x) for x in (A, B, cma, sqrtc, sqrta, t0)
            )
            assert np.array_equal(
                np.asarray(V._tl_stable(*args, lmax)),
                np.asarray(V._tl_forward(*args, lmax)),
            )
    finally:
        jax.config.update("jax_disable_jit", prev_disable_jit)

    # 3. jitted: reordering-level agreement only (documented above)
    jax.config.update("jax_disable_jit", False)
    try:
        new_j = V.make_vacuum_solver(basis, signgs=signgs).full(boundary, bexni)
        with monkeypatch.context() as mp:
            mp.setattr(V, "_tl_stable", forward_only)
            old_j = V.make_vacuum_solver(basis, signgs=signgs).full(boundary, bexni)
        for name, a, b in zip(names, new_j, old_j, strict=True):
            a = np.asarray(a)
            b = np.asarray(b)
            scale = float(np.max(np.abs(b)))
            diff = float(np.max(np.abs(a - b)))
            assert diff <= 1e-11 * scale, (
                f"jit {name}: {diff:.3e} vs scale {scale:.3e}"
            )
    finally:
        jax.config.update("jax_disable_jit", prev_disable_jit)


def test_solver_dataclass_unchanged():
    """The public surface is untouched: same signature, same closures."""
    fields = {f.name for f in dataclasses.fields(V.VacuumSolver)}
    assert fields == {"basis", "signgs", "full", "skip", "assemble"}
