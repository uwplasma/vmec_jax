"""High-mode fixed/free stress case reproducing the reported feature stack.

One public, fully generated deck combining every ingredient of the reported
production failures — at the same 238-mode scale — so the combination is
guarded, not just each ingredient in isolation:

* **238 Fourier modes** (``MPOL=13, NTOR=9``: VMEC stores the m = 0 row only
  for n >= 0, so ``mnmax = (ntor+1) + (mpol-1)*(2*ntor+1) = 10 + 12*19 = 238``),
  matching the reported deck's mode count exactly;
* **finite pressure** (``AM`` power series with a nonzero ``PRES_SCALE``);
* **``LFORBAL = T``** — the alternative radial force-balance formulation;
* **``PRECON_TYPE = 'NONE'`` with ``PREC2D_THRESHOLD = 1e-30``**;
* **``APHI``** and the other 1-D profile arrays;
* **indexed Fortran array sections** (``RBC(-2:2,0) = ...``);
* **no supplied magnetic axis** — the solver must construct and, if needed,
  recover its own;
* **automatic angular resolution** (``NTHETA = 0``, ``NZETA = 0``);
* **a radial ladder** in both fixed- and free-boundary variants.

The deck is written as ``&INDATA`` *text*, not a constructed dataclass, so the
Fortran-compatibility surface itself is what gets parsed.

Outcome policy (per review): a test that accepts any ``VmecError`` as success
guards nothing.  Here the *only* accepted terminations are convergence or the
typed iteration-budget outcome (``VmecConvergenceError``); Jacobian failures,
numerical failures, or crashes fail the test.  Residual channels must be finite
and iterations must advance in every accepted outcome.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from vmex.core.errors import VmecConvergenceError  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.multigrid import (  # noqa: E402
    solve_free_boundary_multigrid,
    solve_multigrid,
)
from vmex.core.solver import resolution_from_input  # noqa: E402

from tests.test_qi_free_boundary_case import qi_free_field  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"

#: (ntor+1) + (mpol-1)*(2*ntor+1) = 10 + 12*19 = 238 active Fourier modes,
#: the reported deck's count exactly.
STRESS_MPOL, STRESS_NTOR = 13, 9


def _boundary_rows() -> str:
    """The nfp=2 QI boundary as INDATA rows, plus indexed-section forms.

    The m = 0 row is written as a Fortran ``lo:hi`` array section on purpose:
    a parser that mishandles indexed sections corrupts the mean surface, which
    is exactly the reported failure mode.
    """
    inp = VmecInput.from_file(str(DATA / "input.nfp2_QI"))
    rbc = np.asarray(inp.rbc, dtype=float)
    zbs = np.asarray(inp.zbs, dtype=float)
    ntor = int(inp.ntor)

    # m = 0 row as indexed sections over n in [-2, 2].
    n_lo, n_hi = -2, 2
    r_sec = " ".join(f"{rbc[ntor + n, 0]:.12e}" for n in range(n_lo, n_hi + 1))
    z_sec = " ".join(f"{zbs[ntor + n, 0]:.12e}" for n in range(n_lo, n_hi + 1))
    lines = [
        f"  RBC({n_lo}:{n_hi},0) = {r_sec}",
        f"  ZBS({n_lo}:{n_hi},0) = {z_sec}",
    ]
    # remaining nonzero coefficients element-by-element
    for m in range(int(inp.mpol)):
        for n in range(-ntor, ntor + 1):
            if m == 0 and n_lo <= n <= n_hi:
                continue
            r, z = rbc[ntor + n, m], zbs[ntor + n, m]
            if r != 0.0:
                lines.append(f"  RBC({n},{m}) = {r:.12e}")
            if z != 0.0:
                lines.append(f"  ZBS({n},{m}) = {z:.12e}")
    return "\n".join(lines)


def stress_indata_text(*, lfreeb: bool, ns_array=(21, 34), niter: int = 30) -> str:
    """The full 238-mode deck with every reported feature enabled at once."""
    ns = " ".join(str(int(n)) for n in ns_array)
    ftol = " ".join("1.0E-11" for _ in ns_array)
    nit = " ".join(str(int(niter)) for _ in ns_array)
    return f"""&INDATA
  MGRID_FILE = '{"qi_modular(generated)" if lfreeb else ""}'
  LFREEB = {"T" if lfreeb else "F"}
  LFORBAL = T
  PRECON_TYPE = 'NONE'
  PREC2D_THRESHOLD = 1.0E-30
  DELT = 0.9
  NFP = 2
  NCURR = 1
  CURTOR = 0.0
  PHIEDGE = 0.03074694979
  EXTCUR = 1.0
  MPOL = {STRESS_MPOL}
  NTOR = {STRESS_NTOR}
  NTHETA = 0
  NZETA = 0
  NS_ARRAY = {ns}
  FTOL_ARRAY = {ftol}
  NITER_ARRAY = {nit}
  NSTEP = 200
  GAMMA = 0.0
  AM = 1.0 -1.0 0.0
  AI = 0.0 0.0
  AC = 0.0 0.0
  APHI = 1.0 0.0 0.0
  PRES_SCALE = 2.0E3
  SPRES_PED = 1.0
{_boundary_rows()}
/
"""


@pytest.fixture(scope="module")
def fixed_input(tmp_path_factory) -> VmecInput:
    path = tmp_path_factory.mktemp("hm_fixed") / "input.hm_fixed"
    path.write_text(stress_indata_text(lfreeb=False))
    return VmecInput.from_file(str(path))


def test_all_reported_features_parse_together(fixed_input: VmecInput) -> None:
    """Every reported ingredient survives one combined parse."""
    inp = fixed_input
    res = resolution_from_input(inp, ns=21)
    assert int(res.mnmax) == 238  # the reported mode count, exactly
    assert bool(inp.lforbal) is True
    assert str(inp.precon_type).strip().lower() in ("none", "'none'")
    assert float(inp.prec2d_threshold) <= 1.0e-29
    assert np.asarray(inp.aphi, dtype=float)[0] == pytest.approx(1.0)
    # no supplied axis: the deck omits RAXIS/ZAXIS entirely
    assert not np.any(np.asarray(inp.raxis_c, dtype=float))
    # automatic angular resolution floors (read_indata.f)
    assert int(res.ntheta) >= 2 * STRESS_MPOL + 6
    assert int(res.nzeta) >= 2 * STRESS_NTOR + 4
    # the indexed m=0 section landed on the mean surface
    rbc = np.asarray(inp.rbc, dtype=float)
    ref = VmecInput.from_file(str(DATA / "input.nfp2_QI"))
    ref_rbc = np.asarray(ref.rbc, dtype=float)
    assert rbc[inp.ntor + 0, 0] == pytest.approx(ref_rbc[ref.ntor + 0, 0])
    assert rbc[inp.ntor + 1, 0] == pytest.approx(ref_rbc[ref.ntor + 1, 0])


def _require_lawful(run) -> tuple[float, float, float]:
    """Run a solve; only convergence or the typed budget outcome may pass."""
    try:
        result = run()
    except VmecConvergenceError as err:
        # iteration budget exhausted: lawful, but must carry finite residuals
        fsq = getattr(err, "fsq", None)
        assert fsq is not None and all(np.isfinite(v) for v in fsq), (
            "budget outcome without finite residuals")
        assert int(getattr(err, "iteration", 0)) >= 1
        return tuple(float(v) for v in fsq)
    # any other VmecError (Jacobian, numerical, input) propagates and FAILS
    fsq = (float(result.fsqr), float(result.fsqz), float(result.fsql))
    assert all(np.isfinite(v) for v in fsq)
    assert int(result.iterations) >= 1
    return fsq


@pytest.mark.full  # ~minutes: 238-mode fixed ladder with LFORBAL + recovery
def test_fixed_boundary_238_mode_ladder_converges(tmp_path) -> None:
    """The public fixed 238-mode case CONVERGES at a realistic budget.

    Review finding: a 30-iteration budget passed via the lawful
    ``VmecConvergenceError`` before proving anything.  With 1000 iterations
    the deck converges (~453 iterations to ~1e-11), so the test now demands
    convergence outright.
    """
    path = tmp_path / "input.hm_fixed_1000"
    path.write_text(stress_indata_text(lfreeb=False, niter=1000))
    inp = VmecInput.from_file(str(path))
    result = solve_multigrid(inp, verbose=False, raise_on_max_iterations=False)
    assert bool(result.converged), (
        f"fixed 238-mode ladder failed to converge: fsqr={float(result.fsqr):.2e}")


@pytest.mark.full  # ~minutes: generated-coils free ladder PAST vacuum activation
def test_free_boundary_238_mode_ladder_survives_activation(tmp_path) -> None:
    """The generated-coils free case stays finite THROUGH vacuum activation.

    Review finding: the old 30-iteration budget ended before activation
    (iteration ~68) and therefore could not catch the post-activation NaN
    produced by an angularly incompatible mgrid/NZETA pairing.  With the
    compatibility policy the automatic resolution selects the field table's
    24 planes, and the run must remain finite well past activation.  It is
    NOT expected to converge: the generated coil set is deliberately poor
    (vacuum and plasma ``R*BTOR`` disagree), and the reference code on a
    compatible grid does not converge it either -- the combined CTH case
    below carries the convergence requirement.
    """
    path = tmp_path / "input.hm_free_200"
    path.write_text(stress_indata_text(lfreeb=True, niter=200))
    inp = VmecInput.from_file(str(path))
    field = qi_free_field(int(inp.nfp))

    lines: list[str] = []

    def collect(text: str = "", end: str = "\n") -> None:
        lines.append(str(text))

    fsq = _require_lawful(lambda: solve_free_boundary_multigrid(
        inp, external_field=field, verbose=True, emit=collect,
        raise_on_max_iterations=True))
    output = "\n".join(lines)
    assert "VACUUM PRESSURE TURNED ON" in output, (
        "budget ended before vacuum activation -- the post-activation "
        "regression surface is not exercised")
    assert all(np.isfinite(v) for v in fsq)


def combined_cth_indata_text() -> str:
    """The 238-mode COMBINED case: every reported feature on a deck that
    converges in both codes.

    Base: the public CTH-like free-boundary fixture, raised to
    ``MPOL=13/NTOR=9`` (mnmax = 238 exactly; the added modes start at zero),
    with ``LFORBAL=T``, ``PRECON_TYPE='NONE'``, ``PREC2D_THRESHOLD=1e-30``,
    ``APHI``, the magnetic axis REMOVED (recovery path), the m=0 boundary
    row as indexed Fortran sections, automatic toroidal resolution
    (``NZETA=0`` -- the mgrid compatibility policy selects the table's 36
    planes), and a 15->25 radial ladder crossing vacuum activation.
    """
    import re

    text = (DATA / "input.cth_like_free_bdy").read_text().split("&END")[0]
    text = text.replace("  MPOL = 5,", "  MPOL = 13,")
    text = text.replace("  NTOR = 4,", "  NTOR = 9,")
    text = text.replace("  NZETA = 36,", "  NZETA = 0,")
    text = text.replace("  NS_ARRAY    = 15,", "  NS_ARRAY    = 15, 25,")
    text = text.replace("  FTOL_ARRAY  = 1.0E-10,",
                        "  FTOL_ARRAY  = 1.0E-8, 1.0E-8,")
    text = text.replace("  NITER_ARRAY = 2500,", "  NITER_ARRAY = 2500, 2500,")
    text = text.replace(
        "  LFREEB = T,",
        "  LFREEB = T,\n  LFORBAL = T,\n  PRECON_TYPE = 'NONE',\n"
        "  PREC2D_THRESHOLD = 1.0E-30,\n  APHI = 1.0, 0.0, 0.0,")
    text = re.sub(r"  RAXIS_CC\(\:\) =[^\n]*\n", "", text)
    text = re.sub(r"  ZAXIS_CS\(\:\) =[^\n]*\n", "", text)
    m0_r, m0_z = {}, {}
    for n in range(0, 5):
        mr = re.search(rf"  RBC\({n},0\) = ([^,\n]+),\n", text)
        mz = re.search(rf"  ZBS\({n},0\) = ([^,\n]+),\n", text)
        m0_r[n], m0_z[n] = mr.group(1), mz.group(1)
        text = text.replace(mr.group(0), "")
        text = text.replace(mz.group(0), "")
    r_sec = " ".join(m0_r[n] for n in range(0, 5))
    z_sec = " ".join(m0_z[n] for n in range(0, 5))
    text = text.replace(
        "  RBC(-4,1)",
        f"  RBC(0:4,0) = {r_sec}\n  ZBS(0:4,0) = {z_sec}\n  RBC(-4,1)")
    return text + "&END\n"


@pytest.mark.full  # ~15 min: the full claimed path in ONE deck, vs VMEC2000
def test_combined_238_mode_cth_free_ladder_matches_vmec2000(tmp_path) -> None:
    """All reported ingredients at once, on a CONVERGENT deck, vs VMEC2000.

    Fresh local xvmec2000/PARVMEC on this exact generated deck (recorded
    2026-07-27, ``NZETA = 36`` explicit since VMEC2000 has no automatic
    compatible selection):

    * 238 Fourier modes; vacuum on at iteration 38 (rung 1);
    * rung 1 (ns=15) converges at 260 iterations (fsqr 9.73e-9);
    * rung 2 (ns=25) converges at 156 iterations (fsqr 9.84e-9);
    * wout: ``wb = 1.283590394747e-3``, axis ``R(v=0) = 0.74414896``
      (``sum raxis_cc``), ``iotaf(edge) = 0.8690375``,
      ``aspect = 5.4332138``.

    Measured VMEX on the same deck (automatic ``NZETA = 0`` -> 36): vacuum
    on at 38, rung 2 converges in **156 iterations** (fsqr 9.74e-9),
    ``r00 = 0.74416249`` -- iteration-for-iteration and axis agreement at
    the shared ftol.

    VMEX must activate vacuum on rung 1, carry it across the transition,
    converge, and land on the same equilibrium.
    """
    mgrid = DATA / "mgrid_cth_like.nc"
    if not mgrid.exists():
        pytest.skip("mgrid fixture not fetched")
    path = tmp_path / "input.combined_238"
    path.write_text(combined_cth_indata_text())
    inp = VmecInput.from_file(str(path))
    res = resolution_from_input(inp, ns=15)
    assert int(res.mnmax) == 238
    assert bool(inp.lforbal) and not np.any(np.asarray(inp.raxis_c))

    lines: list[str] = []

    def collect(text: str = "", end: str = "\n") -> None:
        lines.append(str(text))

    result = solve_free_boundary_multigrid(
        inp, mgrid_path=str(mgrid), verbose=True, emit=collect,
        raise_on_max_iterations=False)

    import re

    output = "\n".join(lines)
    banner_at = output.find("VACUUM PRESSURE TURNED ON")
    second_rung_at = output.rfind("NS = ")
    assert banner_at != -1 and second_rung_at > banner_at, (
        "vacuum did not activate before the radial transition")
    assert bool(result.converged), (
        f"combined 238-mode ladder failed to converge "
        f"(fsqr={float(result.fsqr):.2e})")

    # activation iteration: VMEC2000 and VMEX both print 38 on this deck;
    # a small band absorbs cross-platform float jitter in the 1e-3 crossing
    m = re.search(r"VACUUM PRESSURE TURNED ON AT\s+(\d+)", output)
    assert m is not None
    assert 36 <= int(m.group(1)) <= 40, (
        f"vacuum activated at {m.group(1)}; both codes activate at 38")

    # carried-vacuum rung: both codes converge it in 156 iterations; the
    # band rejects a silent fall-back to fresh reactivation (~260) while
    # absorbing chaotic cross-platform drift
    assert 130 <= int(result.iterations) <= 200, (
        f"carried-vacuum rung took {int(result.iterations)} iterations; "
        "both codes need 156")

    # residual triplet at the recorded VMEC2000 magnitudes (converged just
    # under ftol, with fsqz/fsql well below fsqr)
    assert float(result.fsqr) <= 1.0e-8
    assert float(result.fsqz) <= 2.0e-8 and float(result.fsql) <= 2.0e-8

    # same equilibrium as the recorded VMEC2000 wout: the v=0 axis radius
    # (sum raxis_cc: 0.74414896), the magnetic energy scalar
    # (wb = 1.283590394747e-3) and the edge rotational transform
    # (iotaf(edge) = 0.869037528)
    assert float(result.r00) == pytest.approx(0.7441489627, rel=1e-4)
    assert float(result.wb) == pytest.approx(1.283590394747e-3, rel=2e-4)
    assert float(np.asarray(result.iotaf)[-1]) == pytest.approx(
        0.869037528260457, rel=2e-4)


@pytest.mark.full
def test_vacuum_survives_a_radial_transition() -> None:
    """A free-boundary ladder must carry ACTIVE vacuum across a grid change.

    The reported ladders never reached vacuum activation, so no prior test
    proved the NESTOR/vacuum continuation is rebuilt correctly at the next
    radial resolution once the vacuum pressure is already on.  The CTH-like
    fixture converges quickly enough for the first rung to activate vacuum
    inside an ordinary budget; the second rung then *starts* with active
    vacuum state and must remain finite and converge.

    Measured parity on this exact ladder: VMEC2000 activates vacuum at
    iteration 53, carries it across the transition, and converges the ns=25
    rung in 143 iterations (fsqr 9.7e-9); VMEX converges the same rung in 143
    iterations (fsqr 9.5e-9) -- the carried-vacuum restart is
    iteration-for-iteration faithful.  The generous iteration bound below
    absorbs cross-platform float jitter without weakening the gate.

    The external field MUST come from the deck-aware loader (``mgrid_path``):
    ``MgridField.from_mgrid_data(read_mgrid(...))`` without ``extcur``
    defaults to the file's raw currents and silently ignores the deck's
    ``EXTCUR`` scaling, which turns this case into a different (and
    non-convergent) physics problem.
    """
    import dataclasses

    mgrid = DATA / "mgrid_cth_like.nc"
    if not mgrid.exists():
        pytest.skip("mgrid fixture not fetched")
    inp = VmecInput.from_file(str(DATA / "input.cth_like_free_bdy"))
    inp = dataclasses.replace(
        inp, ns_array=[15, 25], ftol_array=[1.0e-8, 1.0e-8],
        niter_array=[2500, 2500])

    lines: list[str] = []

    def collect(text: str = "", end: str = "\n") -> None:
        lines.append(str(text))

    result = solve_free_boundary_multigrid(
        inp, mgrid_path=str(mgrid), verbose=True, emit=collect,
        raise_on_max_iterations=False)

    output = "\n".join(lines)
    banner_at = output.find("VACUUM PRESSURE TURNED ON")
    second_rung_at = output.rfind("NS = ")
    assert banner_at != -1, "first rung never activated vacuum"
    assert second_rung_at > banner_at, (
        "vacuum activated only after the last grid change -- the transition "
        "never carried active vacuum state")
    for name in ("fsqr", "fsqz", "fsql"):
        assert np.isfinite(float(getattr(result, name)))
    assert bool(result.converged), "post-transition rung failed to converge"
    # VMEC2000 needs 143 iterations on the carried-vacuum ns=25 rung; a
    # faithful restart lands in the same neighbourhood, not at a fresh
    # activation's cost (a full reactivation restarts the residual at ~1e0).
    assert int(result.iterations) <= 500, (
        f"post-transition rung took {int(result.iterations)} iterations; "
        "VMEC2000 needs 143 -- the carried vacuum state is not being reused")


def test_mgrid_nzeta_policy_matches_vmec2000() -> None:
    """VMEC2000's angular-compatibility rule (``mgrid_mod.f`` ier=9).

    A tabulated field with ``kp`` planes per period constrains the solver's
    toroidal grid: NZETA must divide ``kp`` evenly.  Automatic resolution
    (``NZETA = 0``) must therefore select the smallest DIVISOR of ``kp`` at
    or above the ``2*ntor + 4`` floor — for the generated 24-plane field at
    ``NTOR = 9`` that is 24, not the floor 22 (the incompatible 24/22
    pairing produced NaN after vacuum activation; VMEC2000 rejects it
    before solving).  An explicitly incompatible NZETA raises the typed
    input error before iteration one.
    """
    from vmex.core.errors import VmecInputError
    from vmex.core.freeboundary import free_boundary_resolution

    field = qi_free_field(2)
    kp = int(field.br.shape[1])
    assert kp == 24  # the generated fixture's plane count

    def parse(text: str, tmp=[0]) -> VmecInput:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "input.nzeta_policy"
            path.write_text(text)
            return VmecInput.from_file(str(path))

    # automatic: smallest divisor of kp at/above the 2*ntor+4 floor
    auto = parse(stress_indata_text(lfreeb=True))
    res = free_boundary_resolution(auto, field, ns=21)
    assert int(res.nzeta) == 24
    assert kp % int(res.nzeta) == 0

    # explicit compatible divisors pass through unchanged
    ok = parse(
        stress_indata_text(lfreeb=True).replace("NZETA = 0", "NZETA = 12"))
    assert int(free_boundary_resolution(ok, field, ns=21).nzeta) == 12

    # explicit incompatible NZETA: typed error before iteration one
    bad = parse(
        stress_indata_text(lfreeb=True).replace("NZETA = 0", "NZETA = 22"))
    with pytest.raises(VmecInputError, match="divide evenly"):
        free_boundary_resolution(bad, field, ns=21)

    # a non-tabulated (analytic) field imposes no constraint
    class _Analytic:
        def b_cyl(self, r, phi, z):  # pragma: no cover - never called
            return r, phi, z

    assert int(free_boundary_resolution(bad, _Analytic(), ns=21).nzeta) == 22


def test_use_fft_reaches_every_free_boundary_lane(tmp_path, monkeypatch):
    """``use_fft=True`` must reach the traced body of EVERY free lane.

    Review finding: the resolver computed ``use_fft`` and the stage accepted
    it, but the iteration lanes called ``_make_body`` with its default dense
    transform — the FFT kernel was never actually used in free boundary.
    The lanes now take ``use_fft`` as a static jit argument; this spy records
    what ``_make_body`` receives during a short real solve.
    """
    import vmex.core.freeboundary as FBmod

    mgrid = DATA / "mgrid_cth_like.nc"
    if not mgrid.exists():
        pytest.skip("mgrid fixture not fetched")

    seen: list[bool] = []
    original = FBmod._make_body

    def recording(rt, *, evaluation_state=None, use_fft=False):
        seen.append(bool(use_fft))
        return original(rt, evaluation_state=evaluation_state, use_fft=use_fft)

    monkeypatch.setattr(FBmod, "_make_body", recording)
    # fresh vacuum-lane cache: the steady lane bakes use_fft into its traced
    # body, so a cached lane from another test would bypass the spy
    monkeypatch.setattr(FBmod, "_VACUUM_EXECUTABLE_CACHE", {})

    import dataclasses

    inp = dataclasses.replace(
        VmecInput.from_file(str(DATA / "input.cth_like_free_bdy")),
        ns_array=[15], ftol_array=[1.0e-8], niter_array=[80])
    from vmex.core.freeboundary import solve_free_boundary

    solve_free_boundary(inp, mgrid_path=str(mgrid), use_fft=True,
                        error_on_no_convergence=False)
    assert seen, "no lane was traced -- the spy never fired"
    assert all(seen), (
        f"{seen.count(False)} lane trace(s) fell back to the dense body")


@pytest.mark.full  # ~minutes: two short solves through vacuum activation
def test_dense_and_fft_free_boundary_trajectories_match(tmp_path):
    """The separable FFT synthesis is the SAME math as the dense transform.

    Runs the CTH free-boundary case through vacuum activation with both
    kernels and requires matching residual trajectories and final states.
    """
    import dataclasses

    mgrid = DATA / "mgrid_cth_like.nc"
    if not mgrid.exists():
        pytest.skip("mgrid fixture not fetched")
    inp = dataclasses.replace(
        VmecInput.from_file(str(DATA / "input.cth_like_free_bdy")),
        ns_array=[15], ftol_array=[1.0e-8], niter_array=[120])
    from vmex.core.freeboundary import solve_free_boundary

    results = {
        flag: solve_free_boundary(
            inp, mgrid_path=str(mgrid), use_fft=flag,
            error_on_no_convergence=False)
        for flag in (False, True)
    }
    np.testing.assert_allclose(
        np.asarray(results[True].fsq_history),
        np.asarray(results[False].fsq_history),
        rtol=5e-9, atol=1e-15,
        err_msg="FFT and dense free-boundary trajectories diverged")
    for a, b in zip(jax.tree.leaves(results[True].state),
                    jax.tree.leaves(results[False].state), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                   rtol=1e-9, atol=1e-13)


@pytest.mark.full  # ~15 min: above the 512-mode automatic-FFT threshold
def test_free_boundary_537_modes_converges_with_fft(tmp_path, monkeypatch):
    """A CONVERGENT free-boundary case that AUTO-selects the FFT kernel.

    ``MPOL=19/NTOR=14`` on the CTH fixture gives ``mnmax = 15 + 18*29 = 537``
    — above ``GPU_MAX_SPECTRAL_MODES = 512``, the automatic-selection
    threshold that the 238-mode suite cannot reach.

    Review finding: the previous version passed ``use_fft=True`` explicitly,
    so the automatic path (``use_fft=None`` through ``_resolve_use_fft``)
    was never exercised at high mode count.  The solve below therefore omits
    ``use_fft`` entirely; a resolver spy proves the automatic rule was
    consulted with ``None`` and resolved to the FFT kernel, and a
    ``_make_body`` spy (the same mechanism as
    ``test_use_fft_reaches_every_free_boundary_lane``) proves the resolved
    value reached the traced body of every lane.  The host-architecture
    input of the rule is pinned to an FFT-favorable answer so the
    537 > 512 threshold — not the CI runner's CPU vendor — decides,
    deterministically on every host.

    The FFT kernel must then carry the converging high-mode free-boundary
    solve end to end (the fixed-boundary lanes already had this; free
    boundary did not, which is exactly what the review flagged).
    """
    import types

    import vmex.core.freeboundary as FBmod

    mgrid = DATA / "mgrid_cth_like.nc"
    if not mgrid.exists():
        pytest.skip("mgrid fixture not fetched")
    text = (DATA / "input.cth_like_free_bdy").read_text().split("&END")[0]
    text = text.replace("  MPOL = 5,", "  MPOL = 19,")
    text = text.replace("  NTOR = 4,", "  NTOR = 14,")
    text = text.replace("  NZETA = 36,", "  NZETA = 36,")  # 36 % 36 == 0
    text = text.replace("  FTOL_ARRAY  = 1.0E-10,", "  FTOL_ARRAY  = 1.0E-8,")
    path = tmp_path / "input.cth_537"
    path.write_text(text + "&END\n")
    inp = VmecInput.from_file(str(path))
    assert int(resolution_from_input(inp, ns=15).mnmax) == 537

    # Pin the host-architecture input of the automatic rule (solver.platform
    # is consulted only by _resolve_use_fft): CI's x86 CPU runners would
    # otherwise answer the arch gate with "dense" and mask the threshold.
    monkeypatch.setattr("vmex.core.solver.platform",
                        types.SimpleNamespace(machine=lambda: "arm64"))

    resolved: list[tuple[bool | None, bool]] = []
    original_resolve = FBmod._resolve_use_fft

    def recording_resolve(use_fft, device, resolution):
        out = original_resolve(use_fft, device, resolution)
        resolved.append((use_fft, bool(out)))
        return out

    monkeypatch.setattr(FBmod, "_resolve_use_fft", recording_resolve)

    seen: list[bool] = []
    original_body = FBmod._make_body

    def recording_body(rt, *, evaluation_state=None, use_fft=False):
        seen.append(bool(use_fft))
        return original_body(
            rt, evaluation_state=evaluation_state, use_fft=use_fft)

    monkeypatch.setattr(FBmod, "_make_body", recording_body)
    # fresh vacuum-lane cache: the steady lane bakes use_fft into its traced
    # body, so a cached lane from another test would bypass the spy
    monkeypatch.setattr(FBmod, "_VACUUM_EXECUTABLE_CACHE", {})

    from vmex.core.freeboundary import solve_free_boundary

    # use_fft OMITTED on purpose: this drives the automatic selection path.
    result = solve_free_boundary(
        inp, mgrid_path=str(mgrid),
        error_on_no_convergence=False)
    assert resolved == [(None, True)], (
        f"automatic selection did not resolve the FFT kernel above the "
        f"512-mode threshold: {resolved}")
    assert seen and all(seen), (
        f"{seen.count(False)} traced lane bod(y/ies) received the dense "
        f"transform despite the FFT auto-selection")
    assert bool(result.converged), (
        f"537-mode FFT free-boundary solve failed to converge "
        f"(fsqr={float(result.fsqr):.2e})")
