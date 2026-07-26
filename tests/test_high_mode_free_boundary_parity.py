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


@pytest.fixture(scope="module")
def free_input(tmp_path_factory) -> VmecInput:
    path = tmp_path_factory.mktemp("hm_free") / "input.hm_free"
    path.write_text(stress_indata_text(lfreeb=True))
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
def test_fixed_boundary_238_mode_ladder_is_lawful(fixed_input: VmecInput) -> None:
    fsq = _require_lawful(lambda: solve_multigrid(
        fixed_input, verbose=False, raise_on_max_iterations=True))
    assert all(v < 1.0e12 for v in fsq), f"residuals diverged: {fsq}"


@pytest.mark.full  # ~minutes: 238-mode free ladder against the generated coils
def test_free_boundary_238_mode_ladder_is_lawful(free_input: VmecInput) -> None:
    field = qi_free_field(int(free_input.nfp))
    fsq = _require_lawful(lambda: solve_free_boundary_multigrid(
        free_input, external_field=field, verbose=False,
        raise_on_max_iterations=True))
    assert all(v < 1.0e12 for v in fsq), f"residuals diverged: {fsq}"


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
