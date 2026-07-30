"""High-mode free-boundary resilience case: a *regression barrier*, not an
accuracy benchmark.  One deck combines every reported free-boundary failure
mode: Fortran indexed array sections (``RBC(-2:2,0) = ...``), ``LFORBAL=T``
actually selecting the alternative force balance, 1-D profile arrays
(``APHI``), a deliberately bad magnetic axis (initial Jacobian sign change
-> eqsolve.f axis re-guess), a large first-iteration force, a free-boundary
``NS_ARRAY`` ladder carrying vacuum/NESTOR state across grid changes, high
mode count, and ``LFULL3D1OUT`` on a run that misses ``ftol``.

The physics need not converge: the contract is that the solver stays
finite, reports a *typed* outcome, and never crashes; every failure mode
fires in the first stage, so budgets stay small.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from vmex.core.errors import VmecError  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.mgrid import MgridField, read_mgrid  # noqa: E402
from vmex.core.multigrid import solve_free_boundary_multigrid  # noqa: E402
from vmex.core.solver import resolution_from_input  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
MGRID = DATA / "mgrid_cth_like.nc"

#: Above the fixture defaults (5/4) to exercise the high-mode paths within
#: the ordinary test budget.
STRESS_MPOL, STRESS_NTOR = 8, 6

#: Coarse rung surviving the bad axis, then a refine step so the
#: vacuum/NESTOR continuation crosses a grid change; every failure mode
#: fires in the first iterations, so a larger ladder buys nothing.
STRESS_NS = (9, 11)

#: Per-rung budget; "stays finite, typed outcome" is decided immediately.
STRESS_NITER = 5


def stress_indata_text() -> str:
    """A free-boundary deck exercising every reported failure mode at once,
    built as INDATA *text* so the Fortran-compatibility surface is tested."""
    return f"""&INDATA
  MGRID_FILE = '{MGRID.name}',
  LFREEB = T,
  LFORBAL = T,
  LFULL3D1OUT = T,
  DELT = 0.7,
  NFP = 5,
  NCURR = 1,
  CURTOR = 43229.08092460368,
  PHIEDGE = -0.035,
  EXTCUR = 4700.0, 1000.0,
  MPOL = {STRESS_MPOL},
  NTOR = {STRESS_NTOR},
  NZETA = 36,
  NS_ARRAY = {STRESS_NS[0]} {STRESS_NS[1]},
  FTOL_ARRAY = 1.0E-12 1.0E-12,
  NITER_ARRAY = {STRESS_NITER} {STRESS_NITER},
  NSTEP = 100,
  GAMMA = 0.0,
  AM = 0.0 0.0,
  AI = 0.0 0.0,
  AC = 0.0 0.0,
  APHI = 1.0 0.0 0.0,
  ! Deliberately displaced axis: pushes the initial Jacobian through a sign
  ! change so the axis re-guess path has to run.
  RAXIS_CC = 0.60 0.0 0.0 0.0 0.0 0.0 0.0,
  ZAXIS_CS = 0.0 0.0 0.0 0.0 0.0 0.0 0.0,
  ! Indexed array sections (the Fortran `lo:hi` form) rather than one
  ! element per line.
  RBC(-2:2,0) = 0.0 0.0 0.7867 0.0 0.0
  ZBS(-2:2,0) = 0.0 0.0 0.0 0.0 0.0
  RBC(-1:1,1) = 0.01 0.1494 -0.01
  ZBS(-1:1,1) = -0.01 0.1494 0.01
/
"""


@pytest.fixture(scope="module")
def stress_input(tmp_path_factory) -> VmecInput:
    """Parse the stress deck from disk (exercises the file path, not a dict)."""
    path = tmp_path_factory.mktemp("fb_stress") / "input.fb_stress"
    path.write_text(stress_indata_text())
    return VmecInput.from_file(str(path))


def _stress_field(inp: VmecInput) -> MgridField:
    """The deck's external field with EXTCUR applied.

    ``from_mgrid_data`` without ``extcur`` defaults to the file's raw currents
    and would silently drop the deck's ``EXTCUR`` scaling.
    """
    data = read_mgrid(MGRID)
    return MgridField.from_mgrid_data(
        data,
        extcur=np.asarray(inp.extcur, dtype=float)[: data.nextcur],
    )


def test_indexed_sections_and_profile_arrays_parse(stress_input: VmecInput) -> None:
    """Indexed ``RBC``/``ZBS`` sections, ``LFORBAL`` and ``APHI`` all survive parsing."""
    inp = stress_input
    # the ``lo:hi`` sections landed in the right (n, m) slots
    rbc = np.asarray(inp.rbc, dtype=float)
    assert rbc[inp.ntor + 0, 0] == pytest.approx(0.7867)
    assert rbc[inp.ntor + 1, 1] == pytest.approx(-0.01)
    assert rbc[inp.ntor - 1, 1] == pytest.approx(0.01)
    # a parser that dropped the section would leave the boundary all-zero
    assert np.count_nonzero(rbc) >= 3
    # the alternative force-balance formulation is selected, not merely accepted
    assert bool(inp.lforbal) is True
    # 1-D profile arrays parse without disturbing the others
    assert np.asarray(inp.aphi, dtype=float)[0] == pytest.approx(1.0)
    assert bool(inp.lfreeb) is True
    assert bool(inp.lfull3d1out) is True


def test_stress_case_is_high_mode(stress_input: VmecInput) -> None:
    """The deck really is in the high-mode regime the perf work targets."""
    res = resolution_from_input(stress_input, ns=STRESS_NS[0])
    # mnmax = mpol * (2*ntor + 1); well above the fixture's 5 x 9.
    assert int(res.mnmax) >= STRESS_MPOL * (2 * STRESS_NTOR + 1) // 2
    assert int(res.ntheta) >= 2 * STRESS_MPOL + 6  # automatic-resolution floor


@pytest.mark.full  # multi-minute real two-rung free-boundary recovery
@pytest.mark.skipif(not MGRID.exists(), reason="mgrid fixture not fetched")
def test_bad_axis_free_boundary_ladder_stays_finite(stress_input: VmecInput) -> None:
    """Headline contract: the hostile deck degrades gracefully — residuals
    stay finite and the outcome is typed; never a raw crash or a silently
    carried NaN force (ftol is not expected to be reached)."""
    field = _stress_field(stress_input)
    lines: list[str] = []

    def collect(text: str = "", end: str = "\n") -> None:
        """Console sink with ``print``'s signature (the driver passes ``end``)."""
        lines.append(str(text))

    try:
        result = solve_free_boundary_multigrid(
            stress_input, external_field=field, verbose=True,
            emit=collect, raise_on_max_iterations=False,
        )
    except VmecError:
        return  # a typed, diagnosed termination is an acceptable outcome
    for name in ("fsqr", "fsqz", "fsql"):
        value = float(getattr(result, name))
        assert np.isfinite(value), f"{name} went non-finite on the stress deck"
    assert int(result.iterations) >= 1
    # the recovery machinery must be reachable, not silently bypassed
    assert any("FORCE ITERATIONS" in ln for ln in lines)


@pytest.mark.full  # multi-minute forced 75-reset event
@pytest.mark.skipif(not MGRID.exists(), reason="mgrid fixture not fetched")
def test_jacobian_recovery_uses_checkpoint_and_reduces_delt(monkeypatch) -> None:
    """The 75-reset recovery restarts the CHECKPOINT with a HALVED time step
    (eqsolve.f aborts fatally at ``ijacob >= 75``).  ``DELT = 1e4`` on the
    converging CTH deck forces the event deterministically; a stage-recursion
    spy asserts: (a) the recovery fires (banner + recursive call); (b) the
    restored state is the recorded checkpoint — present and fully finite,
    never ``None``/cold (bitwise restore semantics live in
    ``tests/test_step_control.py``); (c) each retry uses
    ``min(0.5, 0.5 * prev)`` so the trajectory cannot replay; (d) the
    recovered trajectory converges, which the failing attempt never does."""
    import dataclasses

    import vmex.core.freeboundary as FBmod

    inp = dataclasses.replace(
        VmecInput.from_file(str(DATA / "input.cth_like_free_bdy")),
        delt=1.0e4,
    )

    recorded: list[dict] = []
    original = FBmod._solve_free_boundary_stage

    def recording(deck, **kw):
        state = kw.get("initial_state")
        recorded.append({
            "time_step": kw.get("time_step"),
            "retries": kw.get("jacobian_retries"),
            "has_state": state is not None,
            "state_finite": bool(
                np.all(np.isfinite(np.asarray(state.R_cos)))
                and np.all(np.isfinite(np.asarray(state.Z_sin)))
            ) if state is not None else None,
        })
        return original(deck, **kw)

    monkeypatch.setattr(FBmod, "_solve_free_boundary_stage", recording)

    lines: list[str] = []

    def collect(text: str = "", end: str = "\n") -> None:
        lines.append(str(text))

    result = solve_free_boundary_multigrid(
        inp, mgrid_path=str(MGRID), verbose=True, emit=collect,
        raise_on_max_iterations=False,
    )

    # (a) the recovery event happened, visibly and structurally
    assert any("JACOBIAN RECOVERY RETRY" in ln for ln in lines), (
        "DELT=1e4 no longer reaches the 75-reset recovery")
    retries = [c for c in recorded if c["time_step"] is not None]
    assert retries, "no recursive recovery call was recorded"

    # (b) every retry restarts a real, fully finite checkpoint state
    for call in retries:
        assert call["has_state"], (
            "recovery restarted a cold state instead of the checkpoint")
        assert call["state_finite"], "restored checkpoint has non-finite leaves"

    # (c) the halving chain min(0.5, 0.5*prev): the retry time step genuinely
    # differs from the failing attempt's, so the trajectory cannot replay
    prev = float(inp.delt)
    for call in retries:
        expected = min(0.5, 0.5 * prev)
        assert call["time_step"] == pytest.approx(expected), (
            f"retry DELT {call['time_step']} != min(0.5, 0.5*{prev})")
        assert call["retries"] is not None and call["retries"] >= 0
        prev = expected

    # (d) the recovered trajectory converges; the DELT=1e4 attempt never does
    assert bool(result.converged), (
        f"recovered run failed to converge (fsqr={float(result.fsqr):.2e})")


@pytest.mark.full  # long-running two-reset recovery plus converging retry
@pytest.mark.skipif(not MGRID.exists(), reason="mgrid fixture not fetched")
def test_consecutive_jacobian_recoveries_free_boundary(monkeypatch) -> None:
    """MULTIPLE consecutive JAC75 events in the (pre-vacuum) free lane — the
    confidential-case signature.  The first retry is forced back to the
    hostile input ``DELT``, producing a SECOND genuine 75-reset event inside
    the recovery chain.  Asserted across BOTH events: every retry restarts a
    present, fully finite checkpoint; the requested ``DELT`` follows
    ``min(0.5, 0.5 * prev)`` from the runtime the failing attempt used; the
    retry budget decrements 2 -> 1 -> 0; retries never re-enable the axis
    re-guess nor reuse the failed vacuum cache; both events restore the SAME
    best checkpoint (the sabotaged attempt records no lower residual —
    fingerprints equal, no-replay carried by the differing time step); the
    final attempt converges.  All events precede vacuum activation."""
    import dataclasses

    import vmex.core.freeboundary as FBmod

    inp = dataclasses.replace(
        VmecInput.from_file(str(DATA / "input.cth_like_free_bdy")),
        delt=1.0e4,
    )

    recorded: list[dict] = []
    original = FBmod._solve_free_boundary_stage

    def hostile_first_retry(deck, **kw):
        n = len(recorded)
        forced = dict(kw)
        if n == 1:  # first retry: as hostile as the initial attempt
            forced["time_step"] = float(inp.delt)
        state = kw.get("initial_state")
        recorded.append({
            "requested_time_step": kw.get("time_step"),
            "retries": kw.get("jacobian_retries"),
            "allow_reguess": kw.get("allow_initial_axis_reguess", True),
            "reuse_vacuum": kw.get("reuse_vacuum_cache", False),
            "has_state": state is not None,
            "state_finite": bool(
                np.all(np.isfinite(np.asarray(state.R_cos)))
                and np.all(np.isfinite(np.asarray(state.Z_sin)))
            ) if state is not None else None,
            "fingerprint": float(np.asarray(state.R_cos).sum())
            if state is not None else None,
        })
        return original(deck, **forced)

    monkeypatch.setattr(FBmod, "_solve_free_boundary_stage", hostile_first_retry)

    lines: list[str] = []

    def collect(text: str = "", end: str = "\n") -> None:
        lines.append(str(text))

    result = solve_free_boundary_multigrid(
        inp, mgrid_path=str(MGRID), verbose=True, emit=collect,
        raise_on_max_iterations=False,
    )

    banners = [ln for ln in lines if "JACOBIAN RECOVERY RETRY" in ln]
    assert len(banners) == 2, "expected TWO consecutive 75-reset recoveries"
    assert len(recorded) == 3
    assert recorded[0]["requested_time_step"] is None
    assert recorded[0]["has_state"] is False    # fresh profil3d start

    # both events: real finite checkpoint + halved/capped DELT law
    for event, call in enumerate(recorded[1:], start=1):
        assert call["has_state"], (
            f"recovery {event} restarted a cold state instead of the "
            "checkpoint")
        assert call["state_finite"], (
            f"recovery {event} checkpoint has non-finite leaves")
        assert call["requested_time_step"] == pytest.approx(
            min(0.5, 0.5 * float(inp.delt)))
        assert call["allow_reguess"] is False
        assert call["reuse_vacuum"] is False
    assert recorded[1]["retries"] == 1
    assert recorded[2]["retries"] == 0

    # same best checkpoint restored again; no-replay carried by the DELT
    assert recorded[2]["fingerprint"] == recorded[1]["fingerprint"]
    assert recorded[2]["requested_time_step"] != pytest.approx(
        float(inp.delt))

    assert bool(result.converged), (
        f"chain failed to converge (fsqr={float(result.fsqr):.2e})")
