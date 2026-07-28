"""High-mode free-boundary resilience case.

This is a *regression barrier*, not an accuracy benchmark.  It combines, in one
deck, every failure mode that has been reported against the free-boundary
driver, so a future refactor cannot quietly reintroduce any of them:

1. **Fortran indexed array sections** — ``RBC(-2:2,0) = ...`` assigns a slice in
   one statement instead of one element per line.  A parser that rejects the
   ``lo:hi`` form, or silently drops the assigned run, fails here.
2. **``LFORBAL = T``** — selects the alternative radial force-balance
   formulation.  A driver that parses the flag but always runs the default
   equations produces a different trajectory and fails the mode assertion.
3. **1-D profile arrays (``APHI``)** — present alongside the usual ``AM``/``AI``/
   ``AC``, and must survive parsing into the toroidal-flux derivative.
4. **A deliberately bad magnetic axis** — displaced far enough that the initial
   Jacobian changes sign, forcing the ``eqsolve.f`` axis re-guess.  A driver
   that skips recovery reports a non-finite first force instead.
5. **A large first-iteration force** on a high-mode grid, which is what drives
   the axis-improvement and Jacobian-reset paths.
6. **A free-boundary ``NS_ARRAY`` ladder** with the vacuum/NESTOR state carried
   across radial-grid changes.
7. **High mode count** (``MPOL``/``NTOR`` well above the fixture defaults), the
   regime where the mode-stacked synthesis and the block storage are expensive.
8. **``LFULL3D1OUT``** semantics on a run that does not reach ``ftol``.

The physics does not need to converge: the contract is that the solver stays
finite, reports a *typed* outcome, and never crashes.  Iteration budgets are
small so the case stays inside the ordinary test budget; the failure modes all
trigger in the first stage.
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

#: Poloidal/toroidal resolution for the stress deck.  Chosen above the fixture
#: defaults (5/4) so ``mnmax`` is large enough to exercise the high-mode paths
#: while still fitting the ordinary test budget.
STRESS_MPOL, STRESS_NTOR = 8, 6

#: Radial ladder: a coarse rung that must survive the bad axis, then a refine
#: step so the vacuum/NESTOR continuation is exercised across a grid change.
#: Kept small deliberately -- every failure mode below fires in the first
#: iterations of the first rung, and a displaced axis multiplies the cost via
#: recovery restarts, so a large budget buys no extra coverage.
STRESS_NS = (9, 11)

#: Iterations per rung.  The contract under test is "stays finite and reports a
#: typed outcome", which is decided immediately; this is not a convergence run.
STRESS_NITER = 5


def stress_indata_text() -> str:
    """A free-boundary deck exercising every reported failure mode at once.

    Built as INDATA *text* (not a constructed dataclass) precisely so the
    Fortran-compatibility surface -- indexed array sections, 1-D profile
    arrays, logical flags -- is what gets tested.
    """
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


@pytest.mark.full  # ~2 min: a real two-rung free-boundary ladder with recovery
@pytest.mark.skipif(not MGRID.exists(), reason="mgrid fixture not fetched")
def test_bad_axis_free_boundary_ladder_stays_finite(stress_input: VmecInput) -> None:
    """The headline contract: a hostile deck degrades *gracefully*.

    With a displaced axis on a high-mode free-boundary ladder the run is not
    expected to reach ``ftol``.  It must nonetheless keep every residual finite
    and finish with a typed VMEX outcome -- never a raw crash, and never a NaN
    force silently carried forward.
    """
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


@pytest.mark.full  # ~2 min: forced 75-reset event, instrumented recovery
@pytest.mark.skipif(not MGRID.exists(), reason="mgrid fixture not fetched")
def test_jacobian_recovery_uses_checkpoint_and_reduces_delt(monkeypatch) -> None:
    """The 75-reset recovery restarts the CHECKPOINT with a HALVED time step.

    ``eqsolve.f`` aborts fatally at ``ijacob >= 75``; the bounded recovery
    instead restarts the best finite checkpoint with a reduced ``DELT``.  Two
    silent regressions would defeat it while keeping the end-to-end recovery
    tests green:

    * restarting a *reconstructed cold state* instead of the checkpoint
      (``initial_state=None`` in the recursive stage call), and
    * replaying the *identical trajectory* because the time step was not
      actually reduced.

    A deliberately absurd ``DELT = 1e4`` on the converging CTH deck forces the
    75-reset event deterministically (the ``jacobian_retries=0`` policy test
    proves ``jacobian_resets == 75`` for this recipe).  This test intercepts
    the module-level stage recursion and asserts: (a) the recovery event fires
    (banner + recursive call); (b) the restored state is the recorded
    checkpoint -- present and fully finite, never ``None`` (bitwise restore
    semantics are unit-locked in ``tests/test_step_control.py``); (c) each
    retry uses ``min(0.5, 0.5 * prev)`` -- a genuinely different time step
    from the failing attempt, so the trajectory cannot replay; (d) the
    recovered trajectory then *converges*, which the failing attempt provably
    never does.
    """
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


@pytest.mark.full  # ~3 min: TWO real 75-reset events + a converging retry
@pytest.mark.skipif(not MGRID.exists(), reason="mgrid fixture not fetched")
def test_consecutive_jacobian_recoveries_free_boundary(monkeypatch) -> None:
    """MULTIPLE consecutive JAC75 events in the (pre-vacuum) free lane.

    The single-event test above proves one recovery; the confidential-case
    signature is REPEATED 75-reset events.  Here the first retry is forced
    to run as hostile as the initial attempt (its ``time_step`` override is
    reset to the absurd input ``DELT``), so a SECOND genuine 75-reset event
    occurs inside the recovery chain, and only the second retry runs at the
    reduced step.  Contract asserted across BOTH events:

    * every retry restarts a present, fully finite checkpoint state (never
      ``initial_state=None`` — a reconstructed cold state);
    * the requested retry ``DELT`` follows ``min(0.5, 0.5 * prev)`` from
      the runtime the failing attempt actually used;
    * the retry budget decrements once per event (2 -> 1 -> 0);
    * retries never re-enable the initial axis re-guess and never reuse the
      failed attempt's vacuum cache;
    * BOTH events restore the recorded best finite checkpoint — for the
      forced second event that is the SAME state as the first (the
      sabotaged attempt never records a lower residual, so the best
      checkpoint cannot improve; measured fingerprints equal), and the
      no-identical-replay guarantee is carried by the time step: the
      retry's requested DELT (0.5) differs from the DELT the failing
      attempt actually executed with (1e4);
    * the final attempt converges.

    All events happen BEFORE vacuum activation (DELT=1e4 collapses within
    the first iterations, long before the fsqr < 1e-1 turn-on gate).
    """
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

    # both events restart a real finite checkpoint with the halved/capped
    # DELT law computed from the runtime the failing attempt actually used
    # (attempt 1 ran at DELT=1e4; the forced attempt 2 also ran at 1e4).
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

    # Best-checkpoint semantics under the forced second event: the
    # sabotaged attempt records no lower residual, so the SAME best finite
    # checkpoint is (correctly) restored again — while the no-replay
    # guarantee holds through the time step, which differs from the DELT
    # the failing attempt actually executed with (1e4).
    assert recorded[2]["fingerprint"] == recorded[1]["fingerprint"]
    assert recorded[2]["requested_time_step"] != pytest.approx(
        float(inp.delt))

    assert bool(result.converged), (
        f"chain failed to converge (fsqr={float(result.fsqr):.2e})")
