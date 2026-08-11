"""Public confidential-proxy stress case: genuinely huge initial force.

The confidential signature: initial ``FSQR ~ 1e8`` (public proxies start
~8.5e3), repeated JAC75 resets, ``FSQL`` pinned near 0.5, and a PARVMEC
run that completes without tolerance yet writes a WOUT (``LFULL3D1OUT=T``).
This module reproduces the class on a PUBLIC deck: the 238-mode boundary
``input.serial2500170_surface_points_mpol12_ntor12`` with major radius 3.5
(R0/|RBC(0,1)| ~ 24), no axis, LFORBAL, PRECON NONE, PREC2D 1e-30, indexed
sections, NS 21..144, LFULL3D1OUT.

Only two levers defeat VMEC's dimensionless force normalization (everything
that rescales the physics cancels in bcovar.f; measured row 1 stayed
~4.6e3 under pressure/PHIEDGE/CURTOR/scale/TCON0/elongation/axis attempts):
an extreme prescribed transform (``AI = 3, -300``) and an ``APHI`` flux
remap whose derivative reverses sign in s in (0.20, 0.69).  The APHI lever
is an INVALID flux parameterization (the s -> Phi map folds; the
profil3d.f-style interior guess assumes monotone flux, so no axis recovery
can remove the resulting force) and VMEX now refuses it with a typed
``VmecInputError`` at setup (``vmex.core.setup.validate_torflux_monotone``).
The lawful stress deck therefore uses the AI lever alone; measurement
(both codes print identical rows, ns=21 rung, 2026-08): row 1 =
``FSQR 1.17E+06, FSQZ 1.41E+04, FSQL 4.22``, WMHD 1.2277E+02, 14 resets in
40 iterations, FSQL pinned O(1) while FSQR stays above 1e3.

Historical cross-code findings, kept for the record: WITH the sign-
reversing APHI both codes reached row 1 ``FSQR 1.65E+07`` but the row-1
energy differed (VMEX WMHD 1.4762E+03 vs VMEC2000 1.4768E+03, ~4e-4 at the
bcovar-level field assembly near the phips zero crossing) and seeded a
one-iteration phase shift; on the FULL 5-rung ladder PARVMEC NaN-marches
at the interpolated ns=55 rung (jacobian.f's ``taumax*taumin < 0`` restart
test is NaN-blind) and still writes a WOUT with ``fsqr = NaN``
(``ier_flag = 16``, exit code 0).  ``test_full_ladder_termination_classes``
pins that divergence as OUR-typed-error vs THEIR-NaN-write-through; the
AI-only deck removed the last known bounded cross-code row divergence.
No free-boundary variant: no public mgrid exists for this boundary; the
free-side recovery contract lives in ``tests/test_freeboundary_stress.py``.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.force_oracle import ROW as V2K_ROW
from vmex.core.errors import (
    INPUT_ERROR_FLAG, MORE_ITER_FLAG, VmecInputError,
)
from vmex.core.fourier import mode_table
from vmex.core.input import VmecInput
from vmex.core.multigrid import solve_multigrid

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BOUNDARY = (
    ROOT / "examples" / "data"
    / "input.serial2500170_surface_points_mpol12_ntor12"
)

#: Measured first-rung row 1 of the (AI-lever) stress deck — VMEX and
#: xvmec2000 print identical rows.  The hard requirement is FSQR >= 1e6.
MEASURED_ROW1 = (1.17e6, 1.41e4, 4.22)

STRESS_NS = (21, 34, 55, 89, 144)
STRESS_NITER = 40


def _replace_assignment(text: str, name: str, replacement: str) -> str:
    result, count = re.subn(
        rf"(?im)^\s*{re.escape(name)}\s*=.*$", replacement, text, count=1,
    )
    assert count == 1, name
    return result


def stress_indata_text(*, single_stage: bool = False,
                       niter: int = STRESS_NITER,
                       reversed_flux: bool = False) -> str:
    """Build the public high-force stress deck from the tracked boundary.

    ``single_stage=True`` restricts the ladder to the first rung (ns=21) —
    the bounded variant used for the cross-code row comparison and the
    LFULL3D1OUT WOUT contract; the deck is otherwise identical.

    ``reversed_flux=True`` adds the historical sign-reversing flux remap
    ``APHI = 5, -16, 12`` (``Phi'(s) = 5 - 32 s + 36 s**2`` < 0 on s in
    (0.2023, 0.6866)) — an invalid parameterization VMEX rejects at setup;
    used only by the parse test and the termination-class divergence pins.
    """
    text = PUBLIC_BOUNDARY.read_text()
    text = _replace_assignment(text, "MPOL", "  MPOL = 13")
    text = _replace_assignment(text, "NTOR", "  NTOR = 9")
    if single_stage:
        ns_lines = "  NS_ARRAY = 21\n  NS_ARRAY(1) = 21"
        counts = "1"
    else:
        ns_lines = (
            "  NS_ARRAY = " + ", ".join(str(n) for n in STRESS_NS)
            + "\n  NS_ARRAY(1) = 21, 34"      # indexed no-op overlay
        )
        counts = str(len(STRESS_NS))
    text = _replace_assignment(text, "ns_array", ns_lines)
    text = _replace_assignment(
        text, "niter_array",
        "  NITER_ARRAY = " + ", ".join([str(niter)] * int(counts)),
    )
    text = _replace_assignment(
        text, "ftol_array",
        "  FTOL_ARRAY = " + ", ".join(["1.0e-13"] * int(counts)),
    )
    text = _replace_assignment(text, "NSTEP", "  NSTEP   = 1")
    text = _replace_assignment(text, "NCURR", "  NCURR   = 0")
    # major radius 3.5 (high aspect); the boundary shape is untouched
    text = re.sub(r"RBC\(\s*0,\s*0\)\s*=\s*[0-9.eE+-]+",
                  "RBC(   0,   0) =  3.5000000000000000e+00", text, count=1)
    aphi_line = (
        "  APHI(1) = 5.0, -16.0, 12.0\n"     # indexed start-element form
        if reversed_flux else ""
    )
    text = text.replace(
        "  NCURR   = 0",
        aphi_line
        + "  AI = 3.0, -300.0\n"
        "  LFORBAL = T\n"
        "  LFULL3D1OUT = T\n"
        "  PRECON_TYPE = 'NONE'\n"
        "  PREC2D_THRESHOLD = 1.0e-30\n"
        "  NCURR   = 0",
        1,
    )
    # Fortran indexed array sections; the boundary statements below
    # overwrite the touched values in source order (no-op overlay).
    text = text.replace(
        "  ! VMEC coefficient order is (n,m): RBC(n,m), ZBS(n,m).",
        "  ! VMEC coefficient order is (n,m): RBC(n,m), ZBS(n,m).\n"
        "  RBC(-6:6,0) = 13*0.0\n"
        "  ZBS(-6:6,0) = 13*0.0",
        1,
    )
    return text


def _v2k_first_stage_rows(text: str) -> dict[int, tuple[float, ...]]:
    """First-rung ``it -> (fsqr, fsqz, fsql, rax, delt, wmhd)`` rows."""
    rows: dict[int, tuple[float, ...]] = {}
    last = 0
    for m in V2K_ROW.finditer(text):
        it = int(m.group(1))
        if it < last:
            break
        last = it
        rows[it] = tuple(float(m.group(i)) for i in range(2, 8))
    return rows


def test_deck_features_parse() -> None:
    """Every required confidential-adjacent feature survives parsing.

    Parsing is deliberately lawful even for the invalid ``reversed_flux``
    variant: rejection is a SETUP-time typed error, so decks remain
    inspectable (``VmecInput.from_indata_text`` -> diagnosis tooling).
    """
    inp = VmecInput.from_indata_text(stress_indata_text(reversed_flux=True))
    assert mode_table(inp.mpol, inp.ntor).mnmax == 238
    np.testing.assert_array_equal(inp.ns_array, list(STRESS_NS))
    rbc = np.asarray(inp.rbc, dtype=float)
    assert rbc[inp.ntor, 0] == pytest.approx(3.5)             # major radius
    assert 3.5 / abs(rbc[inp.ntor, 1]) > 20.0                 # high aspect
    assert not np.any(inp.raxis_c) and not np.any(inp.zaxis_s)  # no axis
    assert bool(inp.lforbal) and bool(inp.lfull3d1out)
    assert inp.precon_type.strip().upper() == "NONE"
    assert float(inp.prec2d_threshold) == pytest.approx(1.0e-30)
    np.testing.assert_allclose(inp.aphi[:3], [5.0, -16.0, 12.0])
    np.testing.assert_allclose(inp.ai[:2], [3.0, -300.0])
    assert int(inp.ncurr) == 0
    # the APHI remap really reverses the flux derivative inside the plasma
    aphi = np.asarray(inp.aphi, dtype=float)
    s = np.linspace(0.0, 1.0, 201)
    dphi = sum((k + 1) * aphi[k] * s**k for k in range(3))
    assert dphi.min() < 0.0 < dphi.max()
    # the lawful stress deck carries the identity flux map (AI lever only)
    lawful = VmecInput.from_indata_text(stress_indata_text())
    np.testing.assert_array_equal(np.asarray(lawful.aphi)[:3], [1.0, 0.0, 0.0])


def test_reversed_flux_deck_raises_typed_input_error_at_setup() -> None:
    """The historical sign-reversing APHI deck now fails lawfully, up front.

    ``solve_multigrid`` must raise ``VmecInputError`` (``ier_flag = 5``)
    naming the exact reversal interval of ``Phi'(s) = 5 - 32 s + 36 s**2``
    before any iteration runs — not diverge, not NaN, not JAC75-storm.
    """
    inp = VmecInput.from_indata_text(
        stress_indata_text(single_stage=True, reversed_flux=True)
    )
    with pytest.raises(VmecInputError) as excinfo:
        solve_multigrid(inp, raise_on_max_iterations=False, device="cpu")
    err = excinfo.value
    assert int(err.ier_flag) == INPUT_ERROR_FLAG
    assert "non-monotonic toroidal-flux map" in err.message
    assert "(0.202283, 0.686605)" in err.message


@pytest.mark.full
def test_initial_force_exceeds_1e6_with_reset_storm_and_fsql_pinned() -> None:
    """Measured first-rung behavior: FSQR >= 1e6, reset storm, FSQL ~ O(1).

    The FSQL row-1 value and the O(1) pinning on successful-step rows are
    the public reproduction of the confidential ``FSQL ~ 0.5`` signature
    (AI-lever deck; both codes print identical rows, measured 2026-08).
    """
    inp = VmecInput.from_indata_text(stress_indata_text(single_stage=True))
    result = solve_multigrid(
        inp, raise_on_max_iterations=False, device="cpu",
    )
    rows = np.asarray(result.fsq_history, dtype=float)
    assert rows.shape[0] == STRESS_NITER
    fsqr1, fsqz1, fsql1 = rows[0, :3]
    assert fsqr1 >= 1.0e6, f"initial FSQR {fsqr1:.3e} fell below 1e6"
    assert fsqr1 == pytest.approx(MEASURED_ROW1[0], rel=1e-2)
    assert fsqz1 == pytest.approx(MEASURED_ROW1[1], rel=1e-2)
    assert fsql1 == pytest.approx(MEASURED_ROW1[2], rel=1e-2)
    # repeated Jacobian resets (the confidential JAC75-storm analogue)
    assert int(result.jacobian_resets) >= 10
    assert int(result.ier_flag) == MORE_ITER_FLAG
    # FSQL pinned at O(1): successful-step rows dip toward ~0.5 while the
    # R-channel stays large — lambda is stuck at its initial force level.
    fsqr, fsql = rows[:, 0], rows[:, 2]
    success = fsql < 2.0
    assert np.any(success), "no successful damped step in the first rung"
    assert fsqr[success].min() > 1.0e3
    assert fsql.max() < 10.0 and fsql[success].min() < 1.0


@pytest.mark.full
def test_wout_written_on_budget_exhaustion_with_lfull3d1out(
    tmp_path: Path,
) -> None:
    """LFULL3D1OUT=T: the CLI writes a WOUT on non-converged exit.

    VMEC2000 semantics: with the flag the run returns ``ier_flag = 2``
    (more_iter) AND leaves a WOUT; without it there is no WOUT.  VMEX's CLI
    implements the same policy (``raise_on_max_iterations`` gating in
    ``vmex/core/cli.py``); this pins it on the stress deck.
    """
    deck = tmp_path / "input.hf_stress_ns21"
    deck.write_text(stress_indata_text(single_stage=True, niter=8))
    env = dict(os.environ, JAX_ENABLE_X64="1")
    completed = subprocess.run(
        [sys.executable, "-m", "vmex.core.cli", str(deck),
         "--outdir", str(tmp_path), "--device", "cpu"],
        cwd=ROOT, capture_output=True, text=True, timeout=1800, env=env,
    )
    assert completed.returncode == MORE_ITER_FLAG, completed.stdout[-2000:]
    wout = tmp_path / "wout_hf_stress_ns21.nc"
    assert wout.exists(), "LFULL3D1OUT=T did not produce a WOUT"
    from vmex.core.wout import read_wout

    data = read_wout(wout)
    assert int(data.ier_flag) == MORE_ITER_FLAG
    assert "Wrote WOUT file" in completed.stdout


def _banded(a: float, b: float, band: float) -> bool:
    return a > 0 and b > 0 and abs(math.log10(a) - math.log10(b)) <= band


def _executable(pytestconfig) -> Path:
    configured = str(pytestconfig.getoption("--vmec2000-executable")).strip()
    candidates = [Path(configured) if configured else None]
    discovered = shutil.which("xvmec2000")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    pytest.fail("--run-vmec2000 requested but xvmec2000 was not found")


@pytest.mark.full
@pytest.mark.vmec2000_live
def test_fixed_termination_class_and_first_25_rows_match_vmec2000(
    pytestconfig, tmp_path: Path,
) -> None:
    """Acceptance on the bounded first-rung deck: both codes exhaust the
    budget without tolerance (neither converges nor aborts), VMEX writes a
    WOUT under ``LFULL3D1OUT=T``, and the first 25 rows agree channel-by-
    channel within a factor-2 log band allowing a +-1 iteration phase
    shift (the AI-lever deck measured digit-identical rows; the band keeps
    the comparison robust to platform jitter)."""
    deck_text = stress_indata_text(single_stage=True)
    v2k_dir = tmp_path / "vmec2000"
    vmex_dir = tmp_path / "vmex"
    v2k_dir.mkdir()
    vmex_dir.mkdir()
    (v2k_dir / "input.hf_stress_ns21").write_text(deck_text)
    (vmex_dir / "input.hf_stress_ns21").write_text(deck_text)

    ref = subprocess.run(
        [str(_executable(pytestconfig)), "input.hf_stress_ns21"],
        cwd=v2k_dir, capture_output=True, text=True, timeout=1800,
    )
    ref_text = ref.stdout + "\n" + ref.stderr
    env = dict(os.environ, JAX_ENABLE_X64="1")
    got = subprocess.run(
        [sys.executable, "-m", "vmex.core.cli",
         str(vmex_dir / "input.hf_stress_ns21"),
         "--outdir", str(vmex_dir), "--device", "cpu"],
        cwd=ROOT, capture_output=True, text=True, timeout=1800, env=env,
    )

    # -- termination class: budget exhausted WITHOUT tolerance, both codes --
    assert "Try increasing NITER" in ref_text, ref_text[-2000:]
    assert not re.search(r"^\s*\d+\s+\S*NaN", ref_text, re.M), (
        "VMEC2000 went non-finite on the bounded deck")
    assert got.returncode == MORE_ITER_FLAG, got.stdout[-2000:]
    # -- WOUT on non-converged exit (both sides carry LFULL3D1OUT=T) --------
    assert (vmex_dir / "wout_hf_stress_ns21.nc").exists()
    assert (v2k_dir / "wout_hf_stress_ns21.nc").exists()

    # -- per-iteration row agreement over the first 25 iterations -----------
    ref_rows = _v2k_first_stage_rows(ref_text)
    got_rows = _v2k_first_stage_rows(got.stdout)
    assert set(range(1, 26)) <= set(ref_rows), "VMEC2000 rows missing"
    assert set(range(1, 26)) <= set(got_rows), "VMEX rows missing"
    assert ref_rows[1][0] >= 1.0e6 and got_rows[1][0] >= 1.0e6
    band = math.log10(2.0)
    for it in range(1, 26):
        a = got_rows[it]
        shifted = [ref_rows.get(it + s) for s in (0, -1, 1)]
        ok = any(
            b is not None
            and all(_banded(a[c], b[c], band) for c in (0, 1, 2))
            for b in shifted
        )
        assert ok, (
            f"iteration {it}: VMEX row {a[:3]} outside the factor-2 band of "
            f"VMEC2000 rows {[b[:3] for b in shifted if b is not None]}"
        )


@pytest.mark.full
@pytest.mark.vmec2000_live
def test_full_ladder_termination_classes(pytestconfig, tmp_path: Path) -> None:
    """Sign-reversing-APHI ladder: OUR lawful typed error vs THEIR NaN
    write-through, deliberately pinned.

    PARVMEC accepts the invalid flux map, interpolates to ns=55 where the
    re-folded state has no valid axis, re-guesses, and NaN-marches through
    its full budget (``jacobian.f``'s ``taumax*taumin < 0`` restart test is
    NaN-blind and ``fsq < ftol`` never holds for NaN) to a WOUT-writing
    exit with code 0 — the WOUT carries ``fsqr = NaN`` (measured
    ``ier_flag = 16``, really_bad_flag).  VMEX raises ``VmecInputError``
    naming the reversal interval before any iteration."""
    deck_text = stress_indata_text(reversed_flux=True)
    v2k_dir = tmp_path / "vmec2000"
    v2k_dir.mkdir()
    (v2k_dir / "input.hf_stress").write_text(deck_text)
    ref = subprocess.run(
        [str(_executable(pytestconfig)), "input.hf_stress"],
        cwd=v2k_dir, capture_output=True, text=True, timeout=3600,
    )
    ref_text = ref.stdout + "\n" + ref.stderr
    # PARVMEC: NaN rows on the printed march, exit code 0, WOUT on disk.
    assert ref.returncode == 0
    assert re.search(r"^\s*\d+\s+.*NaN", ref_text, re.M), (
        "expected the measured PARVMEC NaN-march on the full ladder")
    assert (v2k_dir / "wout_hf_stress.nc").exists()

    inp = VmecInput.from_indata_text(deck_text)
    with pytest.raises(VmecInputError) as excinfo:
        solve_multigrid(inp, raise_on_max_iterations=False, device="cpu")
    assert "(0.202283, 0.686605)" in excinfo.value.message
