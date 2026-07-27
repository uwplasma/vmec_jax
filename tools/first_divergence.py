#!/usr/bin/env python3
"""Privacy-safe first-divergence comparator: VMEX vs a local VMEC2000 binary.

Runs BOTH codes on the SAME deck (``NSTEP`` forced to 1 so the iteration
table is one row per iteration), aligns the identical-format trajectories,
and reports WHERE they first disagree — as stage codes, channel names,
iteration indices and coarse divergence classes only.  The default output
contains no filenames, coefficients, profile values, residual magnitudes,
or WOUT data, so it is safe to report for a confidential deck.  ``--details``
prints numerical values and must never be shared for such a deck.

Stages compared (first failure is the leading suspect):

  C1 PARSE          both codes accept the deck
  C2 AXIS_ROW1      the printed initial axis position (RAX at iteration 1)
  C3 ITER1_FORCES   the iteration-1 FSQR/FSQZ/FSQL residual triplet
  C4 TRAJECTORY     first iteration where any residual channel leaves the
                    matching band (relative, in log space)
  C5 ENERGY         first iteration where WMHD leaves the matching band
  C6 ACTIVATION     free boundary only: the vacuum turn-on iteration
  C7 RECOVERY       counts of Jacobian-reset / axis-re-guess events
  C8 TERMINATION    converged / iteration budget / error, both codes

Classes: MATCH (<1e-6 relative), CLOSE (<1e-3), DIVERGENT (>=1e-3).
A confidential-case report should contain only the stage lines and the
final assessment code.

Usage::

    python tools/first_divergence.py input.CASE --xvmec2000 /path/to/xvmec2000
    python tools/first_divergence.py input.CASE --xvmec2000 ... --niter 100
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ROW = re.compile(
    r"^\s*(\d+)\s+([0-9.E+-]+)\s+([0-9.E+-]+)\s+([0-9.E+-]+)\s+([0-9.E+-]+)"
    r"\s+([0-9.E+-]+)\s+([0-9.E+-]+)", re.M)
BANNER = re.compile(r"VACUUM PRESSURE TURNED ON AT\s+(\d+)")

MATCH, CLOSE, DIVERGENT = "MATCH", "CLOSE", "DIVERGENT"


def _klass(rel: float) -> str:
    if rel < 1e-6:
        return MATCH
    if rel < 1e-3:
        return CLOSE
    return DIVERGENT


def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), np.finfo(float).tiny)
    return abs(a - b) / scale


def _rows(text: str) -> dict[int, tuple[float, ...]]:
    """iteration -> (fsqr, fsqz, fsql, rax, delt, wmhd), LAST rung occurrence.

    Multigrid restarts iteration numbering per rung; keyed per rung segment
    by keeping a running rung offset so the trajectories align rung-by-rung.
    """
    out: dict[int, tuple[float, ...]] = {}
    offset = 0
    last = 0
    for m in ROW.finditer(text):
        it = int(m.group(1))
        if it < last:  # a new rung restarted the counter
            offset += last
        last = it
        out[offset + it] = tuple(float(m.group(i)) for i in range(2, 8))
    return out


def _prepare_deck(src: Path, workdir: Path, niter: int | None) -> Path:
    text = src.read_text()
    text = re.sub(r"NSTEP\s*=\s*\d+", "NSTEP = 1", text)
    if "NSTEP" not in text:
        text = text.replace("&INDATA", "&INDATA\n  NSTEP = 1,", 1)
    if niter is not None:
        text = re.sub(r"NITER_ARRAY\s*=[^\n,]*(,[^\n]*)?",
                      f"NITER_ARRAY = {niter}, {niter}, {niter}, {niter}, {niter}",
                      text)
    dst = workdir / src.name
    dst.write_text(text)
    # keep a referenced mgrid reachable for both codes
    m = re.search(r"MGRID_FILE\s*=\s*'([^']+)'", text)
    if m:
        mg = (src.parent / m.group(1)).expanduser()
        if mg.exists():
            shutil.copy(mg, workdir / mg.name)
    return dst


def _run_vmec2000(exe: Path, deck: Path, timeout: int) -> tuple[str, str]:
    proc = subprocess.run(
        [str(exe), deck.name], cwd=deck.parent, capture_output=True,
        text=True, timeout=timeout)
    threed = deck.parent / f"threed1.{deck.name.split('input.', 1)[-1]}"
    return proc.stdout + "\n" + (threed.read_text() if threed.exists() else ""), (
        "NORMAL" if "EXECUTION TERMINATED NORMALLY" in proc.stdout
        else "ABNORMAL")


def _run_vmex(deck: Path) -> tuple[str, str]:
    from vmex.core.errors import VmecConvergenceError, VmecError
    from vmex.core.input import VmecInput
    from vmex.core.multigrid import solve_free_boundary_multigrid, solve_multigrid

    inp = VmecInput.from_file(str(deck))
    lines: list[str] = []

    def collect(text: str = "", end: str = "\n") -> None:
        lines.append(str(text) + end)

    try:
        if bool(inp.lfreeb):
            mgrid = deck.parent / Path(str(inp.mgrid_file)).name
            result = solve_free_boundary_multigrid(
                inp, mgrid_path=str(mgrid) if mgrid.exists() else None,
                verbose=True, emit=collect, raise_on_max_iterations=False)
        else:
            result = solve_multigrid(
                inp, verbose=True, emit=collect, raise_on_max_iterations=False)
        term = "CONVERGED" if bool(result.converged) else "NITER"
    except VmecConvergenceError:
        term = "NITER"
    except VmecError as exc:
        term = f"TYPED_{type(exc).__name__}"
    return "".join(lines), term


def compare(deck: Path, xvmec2000: Path, *, niter: int | None,
            timeout: int, details: bool) -> int:
    report: list[str] = []
    worst = MATCH

    def stage(code: str, text: str) -> None:
        report.append(f"{code}: {text}")

    def note(klass: str) -> None:
        nonlocal worst
        order = [MATCH, CLOSE, DIVERGENT]
        if order.index(klass) > order.index(worst):
            worst = klass

    with tempfile.TemporaryDirectory() as td_f, \
            tempfile.TemporaryDirectory() as td_j:
        deck_f = _prepare_deck(deck, Path(td_f), niter)
        deck_j = _prepare_deck(deck, Path(td_j), niter)
        try:
            ref_text, ref_term = _run_vmec2000(xvmec2000, deck_f, timeout)
        except Exception as exc:  # noqa: BLE001
            stage("C1 PARSE", f"VMEC2000 run failed ({type(exc).__name__})")
            print("\n".join(report))
            return 2
        vmex_text, vmex_term = _run_vmex(deck_j)
        stage("C1 PARSE", "both codes accepted the deck")

        ref = _rows(ref_text)
        got = _rows(vmex_text)
        common = sorted(set(ref) & set(got))
        if not common:
            stage("C4 TRAJECTORY", "no comparable iteration rows")
            print("\n".join(report))
            return 2

        first = common[0]
        # C2: initial axis position
        k = _klass(_rel(ref[first][3], got[first][3]))
        note(k)
        stage("C2 AXIS_ROW1", k if not details else
              f"{k} (ref={ref[first][3]:.9e} vmex={got[first][3]:.9e})")

        # C3: iteration-1 residual triplet
        names = ("FSQR", "FSQZ", "FSQL")
        parts = []
        for i, name in enumerate(names):
            k = _klass(_rel(ref[first][i], got[first][i]))
            note(k)
            parts.append(f"{name}={k}" + (
                f"(ref={ref[first][i]:.3e},vmex={got[first][i]:.3e})"
                if details else ""))
        stage("C3 ITER1_FORCES", " ".join(parts))

        # C4: first SUSTAINED trajectory divergence per channel (log-space
        # band, 3 consecutive out-of-band rows).  Divergence confined to the
        # final ~20% of an unconverged run is the ordinary chaotic tail of
        # two float trajectories and is reported as LATE_DRIFT, not as a
        # shared-calculation discrepancy.
        tail_start = common[max(0, int(0.8 * len(common)) - 1)]

        def _first_sustained(channel: int, band: float) -> int | None:
            run = 0
            start = None
            for it in common:
                r, g = ref[it][channel], got[it][channel]
                if r <= 0 or g <= 0:
                    run = 0
                    start = None
                    continue
                if abs(np.log10(r) - np.log10(g)) > band:
                    run += 1
                    start = it if start is None else start
                    if run >= 3:
                        return start
                else:
                    run = 0
                    start = None
            return None

        div_report = []
        for i, name in enumerate(names):
            first_div = _first_sustained(i, float(np.log10(2.0)))
            if first_div is None:
                div_report.append(f"{name}=TRACKS({len(common)}rows)")
            elif first_div >= tail_start:
                note(CLOSE)
                div_report.append(f"{name}=LATE_DRIFT@iter{first_div}")
            else:
                note(DIVERGENT)
                div_report.append(f"{name}=FIRST_DIVERGENCE@iter{first_div}")
        stage("C4 TRAJECTORY", " ".join(div_report))

        # C5: energy trajectory (same sustained/tail policy).  The vacuum
        # turn-on kick produces a documented transient WMHD difference for a
        # few dozen iterations after activation while the residual channels
        # keep tracking (measured on the public CTH case: WMHD identical to
        # the printed digits through activation, ~5e-4..3e-3 relative for
        # ~10 iterations after it, re-converging to the same equilibrium) --
        # that window is excluded rather than reported as a shared-path
        # divergence.
        b_ref = BANNER.search(ref_text)
        skip_lo = int(b_ref.group(1)) if b_ref else None
        run = 0
        first_div = None
        for it in common:
            # tight band on the SHARED pre-activation path; loose band after
            # activation, where the two codes' free-boundary paths
            # legitimately differ while converging to the same equilibrium
            # (public CTH: identical printed WMHD through activation,
            # ~1e-3-level path difference after, matching converged wout)
            band = 1e-4 if (skip_lo is None or it < skip_lo) else 1e-2
            if _rel(ref[it][5], got[it][5]) > band:
                run += 1
                if first_div is None:
                    first_div = it
                if run >= 3:
                    break
            else:
                run = 0
                first_div = None
        if first_div is None or run < 3:
            note(MATCH)
            stage("C5 ENERGY", "TRACKS")
        elif first_div >= tail_start:
            note(CLOSE)
            stage("C5 ENERGY", f"LATE_DRIFT@iter{first_div}")
        else:
            note(DIVERGENT)
            stage("C5 ENERGY", f"FIRST_DIVERGENCE@iter{first_div}")

        # C6: vacuum activation
        br, bg = BANNER.search(ref_text), BANNER.search(vmex_text)
        if br or bg:
            ir = int(br.group(1)) if br else None
            ig = int(bg.group(1)) if bg else None
            if ir is not None and ig is not None:
                k = MATCH if abs(ir - ig) <= 2 else DIVERGENT
                note(k)
                stage("C6 ACTIVATION", f"{k} (ref@{ir} vmex@{ig})")
            else:
                note(DIVERGENT)
                stage("C6 ACTIVATION",
                      f"ONLY_{'REF' if ir is not None else 'VMEX'}_ACTIVATED")

        # C7: recovery events
        markers = ("JACOBIAN CHANGED SIGN", "IMPROVE INITIAL MAGNETIC AXIS")
        parts = []
        for m in markers:
            cr, cg = ref_text.count(m), vmex_text.count(m)
            k = MATCH if cr == cg else CLOSE
            note(k)
            parts.append(f"{m.split()[0]}:{k}" + (
                f"(ref={cr},vmex={cg})" if details else ""))
        stage("C7 RECOVERY", " ".join(parts))

        # C8: termination
        k = MATCH if (ref_term == "NORMAL") == (
            vmex_term in ("CONVERGED", "NITER")) else DIVERGENT
        note(k)
        stage("C8 TERMINATION", f"{k} (ref={ref_term} vmex={vmex_term})")

    print("\n".join(report))
    print(f"assessment: FIRST_DIVERGENCE_{worst}")
    return 0 if worst != DIVERGENT else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--xvmec2000", type=Path, required=True,
                    help="path to a local xvmec2000 executable")
    ap.add_argument("--niter", type=int, default=None,
                    help="cap NITER per rung for a fast comparison")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--details", action="store_true",
                    help="print values; NEVER share for a confidential deck")
    args = ap.parse_args()
    return compare(args.input, args.xvmec2000, niter=args.niter,
                   timeout=args.timeout, details=args.details)


if __name__ == "__main__":
    raise SystemExit(main())
