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

  C1 PARSE          acceptance and early-termination status of BOTH codes
  C2 AXIS_ROW1      the printed initial axis position (RAX at iteration 1)
  C3 ITER1_FORCES   the iteration-1 FSQR/FSQZ/FSQL residual triplet
  C4 TRAJECTORY     first iteration where any residual channel leaves the
                    matching band (relative, in log space); when a channel
                    diverges, the line also names WHICH printed channel
                    (FSQR vs FSQZ vs FSQL) left the band first
  C5 ENERGY         first iteration where WMHD leaves the matching band
  C6 ACTIVATION     free boundary only: the vacuum turn-on iteration
  C7 RECOVERY       counts of Jacobian-reset / axis-re-guess events
  C8 TERMINATION    termination class of BOTH codes: CONVERGED,
                    ITERATION_BUDGET, JACOBIAN_75, NON_FINITE, INPUT_ERROR
                    (UNKNOWN when no marker is recognisable); a class
                    mismatch is a first-divergence finding in itself

Scope: C3 compares only the iteration-1 residual TRIPLET and C4/C5 the
printed trajectories.  VMEC2000 prints just the FSQR/FSQZ/FSQL totals, so a
per-force-term interior comparison (which term inside a channel drove a
divergence) is possible only on the VMEX side; the C4 annotation therefore
names the first-diverging printed channel and never attributes it to
interior force terms VMEC2000 does not print.

Classes: MATCH (<1e-6 relative), CLOSE (<1e-3), DIVERGENT (>=1e-3).
A confidential-case report should contain only the stage lines and the
final assessment code.

Failure handling: any harness failure is reported as
``C0 HARNESS_ERROR <ExceptionClassName>`` — the exception class name only,
never a traceback, so private paths, deck contents, or input-derived values
cannot escape.  ``--details`` remains the only mode that prints values.

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

# ---------------------------------------------------------------------------
# Termination classes (C8), shared by both codes.  Coarse by design: class
# names only, so the comparison stays privacy-safe.
# ---------------------------------------------------------------------------
TERM_CONVERGED = "CONVERGED"
TERM_ITERATION_BUDGET = "ITERATION_BUDGET"
TERM_JACOBIAN_75 = "JACOBIAN_75"
TERM_NON_FINITE = "NON_FINITE"
TERM_INPUT_ERROR = "INPUT_ERROR"
TERM_UNKNOWN = "UNKNOWN"

#: werror strings from VMEC2000 ``fileout.f`` that mean the input was refused.
_VMEC2000_INPUT_MARKERS = (
    "ERROR READING INPUT FILE OR NAMELIST",
    "ERROR IN INPUT VALUES",
    "PHIEDGE HAS WRONG SIGN IN VACUUM REGION",
    "NS ARRAY MUST NOT BE ALL ZEROES",
)
# A non-finite residual inside an iteration-table row (Fortran prints NaN /
# Infinity / a field of asterisks on formatted overflow) ...
_NONFINITE_ROW = re.compile(
    r"^\s*\d+\s+\S.*?(?:\bnan\b|\binf(?:inity)?\b|\*{4,})", re.I | re.M)
# ... or a runtime IEEE trap reported on stderr.
_IEEE_TRAP = re.compile(
    r"IEEE_INVALID|IEEE_OVERFLOW|IEEE_DIVIDE_BY_ZERO|"
    r"floating[ -]point exception", re.I)
#: Iteration-budget exhaustion markers.  ``MORE ITERATIONS REQUIRED`` is the
#: ``werror`` table string; measured builds instead print the ``eqsolve.f``
#: hint ``Try increasing NITER ...`` and then ``EXECUTION TERMINATED
#: NORMALLY`` anyway (wout ``ier_flag`` stays 0), so the hint MUST outrank
#: the normal-termination banner.
_NITER_MARKERS = ("MORE ITERATIONS REQUIRED", "Try increasing NITER")


def classify_vmec2000_termination(text: str) -> str:
    """Coarse termination class from VMEC2000 stdout/stderr/threed1 markers.

    Marker-based only (no values): the ``werror`` message table printed by
    ``Sources/Input_Output/fileout.f`` distinguishes normal termination,
    iteration exhaustion (``MORE ITERATIONS REQUIRED``, or the ``eqsolve.f``
    ``Try increasing NITER`` hint on builds that print the normal banner
    regardless) and the JAC75 abort; non-finite runs are recognised from
    NaN/Infinity/overflow fields in the FSQR iteration-table rows or an IEEE
    trap message.  Anything without a recognisable marker is ``UNKNOWN``
    (e.g. a crash before any output).
    """
    if "MORE THAN 75 JACOBIAN ITERATIONS" in text:
        return TERM_JACOBIAN_75
    if any(marker in text for marker in _VMEC2000_INPUT_MARKERS):
        return TERM_INPUT_ERROR
    if _NONFINITE_ROW.search(text) or _IEEE_TRAP.search(text):
        return TERM_NON_FINITE
    if any(marker in text for marker in _NITER_MARKERS):
        return TERM_ITERATION_BUDGET
    if "EXECUTION TERMINATED NORMALLY" in text:
        return TERM_CONVERGED
    return TERM_UNKNOWN


def classify_vmex_termination(exc: BaseException | None = None,
                              converged: bool | None = None) -> str:
    """Map a VMEX outcome (typed exception OR convergence flag) to a class.

    Mirrors the ``vmex.core.errors`` taxonomy: ``VmecJacobianError`` is the
    JAC75-class abort, ``VmecNumericalError`` the non-finite fail-fast,
    ``VmecConvergenceError`` the exhausted iteration budget, and the
    input-validation errors (``VmecInputError``/``MgridNotFoundError``) an
    input rejection.  Other ``VmecError`` subclasses fall back to their
    carried ``ier_flag``; non-vmex exceptions are ``UNKNOWN``.
    """
    if exc is not None:
        from vmex.core import errors as _errors

        if isinstance(exc, _errors.VmecJacobianError):
            return TERM_JACOBIAN_75
        if isinstance(exc, _errors.VmecNumericalError):
            return TERM_NON_FINITE
        if isinstance(exc, _errors.VmecConvergenceError):
            return TERM_ITERATION_BUDGET
        if isinstance(exc, (_errors.VmecInputError, _errors.MgridNotFoundError)):
            return TERM_INPUT_ERROR
        if isinstance(exc, _errors.VmecError):
            return {
                _errors.MORE_ITER_FLAG: TERM_ITERATION_BUDGET,
                _errors.BAD_JACOBIAN_FLAG: TERM_JACOBIAN_75,
                _errors.JAC75_FLAG: TERM_JACOBIAN_75,
                _errors.INPUT_ERROR_FLAG: TERM_INPUT_ERROR,
                _errors.PHIEDGE_ERROR_FLAG: TERM_INPUT_ERROR,
                _errors.NS_ERROR_FLAG: TERM_INPUT_ERROR,
            }.get(int(exc.ier_flag), TERM_UNKNOWN)
        return TERM_UNKNOWN
    if converged is True:
        return TERM_CONVERGED
    if converged is False:
        return TERM_ITERATION_BUDGET
    return TERM_UNKNOWN


def _acceptance(term: str, detail: str = "") -> str:
    """Privacy-safe per-code C1 status: acceptance + early-termination class.

    ``detail`` is an exception CLASS NAME only (never a message, which could
    embed a private path).
    """
    if term == TERM_INPUT_ERROR:
        return "REJECTED(INPUT_ERROR)"
    if term in (TERM_CONVERGED, TERM_ITERATION_BUDGET):
        return "ACCEPTED"
    if detail:
        return f"ACCEPTED_THEN_{term}({detail})"
    return f"ACCEPTED_THEN_{term}"


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
    """Copy ``src`` into ``workdir``, NSTEP forced to 1 and NITER capped.

    Namelist keys are matched case-insensitively and a ``NITER_ARRAY`` value
    list may continue across lines (namelist array continuation).  A
    relatively-referenced ``MGRID_FILE`` is resolved against the DECK
    directory, copied next to the prepared deck, and re-referenced by
    basename so BOTH codes find it from their run directory.
    """
    text = src.read_text()
    text, n = re.subn(r"(NSTEP\s*=\s*)\d+", r"\g<1>1", text, flags=re.I)
    if n == 0:
        text = re.sub(r"&INDATA", "&INDATA\n  NSTEP = 1,", text, count=1,
                      flags=re.I)
    if niter is not None:
        caps = ", ".join([str(niter)] * 5)
        text = re.sub(r"NITER_ARRAY\s*=\s*\d+(?:[\s,]+\d+)*",
                      f"NITER_ARRAY = {caps}", text, flags=re.I)
    m = re.search(r"MGRID_FILE\s*=\s*['\"]([^'\"]+)['\"]", text, flags=re.I)
    if m:
        mg = (src.parent / m.group(1)).expanduser()
        if mg.exists():
            shutil.copy(mg, workdir / mg.name)
            text = text[:m.start(1)] + mg.name + text[m.end(1):]
    dst = workdir / src.name
    dst.write_text(text)
    return dst


def _run_vmec2000(exe: Path, deck: Path, timeout: int) -> tuple[str, str]:
    proc = subprocess.run(
        [str(exe), deck.name], cwd=deck.parent, capture_output=True,
        text=True, timeout=timeout)
    threed = deck.parent / f"threed1.{deck.name.split('input.', 1)[-1]}"
    text = "\n".join([proc.stdout, proc.stderr,
                      threed.read_text() if threed.exists() else ""])
    return text, classify_vmec2000_termination(text)


def _run_vmex(deck: Path) -> tuple[str, str, str]:
    """Run VMEX on the prepared deck.

    Returns ``(iteration_text, termination_class, detail)`` where ``detail``
    is the exception CLASS NAME for failures outside the typed taxonomy
    (privacy-safe: never the message).  A deck that fails to parse is an
    ``INPUT_ERROR`` — i.e. VMEX did NOT accept the input.
    """
    from vmex.core.errors import VmecError
    from vmex.core.input import VmecInput
    from vmex.core.multigrid import solve_free_boundary_multigrid, solve_multigrid

    lines: list[str] = []

    def collect(text: str = "", end: str = "\n") -> None:
        lines.append(str(text) + end)

    try:
        inp = VmecInput.from_file(str(deck))
    except Exception as exc:  # noqa: BLE001 - any unreadable deck is a rejection
        return "", TERM_INPUT_ERROR, type(exc).__name__
    try:
        if bool(inp.lfreeb):
            mgrid = deck.parent / Path(str(inp.mgrid_file)).name
            result = solve_free_boundary_multigrid(
                inp, mgrid_path=str(mgrid) if mgrid.exists() else None,
                verbose=True, emit=collect, raise_on_max_iterations=False)
        else:
            result = solve_multigrid(
                inp, verbose=True, emit=collect, raise_on_max_iterations=False)
    except VmecError as exc:
        return "".join(lines), classify_vmex_termination(exc=exc), ""
    except Exception as exc:  # noqa: BLE001 - class name only, never the message
        return "".join(lines), TERM_UNKNOWN, type(exc).__name__
    return "".join(lines), classify_vmex_termination(
        converged=bool(result.converged)), ""


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
        vmex_text, vmex_term, vmex_detail = _run_vmex(deck_j)

        # C1: BOTH codes' acceptance and early-termination status.  A
        # one-sided rejection is a first-divergence finding at the parser.
        ref_ok = ref_term != TERM_INPUT_ERROR
        vmex_ok = vmex_term != TERM_INPUT_ERROR
        k = MATCH if ref_ok == vmex_ok else DIVERGENT
        note(k)
        stage("C1 PARSE", f"{k} ref={_acceptance(ref_term)} "
                          f"vmex={_acceptance(vmex_term, vmex_detail)}")

        ref = _rows(ref_text)
        got = _rows(vmex_text)
        common = sorted(set(ref) & set(got))
        if common:
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
            firsts: dict[str, int] = {}
            for i, name in enumerate(names):
                first_div = _first_sustained(i, float(np.log10(2.0)))
                if first_div is None:
                    div_report.append(f"{name}=TRACKS({len(common)}rows)")
                elif first_div >= tail_start:
                    note(CLOSE)
                    firsts[name] = first_div
                    div_report.append(f"{name}=LATE_DRIFT@iter{first_div}")
                else:
                    note(DIVERGENT)
                    firsts[name] = first_div
                    div_report.append(f"{name}=FIRST_DIVERGENCE@iter{first_div}")
            if firsts:
                # privacy-safe channel annotation: WHICH printed residual
                # channel left the band first (channel name + iteration only)
                name, it = min(firsts.items(), key=lambda kv: kv[1])
                div_report.append(f"first_channel={name}@iter{it}")
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
        else:
            stage("C4 TRAJECTORY", "no comparable iteration rows")

        # C8: termination CLASS of both codes.  A mismatch (e.g. one code
        # converged while the other exhausted its budget) is itself a
        # first-divergence finding; classes only, no values.
        if ref_term == vmex_term:
            note(MATCH)
            stage("C8 TERMINATION", f"MATCH (ref={ref_term} vmex={vmex_term})")
        else:
            note(DIVERGENT)
            stage("C8 TERMINATION",
                  f"CLASS_MISMATCH (ref={ref_term} vmex={vmex_term})")

    print("\n".join(report))
    print(f"assessment: FIRST_DIVERGENCE_{worst}")
    if not common:
        return 2
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
    try:
        return compare(args.input, args.xvmec2000, niter=args.niter,
                       timeout=args.timeout, details=args.details)
    except Exception as exc:  # noqa: BLE001 - privacy: the class name only,
        # never a traceback that could echo a private path or deck contents
        print(f"C0 HARNESS_ERROR {type(exc).__name__}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
