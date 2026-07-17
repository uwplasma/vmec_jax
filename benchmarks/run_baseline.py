#!/usr/bin/env python3
"""Baseline benchmark harness: VMEC2000 vs vmex vs VMEC++.

Runs a fixed suite of input decks (from ``examples/data``) through each
available solver and records wall time, peak RSS, iteration counts, and
convergence status into ``benchmarks/baseline.json``.  This is the single
source for the README performance figure; re-run it after performance work
(overhaul plan, Phases 0 and 10).

Usage:
    python benchmarks/run_baseline.py [--out baseline.json] [--cases a,b,...]
        [--timeout 600] [--skip vmecpp]

Solvers exercised per case:
- ``vmec2000``       — /Users/rogerio/local/STELLOPT/VMEC2000/Release/xvmec2000
- ``vmex_cold``  — fresh ``vmec`` CLI subprocess (includes JAX/XLA setup)
- ``vmex_warm``  — second in-process core solve (structural executable
  cache hot; same ``vmex.core`` route as the CLI)
- ``vmecpp``         — VMEC++ python API, where the case converges cleanly

Every benchmark row runs at ``ns >= RAMP_NS`` (201): decks whose finest
NS_ARRAY stage is below that get a generated variant with the final stage
rewritten to 201 (``grid: ns201``); decks already at or above 201 run as-is
(``grid: input``).  ns=201 is chosen so the *solve* dominates the wall-clock
and the one-time JIT compile is a small fraction of it — a fairer warm
comparison than ns=51, where compile time is a larger share.  FTOL_ARRAY is
left untouched (each deck's own final-ftol semantics are kept); the final
iteration cap is raised to >= 10000 when ramping, since a cap tuned for the
deck's native coarse grid would turn ns=201 into a spurious non-convergence
for every code.  Cases listed in MULTIGRID_CASES
additionally run with a generated coarse->fine NS_ARRAY ladder ending at
``max(deck_ns, 201)`` (``grid: multigrid``) to compare multigrid behavior.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "examples" / "data"
XVMEC2000 = Path("/Users/rogerio/local/STELLOPT/VMEC2000/Release/xvmec2000")
VMECJAX_PY = Path.home() / ".venvs/vmecjax/bin/python"
VMECPP_PY = Path.home() / ".venvs/vmecpp/bin/python"

# (case name, aux files to copy alongside the deck)
CASES: dict[str, list[str]] = {
    "solovev": [],
    "DSHAPE": [],
    "circular_tokamak": [],
    "cth_like_fixed_bdy": [],
    "li383_low_res": [],
    "LandremanPaul2021_QA_lowres": [],
    "LandremanPaul2021_QH_reactorScale_lowres": [],
    "nfp4_QH_warm_start": [],
    "NuhrenbergZille_1988_QHS": [],
    # Free boundary with the real mgrid (R15.1: now converges to VMEC2000
    # parity).  Ramped to ns >= RAMP_NS like every other row.
    "cth_like_free_bdy": ["mgrid_cth_like.nc"],
    "cth_like_free_bdy_lasym_small": ["mgrid_cth_like_lasym_small.nc"],
}

# Minimum final-stage ns for every benchmark row (see module docstring).
# 201 so the solve dominates the wall-clock and JIT compile is a small share.
RAMP_NS = 201

# Cases that additionally run with a generated coarse->fine NS_ARRAY ladder.
# Ladder ends at max(deck ns, RAMP_NS) so results stay comparable.
MULTIGRID_CASES = {
    "cth_like_fixed_bdy": "15 31 51 101 201",
    "nfp4_QH_warm_start": "17 35 71 101 201",
    "LandremanPaul2021_QA_lowres": None,  # ladder derived from deck ns below
}

TIME_RE = re.compile(r"(\d+\.\d+)\s+real.*?(\d+)\s+maximum resident set size", re.S)

# Active (non-comment) NS_ARRAY assignment; `!`-commented lines don't match.
NS_ARRAY_RE = re.compile(r"^(\s*NS_ARRAY\s*=\s*)([\d,\s]+)", re.I | re.M)


def read_deck_ns(deck: Path) -> int:
    m = NS_ARRAY_RE.search(deck.read_text())
    return int(m.group(2).replace(",", " ").split()[-1]) if m else 0


def make_ramped_deck(deck: Path, dest: Path, min_ns: int = RAMP_NS) -> None:
    """Rewrite the final NS_ARRAY stage to max(deck_ns, min_ns).

    FTOL_ARRAY is left untouched (same stage count) so the deck's own
    final-ftol semantics are preserved.  The final NITER(_ARRAY) stage is
    raised to at least 10000 when ramping: iteration counts grow with ns,
    and a cap tuned for the deck's native coarse grid (e.g. solovev's
    NITER=500 at ns=11) would otherwise turn the ns=201 row into a
    spurious non-convergence for every code.
    """
    text = deck.read_text()

    def repl(m: re.Match) -> str:
        stages = m.group(2).replace(",", " ").split()
        stages[-1] = str(max(int(stages[-1]), min_ns))
        return m.group(1) + " ".join(stages) + "\n "

    text = NS_ARRAY_RE.sub(repl, text, count=1)

    def repl_niter(m: re.Match) -> str:
        head, vals = m.group(0).split("=", 1)
        stages = vals.replace(",", " ").split()
        stages[-1] = str(max(int(stages[-1]), 10000))
        return head + "= " + " ".join(stages) + "\n "

    # When both are present, the per-stage NITER_ARRAY governs and bare NITER
    # is ignored — several bundled decks (circular_tokamak, QA/QH lowres) put
    # bare NITER first, so "first match wins" would rewrite the ineffective
    # line.  Prefer NITER_ARRAY; fall back to bare NITER.
    text, n = re.subn(r"^\s*NITER_ARRAY\s*=\s*[\d,\s]+", repl_niter, text,
                      count=1, flags=re.I | re.M)
    if n == 0:
        text = re.sub(r"^\s*NITER\s*=\s*[\d,\s]+", repl_niter, text,
                      count=1, flags=re.I | re.M)
    dest.write_text(text)


def make_multigrid_deck(deck: Path, ladder: str | None, dest: Path) -> None:
    """Rewrite NS_ARRAY/FTOL_ARRAY/NITER_ARRAY as a coarse->fine ladder."""
    ns_final = max(read_deck_ns(deck), RAMP_NS)
    if ladder is None:
        third = max(5, ns_final // 4) | 1
        half = max(third + 2, ns_final // 2) | 1
        ladder = f"{third} {half} {ns_final}"
    stages = ladder.split()
    text = deck.read_text()
    text = NS_ARRAY_RE.sub(lambda m: f"{m.group(1)}{' '.join(stages)}\n ", text, count=1)
    # Active-line anchored like NS_ARRAY_RE (a `!`-commented line must not
    # match); prefer NITER_ARRAY over bare NITER for the same reason as in
    # make_ramped_deck.
    text = re.sub(r"^\s*FTOL_ARRAY\s*=\s*[\deE.+\- \t,]+",
                  f"FTOL_ARRAY = {' '.join(['1e-8'] * (len(stages) - 1))} 1e-14\n ",
                  text, count=1, flags=re.I | re.M)
    # Final stage gets the same >=10000 budget as make_ramped_deck: at the
    # 1e-14 final ftol the ns=201 stage can legitimately need >4000 sweeps
    # (measured: QA_lowres multigrid raised VmecConvergenceError at 4000).
    niter_line = ("NITER_ARRAY = "
                  + " ".join(["4000"] * (len(stages) - 1)) + " 10000\n ")
    text, n = re.subn(r"^\s*NITER_ARRAY\s*=\s*[\d,\s]+", niter_line, text,
                      count=1, flags=re.I | re.M)
    if n == 0:
        text = re.sub(r"^\s*NITER\s*=\s*[\d,\s]+", niter_line, text,
                      count=1, flags=re.I | re.M)
    dest.write_text(text)


def timed_subprocess(cmd: list[str], cwd: Path, timeout: int) -> dict:
    """Run under /usr/bin/time -l, return wall seconds, peak RSS, output."""
    full = ["/usr/bin/time", "-l"] + cmd
    t0 = time.time()
    try:
        proc = subprocess.run(full, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "wall_s": timeout}
    wall = time.time() - t0
    m = TIME_RE.search(proc.stderr)
    return {
        "ok": proc.returncode == 0,
        "wall_s": round(float(m.group(1)), 3) if m else round(wall, 3),
        "peak_rss_mb": round(int(m.group(2)) / 2**20, 1) if m else None,
        "stdout": proc.stdout[-4000:],
        "stderr": "" if proc.returncode == 0 else proc.stderr[-2000:],
    }


def parse_vmec2000(out: dict) -> dict:
    txt = out.pop("stdout", "")
    out["converged"] = "EXECUTION TERMINATED NORMALLY" in txt
    iters = re.findall(r"^\s*(\d+)\s+[\d.E+-]+\s+[\d.E+-]+", txt, re.M)
    out["iterations"] = int(iters[-1]) if iters else None
    return out


def parse_vmecjax(out: dict) -> dict:
    txt = out.pop("stdout", "")
    out["converged"] = out["ok"] and ("Wrote WOUT" in txt or "wout" in txt.lower())
    iters = re.findall(r"^\s*(\d+)\s+[\d.E+-]+\s+[\d.E+-]+", txt, re.M)
    out["iterations"] = int(iters[-1]) if iters else None
    return out


def run_vmec2000(deck: Path, aux: list[str], timeout: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        shutil.copy(deck, wd)
        for a in aux:
            shutil.copy(DATA / a, wd)
        return parse_vmec2000(timed_subprocess([str(XVMEC2000), deck.name], wd, timeout))


def run_vmecjax_cold(deck: Path, aux: list[str], timeout: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        shutil.copy(deck, wd)
        for a in aux:
            shutil.copy(DATA / a, wd)
        vmec = VMECJAX_PY.parent / "vmec"
        return parse_vmecjax(timed_subprocess([str(vmec), deck.name], wd, timeout))


WARM_SNIPPET = r"""
import json, sys, time, resource
import numpy as np
import vmex  # sets the persistent XLA compilation cache dir
from vmex.core.input import VmecInput
from vmex.core.multigrid import solve_multigrid

def run(path):
    # Same route as the ``vmec`` CLI (core/cli.py): fixed-boundary decks run
    # the full NS_ARRAY ladder; free-boundary decks run the final stage.
    inp = VmecInput.from_file(path)
    if bool(getattr(inp, "lfreeb", False)):
        from vmex.core.freeboundary import solve_free_boundary
        from vmex.core.solver import resolution_from_input

        ns = int(np.atleast_1d(np.asarray(inp.ns_array))[-1])
        return solve_free_boundary(
            inp, resolution=resolution_from_input(inp, ns=ns),
            error_on_no_convergence=False,
        )
    return solve_multigrid(inp)

path = sys.argv[1]
t0 = time.time(); run(path); t1 = time.time()
t2 = time.time(); run(path); t3 = time.time()
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
print(json.dumps({"cold_inproc_s": round(t1-t0,3), "warm_s": round(t3-t2,3),
                  "peak_rss_mb": round(rss,1)}))
"""


def run_vmecjax_warm(deck: Path, aux: list[str], timeout: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        shutil.copy(deck, wd)
        for a in aux:
            shutil.copy(DATA / a, wd)
        try:
            proc = subprocess.run([str(VMECJAX_PY), "-c", WARM_SNIPPET, deck.name],
                                  cwd=wd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout"}
        if proc.returncode != 0:
            return {"ok": False, "error": proc.stderr[-1500:]}
        line = proc.stdout.strip().splitlines()[-1]
        return {"ok": True, "converged": True, **json.loads(line)}


VMECPP_SNIPPET = r"""
import json, sys, time, resource
import vmecpp
inp = vmecpp.VmecInput.from_file(sys.argv[1])
t0 = time.time()
out = vmecpp.run(inp)
wall = time.time() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
print(json.dumps({"wall_s": round(wall,3), "peak_rss_mb": round(rss,1),
                  "iterations": int(out.wout.niter), "converged": True}))
"""


def run_vmecpp(deck: Path, aux: list[str], timeout: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        shutil.copy(deck, wd)
        for a in aux:
            shutil.copy(DATA / a, wd)
        try:
            proc = subprocess.run([str(VMECPP_PY), "-c", VMECPP_SNIPPET, deck.name],
                                  cwd=wd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout"}
        if proc.returncode != 0:
            return {"ok": False, "converged": False, "error": proc.stderr[-800:]}
        return {"ok": True, **json.loads(proc.stdout.strip().splitlines()[-1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "baseline.json"))
    ap.add_argument("--cases", default=None, help="comma-separated subset")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--skip", default="", help="comma-separated solvers to skip")
    args = ap.parse_args()

    cases = args.cases.split(",") if args.cases else list(CASES)
    skip = set(args.skip.split(",")) if args.skip else set()
    # Merge semantics: --cases reruns update their rows in the existing file
    # instead of clobbering the other cases' results.
    out_path = Path(args.out)
    results: dict[str, dict] = (
        json.loads(out_path.read_text()) if out_path.exists() else {})

    for name in cases:
        aux = CASES.get(name, [])
        src = DATA / f"input.{name}"
        if read_deck_ns(src) >= RAMP_NS:
            variants = [("input", src)]
        else:
            ramped = Path(tempfile.mkdtemp()) / f"input.{name}"
            make_ramped_deck(src, ramped)
            variants = [(f"ns{RAMP_NS}", ramped)]
        if name in MULTIGRID_CASES:
            mg = Path(tempfile.mkdtemp()) / f"input.{name}"
            make_multigrid_deck(src, MULTIGRID_CASES[name], mg)
            variants.append(("multigrid", mg))
        for grid, deck in variants:
            key = f"{name}[{grid}]"
            row: dict[str, dict] = {"ns": read_deck_ns(deck)}
            print(f"=== {key} (ns={row['ns']}) ===", flush=True)
            for solver, fn in [("vmec2000", run_vmec2000),
                               ("vmex_cold", run_vmecjax_cold),
                               ("vmex_warm", run_vmecjax_warm),
                               ("vmecpp", run_vmecpp)]:
                if solver in skip:
                    continue
                if solver in ("vmex_warm", "vmecpp") and name.endswith("free_bdy_lasym_small"):
                    if solver == "vmecpp":
                        row[solver] = {"ok": False, "error": "lasym unsupported"}
                        continue
                r = fn(deck, aux, args.timeout)
                row[solver] = r
                print(f"  {solver:15s} wall={r.get('wall_s', r.get('warm_s'))} ok={r.get('ok')}", flush=True)
            results[key] = row

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
