#!/usr/bin/env python3
"""CPU-vs-GPU benchmark matrix for vmex (§7.8).

Runs, one cell at a time (the machine may be shared):

  {decks + synthetic nfp4_QH size scan} x {JAX_PLATFORMS=cpu, cuda}
      x {legacy vj.run_fixed_boundary, core solver.solve(mode="jit")}

recording cold wall (first in-process solve, includes compile), warm wall
(second in-process solve, compile cache hot), compile-vs-run split
(cold - warm), per-iteration step time (warm / iterations), and peak device
memory (``jax.local_devices()[0].memory_stats()`` on cuda).

Plus two microbenchmarks of ``vmex.core.preconditioner.tridiagonal_solve``
(hypotheses c/d of §7.8.3): CPU-vs-GPU across (ns, ncols) at fp64, and
fp32-vs-fp64 on GPU.

Every cell is a fresh subprocess with ``JAX_PLATFORMS`` set, so device
selection and the compile cache are per-cell.

Usage (orchestrator):
    python benchmarks/run_gpu_matrix.py [--out benchmarks/gpu_baseline.json]
        [--only substr] [--timeout 1800] [--skip-tridiag]

Internal worker modes (spawned by the orchestrator):
    --worker solve  --deck PATH --lane {legacy,core_jit}
    --worker tridiag --dtype {f32,f64}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "examples" / "data"
MARKER = "RESULT_JSON:"

DECKS = [
    "solovev",
    "cth_like_fixed_bdy",
    "nfp4_QH_warm_start",
    "LandremanPaul2021_QA_lowres",
    "NuhrenbergZille_1988_QHS",
]

# Synthetic size scan: nfp4_QH_warm_start deck, NS_ARRAY in {35, 75, 151},
# modes as in the deck (mpol=2, ntor=2) and doubled (mpol=4, ntor=4).
SYNTH_NS = [35, 75, 151]
SYNTH_MODES = [(2, 2), (4, 4)]
SYNTH_NITER = 150  # fixed iteration budget -> exact per-iteration throughput

TRIDIAG_NS = [16, 35, 75, 151, 301, 601]
TRIDIAG_NCOLS = [30, 150, 600, 2400]


# --------------------------------------------------------------------------
# workers (run in a subprocess with JAX_PLATFORMS already set)
# --------------------------------------------------------------------------

def _device_mem_mb():
    import jax
    stats = jax.local_devices()[0].memory_stats() or {}
    peak = stats.get("peak_bytes_in_use")
    return round(peak / 2**20, 1) if peak else None


def _extract_iterations(result) -> int | None:
    inner = getattr(result, "result", None)  # legacy FixedBoundaryRun.result
    if inner is not None and inner is not result:
        v = _extract_iterations(inner)
        if v:
            return v
    for attr in ("iterations", "n_iter", "niter"):
        v = getattr(result, attr, None)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    diag = getattr(result, "diagnostics", None)
    if isinstance(diag, dict):
        for key in ("iterations", "n_iter", "niter"):
            v = diag.get(key)
            if isinstance(v, (int, float)) and v > 0:
                return int(v)
    return None


def worker_solve(deck: str, lane: str) -> dict:
    t_import0 = time.perf_counter()
    import jax
    out: dict = {
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
    }

    def one_solve():
        if lane == "legacy":
            import vmex as vj
            t0 = time.perf_counter()
            res = vj.run_fixed_boundary(deck, verbose=False)
            wall = time.perf_counter() - t0
            return wall, _extract_iterations(res), True
        elif lane == "core_jit":
            from vmex.core.input import VmecInput
            from vmex.core import solver
            from vmex.core.errors import VmecConvergenceError
            inp = VmecInput.from_file(deck)
            t0 = time.perf_counter()
            try:
                res = solver.solve(inp, mode="jit", verbose=False)
                wall = time.perf_counter() - t0
                return wall, int(res.iterations), True
            except VmecConvergenceError as e:
                wall = time.perf_counter() - t0
                return wall, int(getattr(e, "iteration", 0)) or None, False
        raise ValueError(lane)

    out["import_s"] = round(time.perf_counter() - t_import0, 3)
    cold_wall, iters1, conv1 = one_solve()
    warm_wall, iters2, conv2 = one_solve()
    iters = iters2 or iters1
    out.update({
        "cold_wall_s": round(cold_wall, 3),
        "warm_wall_s": round(warm_wall, 3),
        "compile_est_s": round(cold_wall - warm_wall, 3),
        "iterations": iters,
        "per_iter_ms": round(1e3 * warm_wall / iters, 3) if iters else None,
        "converged": bool(conv1 and conv2),
        "peak_device_mem_mb": _device_mem_mb(),
    })
    return out


def worker_stepscan(deck150: str, deck450: str, lane: str) -> dict:
    """True per-iteration step time via marginal iterations.

    ``solve()`` retraces/recompiles per call (per-solve closures), so a plain
    warm wall is trace+compile(-cache-load)+run.  Timing the same deck at
    NITER=150 and NITER=450 (FTOL=1e-30, never converges) and differencing
    isolates the pure iteration throughput:
        per_iter = (wall_450 - wall_150) / 300
        per_solve_overhead = wall_150 - 150 * per_iter
    """
    import jax

    def one(deck):
        if lane == "legacy":
            import vmex as vj
            t0 = time.perf_counter()
            try:
                vj.run_fixed_boundary(deck, verbose=False)
            except Exception:
                pass
            return time.perf_counter() - t0
        from vmex.core.input import VmecInput
        from vmex.core import solver
        inp = VmecInput.from_file(deck)
        t0 = time.perf_counter()
        try:
            solver.solve(inp, mode="jit", verbose=False)
        except Exception:
            pass
        return time.perf_counter() - t0

    one(deck150)              # cold warmup (compile both shapes not needed; 150 only)
    one(deck450)              # warm up the 450 shape too
    w150 = min(one(deck150) for _ in range(2))
    w450 = min(one(deck450) for _ in range(2))
    per_iter_ms = (w450 - w150) / 300.0 * 1e3
    return {"backend": jax.default_backend(),
            "wall_150_s": round(w150, 3), "wall_450_s": round(w450, 3),
            "per_iter_ms_marginal": round(per_iter_ms, 4),
            "per_solve_overhead_s": round(w150 - 0.150 * per_iter_ms, 3),
            "peak_device_mem_mb": _device_mem_mb()}


def worker_tridiag(dtype: str) -> dict:
    import numpy as np
    import jax
    import jax.numpy as jnp
    from vmex.core.preconditioner import tridiagonal_solve

    dt = jnp.float32 if dtype == "f32" else jnp.float64
    solve = jax.jit(tridiagonal_solve)
    rng = np.random.default_rng(0)
    rows = {}
    for ns in TRIDIAG_NS:
        for ncols in TRIDIAG_NCOLS:
            a = rng.standard_normal((ns, ncols))
            d = 4.0 + np.abs(rng.standard_normal((ns, ncols)))
            b = rng.standard_normal((ns, ncols))
            r = rng.standard_normal((ns, ncols))
            args = [jnp.asarray(x, dtype=dt) for x in (a, d, b, r)]
            solve(*args).block_until_ready()  # compile + warm
            reps = 50 if ns * ncols < 200_000 else 10
            best = min(
                _timed(lambda: solve(*args).block_until_ready())
                for _ in range(reps)
            )
            rows[f"ns={ns},ncols={ncols}"] = round(best * 1e3, 4)  # ms
    return {"backend": jax.default_backend(), "dtype": dtype,
            "best_ms": rows}


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------

def make_synth_deck(ns: int, mpol: int, ntor: int, dest_dir: Path,
                    niter: int = SYNTH_NITER) -> Path:
    text = (DATA / "input.nfp4_QH_warm_start").read_text()
    text = re.sub(r"NS_ARRAY\s*=\s*\d+", f"NS_ARRAY    = {ns}", text)
    text = re.sub(r"NITER_ARRAY\s*=\s*\d+", f"NITER_ARRAY = {niter}", text)
    text = re.sub(r"NITER\s*=\s*\d+", f"NITER = {niter}", text, count=1)
    text = re.sub(r"FTOL_ARRAY\s*=\s*\S+", "FTOL_ARRAY  = 1e-30", text)
    text = re.sub(r"MPOL\s*=\s*\d+", f"MPOL = {mpol:03d}", text)
    text = re.sub(r"NTOR\s*=\s*\d+", f"NTOR = {ntor:03d}", text)
    dest = dest_dir / f"input.synth_ns{ns}_m{mpol}n{ntor}_it{niter}"
    dest.write_text(text)
    return dest


def run_cell(platform: str, worker_args: list[str], timeout: int,
             cwd: Path | None = None) -> dict:
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = platform
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    cmd = [sys.executable, str(Path(__file__).resolve())] + worker_args
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env, cwd=cwd)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "subprocess_wall_s": timeout}
    wall = time.perf_counter() - t0
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(MARKER):
            out = json.loads(line[len(MARKER):])
            out["ok"] = proc.returncode == 0
            out["subprocess_wall_s"] = round(wall, 3)
            return out
    return {"ok": False, "error": (proc.stderr or proc.stdout)[-2000:],
            "subprocess_wall_s": round(wall, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["solve", "tridiag", "stepscan"])
    ap.add_argument("--deck")
    ap.add_argument("--deck150")
    ap.add_argument("--deck450")
    ap.add_argument("--lane", choices=["legacy", "core_jit"])
    ap.add_argument("--dtype", choices=["f32", "f64"], default="f64")
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "gpu_baseline.json"))
    ap.add_argument("--only", default=None, help="substring filter on case names")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--skip-tridiag", action="store_true")
    ap.add_argument("--skip-matrix", action="store_true")
    args = ap.parse_args()

    if args.worker == "solve":
        print(MARKER + json.dumps(worker_solve(args.deck, args.lane)))
        return
    if args.worker == "tridiag":
        print(MARKER + json.dumps(worker_tridiag(args.dtype)))
        return
    if args.worker == "stepscan":
        print(MARKER + json.dumps(
            worker_stepscan(args.deck150, args.deck450, args.lane)))
        return

    synth_dir = Path(tempfile.mkdtemp(prefix="vmecjax_synth_"))
    cases: list[tuple[str, Path]] = [(n, DATA / f"input.{n}") for n in DECKS]
    for ns in SYNTH_NS:
        for mpol, ntor in SYNTH_MODES:
            cases.append((f"synth_nfp4QH_ns{ns}_mpol{mpol}_ntor{ntor}",
                          make_synth_deck(ns, mpol, ntor, synth_dir)))

    out_path = Path(args.out)
    if out_path.exists():
        results = json.loads(out_path.read_text())  # merge into previous runs
        results.setdefault("matrix", {})
        results.setdefault("tridiag", {})
    else:
        results = {"matrix": {}, "tridiag": {}}
    results["meta"] = {"host": os.uname().nodename,
                       "date": time.strftime("%Y-%m-%d %H:%M"),
                       "synth_niter": SYNTH_NITER}
    for name, deck in cases:
        if args.skip_matrix:
            break
        if args.only and args.only not in name:
            continue
        results["matrix"][name] = {}
        for platform in ("cpu", "cuda"):
            for lane in ("legacy", "core_jit"):
                key = f"{platform}/{lane}"
                print(f"=== {name} [{key}] ===", flush=True)
                r = run_cell(platform,
                             ["--worker", "solve", "--deck", str(deck),
                              "--lane", lane], args.timeout)
                results["matrix"][name][key] = r
                print(f"    cold={r.get('cold_wall_s')} warm={r.get('warm_wall_s')}"
                      f" it={r.get('iterations')} per_it_ms={r.get('per_iter_ms')}"
                      f" ok={r.get('ok')}", flush=True)
                Path(args.out).write_text(json.dumps(results, indent=1))

    # True per-iteration step time (marginal NITER=150 vs 450) on the
    # synthetic size scan — solve() retraces per call, so plain warm walls
    # overstate iteration cost; see worker_stepscan.
    results.setdefault("stepscan", {})
    if not args.only:
        for ns in SYNTH_NS:
            for mpol, ntor in SYNTH_MODES:
                d150 = make_synth_deck(ns, mpol, ntor, synth_dir, niter=150)
                d450 = make_synth_deck(ns, mpol, ntor, synth_dir, niter=450)
                name = f"ns{ns}_mpol{mpol}_ntor{ntor}"
                results["stepscan"][name] = {}
                for platform in ("cpu", "cuda"):
                    for lane in ("legacy", "core_jit"):
                        key = f"{platform}/{lane}"
                        print(f"=== stepscan {name} [{key}] ===", flush=True)
                        r = run_cell(platform,
                                     ["--worker", "stepscan",
                                      "--deck150", str(d150),
                                      "--deck450", str(d450),
                                      "--lane", lane], args.timeout)
                        results["stepscan"][name][key] = r
                        print(f"    per_iter_ms={r.get('per_iter_ms_marginal')}"
                              f" overhead_s={r.get('per_solve_overhead_s')}"
                              f" ok={r.get('ok')}", flush=True)
                        Path(args.out).write_text(json.dumps(results, indent=1))

    if not args.skip_tridiag and not args.only:
        for platform, dtype in [("cpu", "f64"), ("cuda", "f64"), ("cuda", "f32")]:
            key = f"{platform}/{dtype}"
            print(f"=== tridiag microbench [{key}] ===", flush=True)
            results["tridiag"][key] = run_cell(
                platform, ["--worker", "tridiag", "--dtype", dtype],
                args.timeout)
            Path(args.out).write_text(json.dumps(results, indent=1))

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
