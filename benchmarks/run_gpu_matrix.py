#!/usr/bin/env python3
"""CPU-vs-GPU benchmark matrix for vmex (§7.8).

Runs, one cell at a time (the machine may be shared):

  {decks + synthetic nfp4_QH size scan} x {device=cpu, device=gpu}
      x {solve_multigrid, solver.solve(mode="jit")}

recording cold wall (first in-process solve, includes compile), warm wall
(second in-process solve, compile cache hot), compile-vs-run split
(cold - warm), per-iteration step time (warm / iterations), and peak device
memory (``jax.local_devices()[0].memory_stats()`` on GPU).

Plus two microbenchmarks of ``vmex.core.preconditioner.tridiagonal_solve``
(hypotheses c/d of §7.8.3): CPU-vs-GPU across (ns, ncols) at fp64, and
fp32-vs-fp64 on GPU.

Every cell is a fresh subprocess and selects hardware through VMEX's public
``device=`` API.  No JAX platform environment variable is required.

Usage (orchestrator):
    python benchmarks/run_gpu_matrix.py [--out benchmarks/gpu_baseline.json]
        [--only substr] [--timeout 1800] [--skip-tridiag]

Office decision sweep (one command; produces every CPU-vs-GPU crossover
curve: fixed-boundary per-iteration marginals on the ns x mnmax grid, the
free-boundary NS sweep, the >512-mode FFT probe, and the fixed/free/gradient
workflow profiles via ``profile_resources.py``):

    python benchmarks/run_gpu_matrix.py --office \
        --out benchmarks/gpu_office.json

Repeat with ``--xla-flags "--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL"``
and a different ``--out`` for the CUDA-graph A/B; the flags are recorded in
``meta`` and applied to every child process.

Internal worker modes (spawned by the orchestrator):
    --worker solve  --deck PATH --lane {multigrid,single_cli,single_jit}
        --device {cpu,gpu}
    --worker stepscan --deck150 PATH --deck450 PATH --lane ... --device ...
    --worker tridiag --dtype {f32,f64} --device {cpu,gpu}
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

# --office decision grid: production radial range x mnmax 8..288 (the
# device-policy calibration range), production CLI lane included, plus a
# free-boundary NS sweep (vacuum steady lane / NESTOR cost per iteration)
# and one >512-mode probe where the GPU default switches to the separable
# FFT synthesis (device policy currently routes that regime to CPU).
OFFICE_NS = [51, 101, 201]
OFFICE_MODES = [(2, 2), (8, 8), (12, 12)]     # mnmax 8, 128, 288
OFFICE_LANES = ("multigrid", "single_cli")
OFFICE_FREE_NS = [15, 25, 51]
OFFICE_HIGH_MODE = (51, 19, 14)               # mnmax 537 > GPU_MAX_SPECTRAL_MODES

TRIDIAG_NS = [16, 35, 75, 151, 301, 601]
TRIDIAG_NCOLS = [30, 150, 600, 2400]


# --------------------------------------------------------------------------
# workers (each subprocess receives an explicit device selector)
# --------------------------------------------------------------------------

def _device_mem_mb():
    import jax
    stats = jax.local_devices()[0].memory_stats() or {}
    peak = stats.get("peak_bytes_in_use")
    return round(peak / 2**20, 1) if peak else None


def _extract_iterations(result) -> int | None:
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


def worker_solve(deck: str, lane: str, device: str) -> dict:
    t_import0 = time.perf_counter()
    import jax
    out: dict = {
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
    }

    def one_solve():
        if lane == "multigrid":
            import vmex as vj
            from vmex.core.input import VmecInput
            inp = VmecInput.from_file(deck)
            t0 = time.perf_counter()
            res = vj.solve_multigrid(inp, verbose=False, device=device)
            wall = time.perf_counter() - t0
            return wall, _extract_iterations(res), True
        elif lane in ("single_jit", "single_cli"):
            from vmex.core.input import VmecInput
            from vmex.core import solver
            from vmex.core.errors import VmecConvergenceError
            inp = VmecInput.from_file(deck)
            mode = "jit" if lane == "single_jit" else "cli"
            t0 = time.perf_counter()
            try:
                res = solver.solve(inp, mode=mode, verbose=False, device=device)
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


def worker_stepscan(deck150: str, deck450: str, lane: str, device: str) -> dict:
    """True per-iteration step time via marginal iterations.

    ``solve()`` retraces/recompiles per call (per-solve closures), so a plain
    warm wall is trace+compile(-cache-load)+run.  Timing the same deck at
    NITER=150 and NITER=450 (FTOL=1e-30, never converges) and differencing
    isolates the pure iteration throughput:
        per_iter = (wall_450 - wall_150) / 300
        per_solve_overhead = wall_150 - 150 * per_iter
    """
    import jax

    failures = 0

    def one(deck):
        nonlocal failures
        if lane == "multigrid":
            import vmex as vj
            from vmex.core.input import VmecInput
            from vmex.core.errors import VmecConvergenceError
            inp = VmecInput.from_file(deck)
            t0 = time.perf_counter()
            try:
                vj.solve_multigrid(inp, verbose=False, device=device)
            except VmecConvergenceError:
                pass                # FTOL=1e-30 never converges, by design
            except Exception:
                failures += 1
            return time.perf_counter() - t0
        from vmex.core.input import VmecInput
        from vmex.core import solver
        from vmex.core.errors import VmecConvergenceError
        inp = VmecInput.from_file(deck)
        mode = "cli" if lane == "single_cli" else "jit"
        t0 = time.perf_counter()
        try:
            solver.solve(inp, mode=mode, verbose=False, device=device)
        except VmecConvergenceError:
            pass                    # FTOL=1e-30 never converges, by design
        except Exception:
            failures += 1
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
            "unexpected_failures": failures,
            "peak_device_mem_mb": _device_mem_mb()}


def worker_tridiag(dtype: str, device: str) -> dict:
    import numpy as np
    import jax
    import jax.numpy as jnp
    from vmex.core.preconditioner import tridiagonal_solve

    dt = jnp.float32 if dtype == "f32" else jnp.float64
    target = jax.devices(device)[0]
    solve = jax.jit(tridiagonal_solve, device=target)
    rng = np.random.default_rng(0)
    rows = {}
    for ns in TRIDIAG_NS:
        for ncols in TRIDIAG_NCOLS:
            a = rng.standard_normal((ns, ncols))
            d = 4.0 + np.abs(rng.standard_normal((ns, ncols)))
            b = rng.standard_normal((ns, ncols))
            r = rng.standard_normal((ns, ncols))
            args = [jax.device_put(jnp.asarray(x, dtype=dt), target) for x in (a, d, b, r)]
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


def make_synth_free_deck(ns: int, dest_dir: Path, niter: int) -> Path:
    """Single-stage cth-like free-boundary deck at NS_ARRAY = ns.

    FTOL 1e-30 never converges, so the marginal NITER=150/450 difference
    times the steady vacuum lane (NESTOR cadence included).  Children must
    run with ``cwd`` = ``examples/data`` so the deck's relative
    ``MGRID_FILE`` resolves.
    """
    text = (DATA / "input.cth_like_free_bdy").read_text()
    text = re.sub(r"NS_ARRAY\s*=\s*\d+", f"NS_ARRAY    = {ns}", text)
    text = re.sub(r"NITER_ARRAY\s*=\s*\d+", f"NITER_ARRAY = {niter}", text)
    text = re.sub(r"FTOL_ARRAY\s*=\s*[0-9.eEdD+-]+", "FTOL_ARRAY  = 1e-30", text)
    dest = dest_dir / f"input.synthfree_ns{ns}_it{niter}"
    dest.write_text(text)
    return dest


_CHILD_ENV: dict | None = None   #: orchestrator-set extra env (XLA_FLAGS)


def _git_commit() -> str | None:
    """Provenance: the measured checkout's commit (+ ``-dirty`` marker)."""
    try:
        head = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short=9", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
        if head and subprocess.run(
            ["git", "-C", str(REPO), "diff", "--quiet", "HEAD"],
            capture_output=True, timeout=10,
        ).returncode:
            head += "-dirty"
        return head
    except Exception:
        return None


def run_cell(device: str, worker_args: list[str], timeout: int,
             cwd: Path | None = None) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--device", device] + worker_args
    env = {**os.environ, **_CHILD_ENV} if _CHILD_ENV else None
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, cwd=cwd, env=env)
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
    ap.add_argument("--lane", choices=["multigrid", "single_cli", "single_jit"])
    ap.add_argument("--device", choices=["cpu", "gpu"])
    ap.add_argument("--dtype", choices=["f32", "f64"], default="f64")
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "gpu_baseline.json"))
    ap.add_argument("--only", default=None, help="substring filter on case names")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--skip-tridiag", action="store_true")
    ap.add_argument("--skip-matrix", action="store_true")
    ap.add_argument("--office", action="store_true",
                    help="one-command decision sweep: ns x mnmax stepscan grid "
                         "(production CLI lane), free-boundary NS sweep, "
                         ">512-mode probe, and profile_resources workflows")
    ap.add_argument("--xla-flags", default=None,
                    help="XLA_FLAGS for every child (e.g. the command-buffer "
                         "A/B); recorded in meta")
    args = ap.parse_args()

    global _CHILD_ENV
    if args.xla_flags:
        _CHILD_ENV = {"XLA_FLAGS": args.xla_flags}

    if args.worker == "solve":
        print(MARKER + json.dumps(worker_solve(args.deck, args.lane, args.device)))
        return
    if args.worker == "tridiag":
        print(MARKER + json.dumps(worker_tridiag(args.dtype, args.device)))
        return
    if args.worker == "stepscan":
        print(MARKER + json.dumps(
            worker_stepscan(args.deck150, args.deck450, args.lane, args.device)))
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
                       "commit": _git_commit(),
                       "synth_niter": SYNTH_NITER,
                       "xla_flags": args.xla_flags or os.environ.get("XLA_FLAGS"),
                       "office": bool(args.office)}
    for name, deck in cases:
        if args.skip_matrix or args.office:
            break
        if args.only and args.only not in name:
            continue
        results["matrix"][name] = {}
        for device in ("cpu", "gpu"):
            for lane in ("multigrid", "single_jit"):
                key = f"{device}/{lane}"
                print(f"=== {name} [{key}] ===", flush=True)
                r = run_cell(device,
                             ["--worker", "solve", "--deck", str(deck),
                              "--lane", lane], args.timeout)
                results["matrix"][name][key] = r
                print(f"    cold={r.get('cold_wall_s')} warm={r.get('warm_wall_s')}"
                      f" it={r.get('iterations')} per_it_ms={r.get('per_iter_ms')}"
                      f" ok={r.get('ok')}", flush=True)
                Path(args.out).write_text(json.dumps(results, indent=1))

    def stepscan_cell(name, d150, d450, device, lane, cwd=None):
        key = f"{device}/{lane}"
        print(f"=== stepscan {name} [{key}] ===", flush=True)
        r = run_cell(device,
                     ["--worker", "stepscan",
                      "--deck150", str(d150), "--deck450", str(d450),
                      "--lane", lane], args.timeout, cwd=cwd)
        results["stepscan"].setdefault(name, {})[key] = r
        print(f"    per_iter_ms={r.get('per_iter_ms_marginal')}"
              f" overhead_s={r.get('per_solve_overhead_s')}"
              f" ok={r.get('ok')}", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=1))

    # True per-iteration step time (marginal NITER=150 vs 450) on the
    # synthetic size scan — solve() retraces per call, so plain warm walls
    # overstate iteration cost; see worker_stepscan.
    results.setdefault("stepscan", {})
    if not args.only:
        ns_grid = OFFICE_NS if args.office else SYNTH_NS
        mode_grid = OFFICE_MODES if args.office else SYNTH_MODES
        lanes = OFFICE_LANES if args.office else ("multigrid", "single_jit")
        for ns in ns_grid:
            for mpol, ntor in mode_grid:
                d150 = make_synth_deck(ns, mpol, ntor, synth_dir, niter=150)
                d450 = make_synth_deck(ns, mpol, ntor, synth_dir, niter=450)
                for device in ("cpu", "gpu"):
                    for lane in lanes:
                        stepscan_cell(f"ns{ns}_mpol{mpol}_ntor{ntor}",
                                      d150, d450, device, lane)

    if args.office and not args.only:
        # Free-boundary NS sweep: steady vacuum-lane iteration cost
        # (children run from DATA so the relative MGRID_FILE resolves).
        for ns in OFFICE_FREE_NS:
            d150 = make_synth_free_deck(ns, synth_dir, niter=150)
            d450 = make_synth_free_deck(ns, synth_dir, niter=450)
            for device in ("cpu", "gpu"):
                stepscan_cell(f"free_ns{ns}", d150, d450, device,
                              "multigrid", cwd=DATA)
        # >512-mode probe: GPU default switches to FFT synthesis here and
        # the auto device policy routes this regime to CPU — measure both.
        ns, mpol, ntor = OFFICE_HIGH_MODE
        d150 = make_synth_deck(ns, mpol, ntor, synth_dir, niter=150)
        d450 = make_synth_deck(ns, mpol, ntor, synth_dir, niter=450)
        for device in ("cpu", "gpu"):
            stepscan_cell(f"highmode_ns{ns}_mpol{mpol}_ntor{ntor}",
                          d150, d450, device, "single_cli")
        # Workflow profiles (fixed/free/gradient, cold+warm+memory) via the
        # resource harness, one JSON per device, embedded under "resources".
        results.setdefault("resources", {})
        for device in ("cpu", "gpu"):
            dest = Path(synth_dir) / f"resources_{device}.json"
            cmd = [sys.executable,
                   str(REPO / "benchmarks" / "profile_resources.py"),
                   "--device", device, "--cases", "fixed,free,implicit",
                   "--out", str(dest)]
            print(f"=== profile_resources [{device}] ===", flush=True)
            env = {**os.environ, **_CHILD_ENV} if _CHILD_ENV else None
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=4 * args.timeout, env=env)
            if dest.exists():
                results["resources"][device] = json.loads(dest.read_text())
            else:
                results["resources"][device] = {
                    "ok": False, "error": (proc.stderr or proc.stdout)[-2000:]}
            Path(args.out).write_text(json.dumps(results, indent=1))

    if not args.skip_tridiag and not args.only and not args.office:
        for device, dtype in [("cpu", "f64"), ("gpu", "f64"), ("gpu", "f32")]:
            key = f"{device}/{dtype}"
            print(f"=== tridiag microbench [{key}] ===", flush=True)
            results["tridiag"][key] = run_cell(
                device, ["--worker", "tridiag", "--dtype", dtype],
                args.timeout)
            Path(args.out).write_text(json.dumps(results, indent=1))

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
