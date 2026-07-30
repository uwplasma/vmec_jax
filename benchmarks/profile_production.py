#!/usr/bin/env python
"""Production-run profiling: compile/warm/per-iteration/memory per use case.

Profiles the five production workflows end-to-end on an explicitly selected
JAX device, using VMEX's public ``device=`` API rather than platform
environment variables:

1. ``fixed_ns201``      — single-grid ns=201 fixed-boundary solve (li383)
2. ``multigrid_ns201``  — coarse->fine ladder 51/101/201 (li383)
3. ``free_boundary``    — NESTOR free boundary (cth_like_free_bdy, deck ns)
4. ``implicit_grad``    — value_and_grad of aspect via the implicit adjoint
5. ``opt_step``         — two least_squares iterations, jac="implicit"

For each: cold wall (first call, includes XLA compile), warm wall (second
call, compiled), iterations, per-iteration ms, and peak RSS.  On GPU also
reports per-device peak bytes from ``jax.local_devices()[i].memory_stats()``.

Usage::

    python benchmarks/profile_production.py --device cpu [--out profile.json]

Wall numbers on a shared machine are indicative; the compile/warm SPLIT and
the per-iteration number are the actionable quantities.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data"
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

import jax  # noqa: E402

import vmex as vj  # noqa: E402


def _peak_rss_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1e9 if platform.system() == "Darwin" else ru * 1024 / 1e9


def _gpu_stats() -> dict:
    out = {}
    for d in jax.local_devices():
        stats = getattr(d, "memory_stats", lambda: None)()
        if stats:
            out[str(d)] = {k: v for k, v in stats.items()
                           if k in ("peak_bytes_in_use", "bytes_in_use", "bytes_limit")}
    return out


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def profile_fixed(device, ns: int = 201):
    inp = vj.VmecInput.from_file(DATA / "input.li383_low_res")
    inp = dataclasses.replace(inp, ns_array=[ns], ftol_array=[1e-11],
                              niter_array=[10000])
    r, cold = _timed(lambda: vj.solve_multigrid(inp, verbose=False, device=device))
    r2, warm = _timed(lambda: vj.solve_multigrid(inp, verbose=False, device=device))
    iters = int(r2.iterations)
    return {"cold_s": cold, "warm_s": warm, "iters": iters,
            "ms_per_iter": 1e3 * warm / max(iters, 1),
            "converged": bool(r2.converged)}


def profile_multigrid(device, ns: int = 201):
    inp = vj.VmecInput.from_file(DATA / "input.li383_low_res")
    inp = dataclasses.replace(inp, ns_array=[51, 101, ns],
                              ftol_array=[1e-8, 1e-8, 1e-11],
                              niter_array=[4000, 4000, 10000])
    r, cold = _timed(lambda: vj.solve_multigrid(inp, verbose=False, device=device))
    r2, warm = _timed(lambda: vj.solve_multigrid(inp, verbose=False, device=device))
    return {"cold_s": cold, "warm_s": warm, "iters": int(r2.iterations),
            "converged": bool(r2.converged)}


def profile_free_boundary(device):
    inp = vj.VmecInput.from_file(DATA / "input.cth_like_free_bdy")
    mg = DATA / "mgrid_cth_like.nc"
    r, cold = _timed(lambda: vj.solve_free_boundary(
        inp, mgrid_path=mg, error_on_no_convergence=False, device=device))
    r2, warm = _timed(lambda: vj.solve_free_boundary(
        inp, mgrid_path=mg, error_on_no_convergence=False, device=device))
    iters = int(r2.iterations)
    return {"cold_s": cold, "warm_s": warm, "iters": iters,
            "ms_per_iter": 1e3 * warm / max(iters, 1),
            "converged": bool(r2.converged)}


def profile_implicit_grad(device):
    from vmex.core import implicit as im
    inp = vj.VmecInput.from_file(DATA / "input.solovev")
    p0 = im.params_from_input(inp, device=device)

    def obj(p):
        return im.run(inp, p, ftol=1e-12, max_iterations=5000,
                      device=device).aspect

    (v, g), cold = _timed(lambda: jax.value_and_grad(obj)(p0))
    _ = float(v), float(np.asarray(g.phiedge))
    (v2, g2), warm = _timed(lambda: jax.value_and_grad(obj)(p0))
    _ = float(v2), float(np.asarray(g2.phiedge))
    return {"cold_s": cold, "warm_s": warm}


def profile_opt_step(device):
    from vmex import optimize as opt
    inp = vj.VmecInput.from_file(DATA / "input.minimal_seed_nfp2")
    qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), 1, 0)
    terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, 6.0, 1.0)]
    res, cold = _timed(lambda: opt.least_squares(
        terms, inp, max_mode=1, jac="implicit", use_ess=True, max_nfev=2,
        ftol=1e-30, xtol=1e-30, device=device))
    res2, warm = _timed(lambda: opt.least_squares(
        terms, inp, max_mode=1, jac="implicit", use_ess=True, max_nfev=2,
        ftol=1e-30, xtol=1e-30, device=device))
    return {"cold_s": cold, "warm_s": warm}


CASES = {
    "fixed_ns201": profile_fixed,
    "multigrid_ns201": profile_multigrid,
    "free_boundary": profile_free_boundary,
    "implicit_grad": profile_implicit_grad,
    "opt_step": profile_opt_step,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("auto", "none", "cpu", "gpu"),
                    default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cases", default=None, help="comma-separated subset")
    args = ap.parse_args()
    names = args.cases.split(",") if args.cases else list(CASES)

    device = None if args.device == "none" else args.device
    backend = jax.default_backend()
    print(f"backend={backend}  requested_device={args.device}  "
          f"devices={[str(d) for d in jax.local_devices()]}")
    results = {"backend": backend, "requested_device": args.device,
               "platform": platform.platform(), "cases": {}}
    for name in names:
        rss0 = _peak_rss_gb()
        try:
            row = CASES[name](device)
        except Exception as exc:  # keep profiling the rest
            row = {"error": f"{type(exc).__name__}: {exc}"}
        row["peak_rss_gb"] = _peak_rss_gb()
        row["rss_delta_gb"] = row["peak_rss_gb"] - rss0
        gpu = _gpu_stats()
        if gpu and backend != "cpu":
            row["gpu_peak_gb"] = {k: v.get("peak_bytes_in_use", 0) / 1e9
                                  for k, v in gpu.items()}
        results["cases"][name] = row
        keys = ("cold_s", "warm_s", "iters", "ms_per_iter", "peak_rss_gb", "error")
        pretty = "  ".join(f"{k}={row[k]:.3g}" if isinstance(row.get(k), float)
                           else f"{k}={row[k]}" for k in keys if k in row)
        print(f"[{name:16s}] {pretty}", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
