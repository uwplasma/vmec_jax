#!/usr/bin/env python
"""Profile the vmex production hot paths (R24).

Reports, for each case, a cold-vs-warm wall-time split (cold = XLA compile +
run, warm = run only) and peak RSS, for the two paths that dominate real use:

  1. fixed-boundary multigrid solve   (``vmex.solve_multigrid``)
  2. differentiable ``value_and_grad`` of ``wb`` (implicit forward + adjoint)

plus a cProfile of a warm solve to confirm Python-level orchestration overhead
is negligible (all compute lives in XLA).

Runs on whatever backend JAX is configured for, so it is the same script for
the CPU numbers in the docs and for a GPU box::

    JAX_ENABLE_X64=1 python tools/profile_hotpaths.py
    JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 python tools/profile_hotpaths.py --cases solovev

On CPU the dominant latency is compile time, which the persistent compilation
cache (on by default; see ``vmex._compat``) amortises across processes; the
warm numbers are the per-call steady state inside one process / an opt loop.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import os
import platform
import pstats
import resource
import time

import jax
import numpy as np

import vmex as vj
from vmex.core import implicit as im

DATA = os.path.join(os.path.dirname(__file__), os.pardir, "examples", "data")
# ru_maxrss is bytes on macOS, KiB on Linux.
_RSS_SCALE = 1.0 if platform.system() == "Darwin" else 1024.0
# per-case forward ftol / iteration cap for the differentiable value_and_grad.
_GRAD_CFG = {"solovev": dict(ftol=1e-14, max_iterations=2000),
             "li383_low_res": dict(ftol=1e-13, max_iterations=6000)}


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_SCALE / 1e6


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    return out, time.perf_counter() - t0


def _load(case: str) -> vj.VmecInput:
    return vj.VmecInput.from_file(os.path.join(DATA, f"input.{case}"))


def profile_solve(cases: list[str]) -> None:
    print("\n1. FIXED-BOUNDARY MULTIGRID SOLVE  (cold = compile+run, warm = run)")
    print(f"{'case':18s} {'cold s':>9} {'warm s':>9} {'compile s':>10} "
          f"{'iters':>7} {'peakRSS MB':>11}")
    for case in cases:
        inp = _load(case)
        _, t_cold = timed(lambda: vj.solve_multigrid(inp, verbose=False))
        res, t_warm = timed(lambda: vj.solve_multigrid(inp, verbose=False))
        print(f"{case:18s} {t_cold:9.2f} {t_warm:9.2f} {t_cold - t_warm:10.2f} "
              f"{int(res.iterations):7d} {peak_mb():11.1f}")


def profile_grad(cases: list[str]) -> None:
    print("\n2. DIFFERENTIABLE value_and_grad(wb)  (implicit forward + adjoint)")
    print(f"{'case':18s} {'cold s':>9} {'warm s':>9} {'compile s':>10} {'peakRSS MB':>11}")
    for case in cases:
        inp = _load(case)
        p0 = im.params_from_input(inp)
        cfg = _GRAD_CFG.get(case, dict(ftol=1e-12, max_iterations=4000))
        vg = jax.value_and_grad(lambda p: im.run(inp, p, **cfg).wb)
        (_, _), t_cold = timed(lambda: vg(p0))
        (v, g), t_warm = timed(lambda: vg(p0))
        print(f"{case:18s} {t_cold:9.2f} {t_warm:9.2f} {t_cold - t_warm:10.2f} "
              f"{peak_mb():11.1f}")
        print(f"   wb = {float(v):.9e}   |grad.rbc| = "
              f"{float(np.linalg.norm(np.asarray(g.rbc))):.3e}")


def profile_python_overhead(case: str) -> None:
    print("\n3. cProfile OF A WARM SOLVE  (top Python-level cumulative time)")
    inp = _load(case)
    vj.solve_multigrid(inp, verbose=False)          # warm the cache
    pr = cProfile.Profile()
    pr.enable()
    vj.solve_multigrid(inp, verbose=False)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(25)
    lines = buf.getvalue().splitlines()
    header = [ln for ln in lines if "ncalls" in ln]
    if header:
        print(header[0])
    shown = 0
    for ln in lines:
        if "vmex/" in ln and shown < 14:
            print(ln)
            shown += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="solovev,li383_low_res",
                    help="comma-separated deck names under examples/data/input.*")
    ap.add_argument("--no-grad", action="store_true", help="skip the value_and_grad path")
    ap.add_argument("--no-cprofile", action="store_true", help="skip the cProfile section")
    args = ap.parse_args()
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]

    print(f"JAX backend: {jax.default_backend()}   x64={jax.config.read('jax_enable_x64')}")
    profile_solve(cases)
    if not args.no_grad:
        profile_grad([c for c in cases if c == "solovev"] or cases[:1])
    if not args.no_cprofile:
        profile_python_overhead(cases[-1])
    print(f"\npeak RSS (whole run): {peak_mb():.1f} MB")


if __name__ == "__main__":
    main()
