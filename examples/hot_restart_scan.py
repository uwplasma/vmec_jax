#!/usr/bin/env python
"""Hot restart across a parameter scan: reuse the converged state, recompile nothing.

A parameter scan solves a sequence of nearby equilibria.  Starting each one from
the previous converged state (a *hot restart*) means the solver begins a hair
from the answer and converges in a handful of iterations instead of hundreds.
Because vmex caches one compiled executable per solver *structure*, every
scan point at fixed resolution reuses it -- zero recompilation.

This script solves a base case cold, then scans the edge toroidal flux
(``phiedge``) and solves each point warm-started from its predecessor.  The
headline is the iteration count (machine-independent); wall time is printed too
but is only indicative on a shared CPU.

Physics: circular tokamak (nfp=1), fixed boundary, single radial grid so the
seed shape matches across the scan.  Runtime a few seconds.
"""

import dataclasses
import os
import time
from pathlib import Path

import numpy as np

import vmex as vj

# --------------------------- parameters ------------------------------------
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.circular_tokamak"
NS = 25                                   # single radial grid (fixed structure)
SCAN = np.linspace(0.95, 1.05, 5)         # phiedge multipliers around the base
CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if CI:
    NS = 15

base = vj.VmecInput.from_file(INPUT_FILE)
base = dataclasses.replace(base, ns_array=[NS], ftol_array=[1e-11], niter_array=[5000])
phiedge0 = float(base.phiedge)


def _solve(inp, seed):
    t0 = time.perf_counter()
    res = vj.solve_multigrid(inp, initial_state=seed, verbose=False)
    return res, time.perf_counter() - t0


# --------------------------- cold base solve -------------------------------
res0, dt0 = _solve(base, None)
print(f"cold base solve: {int(res0.iterations):4d} iters, {dt0:6.2f} s "
      f"(phiedge = {phiedge0:.4f})")
print(f"\n{'phiedge':>10s} {'iters':>6s} {'wall_s':>8s}   restart")
print(f"{'-'*10} {'-'*6} {'-'*8}   {'-'*12}")

# --------------------------- warm-restarted scan ---------------------------
seed = res0.state
cold_iters = []
for mult in SCAN:
    inp = dataclasses.replace(base, phiedge=phiedge0 * float(mult))
    res, dt = _solve(inp, seed)                 # warm: seed from previous point
    seed = res.state                            # carry the state forward
    # reference: how many iterations the SAME point needs cold (no seed)
    res_cold, _ = _solve(inp, None)
    cold_iters.append(int(res_cold.iterations))
    print(f"{phiedge0 * mult:10.4f} {int(res.iterations):6d} {dt:8.2f}   "
          f"warm ({int(res_cold.iterations)} cold)")

print(f"\nWarm restarts converge in far fewer iterations than cold "
      f"(cold ~{int(np.mean(cold_iters))} iters/point); the compiled executable "
      "is reused across every scan point.")
