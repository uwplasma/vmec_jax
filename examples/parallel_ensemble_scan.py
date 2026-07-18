#!/usr/bin/env python
"""Concurrent ensemble of independent solves: strong scaling on CPU.

A parameter scan or an ensemble optimization solves ``N`` *independent*
equilibria.  Because each vmex forward solve runs on the host and releases the
GIL while XLA executes its compiled iteration lanes, a plain thread pool
overlaps those solves and gives real wall-clock speedup -- with every result
byte-identical to solving that input alone.

This script builds a balanced ``phiedge`` scan (same structure, so the members
share one compiled executable) and times it serially and through
``vmex.parallel.solve_ensemble`` at a few worker counts, printing the
strong-scaling curve.  It also verifies bit-identical results against the
serial solve -- the correctness contract, not just the speed.

Physics: nfp=2 QA stellarator, fixed boundary, single radial grid so the
compiled structure is shared across the scan.  Runtime a few seconds.

The headline (speedup) is machine-dependent -- it scales with the free cores
on the box.  See ``docs/parallelization.rst`` for the full mechanism study and
the multi-GPU design.
"""

import dataclasses
import os
import time
from pathlib import Path

import numpy as np

import vmex as vj

# --------------------------- parameters ------------------------------------
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.nfp2_QA"
NS = 35                                    # single radial grid (shared structure)
N = 8                                      # ensemble size (phiedge scan points)
WORKER_COUNTS = (1, 2, 4)                  # thread counts to time
REPS = 2                                   # best-of REPS per configuration
CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if CI:                                     # smoke budget: tiny + cheap
    NS, N, WORKER_COUNTS, REPS = 15, 4, (1, 2), 1

base = vj.VmecInput.from_file(INPUT_FILE)
base = dataclasses.replace(base, ns_array=[NS], ftol_array=[1e-11], niter_array=[3000])
phiedge0 = float(base.phiedge)

# Balanced ensemble: a small phiedge scan around the base value.
inputs = [
    dataclasses.replace(base, phiedge=phiedge0 * (1.0 + 0.01 * (i - N / 2) / N))
    for i in range(N)
]


def _solve(inp):
    return vj.solve(inp, verbose=False)


def _max_state_diff(a, b):
    return max(
        float(np.max(np.abs(np.asarray(getattr(a, f)) - np.asarray(getattr(b, f)))))
        for f in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")
    )


# --------------------------- warm compile ----------------------------------
# Compile the shared executable once so the timings measure execution, not the
# one-off XLA compile.
_ = _solve(base)

# --------------------------- serial baseline -------------------------------
serial_out = [_solve(x) for x in inputs]


def _time_serial():
    t0 = time.perf_counter()
    [_solve(x) for x in inputs]
    return time.perf_counter() - t0


def _time_workers(w):
    t0 = time.perf_counter()
    vj.parallel.solve_ensemble(inputs, workers=w, multigrid=False, verbose=False)
    return time.perf_counter() - t0


ser = min(_time_serial() for _ in range(REPS))

# --------------------------- correctness check -----------------------------
# The whole point: threaded results are byte-identical to the serial solve.
ensemble = vj.parallel.solve_ensemble(inputs, multigrid=False, verbose=False)
max_diff = max(_max_state_diff(a.state, b.state) for a, b in zip(serial_out, ensemble))
iters_match = all(int(a.iterations) == int(b.iterations)
                  for a, b in zip(serial_out, ensemble))
print(f"correctness: max|state diff| vs serial = {max_diff:.1e}  "
      f"(iterations identical: {iters_match})")
assert max_diff == 0.0 and iters_match, "ensemble result must be bit-identical to serial"

# --------------------------- strong scaling --------------------------------
cpu = os.cpu_count()
print(f"\nstrong scaling  (N={N} solves, ns={NS}, {cpu} logical CPUs, best-of-{REPS})")
print(f"{'workers':>8} {'wall_s':>8} {'speedup':>8} {'eff%':>6}")
print(f"{'serial':>8} {ser:8.2f} {'1.00x':>8} {'100':>6}")
for w in WORKER_COUNTS:
    t = min(_time_workers(w) for _ in range(REPS))
    print(f"{w:8d} {t:8.2f} {ser / t:7.2f}x {100 * ser / t / w:5.0f}")

print("\nNote: sub-linear scaling is expected -- XLA already multithreads within "
      "each\nsolve, so ensemble workers and intra-solve threads share the same cores.")
