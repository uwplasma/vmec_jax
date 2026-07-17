#!/usr/bin/env python
"""Precise QA in ONE least-squares call: no mode-continuation ladder, just ESS.

The staged examples (``QA_optimization.py`` etc.) walk ``max_mode`` 1 -> 5 so
the optimizer settles the long-wavelength shape before the fine harmonics are
released.  This script shows the alternative that makes the ladder
unnecessary: hand the optimizer *all* the max_mode-5 boundary harmonics at
once and let **Exponential Spectral Scaling** (``use_ess=True``) do the
ordering — the trust-region radius of each dof is scaled by
``exp(-alpha * max(|m|, |n|))``, so high harmonics move on exponentially
shorter leashes and the optimizer explores the same coarse-to-fine hierarchy
implicitly, in a single stage.

Physics setup is identical to ``QA_optimization.py`` (nfp=2 vacuum QA from a
near-circular torus, quasisymmetry + aspect + iota targets, implicit adjoint
gradients).  Measured 2026-07-12 on the office 36-core CPU (memo + block
Jacobian + perturbation warm start):

    seed QS 2.043e-01 -> final QS 7.155e-06 (aspect 6.000, iota 0.420)
    in ONE call, 868 s (14.5 min) — vs 1532 s for the staged 1->5 ladder
    reaching 3.7e-07.  Same precision class, ~1.8x faster, no ladder.
"""

import dataclasses
import os
from pathlib import Path

import numpy as np

import vmex as vj
from vmex import optimize as opt

# --------------------------- parameters ------------------------------------
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.minimal_seed_nfp2"
OUT_DIR = Path("output_QA_optimization_ess")
QS_SURFACES = np.linspace(0.1, 1.0, 10)
HELICITY_M, HELICITY_N = 1, 0             # QA: |B| = |B|(s, theta)
ASPECT_TARGET = 6.0
IOTA_TARGET = 0.42
SEED_PERTURBATION = 0.01                  # helical kick off the axisymmetric saddle
MAX_MODE = 5                              # ALL harmonics at once — no ladder
ESS_ALPHA = 0.7                           # trust-region decay per harmonic order
MAX_NFEV = 4000                           # single-stage budget (~2 ladder stages)
FTOL = 1e-8
JAC = "implicit"
if os.environ.get("VMEX_EXAMPLES_CI") == "1":  # smoke-test budget
    MAX_MODE, MAX_NFEV, FTOL = 2, 4, 1e-4

# --------------------------- seed equilibrium ------------------------------
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor + 1, 1] += SEED_PERTURBATION  # break the axisymmetric tie
zbs[inp.ntor + 1, 1] += SEED_PERTURBATION
inp = dataclasses.replace(inp, rbc=rbc, zbs=zbs)
qs = opt.QuasisymmetryRatioResidual(QS_SURFACES, HELICITY_M, HELICITY_N)


def report(tag, eq):
    total = float(qs.total(eq))
    print(f"[{tag}] QS total = {total:.6e}, "
          f"aspect = {float(opt.aspect_ratio(eq.state, eq.runtime)):.4f}, "
          f"mean iota = {float(opt.mean_iota(eq.state, eq.runtime)):.4f}")
    return total


qs_seed = report("seed", opt.solve_equilibrium(inp))

# --------------------------- objective (user-authored) ---------------------
objective_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (opt.mean_iota, IOTA_TARGET, 1.0),
]

# --------------------------- ONE least-squares call ------------------------
ndofs = len(opt.boundary_dof_names(inp, MAX_MODE))
print(f"\nsingle stage: max_mode = {MAX_MODE} ({ndofs} boundary dofs), "
      f"ESS alpha = {ESS_ALPHA}")
result = opt.least_squares(
    objective_terms, inp, max_mode=MAX_MODE, jac=JAC,
    use_ess=True, ess_alpha=ESS_ALPHA,      # <-- the ladder-replacement
    verbose=1, max_nfev=MAX_NFEV, ftol=FTOL, xtol=1e-10,
)
inp = result.input

# --------------------------- final results ---------------------------------
qs_final = qs_seed
if result.equilibrium is not None:
    qs_final = report("final", result.equilibrium)
print(f"\nQS total: seed {qs_seed:.3e} -> final {qs_final:.3e} "
      f"(one call, no max_mode ladder)")
OUT_DIR.mkdir(parents=True, exist_ok=True)
inp.to_indata(OUT_DIR / "input.QA_ess_optimized")
if result.equilibrium is not None:
    wout_path = vj.write_wout(OUT_DIR / "wout_QA_ess_optimized.nc",
                              result.equilibrium.wout)
    print(f"wrote {wout_path}")
