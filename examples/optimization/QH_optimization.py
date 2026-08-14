#!/usr/bin/env python
"""Quasi-helically symmetric boundary optimization with a magnetic well."""

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt

nfp = 4  # number of field periods
SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODES = [2,3]
MAX_NFEV = [15, 15]
ASPECT_TARGET = 6.0
# IOTA_TARGET = -1.1
# MAGNETIC_WELL_TARGET = 0.01
MINIMUM_MPOL = 5
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.12

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke: MAX_MODES, MAX_NFEV = [1], [4]

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
# A circular torus cannot acquire iota to first order. Seed a rotating ellipse
# explicitly so the local optimization starts in the QH basin.
inp.rbc[inp.ntor-1, 1] =-SEED_PERTURBATION
inp.zbs[inp.ntor-1, 1] = SEED_PERTURBATION

# Objective function terms
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=-1)
objective_function_terms = [
         (qs, 0.0, 1.0),
         (opt.aspect_ratio, ASPECT_TARGET, 1.0),
        #  (opt.mean_iota, IOTA_TARGET, 10.0),
        #  (opt.magnetic_well, MAGNETIC_WELL_TARGET, 10.0),
         ]

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"))

# Optimize for QH in stages, increasing the maximum mode number each time
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== QH stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_tuples(inp, objective_function_terms, max_mode=max_mode,
                                          vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True)
    print(f"dof_names = {problem.dof_names}")
    if not ci_smoke: problem.compile_residual_and_jacobian()
    result = least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=problem.scales, max_nfev=max_nfev,
        ftol=1e-6, xtol=1e-10, verbose=2
    )
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)
    # inp.to_indata(f"input.QH_max_mode_{max_mode:03d}")

# Print results
final_input = replace(inp,
    ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-14]),
    niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.state,
    verbose=not ci_smoke, raise_on_max_iterations=True)
final_total = report("final", final_equilibrium)["QS total"]
print(f"\nQS total {final_total:.3e}")

# Save results
input_path = final_input.to_indata("input.QH_optimized")
wout_path = vj.write_wout("wout_QH_optimized.nc", final_equilibrium.wout)
print(f"wrote {input_path}\nwrote {wout_path}")

# Plot results
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
