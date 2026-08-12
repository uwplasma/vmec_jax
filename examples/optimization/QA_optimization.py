#!/usr/bin/env python
"""Quasi-axisymmetric boundary optimization with a magnetic well."""

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"

DATA = Path(__file__).resolve().parents[1] / "data" / "input.minimal_seed_nfp2"
SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODES, MAX_NFEV = [3], [20]#[2, 3], [10, 40]
ASPECT_TARGET = 5.0
IOTA_TARGET = 0.42
MAGNETIC_WELL_TARGET = 0.01
MINIMUM_MPOL = 5

if ci_smoke: MAX_MODES, MAX_NFEV = [1], [4]

inp = vj.VmecInput.from_file(DATA)

# If QH use helicity_n=-1
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
         (qs, 0.0, 1.0), (opt.aspect_ratio, ASPECT_TARGET, 1.0),
         (opt.mean_iota, IOTA_TARGET, 10.0),
         (opt.magnetic_well, MAGNETIC_WELL_TARGET, 1.0),
         ]

def report(label, equilibrium):
    total = float(qs.total(equilibrium))
    print(f"[{label}] QS total = {total:.6e}, "
          f"aspect = {float(opt.aspect_ratio(equilibrium.state, equilibrium.runtime)):.4f}, "
          f"mean iota = {float(opt.mean_iota(equilibrium.state, equilibrium.runtime)):.4f}, "
          f"magnetic well = {float(opt.magnetic_well(equilibrium.state, equilibrium.runtime)):.4f}")
    return total

# Optimize for QA in stages, increasing the maximum mode number each time
seed = opt.solve_equilibrium(inp)
seed_total = report("seed", seed)
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== QA stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_tuples(inp, objective_function_terms, max_mode=max_mode, use_ess=True)
    result = least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=problem.scales, max_nfev=max_nfev,
        ftol=1e-6, xtol=1e-10, verbose=2
    )
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)
    # inp.to_indata(f"input.QA_max_mode_{max_mode:03d}")

# Print results, save and plot
final_input = replace(inp,
    ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-14]),
    niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.state,
    verbose=not ci_smoke, raise_on_max_iterations=True)
final_total = report("final", final_equilibrium)
print(f"\nQS total: seed {seed_total:.3e} -> final {final_total:.3e}")
input_path = inp.to_indata("input.QA_optimized")
wout_path = vj.write_wout("wout_QA_optimized.nc", final_equilibrium.wout)
print(f"wrote {input_path}\nwrote {wout_path}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
