#!/usr/bin/env python
"""Quasi-axisymmetric boundary optimization with a SciPy scalar-gradient method."""

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import OptimizationMonitor
from vmex import optimize as opt


nfp = 2  # number of field periods
SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODE = 3
MAXITER  = 200
METHOD   = "L-BFGS-B" # or "BFGS"
PARAMETER_BOUND = 1.0
BOUNDARY_STEP = 0.1   # typical change represented by one scaled variable
ASPECT_TARGET = 5.0
IOTA_TARGET   = 0.42
# MAGNETIC_WELL_TARGET = 0.01
MINIMUM_MPOL = 5
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.05

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODE, MAXITER = 1, 4

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
inp.rbc[inp.ntor-1, 1] =-SEED_PERTURBATION
inp.zbs[inp.ntor-1, 1] = SEED_PERTURBATION
mpol = max(MAX_MODE + 2, MINIMUM_MPOL)
inp = replace(inp, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)

# For QH use helicity_n=-1.
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (opt.mean_iota, IOTA_TARGET, 10.0),
    # (opt.magnetic_well, MAGNETIC_WELL_TARGET, 1.0),
]
problem = opt.VmecProblem.from_tuples(inp, objective_function_terms, max_mode=MAX_MODE,
                                      vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, progress=True)
print(f"dof_names = {problem.dof_names}")
problem.compile_value_and_gradient()
x0, scales = problem.x0, BOUNDARY_STEP * problem.scales

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"))

def x_from_y(y):
    return x0 + scales * y

def cost(y):
    return problem.fun(x_from_y(y))

def gradient(y):
    return scales * problem.grad(x_from_y(y))

monitor = OptimizationMonitor(problem)
def monitor_y(intermediate_result):
    monitor({"x": x_from_y(intermediate_result.x), "fun": intermediate_result.fun,
             "jac": gradient(intermediate_result.x)})

options = {"maxiter": MAXITER, "gtol": 1.0e-6}
if METHOD == "L-BFGS-B":
    options.update(maxls=20, ftol=1.0e-12, maxcor=20)
result = minimize(cost, np.zeros_like(x0), jac=gradient, method=METHOD,
                  bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x0.size if METHOD == "L-BFGS-B" else None,
                  callback=monitor_y, options=options)
result.x = x_from_y(result.x)
equilibrium = problem.equilibrium_from_x(result.x)
inp = problem.input_from_x(result.x)

# Print results
final_input = replace(inp,
    ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-14]),
    niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.state,
    verbose=not ci_smoke, raise_on_max_iterations=True)
final_total = report("final", final_equilibrium)["QS total"]
print(f"\n{METHOD}: final cost = {float(result.fun):.12e}, QS total = {final_total:.3e}")

# Save results
input_path = final_input.to_indata(f"input.QA_scipy_{METHOD}")
wout_path = vj.write_wout(f"wout_QA_scipy_{METHOD}.nc", final_equilibrium.wout)
print(f"wrote {input_path}")
print(f"wrote {wout_path}")

# Plot results
monitor.save(f"QA_scipy_{METHOD}_objectives.csv")
monitor.plot(f"QA_scipy_{METHOD}_objectives.png")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
