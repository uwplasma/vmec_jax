#!/usr/bin/env python
"""Explore QA basins with SciPy basin hopping, then polish by least squares."""

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from scipy.optimize import basinhopping, least_squares

import vmex as vj
from vmex import optimize as opt

nfp = 1
SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODE, N_BASINS, LOCAL_MAXITER, POLISH_NFEV = 3, 10, 15, 30
ASPECT_TARGET, IOTA_FLOOR = 5.0, 0.37
BOUNDARY_STEP, PARAMETER_BOUND = 0.1, 1.0
# Smaller alpha damps high Fourier modes less than the robust default 1.2,
# helping basin exploration produce shapes unlike a low-mode local optimum.
ESS_ALPHA = 0.7
MINIMUM_MPOL, SEED_PERTURBATION = 5, 0.05
VARY_MAJOR_RADIUS = False

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODE, N_BASINS, LOCAL_MAXITER, POLISH_NFEV = 1, 1, 2, 3

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
mpol = max(MAX_MODE + 2, MINIMUM_MPOL)
inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)

# Floor the profile minimum, not its average: a mean target is satisfiable while
# an interior surface sits near zero transform, which is what a current-carried
# finite-beta profile does. opt.mean_iota targets the average instead, and
# opt.soft_min_abs_iota is the smooth-minimum variant.
def iota_floor(equilibrium_state, solver_context):
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)

qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 100.0),
    ]
problem = opt.VmecProblem.from_tuples(inp, objective_function_terms, max_mode=MAX_MODE,
    vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, ess_alpha=ESS_ALPHA, progress=True)
print(f"dof_names = {problem.dof_names}")
problem.compile_value_and_gradient()
x0, scales = problem.x0, BOUNDARY_STEP * problem.scales

def x_from_y(y):
    return x0 + scales * y

def value_and_gradient(y):
    value, gradient = problem.value_and_grad(x_from_y(y))
    return value, scales * gradient

monitor = opt.OptimizationMonitor(problem, stream=None)
best = {"y": np.zeros_like(x0), "value": np.inf}
def basin_report(y, value, accepted):
    gradient = value_and_gradient(y)[1]
    monitor({"x": x_from_y(y), "fun": value, "jac": gradient})
    if accepted and value < best["value"]:
        best.update(y=np.asarray(y).copy(), value=float(value))
    print(f"basin cost = {value:.6e}, accepted = {accepted}")

print("First print can take more than ten minutes")
bounds = [(-PARAMETER_BOUND, PARAMETER_BOUND)] * x0.size
basinhopping(value_and_gradient, np.zeros_like(x0), niter=N_BASINS,
    T=0.05, stepsize=0.25, minimizer_kwargs={"method": "L-BFGS-B", "jac": True,
        "bounds": bounds, "options": {"maxiter": LOCAL_MAXITER, "ftol": 1e-10}},
    callback=basin_report, rng=np.random.default_rng(7), disp=True)

# The global phase selects a basin; exact residual/Jacobian least squares then
# exploits the tuple structure for an efficient local finish.
x_global = x_from_y(best["y"])
polish = least_squares(problem.residual, x_global, jac=problem.residual_jac,
    x_scale=problem.scales, bounds=(x0 - scales, x0 + scales), max_nfev=POLISH_NFEV,
    ftol=1e-7, xtol=1e-10, verbose=2, callback=monitor)
inp, equilibrium = problem.input_from_x(polish.x), problem.equilibrium_from_x(polish.x)

final_input = replace(inp, ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1e-10 if ci_smoke else 1e-14]), niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)
report = opt.EquilibriumReporter(("QS", qs.total, ".4e"),
    ("aspect", opt.aspect_ratio, ".3f"), ("iota", opt.mean_iota, ".3f"))
report("final", final_equilibrium)

input_path = final_input.to_indata("input.QA_global_optimized")
wout_path = vj.write_wout("wout_QA_global_optimized.nc", final_equilibrium.wout)
print(f"wrote {input_path}\nwrote {wout_path}")
monitor.save("QA_global_objectives.csv"); monitor.plot("QA_global_objectives.png")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
