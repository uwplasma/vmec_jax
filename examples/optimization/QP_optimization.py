#!/usr/bin/env python
"""Quasi-poloidal boundary optimization with an explicit mode ladder."""

from dataclasses import replace
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt

nfp = 2  # number of field periods
SURFACES = np.array([0.5, 0.7, 0.9])
MAX_MODES, MAX_NFEV = [3,4,5], [15,15,30]  # mode-ladder alternative: [1, 2, 3], [20, 20, 20]
ASPECT_TARGET = 7.0
IOTA_FLOOR = 0.51
MIRROR_LIMIT = 0.35
ELONGATION_LIMIT = 12.0
ESS_ALPHA = 1.2  # smaller values let high Fourier modes move more
MINIMUM_MPOL = 5
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.05

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [4]

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5,
              niter_array=np.array([300, 8000]),
              ftol_array=np.array([1.0e-11, 1e-12]),
              ns_array=np.array([25, 35]))

# Objective function terms
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=0, helicity_n=1)

def mirror_excess(equilibrium_state, solver_context):
    return jnp.maximum(
        opt.mirror_ratio(equilibrium_state, solver_context) - MIRROR_LIMIT, 0.0)

# Floor the profile minimum, not its average: a mean target is satisfiable while
# an interior surface sits near zero transform, which is what a current-carried
# finite-beta profile does. opt.mean_iota targets the average instead, and
# opt.soft_min_abs_iota is the smooth-minimum variant.
def iota_floor(equilibrium_state, solver_context):
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)

def elongation_excess(equilibrium_state, solver_context):
    return jnp.maximum(
        opt.max_elongation(equilibrium_state, solver_context) - ELONGATION_LIMIT, 0.0)

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("min |iota|", opt.min_abs_iota, ".4f"), ("elongation", opt.max_elongation, ".4f"),
    ("mirror", opt.mirror_ratio, ".4f"))
monitor = opt.OptimizationMonitor(stream=None)

objective_function_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 100.0),
    (mirror_excess, 0.0, 10.0),
    (elongation_excess, 0.0, 10.0)]

for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== QP stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_tuples(inp, objective_function_terms, max_mode=max_mode,
        vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, ess_alpha=ESS_ALPHA)
    print(f"dof_names = {problem.dof_names}")
    monitor.problem = problem
    if not ci_smoke:
        problem.compile_residual_and_jacobian()
    result = least_squares(problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=problem.scales, max_nfev=max_nfev, ftol=1e-6, xtol=1e-10,
        verbose=2, callback=monitor)
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)
    inp.to_indata(f"input.QP_max_mode_{max_mode:03d}")

# Print results
final_input = replace(inp,
    ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-14]),
    niter_array=np.array([35000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)
final_total = report("final", final_equilibrium)["QS total"]
print(f"\nQS total {final_total:.3e}")

# Save results
input_path = final_input.to_indata("input.QP_optimized")
wout_path = vj.write_wout("wout_QP_optimized.nc", final_equilibrium.wout)
print(f"wrote {input_path}")
print(f"wrote {wout_path}")

# Plot results
monitor.save("QP_optimization_objectives.csv")
monitor.plot("QP_optimization_objectives.png")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
