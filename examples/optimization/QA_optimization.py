#!/usr/bin/env python
"""Quasi-axisymmetric boundary optimization with a magnetic well."""

from dataclasses import replace
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt

nfp = 2  # number of field periods
SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODES, MAX_NFEV = [1,2,3], [10, 10, 15]
MAGNETIC_WELL_TARGET = 0.01
ASPECT_TARGET = 5.0
# MAX_MODES, MAX_NFEV = [1,2,3,4,5,6,7,8,9], [10, 10, 15, 20, 25, 40, 50, 60, 60]
# MAGNETIC_WELL_TARGET = 0.07
# ASPECT_TARGET = 3.5
IOTA_FLOOR = 0.42
PARAMETER_STEP, MAX_PARAMETER_CHANGE = 0.02, 5.0
ESS_ALPHA = 1.2  # smaller values let high Fourier modes move more
MINIMUM_MPOL = 5
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.05

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [4]

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
# The exactly circular torus has zero first-order iota sensitivity. This
# explicit rotating-ellipse perturbation gives the local optimizer a QA basin.
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)

# Floor the profile minimum, not its average: a mean target is satisfiable while
# an interior surface sits near zero transform, which is what a current-carried
# finite-beta profile does. opt.mean_iota targets the average instead, and
# opt.soft_min_abs_iota is the smooth-minimum variant.
def iota_floor(equilibrium_state, solver_context):
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)

# Objective function terms
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
         (qs, 0.0, 1.0),
         (opt.aspect_ratio, ASPECT_TARGET, 1.0),
         (iota_floor, 0.0, 10.0),
         (opt.magnetic_well, MAGNETIC_WELL_TARGET, 1.0),
         ]

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"))
monitor = opt.OptimizationMonitor(stream=None)

# Optimize for QA first, then add the pressure-stability proxy locally.
equilibrium = opt.solve_equilibrium(inp)
# If a RuntimeWarning reports uncertified Jacobian columns, it is expected
# once the optimizer leaves the seed and needs no action: the shipped
# jacobian_adjoint_tol=1e-4 and jacobian_adjoint_maxiter=10 are the measured
# optimum, since ten times that budget moved the Jacobian by 2e-8 and
# certified no extra column. Both are from_tuples arguments; pass
# evaluation_progress=False to drop the per-evaluation timing lines.
for stage, (max_mode, max_nfev) in enumerate(zip(MAX_MODES, MAX_NFEV)):
    print(f"\n===== QA stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    stage_terms = objective_function_terms
    problem = opt.VmecProblem.from_tuples(inp, stage_terms, max_mode=max_mode,
        vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, ess_alpha=ESS_ALPHA,
        restart_from=equilibrium)
    print(f"dof_names = {problem.dof_names}")
    monitor.problem = problem
    if not ci_smoke:
        problem.compile_residual_and_jacobian()
    step = PARAMETER_STEP * problem.scales
    result = least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=step,max_nfev=max_nfev, bounds=(
                problem.x0 - MAX_PARAMETER_CHANGE * step,
                problem.x0 + MAX_PARAMETER_CHANGE * step),
        ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor
    )
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)
    # inp.to_indata(f"input.QA_max_mode_{max_mode:03d}")

# Print results
final_input = replace(inp,
    ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-14]),
    niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)
final_total = report("final", final_equilibrium)["QS total"]
print(f"\nQS total {final_total:.3e}")

vacuum_name = "QA_optimized"
vacuum_input_path = final_input.to_indata(f"input.{vacuum_name}")
vacuum_wout_path = vj.write_wout(f"wout_{vacuum_name}.nc", final_equilibrium.wout)
print(f"wrote {vacuum_input_path}\nwrote {vacuum_wout_path}")

# Plot results
monitor.save("QA_optimization_objectives.csv")
monitor.plot("QA_optimization_objectives.png")
for path in vj.plot_wout(vacuum_wout_path, ".").values():
    print(f"wrote {path}")